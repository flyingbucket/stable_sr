from ldm.models.diffusion.ddpm import LatentDiffusion
import torch
import torch.nn.functional as F
from ldm.util import default


class LatentDiffusionWaveletAttn(LatentDiffusion):
    """
    Latent Diffusion model using wavelet maps as cross-attention condition.
    The input is a bicubic-downsampled single-channel image, and the GT latent is computed from the original.
    """
    def get_input(self, batch, k=None, return_first_stage_outputs=False, force_c_encode=False,
                cond_key=None, return_original_cond=False, bs=None):
        x_lq_up = batch["lq_image"].to(self.device).float()   # 已上采样的退化图
        x_gt = batch["gt_image"].to(self.device).float()      # 原始图
        wavelet_cond = batch["wavelet"].to(self.device).float()

        # encode
        z = self.get_first_stage_encoding(self.encode_first_stage(x_lq_up)).detach()
        z_gt = self.get_first_stage_encoding(self.encode_first_stage(x_gt)).detach()
        c = self.get_learned_conditioning(wavelet_cond)

        out = [z, c, z_gt]

        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x_lq_up, x_gt, xrec])
        if return_original_cond:
            out.append(wavelet_cond)
        return out

    # def get_input(self, batch, k=None, return_first_stage_outputs=False, force_c_encode=False,
    #               cond_key=None, return_original_cond=False, bs=None):
    #     # Extract original GT image (assume key 'image')
    #     x_gt = batch[self.first_stage_key].to(self.device).float()  # [B, 1, H, W]

    #     # Downsample + upsample (simulate simple degradation)
    #     scale_factor = 1 / self.scale_factor if hasattr(self, 'scale_factor') else 0.25
    #     x_lq = F.interpolate(x_gt, scale_factor=scale_factor, mode='bicubic', align_corners=False)
    #     x_lq_up = F.interpolate(x_lq, size=x_gt.shape[-2:], mode='bicubic', align_corners=False)

    #     # Encode LQ image (input to diffusion)
    #     encoder_lq = self.encode_first_stage(x_lq_up)
    #     z_lq = self.get_first_stage_encoding(encoder_lq).detach()

    #     # Encode GT image (used as denoising target)
    #     encoder_gt = self.encode_first_stage(x_gt)
    #     z_gt = self.get_first_stage_encoding(encoder_gt).detach()

    #     # Conditioning from wavelet map
    #     wavelet_cond = batch["wavelet"].to(self.device).float()
    #     c = self.get_learned_conditioning(wavelet_cond)

    #     out = [z_lq, c, z_gt]

    #     if return_first_stage_outputs:
    #         xrec = self.decode_first_stage(z_lq)
    #         out.extend([x_lq_up, x_gt, xrec])
    #     if return_original_cond:
    #         out.append(wavelet_cond)

    #     return out

    def shared_step(self, batch, **kwargs):
        x, c,gt = self.get_input(batch, self.first_stage_key)
        loss = self(x, c,gt, **kwargs)
        return loss

    def forward(self, x, c, gt, *args, **kwargs):
        """
        Override forward to provide GT latent (z_gt) for loss target.
        """
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
        return self.p_losses(x, c, t, gt=gt, *args, **kwargs)

    def p_losses(self, x_start, cond, t,  noise=None,**kwargs):
        """
        Override to use GT latent (z_gt) as target instead of x_start itself.
        """
        assert kwargs.get('gt') is not None, "Must provide ground truth latent (z_gt) for loss calculation"
        gt = kwargs['gt']
        noise = default(noise, lambda: torch.randn_like(gt))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        # select target
        if self.parameterization == "x0":
            target = gt
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(gt, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        # logvar optional
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        # optional ELBO loss
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def get_learned_conditioning(self, c):
        """
        If cond_stage_model is identity or None, directly use the tensor.
        """
        if self.cond_stage_model is None or isinstance(self.cond_stage_model, torch.nn.Identity):
            return c
        else:
            return super().get_learned_conditioning(c)
