from ldm.models.diffusion.ddpm import LatentDiffusion
import torch
import torch.nn.functional as F
from ldm.util import default,instantiate_from_config

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class LatentDiffusionWaveletAttn(LatentDiffusion):
    """
    Latent Diffusion model using wavelet maps as cross-attention condition.
    The input is a bicubic-downsampled single-channel image, and the GT latent is computed from the original.
    """
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 structcond_stage_config=None,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 unfrozen_first_stage=True,
                 unfrozen_unet=False,
                 unfrozen_cond_stage=True,
                 *args, **kwargs):
        
        self.unfrozen_first_stage = unfrozen_first_stage
        self.unfrozen_unet = unfrozen_unet
        self.unfrozen_cond_stage = unfrozen_cond_stage

        super().__init__(first_stage_config=first_stage_config,
                         cond_stage_config=cond_stage_config,
                         num_timesteps_cond=num_timesteps_cond,
                         cond_stage_key=cond_stage_key,
                         cond_stage_trainable=cond_stage_trainable,
                         concat_mode=concat_mode,
                         cond_stage_forward=cond_stage_forward,
                         conditioning_key=conditioning_key,
                         scale_factor=scale_factor,
                         scale_by_std=scale_by_std,
                         *args, **kwargs)

        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)

        print('>>>>>>>>>>>>>>>> model >>>>>>>>>>>>>>>>>>')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)

        print('>>>>>>>>>>>>>>>> cond_stage_model >>>>>>>>>>>>>>>>>>')
        for name, param in self.cond_stage_model.named_parameters():
            if param.requires_grad:
                print(name)

        if hasattr(self, 'structcond_stage_model'):
            print('>>>>>>>>>>>>>>>> structcond_stage_model >>>>>>>>>>>>>>>>>>')
            for name, param in self.structcond_stage_model.named_parameters():
                if param.requires_grad:
                    print(name)
 

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model
        if not self.unfrozen_first_stage:
            self.first_stage_model.eval()
            self.first_stage_model.train = disabled_train
            for p in self.first_stage_model.parameters():
                p.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model
                if not self.ufrozen_cond_stage:
                    self.cond_stage_model.eval()
                    self.cond_stage_model.train = disabled_train
                    for p in self.cond_stage_model.parameters():
                        p.requires_grad = False
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        import re
        import traceback
        print(">>> [DDPM] init_from_ckpt() called <<<")
        traceback.print_stack(limit=5)       
        print(f"Model ID (self): {id(self)}")

        # 添加额外自动忽略 Cross-Attn 参数（比如 to_k 和 to_v）
        auto_skip_patterns = [
            "model.diffusion_model.*.attn2.to_k",
            "model.diffusion_model.*.attn2.to_v"
        ]
        # 加载一次 ckpt 的 state_dict，匹配 key 用
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        all_keys = list(sd.keys())

        # 匹配 auto keys → 得到前缀
        auto_ignore = set()
        for pattern in auto_skip_patterns:
            regex = re.compile("^" + pattern.replace(".", r"\.").replace("*", ".*"))
            for k in all_keys:
                if regex.match(k):
                    print(f"[AutoIgnore] Will skip key: {k}")
                    auto_ignore.add(k.split('.')[0])  # 添加顶层前缀

        # 合并手动配置的 ignore_keys 和自动推断的 auto_ignore
        expanded_ignore = list(set(ignore_keys) | auto_ignore)
        # 调用父类实现
        super().init_from_ckpt(path, ignore_keys=expanded_ignore, only_model=only_model)
        
        n_trainable = sum(p.numel() for p in self.first_stage_model.parameters() if p.requires_grad)
        print(f"First stage trainable params: {n_trainable:,}")

        n_trainable = sum(p.numel() for p in self.cond_stage_model.parameters() if p.requires_grad)
        print(f"Cond stage trainable params: {n_trainable:,}")
        breakpoint()

   # def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
    #         print(f"[WaveletAttn] Loading checkpoint from {path}")
            
    #         # 强制忽略 encoder 参数（以支持单通道输入）
    #         additional_ignores = ['first_stage_model.encoder']
    #         for k in additional_ignores:
    #             if k not in ignore_keys:
    #                 ignore_keys.append(k)

    #     # 调用父类的加载方法（实际上在 DDPM 中实现）
    #     super().init_from_ckpt(path, ignore_keys=ignore_keys, only_model=only_model)

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
