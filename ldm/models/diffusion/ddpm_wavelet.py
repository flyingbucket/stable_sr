from ldm.models.diffusion.ddpm import LatentDiffusion
import torch
import torch.nn.functional as F
from ldm.util import default,instantiate_from_config
from einops import repeat,rearrange
from torchvision.utils import make_grid
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
# from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
# from contextlib import contextmanager
# from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.util import default, instantiate_from_config
# from ldm.modules.ema import LitEma
# from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import IdentityFirstStage, AutoencoderKL
# from ldm.models.diffusion.ddim import DDIMSampler

# from basicsr.utils import DiffJPEG, USMSharp
# from basicsr.utils.img_process_util import filter2D
# from basicsr.data.transforms import paired_random_crop, triplet_random_crop
# from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt, bivariate_Gaussian
# import random

# from ldm.modules.diffusionmodules.util import make_ddim_timesteps
# import copy
# import os
# import cv2
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class LatentDiffusionWaveletCS(LatentDiffusion):
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
                 only_model=False,
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
        only_model = kwargs.pop("only_model", False)    
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model=only_model)

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
        # import traceback
        # print(">>> [DDPM] init_from_ckpt() called <<<")
        # traceback.print_stack(limit=5)       
        # print(f"Model ID (self): {id(self)}")

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

    def get_input(self, batch, k=None, return_first_stage_outputs=False, force_c_encode=False,
                cond_key=None, return_original_cond=False, bs=None):

        # 图像数据加载
        x_lq_up = batch["lq_image"].to(self.device).float()
        x_gt = batch["gt_image"].to(self.device).float()
        wavelet_cond = batch["wavelet"].to(self.device).float()

        # 编码图像为 latent
        z = self.get_first_stage_encoding(self.encode_first_stage(x_lq_up)).detach()
        z_gt = self.get_first_stage_encoding(self.encode_first_stage(x_gt)).detach()

        # 结构条件（来自 wavelet 子带）
        struct_cond = self.get_learned_conditioning(wavelet_cond)  # [B, C, 64, 64]

        lq_cond = self.get_first_stage_encoding(self.encode_first_stage(x_lq_up)).detach()  # [B, C, 64, 64]
        # 构造最终条件 dict（不含文本）
        c = {
            "c_crossattn": [torch.zeros((z.shape[0], 77, 768), device=z.device)],
            "c_concat": [lq_cond],     # 参与拼接
            "struct_cond": struct_cond # 参与结构引导
        }
        if c['struct_cond'] is None:
            raise ValueError("Conditioning 'struct_cond' is None in get_input.")

        out = [z, c, z_gt]

        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z_gt)
            out.extend([x_lq_up, x_gt, xrec])
        if return_original_cond:
            out.append(wavelet_cond)
        return out

    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=False,
                   plot_diffusion_rows=True, **kwargs):

        # use_ddim = ddim_steps is not None
        use_ddim=False

        log = dict()
        outs = self.get_input(batch, self.first_stage_key,
                      return_first_stage_outputs=True,
                      force_c_encode=True,
                      return_original_cond=True,
                      bs=N)

        z, c, z_gt, x_lq_up, x_gt, xrec = outs[:6]  # 不需要 wavelet_cond
        # z, c, _, xc, x, xrec = outs[:6]  # 注意顺序重新对齐
        c["c_crossattn"] = c["c_crossattn"][0]  # from [tensor] to tensor
        print(type(c["c_crossattn"]))  # ensure c_crossattn is a tensor
        x = z_gt# GT image
        xc = x_lq_up  # conditioning image
        print(f"x.shape = {x.shape}") 
        print(f"c_concat shape = {c['c_concat'][0].shape}")
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs_hq"] = x_gt
        log["recon_hq"] = xrec
        log["input_lq"] = x_lq_up
        log["recon_lq"] = self.decode_first_stage(z)
        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                             ddim_steps=ddim_steps,eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):

                    samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples


        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
    
    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
            verbose=True, timesteps=None, quantize_denoised=False,
            mask=None, x0=None, shape=None, **kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size//8, self.image_size//8)

        struct_cond = cond.pop("struct_cond", None)  # 提前分离出来

        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                        list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        return self.p_sample_loop(cond, shape,
                                return_intermediates=return_intermediates, x_T=x_T,
                                verbose=verbose, timesteps=timesteps,
                                quantize_denoised=quantize_denoised,
                                mask=mask, x0=x0,
                                struct_cond=struct_cond)  # 明确传入

    def sample_log(self,cond,batch_size,ddim=False, ddim_steps=10,**kwargs):

        # if ddim:
        #     raise NotImplementedError("DDIM sampling is not implemented in this version of the code.")
        #     ddim_sampler = DDIMSampler(self)
        #     shape = (self.channels, self.image_size, self.image_size)
        #     samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
        #                                                 shape,cond,verbose=False,**kwargs)

        # else:
        #     samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
        #                                          return_intermediates=True,**kwargs)
        # print("[WARNING] We should not have to concat c_concat to itself to satisfy the input channels of the model.")
        # if self.model.conditioning_key=='concat':
        #     c_concat = cond['c_concat'][0]
        #     cond['c_concat'] = [torch.cat([c_concat, c_concat], dim=1)]
        # else:
        #     raise NotImplementedError(f"Conditioning key '{self.conditioning_key}' not supported for sampling.")
        samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                return_intermediates=True,**kwargs)

        return samples, intermediates
    def shared_step(self, batch, **kwargs):
        x, c,gt = self.get_input(batch, self.first_stage_key)
        loss = self(x, c,gt, **kwargs)
        return loss

    def forward(self, x, c, gt, *args, **kwargs):
        """
        Override forward to provide GT latent (z_gt) for loss target.
        """
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()

        # 如果 conditioning_key 不为空，条件必需存在
        if self.model.conditioning_key is not None:
            assert c is not None

            if self.cond_stage_trainable and not isinstance(c, dict):
                c = self.get_learned_conditioning(c)

        # 执行扩散损失计算
        return self.p_losses(x, c, t, gt=gt, *args, **kwargs)

    def p_losses(self, x_start, cond, t,  noise=None,**kwargs):
        """
        Override to use GT latent (z_gt) as target instead of x_start itself.
        """
        assert kwargs.get('gt') is not None, "Must provide ground truth latent (z_gt) for loss calculation"
        gt = kwargs['gt']
        noise = default(noise, lambda: torch.randn_like(gt))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # if self.model.conditioning_key in ['concat', 'hybird']:
        #     x_noisy = torch.cat([x_noisy, cond['c_concat'][0]], dim=1)
        if cond['struct_cond'] is None:
            raise ValueError("Conditioning 'struct_cond' is None in p_losses.")
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
        t_cpu = t.detach().cpu()  
        logvar_t = self.logvar[t_cpu].to(self.device)  # this version is compatible with torch 1.13 and newer
        # logvar_t = self.logvar[t].to(self.device) # using pip requirements.txt,torch 1.13 and newer version will be installed causing error

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

    def apply_model(self, x_noisy, t, cond, return_ids=False,struct_cond=None):
        # print("apply_model cond keys:", cond.keys())

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
       
        if not struct_cond:
            struct_cond= cond.get('struct_cond', None)
            assert struct_cond is not None, "Conditioning 'struct_cond' must be provided in apply_model."
        else: 
            struct_cond=struct_cond
            assert struct_cond is not None, "Conditioning 'sturct_cond' must be provided in apply_model."
        
        c_concat = cond.get('c_concat', None)
        c_crossattn = cond.get('c_crossattn', None)
        x_recon = self.model(x_noisy, t, c_concat=c_concat, c_crossattn=c_crossattn, struct_cond=struct_cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def p_mean_variance(self, x, c, t, clip_denoised: bool,
                    return_codebook_ids=False, quantize_denoised=False,
                    return_x0=False, score_corrector=None, corrector_kwargs=None,
                    struct_cond=None):  

        t_in = t

        model_out = self.apply_model(x, t_in, c, struct_cond=struct_cond, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        # 处理 output 的 parameterization
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        elif self.parameterization == "v":
            x_recon = self.predict_start_from_z_and_v(x, model_out, t)
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)

        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                struct_cond=None):  # 添加 struct_cond 参数
        b, *_, device = *x.shape, x.device

        # 传入 struct_cond 给 p_mean_variance
        outputs = self.p_mean_variance(
            x=x, c=c, t=t, clip_denoised=clip_denoised,
            return_codebook_ids=return_codebook_ids,
            quantize_denoised=quantize_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            struct_cond=struct_cond  # 显式传入
        )

        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, cond, shape, return_intermediates=False,
                    x_T=None, verbose=True, callback=None, timesteps=None,
                    quantize_denoised=False, mask=None, x0=None, img_callback=None,
                    start_T=None, log_every_t=None, struct_cond=None):  # 新增 struct_cond

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]

        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]

        if timesteps is None:
            timesteps = self.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)

        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)

            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(
                img, cond, ts,
                clip_denoised=self.clip_denoised,
                quantize_denoised=quantize_denoised,
                struct_cond=struct_cond  # 显式传入
            )

            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img


class LatentDiffusionOriSO(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 structcond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 unfrozen_diff=False,
                 random_size=False,
                 test_gt=False,
                 p2_gamma=None,
                 p2_k=None,
                 time_replace=None,
                 use_usm=False,
                 mix_ratio=0.0,
                 *args, **kwargs):
        # put this in your init
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        self.unfrozen_diff = unfrozen_diff
        self.random_size = random_size
        self.test_gt = test_gt
        self.time_replace = time_replace
        self.use_usm = use_usm
        self.mix_ratio = mix_ratio
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.instantiate_structcond_stage(structcond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        if not self.unfrozen_diff:
            self.model.eval()
            # self.model.train = disabled_train
            for name, param in self.model.named_parameters():
                if 'spade' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        print('>>>>>>>>>>>>>>>>model>>>>>>>>>>>>>>>>>>>>')
        param_list = []
        for name, params in self.model.named_parameters():
            if params.requires_grad:
                param_list.append(name)
        print(param_list)
        param_list = []
        print('>>>>>>>>>>>>>>>>>cond_stage_model>>>>>>>>>>>>>>>>>>>')
        for name, params in self.cond_stage_model.named_parameters():
            if params.requires_grad:
                param_list.append(name)
        print(param_list)
        param_list = []
        print('>>>>>>>>>>>>>>>>structcond_stage_model>>>>>>>>>>>>>>>>>>>>')
        for name, params in self.structcond_stage_model.named_parameters():
            if params.requires_grad:
                param_list.append(name)
        print(param_list)

        # P2 weighting: https://github.com/jychoi118/P2-weighting
        if p2_gamma is not None:
            assert p2_k is not None
            self.p2_gamma = p2_gamma
            self.p2_k = p2_k
            self.snr = 1.0 / (1 - self.alphas_cumprod) - 1
        else:
            self.snr = None

        # Support time respacing during training
        if self.time_replace is None:
            self.time_replace = kwargs['timesteps']
        use_timesteps = set(space_timesteps(kwargs['timesteps'], [self.time_replace]))
        last_alpha_cumprod = 1.0
        new_betas = []
        timestep_map = []
        for i, alpha_cumprod in enumerate(self.alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)
        new_betas = [beta.data.cpu().numpy() for beta in new_betas]
        self.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas), linear_start=kwargs['linear_start'], linear_end=kwargs['linear_end'])
        self.ori_timesteps = list(use_timesteps)
        self.ori_timesteps.sort()

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                # self.cond_stage_model.train = disabled_train
                for name, param in self.cond_stage_model.named_parameters():
                    if 'final_projector' not in name:
                        param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model
            self.cond_stage_model.train()

    def instantiate_structcond_stage(self, config):
        model = instantiate_from_config(config)
        self.structcond_stage_model = model
        self.structcond_stage_model.train()

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                            force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch, taken from Real-ESRGAN:
        https://github.com/xinntao/Real-ESRGAN

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if b == self.configs.data.params.batch_size:
            if not hasattr(self, 'queue_size'):
                self.queue_size = self.configs.data.params.train.params.get('queue_size', b*50)
            if not hasattr(self, 'queue_lr'):
                assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
                self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
                _, c, h, w = self.gt.size()
                self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
                self.queue_ptr = 0
            if self.queue_ptr == self.queue_size:  # the pool is full
                # do dequeue and enqueue
                # shuffle
                idx = torch.randperm(self.queue_size)
                self.queue_lr = self.queue_lr[idx]
                self.queue_gt = self.queue_gt[idx]
                # get first b samples
                lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
                gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
                # update the queue
                self.queue_lr[0:b, :, :, :] = self.lq.clone()
                self.queue_gt[0:b, :, :, :] = self.gt.clone()

                self.lq = lq_dequeue
                self.gt = gt_dequeue
            else:
                # only do enqueue
                self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
                self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
                self.queue_ptr = self.queue_ptr + b

    def randn_cropinput(self, lq, gt, base_size=[64, 128, 256, 512]):
        cur_size_h = random.choice(base_size)
        cur_size_w = random.choice(base_size)
        init_h = lq.size(-2)//2
        init_w = lq.size(-1)//2
        lq = lq[:, :, init_h-cur_size_h//2:init_h+cur_size_h//2, init_w-cur_size_w//2:init_w+cur_size_w//2]
        gt = gt[:, :, init_h-cur_size_h//2:init_h+cur_size_h//2, init_w-cur_size_w//2:init_w+cur_size_w//2]
        assert lq.size(-1)>=64
        assert lq.size(-2)>=64
        return [lq, gt]

    @torch.no_grad()
    def get_input(self, batch, k=None, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, val=False, text_cond=[''], return_gt=False, resize_lq=True):

        """Degradation pipeline, modified from Real-ESRGAN:
        https://github.com/xinntao/Real-ESRGAN
        """

        if not hasattr(self, 'jpeger'):
            jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        if not hasattr(self, 'usm_sharpener'):
            usm_sharpener = USMSharp().cuda()  # do usm sharpening

        im_gt = batch['gt'].cuda()
        if self.use_usm:
            im_gt = usm_sharpener(im_gt)
        im_gt = im_gt.to(memory_format=torch.contiguous_format).float()
        kernel1 = batch['kernel1'].cuda()
        kernel2 = batch['kernel2'].cuda()
        sinc_kernel = batch['sinc_kernel'].cuda()

        ori_h, ori_w = im_gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(im_gt, kernel1)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.configs.degradation['resize_prob'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.configs.degradation['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.configs.degradation['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        gray_noise_prob = self.configs.degradation['gray_noise_prob']
        if random.random() < self.configs.degradation['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.configs.degradation['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.configs.degradation['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        if self.configs.degradation['use_jpeg']:
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if random.random() < self.configs.degradation['second_blur_prob']:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                self.configs.degradation['resize_prob2'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, self.configs.degradation['resize_range2'][1])
        elif updown_type == 'down':
            scale = random.uniform(self.configs.degradation['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(int(ori_h / self.configs.sf * scale),
                      int(ori_w / self.configs.sf * scale)),
                mode=mode,
                )
        # add noise
        gray_noise_prob = self.configs.degradation['gray_noise_prob2']
        if random.random() < self.configs.degradation['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.configs.degradation['noise_range2'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.configs.degradation['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False,
                )

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if random.random() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // self.configs.sf,
                          ori_w // self.configs.sf),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            if self.configs.degradation['use_jpeg']:
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            if self.configs.degradation['use_jpeg']:
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                    out,
                    size=(ori_h // self.configs.sf,
                          ori_w // self.configs.sf),
                    mode=mode,
                    )
            out = filter2D(out, sinc_kernel)

        # clamp and round
        im_lq = torch.clamp(out, 0, 1.0)

        # random crop
        gt_size = self.configs.degradation['gt_size']
        im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, self.configs.sf)
        self.lq, self.gt = im_lq, im_gt

        if resize_lq:
            self.lq = F.interpolate(
                    self.lq,
                    size=(self.gt.size(-2),
                          self.gt.size(-1)),
                    mode='bicubic',
                    )

        if random.random() < self.configs.degradation['no_degradation_prob'] or torch.isnan(self.lq).any():
            self.lq = self.gt

        # training pair pool
        if not val and not self.random_size:
            self._dequeue_and_enqueue()
        # sharpen self.gt again, as we have changed the self.gt with self._dequeue_and_enqueue
        self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        self.lq = self.lq*2 - 1.0
        self.gt = self.gt*2 - 1.0

        if self.random_size:
            self.lq, self.gt = self.randn_cropinput(self.lq, self.gt)

        self.lq = torch.clamp(self.lq, -1.0, 1.0)

        x = self.lq
        y = self.gt
        if bs is not None:
            x = x[:bs]
            y = y[:bs]
        x = x.to(self.device)
        y = y.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        encoder_posterior_y = self.encode_first_stage(y)
        z_gt = self.get_first_stage_encoding(encoder_posterior_y).detach()

        xc = None
        if self.use_positional_encodings:
            assert NotImplementedError
            pos_x, pos_y = self.compute_latent_shifts(batch)
            c = {'pos_x': pos_x, 'pos_y': pos_y}

        while len(text_cond) < z.size(0):
            text_cond.append(text_cond[-1])
        if len(text_cond) > z.size(0):
            text_cond = text_cond[:z.size(0)]
        assert len(text_cond) == z.size(0)

        out = [z, text_cond]
        out.append(z_gt)

        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z_gt)
            out.extend([x, self.gt, xrec])
        if return_original_cond:
            out.append(xc)

        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)


    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        x, c, gt = self.get_input(batch, self.first_stage_key)
        # print("in diffusion shared_step,enter breakpoint to check x,c,gt shape")
        # breakpoint()
        loss = self(x, c, gt)
        return loss

    def forward(self, x, c, gt, *args, **kwargs):
        index = np.random.randint(0, self.num_timesteps, size=x.size(0))
        t = torch.from_numpy(index)
        t = t.to(self.device).long()

        t_ori = torch.tensor([self.ori_timesteps[index_i] for index_i in index])
        t_ori = t_ori.long().to(x.device)
        breakpoint()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            else:
                c = self.cond_stage_model(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                print(s)
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        if self.test_gt:
            struc_c = self.structcond_stage_model(gt, t_ori)
        else:
            struc_c = self.structcond_stage_model(x, t_ori)
        return self.p_losses(gt, c, struc_c, t, t_ori, x, *args, **kwargs)

    def _rescale_annotations(self, bboxes, crop_coordinates):  # TODO: move to dataset
        def rescale_bbox(bbox):
            x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return x0, y0, w, h

        return [rescale_bbox(b) for b in bboxes]

    def apply_model(self, x_noisy, t, cond, struct_cond, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        if hasattr(self, "split_input_params"):
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["image", "LR_image", "segmentation",
                                       'bbox_img'] and self.model.conditioning_key:  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert (len(c) == 1)  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left postions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                        for patch_nr in range(z.shape[-1])]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [(x_tl, y_tl,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device)
                                      for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)
                print(cut_cond.shape)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                print(adapted_cond.shape)
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                print(adapted_cond.shape)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])
                print(adapted_cond.shape)

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0],
                                  tuple)  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            cond['struct_cond'] = struct_cond
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, struct_cond, t, t_ori, z_gt, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if self.mix_ratio > 0:
            if random.random() < self.mix_ratio:
                noise_new = default(noise, lambda: torch.randn_like(x_start))
                noise = noise_new * 0.5 + noise * 0.5
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_output = self.apply_model(x_noisy, t_ori, cond, struct_cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        model_output_ = model_output

        loss_simple = self.get_loss(model_output_, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        #P2 weighting
        if self.snr is not None:
            self.snr = self.snr.to(loss_simple.device)
            weight = extract_into_tensor(1 / (self.p2_k + self.snr)**self.p2_gamma, t, target.shape)
            loss_simple = weight * loss_simple

        logvar_t = self.logvar[t.cpu()].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output_, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def p_mean_variance(self, x, c, struct_cond, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None, t_replace=None, unconditional_conditioning=None, unconditional_guidance_scale=None,
                        reference_sr=None, reference_lr=0.05, reference_step=1, reference_range=[100, 1000]):
        if t_replace is None:
            t_in = t
        else:
            t_in = t_replace

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_out = self.apply_model(x, t_in, c, struct_cond, return_ids=return_codebook_ids)
        else:
            x_in = torch.cat([x] * 2)
            t_in_ = torch.cat([t_in] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.apply_model(x_in, t_in_, c_in, struct_cond, return_ids=False).chunk(2)
            model_out = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            return_codebook_ids=False

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        elif self.parameterization == "v":
            x_recon = self.predict_start_from_z_and_v(x, model_out, t)
        else:
            raise NotImplementedError()

        if reference_sr is not None:
            # apply reference guidance
            if t[0] >= reference_range[0] and t[0] <= reference_range[1]:
                xstart_current = x_recon.detach().clone().requires_grad_(True)
                xstart_pred = x_recon.detach().clone().requires_grad_(False)
                for _ in range(reference_step):
                    with torch.enable_grad():
                        tau0 = torch.ones_like(xstart_current) * reference_lr
                        mask = torch.ones_like(xstart_current)
                        tau0 = tau0*mask
                        delta_y = torch.square(reference_sr - xstart_current).sum() / reference_sr.shape[0]
                        gradient = torch.autograd.grad(delta_y, xstart_current)[0] * tau0
                        assert not torch.isnan(gradient).any()
                        new_xstart = (
                            xstart_current.float().detach() - gradient.float()
                        )
                    xstart_current = new_xstart.detach().requires_grad_(True)
                x_recon = xstart_current.detach().clone()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    def p_mean_variance_canvas(self, x, c, struct_cond, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None, t_replace=None, tile_size=64, tile_overlap=32, batch_size=4, tile_weights=None,
                        unconditional_conditioning=None, unconditional_guidance_scale=None,
                        reference_sr=None, reference_lr=0.05, reference_step=1, reference_range=[100, 1000]):
        assert tile_weights is not None

        if t_replace is None:
            t_in = t
        else:
            t_in = t_replace

        _, _, h, w = x.size()

        grid_rows = 0
        cur_x = 0
        while cur_x < x.size(-1):
            cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
            grid_rows += 1

        grid_cols = 0
        cur_y = 0
        while cur_y < x.size(-2):
            cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
            grid_cols += 1

        input_list = []
        cond_list = []
        noise_preds = []
        for row in range(grid_rows):
            noise_preds_row = []
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = w - tile_size
                if col == grid_cols-1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                # print('input_start_x', input_start_x)
                # print('input_end_x', input_end_x)
                # print('input_start_y', input_start_y)
                # print('input_end_y', input_end_y)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                input_tile = x[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                input_list.append(input_tile)
                cond_tile = struct_cond[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                cond_list.append(cond_tile)

                if len(input_list) == batch_size or col == grid_cols-1:
                    input_list = torch.cat(input_list, dim=0)
                    cond_list = torch.cat(cond_list, dim=0)

                    if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                        struct_cond_input = self.structcond_stage_model(cond_list, t_in[:input_list.size(0)])
                        model_out = self.apply_model(input_list, t_in[:input_list.size(0)], c[:input_list.size(0)], struct_cond_input, return_ids=return_codebook_ids)
                    else:
                        input_list_ = torch.cat([input_list] * 2)
                        t_in_ = torch.cat([t_in[:input_list.size(0)]] * 2)
                        struct_cond_input = self.structcond_stage_model(torch.cat([cond_list] * 2), t_in_)
                        c_in = torch.cat([unconditional_conditioning[:input_list.size(0)], c[:input_list.size(0)]])
                        e_t_uncond, e_t = self.apply_model(input_list_, t_in_, c_in, struct_cond_input, return_ids=False).chunk(2)
                        model_out = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
                        return_codebook_ids=False

                    if score_corrector is not None:
                        assert self.parameterization == "eps"
                        model_out = score_corrector.modify_score(self, model_out, input_list, t[:input_list.size(0)], c[:input_list.size(0)], **corrector_kwargs)

                    if return_codebook_ids:
                        model_out, logits = model_out

                    for sample_i in range(model_out.size(0)):
                        noise_preds_row.append(model_out[sample_i].unsqueeze(0))
                    input_list = []
                    cond_list = []

            noise_preds.append(noise_preds_row)

        # Stitch noise predictions for all tiles
        noise_pred = torch.zeros(x.shape, device=x.device)
        contributors = torch.zeros(x.shape, device=x.device)
        # Add each tile contribution to overall latents
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = w - tile_size
                if col == grid_cols-1:
                    ofs_y = h - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size
                # print(noise_preds[row][col].size())
                # print(tile_weights.size())
                # print(noise_pred.size())
                noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row][col] * tile_weights
                contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
                # contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights * tile_weights
        # Average overlapping areas with more than 1 contributor
        noise_pred /= contributors
        # noise_pred /= torch.sqrt(contributors)
        model_out = noise_pred

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t[:model_out.size(0)], noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        elif self.parameterization == "v":
            x_recon = self.predict_start_from_z_and_v(x, model_out, t[:model_out.size(0)])
        else:
            raise NotImplementedError()

        if reference_sr is not None:
            # apply reference guidance
            if t[0] >= reference_range[0] and t[0] <= reference_range[1]:
                xstart_current = x_recon.detach().clone().requires_grad_(True)
                xstart_pred = x_recon.detach().clone().requires_grad_(False)
                for _ in range(reference_step):
                    with torch.enable_grad():
                        tau0 = torch.ones_like(xstart_current) * reference_lr
                        mask = torch.ones_like(xstart_current)
                        tau0 = tau0*mask
                        delta_y = torch.square(reference_sr - xstart_current).sum() / reference_sr.shape[0]
                        gradient = torch.autograd.grad(delta_y, xstart_current)[0] * tau0
                        assert not torch.isnan(gradient).any()
                        new_xstart = (
                            xstart_current.float().detach() - gradient.float()
                        )
                    xstart_current = new_xstart.detach().requires_grad_(True)
                x_recon = xstart_current.detach().clone()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t[:x_recon.size(0)])
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, struct_cond, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, t_replace=None,
                 unconditional_conditioning=None, unconditional_guidance_scale=None,
                 reference_sr=None, reference_lr=0.05, reference_step=1, reference_range=[100, 1000]):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, struct_cond=struct_cond, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, t_replace=t_replace,
                                       unconditional_conditioning=unconditional_conditioning, unconditional_guidance_scale=unconditional_guidance_scale,
                                       reference_sr=reference_sr, reference_lr=reference_lr, reference_step=reference_step, reference_range=reference_range)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_canvas(self, x, c, struct_cond, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, t_replace=None,
                 tile_size=64, tile_overlap=32, batch_size=4, tile_weights=None, unconditional_conditioning=None, unconditional_guidance_scale=None,
                 reference_sr=None, reference_lr=0.05, reference_step=1, reference_range=[100, 1000]):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance_canvas(x=x, c=c, struct_cond=struct_cond, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, t_replace=t_replace,
                                       tile_size=tile_size, tile_overlap=tile_overlap, batch_size=batch_size, tile_weights=tile_weights,
                                       unconditional_conditioning=unconditional_conditioning, unconditional_guidance_scale=unconditional_guidance_scale,
                                       reference_sr=reference_sr, reference_lr=reference_lr, reference_step=reference_step, reference_range=reference_range)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t[:b] == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, struct_cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, struct_cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, struct_cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None, time_replace=None, adain_fea=None, interfea_path=None,
                      unconditional_conditioning=None,
                      unconditional_guidance_scale=None,
                      reference_sr=None, reference_lr=0.05, reference_step=1, reference_range=[100, 1000]):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        batch_list = []
        for i in iterator:
            if time_replace is None or time_replace == 1000:
                ts = torch.full((b,), i, device=device, dtype=torch.long)
                t_replace=None
            else:
                ts = torch.full((b,), i, device=device, dtype=torch.long)
                t_replace = repeat(torch.tensor([self.ori_timesteps[i]]), '1 -> b', b=img.size(0))
                t_replace = t_replace.long().to(device)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            if t_replace is not None:
                if start_T is not None:
                    if self.ori_timesteps[i] > start_T:
                         continue
                struct_cond_input = self.structcond_stage_model(struct_cond, t_replace)
            else:
                if start_T is not None:
                    if i > start_T:
                        continue
                struct_cond_input = self.structcond_stage_model(struct_cond, ts)

            if interfea_path is not None:
                batch_list.append(struct_cond_input)

            img = self.p_sample(img, cond, struct_cond_input, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised, t_replace=t_replace,
                                unconditional_conditioning=unconditional_conditioning,
                                unconditional_guidance_scale=unconditional_guidance_scale,
                                reference_sr=reference_sr, reference_lr=reference_lr, reference_step=reference_step, reference_range=reference_range)

            if adain_fea is not None:
                if i < 1:
                    img = adaptive_instance_normalization(img, adain_fea)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        if len(batch_list) > 0:
            num_batch = batch_list[0]['64'].size(0)
            for batch_i in range(num_batch):
                batch64_list = []
                batch32_list = []
                for num_i in range(len(batch_list)):
                    batch64_list.append(cal_pca_components(batch_list[num_i]['64'][batch_i], 3))
                    batch32_list.append(cal_pca_components(batch_list[num_i]['32'][batch_i], 3))
                batch64_list = np.array(batch64_list)
                batch32_list = np.array(batch32_list)

                batch64_list = batch64_list - np.min(batch64_list)
                batch64_list = batch64_list / np.max(batch64_list)
                batch32_list = batch32_list - np.min(batch32_list)
                batch32_list = batch32_list / np.max(batch32_list)

                total_num = batch64_list.shape[0]

                for index in range(total_num):
                    os.makedirs(os.path.join(interfea_path, 'fea_'+str(batch_i)+'_64'), exist_ok=True)
                    cur_path = os.path.join(interfea_path, 'fea_'+str(batch_i)+'_64', 'step_'+str(total_num-index)+'.png')
                    visualize_fea(cur_path, batch64_list[index])
                    os.makedirs(os.path.join(interfea_path, 'fea_'+str(batch_i)+'_32'), exist_ok=True)
                    cur_path = os.path.join(interfea_path, 'fea_'+str(batch_i)+'_32', 'step_'+str(total_num-index)+'.png')
                    visualize_fea(cur_path, batch32_list[index])

        if return_intermediates:
            return img, intermediates
        return img

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.betas.device), (nbatches, self.configs.model.params.channels, 1, 1))

    @torch.no_grad()
    def p_sample_loop_canvas(self, cond, struct_cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None, time_replace=None, adain_fea=None, interfea_path=None, tile_size=64, tile_overlap=32, batch_size=4,
                      unconditional_conditioning=None, unconditional_guidance_scale=None,
                      reference_sr=None, reference_lr=0.05, reference_step=1, reference_range=[100, 1000],):

        assert tile_size is not None

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = batch_size
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

        for i in iterator:
            if time_replace is None or time_replace == 1000:
                ts = torch.full((b,), i, device=device, dtype=torch.long)
                t_replace=None
            else:
                ts = torch.full((b,), i, device=device, dtype=torch.long)
                t_replace = repeat(torch.tensor([self.ori_timesteps[i]]), '1 -> b', b=batch_size)
                t_replace = t_replace.long().to(device)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            if interfea_path is not None:
                for batch_i in range(struct_cond_input['64'].size(0)):
                    os.makedirs(os.path.join(interfea_path, 'fea_'+str(batch_i)+'_64'), exist_ok=True)
                    cur_path = os.path.join(interfea_path, 'fea_'+str(batch_i)+'_64', 'step_'+str(i)+'.png')
                    visualize_fea(cur_path, struct_cond_input['64'][batch_i, 0])
                    os.makedirs(os.path.join(interfea_path, 'fea_'+str(batch_i)+'_32'), exist_ok=True)
                    cur_path = os.path.join(interfea_path, 'fea_'+str(batch_i)+'_32', 'step_'+str(i)+'.png')
                    visualize_fea(cur_path, struct_cond_input['32'][batch_i, 0])

            img = self.p_sample_canvas(img, cond, struct_cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised, t_replace=t_replace,
                                tile_size=tile_size, tile_overlap=tile_overlap, batch_size=batch_size, tile_weights=tile_weights,
                                unconditional_conditioning=unconditional_conditioning, unconditional_guidance_scale=unconditional_guidance_scale,
                                reference_sr=reference_sr, reference_lr=reference_lr, reference_step=reference_step, reference_range=reference_range,)

            if adain_fea is not None:
                if i < 1:
                    img = adaptive_instance_normalization(img, adain_fea)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, struct_cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None, time_replace=None, adain_fea=None, interfea_path=None, start_T=None,
               unconditional_conditioning=None,
               unconditional_guidance_scale=None,
               reference_sr=None, reference_lr=0.05, reference_step=1, reference_range=[100, 1000],
               **kwargs):

        if shape is None:
            shape = (batch_size, self.channels, self.image_size//8, self.image_size//8)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  struct_cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0, time_replace=time_replace, adain_fea=adain_fea, interfea_path=interfea_path, start_T=start_T,
                                  unconditional_conditioning=unconditional_conditioning,
                                  unconditional_guidance_scale=unconditional_guidance_scale,
                                  reference_sr=reference_sr, reference_lr=reference_lr, reference_step=reference_step, reference_range=reference_range)

    @torch.no_grad()
    def sample_canvas(self, cond, struct_cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None, time_replace=None, adain_fea=None, interfea_path=None, tile_size=64, tile_overlap=32, batch_size_sample=4, log_every_t=None,
               unconditional_conditioning=None, unconditional_guidance_scale=None, **kwargs):

        if shape is None:
            shape = (batch_size, self.channels, self.image_size//8, self.image_size//8)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key] if not isinstance(cond[key], list) else
                list(map(lambda x: x, cond[key])) for key in cond}
            else:
                cond = [c for c in cond] if isinstance(cond, list) else cond
        return self.p_sample_loop_canvas(cond,
                                  struct_cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0, time_replace=time_replace, adain_fea=adain_fea, interfea_path=interfea_path, tile_size=tile_size, tile_overlap=tile_overlap,
                                  unconditional_conditioning=unconditional_conditioning, unconditional_guidance_scale=unconditional_guidance_scale, batch_size=batch_size_sample, log_every_t=log_every_t)

    @torch.no_grad()
    def sample_log(self,cond,struct_cond,batch_size,ddim, ddim_steps,**kwargs):

        if ddim:
            raise NotImplementedError
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size//8, self.image_size//8)
            samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, struct_cond=struct_cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates


    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=False,
                   plot_diffusion_rows=False, **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        z, c_lq, z_gt, x, gt, yrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N, val=True)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        if self.test_gt:
            log["gt"] = gt
        else:
            log["inputs"] = x
            log["reconstruction"] = gt
            log["recon_lq"] = self.decode_first_stage(z)

        c = self.cond_stage_model(c_lq)
        if self.test_gt:
            struct_cond = z_gt
        else:
            struct_cond = z

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            noise = torch.randn_like(z)
            ddim_sampler = DDIMSampler(self)
            with self.ema_scope("Plotting"):
                if self.time_replace is not None:
                    cur_time_step=self.time_replace
                else:
                    cur_time_step = 1000

                samples, z_denoise_row = self.sample(cond=c, struct_cond=struct_cond, batch_size=N, timesteps=cur_time_step, return_intermediates=True, time_replace=self.time_replace)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c,struct_cond=struct_cond,batch_size=N,ddim=use_ddim,
                                                             ddim_steps=ddim_steps,eta=ddim_eta,
                                                             quantize_denoised=True, x_T=x_T)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                assert NotImplementedError
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):

                    samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c, struct_cond=struct_cond,
                                                               shape=(self.channels, self.image_size//8, self.image_size//8),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        params = params + list(self.cond_stage_model.parameters())
        params = params + list(self.structcond_stage_model.parameters())
        if self.learn_logvar:
            assert not self.learn_logvar
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x