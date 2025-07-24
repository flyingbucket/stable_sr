from ldm.models.diffusion.ddpm import LatentDiffusion, space_timesteps
import torch
import random
import copy
import torch.nn.functional as F
from ldm.util import default, instantiate_from_config
from einops import repeat, rearrange
from torchvision.utils import make_grid
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from pytorch_lightning.utilities.distributed import rank_zero_only
from ldm.modules.diffusionmodules.util import (
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
)
from ldm.models.autoencoder import IdentityFirstStage, AutoencoderKL
from ldm.models.autoencoder_plus import AutoencoderKLPlus


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentDiffusionWaveletCS(LatentDiffusion):
    """
    Latent Diffusion model using wavelet maps as cross-attention condition.
    The input is a bicubic-downsampled single-channel image, and the GT latent is computed from the original.
    """

    def __init__(
        self,
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
        *args,
        **kwargs,
    ):
        self.unfrozen_first_stage = unfrozen_first_stage
        self.unfrozen_unet = unfrozen_unet
        self.unfrozen_cond_stage = unfrozen_cond_stage

        super().__init__(
            first_stage_config=first_stage_config,
            cond_stage_config=cond_stage_config,
            num_timesteps_cond=num_timesteps_cond,
            cond_stage_key=cond_stage_key,
            cond_stage_trainable=cond_stage_trainable,
            concat_mode=concat_mode,
            cond_stage_forward=cond_stage_forward,
            conditioning_key=conditioning_key,
            scale_factor=scale_factor,
            scale_by_std=scale_by_std,
            *args,
            **kwargs,
        )

        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        only_model = kwargs.pop("only_model", False)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model=only_model)

        print(">>>>>>>>>>>>>>>> model >>>>>>>>>>>>>>>>>>")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)

        print(">>>>>>>>>>>>>>>> cond_stage_model >>>>>>>>>>>>>>>>>>")
        for name, param in self.cond_stage_model.named_parameters():
            if param.requires_grad:
                print(name)

        if hasattr(self, "structcond_stage_model"):
            print(">>>>>>>>>>>>>>>> structcond_stage_model >>>>>>>>>>>>>>>>>>")
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
        if not path:
            print("[INFO] No ckpt path provided, skipping weight loading.")
            return
        import re
        # import traceback
        # print(">>> [DDPM] init_from_ckpt() called <<<")
        # traceback.print_stack(limit=5)
        # print(f"Model ID (self): {id(self)}")

        # 添加额外自动忽略 Cross-Attn 参数（比如 to_k 和 to_v）
        auto_skip_patterns = [
            "model.diffusion_model.*.attn2.to_k",
            "model.diffusion_model.*.attn2.to_v",
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
                    auto_ignore.add(k.split(".")[0])  # 添加顶层前缀

        # 合并手动配置的 ignore_keys 和自动推断的 auto_ignore
        expanded_ignore = list(set(ignore_keys) | auto_ignore)
        # 调用父类实现
        super().init_from_ckpt(path, ignore_keys=expanded_ignore, only_model=only_model)
        n_trainable = sum(
            p.numel() for p in self.first_stage_model.parameters() if p.requires_grad
        )
        print(f"First stage trainable params: {n_trainable:,}")

        n_trainable = sum(
            p.numel() for p in self.cond_stage_model.parameters() if p.requires_grad
        )
        print(f"Cond stage trainable params: {n_trainable:,}")

    def get_input(
        self,
        batch,
        k=None,
        return_first_stage_outputs=False,
        force_c_encode=False,
        cond_key=None,
        return_original_cond=False,
        bs=None,
    ):
        # 图像数据加载
        x_lq_up = batch["lq_image"].to(self.device).float()
        x_gt = batch["gt_image"].to(self.device).float()
        wavelet_cond = batch["wavelet"].to(self.device).float()

        # 编码图像为 latent
        z = self.get_first_stage_encoding(self.encode_first_stage(x_lq_up)).detach()
        z_gt = self.get_first_stage_encoding(self.encode_first_stage(x_gt)).detach()

        # 结构条件（来自 wavelet 子带）
        struct_cond = self.get_learned_conditioning(wavelet_cond)  # [B, C, 64, 64]

        lq_cond = self.get_first_stage_encoding(
            self.encode_first_stage(x_lq_up)
        ).detach()  # [B, C, 64, 64]
        # 构造最终条件 dict（不含文本）
        c = {
            "c_crossattn": [torch.zeros((z.shape[0], 77, 768), device=z.device)],
            "c_concat": [lq_cond],  # 参与拼接
            "struct_cond": struct_cond,  # 参与结构引导
        }
        if c["struct_cond"] is None:
            raise ValueError("Conditioning 'struct_cond' is None in get_input.")

        out = [z, c, z_gt]

        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z_gt)
            out.extend([x_lq_up, x_gt, xrec])
        if return_original_cond:
            out.append(wavelet_cond)
        return out

    def log_images(
        self,
        batch,
        N=8,
        n_row=4,
        sample=True,
        ddim_steps=200,
        ddim_eta=1.0,
        return_keys=None,
        quantize_denoised=True,
        inpaint=False,
        plot_denoise_rows=False,
        plot_progressive_rows=False,
        plot_diffusion_rows=True,
        **kwargs,
    ):
        # use_ddim = ddim_steps is not None
        use_ddim = False

        log = dict()
        outs = self.get_input(
            batch,
            self.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=True,
            return_original_cond=True,
            bs=N,
        )

        z, c, z_gt, x_lq_up, x_gt, xrec = outs[:6]  # 不需要 wavelet_cond
        c["c_crossattn"] = c["c_crossattn"][0]  # from [tensor] to tensor
        x = z_gt  # GT image
        xc = x_lq_up  # conditioning image
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["input_hq"] = x_gt
        log["recon_hq"] = xrec
        log["input_lq"] = x_lq_up
        log["recon_lq"] = self.decode_first_stage(z)
        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), "1 -> b", b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, "n b c h w -> b n c h w")
            diffusion_grid = rearrange(diffusion_grid, "b n c h w -> (b n) c h w")
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        def safe_clone(item):
            if isinstance(item, torch.Tensor):
                return item.clone().detach()
            elif isinstance(item, list):
                return [safe_clone(i) for i in item]
            elif isinstance(item, dict):
                return {k: safe_clone(v) for k, v in item.items()}
            else:
                return item

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                c_copy=safe_clone(c)
                samples, z_denoise_row = self.sample_log(
                    cond=c_copy,
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                )
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(
                self.first_stage_model,
                (AutoencoderKL, AutoencoderKLPlus, IdentityFirstStage),
            ):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    c_copy=safe_clone(c)
                    samples, z_denoise_row = self.sample_log(
                        cond=c_copy,
                        batch_size=N,
                        ddim=use_ddim,
                        ddim_steps=ddim_steps,
                        eta=ddim_eta,
                        quantize_denoised=True,
                    )
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
            #     log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0.0
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):
                    c_copy=safe_clone(c)
                    samples, _ = self.sample_log(
                        cond=c_copy,
                        batch_size=N,
                        ddim=use_ddim,
                        eta=ddim_eta,
                        ddim_steps=ddim_steps,
                        x0=z[:N],
                        mask=mask,
                    )
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    c_copy=safe_clone(c)
                    samples, _ = self.sample_log(
                        cond=c_copy,
                        batch_size=N,
                        ddim=use_ddim,
                        eta=ddim_eta,
                        ddim_steps=ddim_steps,
                        x0=z[:N],
                        mask=mask,
                    )
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    @torch.no_grad()
    def sample(
        self,
        cond,
        batch_size=16,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        shape=None,
        **kwargs,
    ):
        if shape is None:
            shape = (
                batch_size,
                self.channels,
                self.image_size // 8,
                self.image_size // 8,
            )

        struct_cond = cond.pop("struct_cond", None)  # 提前分离出来

        if cond is not None:
            if isinstance(cond, dict):
                cond = {
                    key: cond[key][:batch_size]
                    if not isinstance(cond[key], list)
                    else list(map(lambda x: x[:batch_size], cond[key]))
                    for key in cond
                }
            else:
                cond = (
                    [c[:batch_size] for c in cond]
                    if isinstance(cond, list)
                    else cond[:batch_size]
                )

        return self.p_sample_loop(
            cond,
            shape,
            return_intermediates=return_intermediates,
            x_T=x_T,
            verbose=verbose,
            timesteps=timesteps,
            quantize_denoised=quantize_denoised,
            mask=mask,
            x0=x0,
            struct_cond=struct_cond,
        )  # 明确传入

    def sample_log(self, cond, batch_size, ddim=False, ddim_steps=10, **kwargs):
        samples, intermediates = self.sample(
            cond=cond, batch_size=batch_size, return_intermediates=True, **kwargs
        )

        return samples, intermediates

    def shared_step(self, batch, **kwargs):
        x, c, gt = self.get_input(batch, self.first_stage_key)
        loss = self(x, c, gt, **kwargs)
        return loss

    def forward(self, x, c, gt, *args, **kwargs):
        """
        Override forward to provide GT latent (z_gt) for loss target.
        """
        t = torch.randint(
            0, self.num_timesteps, (x.shape[0],), device=self.device
        ).long()

        # 如果 conditioning_key 不为空，条件必需存在
        if self.model.conditioning_key is not None:
            assert c is not None

            if self.cond_stage_trainable and not isinstance(c, dict):
                c = self.get_learned_conditioning(c)

        # 执行扩散损失计算
        return self.p_losses(x, c, t, gt=gt, *args, **kwargs)

    def p_losses(self, x_start, cond, t, noise=None, **kwargs):
        """
        Override to use GT latent (z_gt) as target instead of x_start itself.
        """
        assert kwargs.get("gt") is not None, (
            "Must provide ground truth latent (z_gt) for loss calculation"
        )
        gt = kwargs["gt"]
        noise = default(noise, lambda: torch.randn_like(gt))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # if self.model.conditioning_key in ['concat', 'hybird']:
        #     x_noisy = torch.cat([x_noisy, cond['c_concat'][0]], dim=1)
        if cond["struct_cond"] is None:
            raise ValueError("Conditioning 'struct_cond' is None in p_losses.")
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = "train" if self.training else "val"

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
        loss_dict.update({f"{prefix}/loss_simple": loss_simple.mean()})

        # logvar optional
        t_cpu = t.detach().cpu()
        logvar_t = self.logvar[t_cpu].to(
            self.device
        )  # this version is compatible with torch 1.13 and newer
        # logvar_t = self.logvar[t].to(self.device) # using pip requirements.txt,torch 1.13 and newer version will be installed causing error

        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean()})
            loss_dict.update({"logvar": self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        # optional ELBO loss
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f"{prefix}/loss_vlb": loss_vlb})
        loss += self.original_elbo_weight * loss_vlb
        loss_dict.update({f"{prefix}/loss": loss})

        return loss, loss_dict

    def get_learned_conditioning(self, c):
        """
        If cond_stage_model is identity or None, directly use the tensor.
        """
        if self.cond_stage_model is None or isinstance(
            self.cond_stage_model, torch.nn.Identity
        ):
            return c
        else:
            return super().get_learned_conditioning(c)

    def apply_model(self, x_noisy, t, cond, return_ids=False, struct_cond=None):
        # print("apply_model cond keys:", cond.keys())

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = (
                "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            )
            cond = {key: cond}

        if not struct_cond:
            struct_cond = cond.get("struct_cond", None)
            assert struct_cond is not None, (
                "Conditioning 'struct_cond' must be provided in apply_model."
            )
        else:
            struct_cond = struct_cond
            assert struct_cond is not None, (
                "Conditioning 'sturct_cond' must be provided in apply_model."
            )

        c_concat = cond.get("c_concat", None)
        c_crossattn = cond.get("c_crossattn", None)
        x_recon = self.model(
            x_noisy,
            t,
            c_concat=c_concat,
            c_crossattn=c_crossattn,
            struct_cond=struct_cond,
        )

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def p_mean_variance(
        self,
        x,
        c,
        t,
        clip_denoised: bool,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        score_corrector=None,
        corrector_kwargs=None,
        struct_cond=None,
    ):
        t_in = t

        model_out = self.apply_model(
            x, t_in, c, struct_cond=struct_cond, return_ids=return_codebook_ids
        )

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(
                self, model_out, x, t, c, **corrector_kwargs
            )

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
            x_recon.clamp_(-1.0, 1.0)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )

        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    def p_sample(
        self,
        x,
        c,
        t,
        clip_denoised=False,
        repeat_noise=False,
        return_codebook_ids=False,
        quantize_denoised=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        struct_cond=None,
    ):  # 添加 struct_cond 参数
        b, *_, device = *x.shape, x.device

        # 传入 struct_cond 给 p_mean_variance
        outputs = self.p_mean_variance(
            x=x,
            c=c,
            t=t,
            clip_denoised=clip_denoised,
            return_codebook_ids=return_codebook_ids,
            quantize_denoised=quantize_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            struct_cond=struct_cond,  # 显式传入
        )

        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (
                0.5 * model_log_variance
            ).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (
                0.5 * model_log_variance
            ).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(
        self,
        cond,
        shape,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        start_T=None,
        log_every_t=None,
        struct_cond=None,
    ):  # 新增 struct_cond
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

        iterator = (
            tqdm(
                reversed(range(0, timesteps)),
                desc="Sampling t",
                total=timesteps,
                leave=False,
            )
            if verbose
            else reversed(range(0, timesteps))
        )

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)

            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(
                img,
                cond,
                ts,
                clip_denoised=self.clip_denoised,
                quantize_denoised=quantize_denoised,
                struct_cond=struct_cond,  # 显式传入
            )

            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img
