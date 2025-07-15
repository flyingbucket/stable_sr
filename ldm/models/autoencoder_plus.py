import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Encoder, Decoder, Decoder_Mix
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop, triplet_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt
import random

import torchvision.transforms as transforms


from ldm.models.DGConv import (
    DGConvModule, 
    OptimizedDGConvModule, 
    LightweightDGConvModule, 
    MemoryEfficientDGConvModule
)

class AutoencoderKLPlus(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 learning_rate=4.5e-6,   
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 dgconv_config=None):
        """
        增强版AutoencoderKL，集成DGConv模块
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        #___________________________________________________-
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        # -------- 损失 -------- #
        self.loss = instantiate_from_config(lossconfig)

        # -------- 颜色化（可选） -------- #
        if colorize_nlabels is not None:
            self.register_buffer("colorize",
                                 torch.randn(3, colorize_nlabels, 1, 1))
        ##########
        
        
        
        
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
            
        # 默认DGConv配置
        if dgconv_config is None:
            dgconv_config = {
                'type': 'optimized',
                'kernel_size': 3,
                'use_aiiblock': True,
                'use_vanilla': True,
                'use_checkpoint': False,
                'positions': ['encoder_output']
            }
        
        self.dgconv_config = dgconv_config
        self.dgconv_type = dgconv_config.get('type', 'optimized')
        self.dgconv_kernel_size = dgconv_config.get('kernel_size', 3)
        self.use_aiiblock = dgconv_config.get('use_aiiblock', True)
        self.use_vanilla = dgconv_config.get('use_vanilla', True)
        self.use_checkpoint = dgconv_config.get('use_checkpoint', False)
        self.dgconv_positions = dgconv_config.get('positions', ['encoder_output'])
        
        # 初始化DGConv模块容器
        self.dgconv_modules = nn.ModuleDict()
        
        # 初始化DGConv模块
        self._init_dgconv_modules(ddconfig)
        
        # 兼容性设置 - 使用属性而不是循环引用
        self._setup_compatibility()
        
        # 加载预训练权重（如果提供）
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def _setup_compatibility(self):
        """设置与StableSR框架的兼容性，避免循环引用"""
        # 创建虚拟的diffusion_model
        class DummyDiffusionModel:
            def named_parameters(self):
                return []
            def parameters(self):
                return []
        
        self.diffusion_model = DummyDiffusionModel()
        
        # 创建model包装器
        class ModelWrapper:
            def __init__(self, diffusion_model):
                self.diffusion_model = diffusion_model
        
        self.model = ModelWrapper(self.diffusion_model)
        
        # 设置一个标志，表明这是first_stage_model
        self._is_first_stage = True

    def _init_dgconv_modules(self, ddconfig):
        """初始化DGConv模块"""        
        # 根据配置的位置初始化不同的DGConv模块
        if 'encoder_output' in self.dgconv_positions:
            # encoder输出的通道数通常是z_channels * 2（均值和方差）
            encoder_out_channels = 2 * ddconfig["z_channels"]
            self.dgconv_modules['encoder_output'] = self._create_dgconv(encoder_out_channels)
        
        if 'decoder_input' in self.dgconv_positions:
            # decoder输入的通道数
            decoder_in_channels = ddconfig["z_channels"]
            self.dgconv_modules['decoder_input'] = self._create_dgconv(decoder_in_channels)

    def _create_dgconv(self, channels):
        """创建DGConv模块"""
        if self.dgconv_type == 'memory_efficient':
            # 内存高效版本，可以选择内部使用哪种DGConv
            internal_type = self.dgconv_config.get('internal_dgconv_type', 'optimized')
            return MemoryEfficientDGConvModule(
                channels=channels,
                kernel_size=self.dgconv_kernel_size,
                use_checkpoint=self.use_checkpoint,
                use_vanilla=self.use_vanilla,
                use_aiiblock=self.use_aiiblock,
                dgconv_type=internal_type
            )
        elif self.dgconv_type == 'standard':
            return DGConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=self.dgconv_kernel_size,
                use_vanilla=self.use_vanilla,
                use_aiiblock=self.use_aiiblock
            )
        elif self.dgconv_type == 'optimized':
            return OptimizedDGConvModule(
                channels=channels,
                kernel_size=self.dgconv_kernel_size,
                use_vanilla=self.use_vanilla,
                use_aiiblock=self.use_aiiblock
            )
        elif self.dgconv_type == 'lightweight':
            return LightweightDGConvModule(
                channels=channels,
                kernel_size=self.dgconv_kernel_size,
                use_aiiblock=self.use_aiiblock
            )
        else:
            raise ValueError(f"Unknown dgconv_type: {self.dgconv_type}")

    @property
    def first_stage_model(self):
        """动态属性，返回自身但避免循环引用"""
        return self

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        """从检查点加载权重，兼容StableSR的权重格式"""
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            if 'first_stage_model' in k:
                new_key = k[18:]  # 移除'first_stage_model.'前缀
                sd[new_key] = sd[k]
                del sd[k]
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        
        # 只加载已存在的权重，忽略新增的DGConv模块
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys (including new DGConv modules): {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def encode(self, x, return_encfea=False):
        """增强的编码过程"""
        h = self.encoder(x)
        
        # 在encoder输出处应用DGConv
        if 'encoder_output' in self.dgconv_positions and 'encoder_output' in self.dgconv_modules:
            dgconv_out = self.dgconv_modules['encoder_output'](h)
            h = h + dgconv_out  # 残差连接
        
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        
        if return_encfea:
            return posterior, moments
        return posterior

    def decode(self, z):
        """增强的解码过程"""
        z = self.post_quant_conv(z)
        
        # 在decoder输入处应用DGConv
        if 'decoder_input' in self.dgconv_positions and 'decoder_input' in self.dgconv_modules:
            dgconv_out = self.dgconv_modules['decoder_input'](z)
            z = z + dgconv_out  # 残差连接
        
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar+dgconv
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        # 包含DGConv模块的参数
        dgconv_params = list(self.dgconv_modules.parameters()) if hasattr(self, 'dgconv_modules') else []
        
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()) +
            dgconv_params,
            lr=lr, betas=(0.5, 0.9)
        )
        
        # 检查损失函数是否有判别器
        if hasattr(self.loss, 'discriminator'):
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr, betas=(0.5, 0.9))
            return [opt_ae, opt_disc], []
        else:
            return [opt_ae], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
    

class AutoencoderKLResiPlus(AutoencoderKLResi):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 fusion_w=1.0,
                 freeze_dec=True,
                 synthesis_data=False,
                 use_usm=False,
                 test_gt=False,
                 dgconv_config=None):
        """
        增强版AutoencoderKL，集成DGConv模块
        
        Args:
            dgconv_config (dict): DGConv配置，包含:
                - type: 'standard', 'optimized', 或 'lightweight'
                - kernel_size: 卷积核大小，默认3
                - use_aiiblock: 是否使用AIIBlock，默认True
                - positions: DGConv插入位置列表，如['encoder_output', 'encoder_middle', 'decoder_input']
                - encoder_channels: encoder各阶段的通道数列表
        """
        super().__init__(ddconfig, lossconfig, embed_dim, ckpt_path, ignore_keys, image_key, colorize_nlabels,
                         monitor, fusion_w, freeze_dec, synthesis_data, use_usm, test_gt)

        # 默认DGConv配置
        if dgconv_config is None:
            dgconv_config = {
                'type': 'optimized',
                'kernel_size': 3,
                'use_aiiblock': True,
                'positions': ['encoder_output'],
                'encoder_channels': None
            }
        
        self.dgconv_config = dgconv_config
        self.dgconv_type = dgconv_config.get('type', 'optimized')
        self.dgconv_kernel_size = dgconv_config.get('kernel_size', 3)
        self.use_aiiblock = dgconv_config.get('use_aiiblock', True)
        self.dgconv_positions = dgconv_config.get('positions', ['encoder_output'])
        
        # 初始化DGConv模块
        self._init_dgconv_modules()
        
        # 如果需要在decoder中使用增强特征，则初始化enhanced_decoder
        if 'decoder_input' in self.dgconv_positions or 'decoder_middle' in self.dgconv_positions:
            self.enhanced_decoder = self._create_enhanced_decoder(ddconfig)
            self.enhanced_decoder.fusion_w = fusion_w
        else:
            self.enhanced_decoder = None

    def _init_dgconv_modules(self):
        """初始化DGConv模块"""
        self.dgconv_modules = nn.ModuleDict()
        
        # 根据配置的位置初始化不同的DGConv模块
        if 'encoder_output' in self.dgconv_positions:
            # encoder输出的通道数通常是z_channels * 2（均值和方差）
            encoder_out_channels = self.ddconfig.get('z_channels', 4) * 2
            self.dgconv_modules['encoder_output'] = self._create_dgconv(encoder_out_channels)
        
        if 'encoder_middle' in self.dgconv_positions:
            # 需要修改encoder以支持中间层DGConv
            encoder_channels = self.dgconv_config.get('encoder_channels', None)
            if encoder_channels:
                for i, ch in enumerate(encoder_channels):
                    self.dgconv_modules[f'encoder_stage_{i}'] = self._create_dgconv(ch)
        
        if 'decoder_input' in self.dgconv_positions:
            # decoder输入的通道数
            decoder_in_channels = self.ddconfig.get('z_channels', 4)
            self.dgconv_modules['decoder_input'] = self._create_dgconv(decoder_in_channels)

    def _create_dgconv(self, channels):
        """创建DGConv模块"""
        if self.dgconv_type == 'standard':
            from your_dgconv_module import DGConvModule
            return DGConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=self.dgconv_kernel_size,
                use_aiiblock=self.use_aiiblock
            )
        elif self.dgconv_type == 'optimized':
            from your_dgconv_module import OptimizedDGConvModule
            return OptimizedDGConvModule(
                channels=channels,
                kernel_size=self.dgconv_kernel_size,
                use_aiiblock=self.use_aiiblock
            )
        elif self.dgconv_type == 'lightweight':
            from your_dgconv_module import LightweightDGConvModule
            return LightweightDGConvModule(
                channels=channels,
                kernel_size=self.dgconv_kernel_size,
                use_aiiblock=self.use_aiiblock
            )
        else:
            raise ValueError(f"Unknown dgconv_type: {self.dgconv_type}")

    def _create_enhanced_decoder(self, ddconfig):
        """创建增强的解码器（如果需要）"""
        # 这里假设Decoder_Mix是您的自定义解码器
        # 如果需要在解码器中集成DGConv，可以创建一个新的解码器类
        return Decoder_Mix(**ddconfig)

    def encode(self, x):
        """增强的编码过程"""
        # 如果需要在encoder中间层使用DGConv，需要修改encoder
        if 'encoder_middle' in self.dgconv_positions and hasattr(self, 'enhanced_encoder'):
            h, enc_fea = self.enhanced_encoder(x, return_fea=True)
        else:
            h, enc_fea = self.encoder(x, return_fea=True)
        
        # 在encoder输出处应用DGConv
        if 'encoder_output' in self.dgconv_positions:
            dgconv_out = self.dgconv_modules['encoder_output'](h)
            h = h + dgconv_out  # 残差连接
        
        # 量化
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        
        return posterior, enc_fea

    def decode(self, z, enc_fea):
        """增强的解码过程"""
        z = self.post_quant_conv(z)
        
        # 在decoder输入处应用DGConv
        if 'decoder_input' in self.dgconv_positions:
            dgconv_out = self.dgconv_modules['decoder_input'](z)
            z = z + dgconv_out  # 残差连接
        
        # 使用增强解码器或原始解码器
        if self.enhanced_decoder is not None:
            dec = self.enhanced_decoder(z, enc_fea)
        else:
            dec = self.decoder(z, enc_fea)
        
        return dec

    def forward(self, input, latent, sample_posterior=True):
        posterior, enc_fea_lq = self.encode(input)
        dec = self.decode(latent, enc_fea_lq)
        return dec, posterior    