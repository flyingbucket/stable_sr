import torch.nn as nn
import torch.nn.functional as F
import torch as th
from ldm.modules.diffusionmodules.openaimodel import (
    Downsample,
    AttentionBlock,
    conv_nd,
    convert_module_to_f16,
    convert_module_to_f32,
)
from ldm.modules.diffusionmodules.util import normalization


class ResBlockNoTime(nn.Module):
    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        dims=2,
        use_checkpoint=False,
        use_scale_shift_norm=False,
        down=False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.in_channels = channels
        self.out_channels = out_channels or channels
        self.down = down

        self.norm1 = normalization(channels)
        self.activation = nn.SiLU()
        self.conv1 = conv_nd(dims, channels, self.out_channels, 3, padding=1)

        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)

        if self.in_channels != self.out_channels:
            self.skip_connection = conv_nd(dims, self.in_channels, self.out_channels, 1)
        else:
            self.skip_connection = nn.Identity()

        if down:
            self.downsample = Downsample(self.out_channels, True, dims)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)

        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)

        out = self.skip_connection(x) + h
        out = self.downsample(out)
        return out


class EncoderUNet(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32

        self.input_blocks = nn.ModuleList(
            [nn.Sequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))]
        )

        ch = model_channels
        ds = 1
        input_block_chans = []
        self._feature_size = ch

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlockNoTime(
                        ch,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
            if level != len(channel_mult) - 1:
                out_ch = ch
                down_layer = (
                    ResBlockNoTime(
                        ch,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                    )
                    if resblock_updown
                    else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                )
                self.input_blocks.append(nn.Sequential(down_layer))
                ch = out_ch
                ds *= 2
                input_block_chans.append(ch)
                self._feature_size += ch

        self.middle_block = nn.Sequential(
            ResBlockNoTime(
                ch,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlockNoTime(
                ch,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        input_block_chans.append(ch)
        self._feature_size += ch
        self.input_block_chans = input_block_chans

        self.fea_tran = nn.ModuleList(
            [
                ResBlockNoTime(
                    ch,
                    dropout,
                    out_channels=out_channels,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
                for ch in input_block_chans
            ]
        )

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x):
        h = x.type(self.dtype)
        result_list = []
        results = {}

        for module in self.input_blocks:
            last_h = h
            h = module(h)
            if h.size(-1) != last_h.size(-1):
                result_list.append(last_h)

        h = self.middle_block(h)
        result_list.append(h)

        assert len(result_list) == len(self.fea_tran)

        for i in range(len(result_list)):
            res = str(result_list[i].size(-1))
            results[res] = self.fea_tran[i](result_list[i])

        return results
