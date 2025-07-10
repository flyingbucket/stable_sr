import torch.nn as nn
import torch.nn.functional as F

class WaveletCNNEncoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=320, down_factor=8):
        super().__init__()
        stride = int(down_factor).bit_length() - 1  # 8 -> 3 层 stride=2
        layers = []
        channels = [in_channels, 64, 128, 256, out_channels]
        for i in range(stride):
            layers += [
                nn.Conv2d(
                    channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1
                ),
                nn.ReLU(inplace=True),
            ]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)



class WaveletBicubicResidualEncoder(nn.Module):
    def __init__(self, in_channels=4, out_channels=320, num_resblocks=3, multiscale_sizes=(128, 64, 32, 16, 8)):
        """
        返回结构引导的多尺度特征图，用于 SPADE 中不同分辨率的调制。
        """
        super().__init__()
        self.multiscale_sizes = multiscale_sizes
        self.initial_proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        res_blocks = []
        for _ in range(num_resblocks):
            res_blocks.append(ResidualBlock(out_channels))
        self.res_blocks = nn.Sequential(*res_blocks)

    def forward(self, x):
        """
        输入：
            x: [B, 4, H, W] 小波子带图
        输出：
            dict[str(res)]: [B, C, res, res] 结构引导特征图
        """
        struct_cond = {}

        for res in self.multiscale_sizes:
            x_down = F.interpolate(x, size=(res, res), mode="bicubic", align_corners=False)
            x_proj = self.initial_proj(x_down)
            x_feat = x_proj + self.res_blocks(x_proj)
            struct_cond[str(res)] = x_feat

        return struct_cond


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)
