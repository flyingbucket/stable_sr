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
    def __init__(self, in_channels=4, out_channels=320, down_factor=8, num_resblocks=3):
        super().__init__()
        self.down_factor = down_factor
        self.initial_proj = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )

        res_blocks = []
        for _ in range(num_resblocks):
            res_blocks.append(ResidualBlock(out_channels))
        self.res_blocks = nn.Sequential(*res_blocks)

    def forward(self, x):
        H, W = x.shape[-2:]
        target_size = (H // self.down_factor, W // self.down_factor)
        x = F.interpolate(x, size=target_size, mode="bicubic", align_corners=False)
        x = self.initial_proj(x)
        return x + self.res_blocks(x)


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
