import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from glob import glob
from PIL import Image
import pywt  # 小波变换


class WaveletSRDataset(Dataset):
    def __init__(self, image_dir, crop_size=None, wavelet="haar"):
        """
        Args:
            image_dir (str): 包含单通道 PNG 图像的目录。
            crop_size (int or None): 可选裁剪尺寸，默认不裁剪。
            wavelet (str): 小波基名称，如 'haar'、'db1' 等。
        """
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
        self.crop_size = crop_size
        self.wavelet = wavelet

    def _load_image(self, path):
        img = Image.open(path).convert("L")  # 转为单通道灰度图
        tensor = to_tensor(img)  # [1, H, W], range [0,1]
        return tensor

    def _crop_center(self, img, size):
        _, h, w = img.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return img[:, top : top + size, left : left + size]

    def _dwt_tensor(self, img_tensor):
        # img_tensor: [1, H, W]
        img_np = img_tensor.squeeze(0).numpy()  # → [H, W]
        coeffs2 = pywt.dwt2(img_np, self.wavelet)
        LL, (LH, HL, HH) = coeffs2  # 每个 shape: [H/2, W/2]

        # 转成 tensor，并拼接
        dwt_tensor = torch.stack(
            [
                torch.from_numpy(LL),
                torch.from_numpy(LH),
                torch.from_numpy(HL),
                torch.from_numpy(HH),
            ],
            dim=0,
        ).float()  # [4, H/2, W/2]

        return dwt_tensor

    def __getitem__(self, index):
        path = self.image_paths[index]
        img_gt = self._load_image(path)  # [1, H, W]

        if self.crop_size:
            img_gt = self._crop_center(img_gt, self.crop_size)

        # 下采样 + 上采样（bicubic）
        lq = F.interpolate(
            img_gt, scale_factor=0.25, mode="bicubic", align_corners=False
        )
        lq_up = F.interpolate(
            lq, size=img_gt.shape[-2:], mode="bicubic", align_corners=False
        )

        # 小波变换在 lq 图上（上采样前/后均可）
        wavelet = self._dwt_tensor(lq_up)  # shape: [4, H/2, W/2]

        # 如果你希望 wavelet 与 image 对齐大小，可以上采样：
        # wavelet = F.interpolate(
        #     wavelet.unsqueeze(0),
        #     size=img_gt.shape[-2:],
        #     mode="bilinear",
        #     align_corners=False,
        # ).squeeze(0)

        return {
            "gt_image": img_gt,  # [1, H, W]
            "lq_image": lq_up,  # [1, H, W]
            "wavelet": wavelet,  # [4, H/2, W/2]
        }

    def __len__(self):
        return len(self.image_paths)
