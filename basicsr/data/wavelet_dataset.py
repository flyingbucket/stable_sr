import os
import torch
import torch.nn.functional as F
import pywt  
import numpy as np
import cv2  
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from glob import glob
from PIL import Image
from omegaconf import ListConfig, DictConfig

class WaveletSRDataset(Dataset):
    def __init__(self, gt_path=None, crop_size=None, wavelet="haar", **kwargs):
        """
        Args:
            gt_path (str or list): 图像路径根目录
            crop_size (int): 裁剪大小
            wavelet (str): 小波基名称
            kwargs: 兼容 config 调用时传入的额外字段
        """

        # 如果是 config 字典，说明传的是整个 params
        if isinstance(gt_path, (DictConfig, dict)):
            params = gt_path
            gt_path = params.get("gt_path", None)
            crop_size = params.get("crop_size", crop_size)
            wavelet = params.get("wavelet", wavelet)

        # 兼容 ListConfig（配置中是列表形式）
        if isinstance(gt_path, (ListConfig, list)):
            assert len(gt_path) > 0, "gt_path 不能是空列表"
            gt_path = gt_path[0]

        assert isinstance(gt_path, str), f"gt_path 应为字符串，但实际为: {type(gt_path)}"

        self.image_paths = sorted(glob(os.path.join(gt_path, "*.png")))
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

        # => [1, H, W] → [1, 1, H, W]
        img_gt_batched = img_gt.unsqueeze(0)
        # 下采样 + 上采样（bicubic）
        lq = F.interpolate(
            img_gt_batched, scale_factor=0.25, mode="bicubic", align_corners=False
        )
        lq_up = F.interpolate(
            lq, size=img_gt.shape[-2:], mode="bicubic", align_corners=False
        )
        # 去掉 batch 维度 → [1, H, W]
        lq_up = lq_up.squeeze(0)

        # 小波变换在 lq 图上（上采样前/后均可）
        wavelet = self._dwt_tensor(lq_up)  # shape: [4, H/2, W/2]

        return {
            "gt_image": img_gt,  # [1, H, W]
            "lq_image": lq_up,  # [1, H, W]
            "wavelet": wavelet,  # [4, H/2, W/2]
        }

    def __len__(self):
        return len(self.image_paths)

class WaveletSRDGDataset(Dataset):
    def __init__(self, gt_path=None, crop_size=None, wavelet="haar", **kwargs):
        """
        Args:
            gt_path (str or list): 图像路径根目录
            crop_size (int): 裁剪大小
            wavelet (str): 小波基名称
            kwargs: 兼容 config 调用时传入的额外字段
        """

        # 如果是 config 字典，说明传的是整个 params
        if isinstance(gt_path, (DictConfig, dict)):
            params = gt_path
            gt_path = params.get("gt_path", None)
            crop_size = params.get("crop_size", crop_size)
            wavelet = params.get("wavelet", wavelet)

        # 兼容 ListConfig（配置中是列表形式）
        if isinstance(gt_path, (ListConfig, list)):
            assert len(gt_path) > 0, "gt_path 不能是空列表"
            gt_path = gt_path[0]

        assert isinstance(gt_path, str), f"gt_path 应为字符串，但实际为: {type(gt_path)}"

        self.image_paths = sorted(glob(os.path.join(gt_path, "*.png")))
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

        degraded=img_gt.clone()
        speckle = np.random.gamma(shape=1/0.003, scale=0.003, size=degraded.shape)
        degraded = degraded * speckle
        # 轻微模糊
        degraded = cv2.GaussianBlur(degraded, (3, 3), 0.3)
        # 极弱加性噪声
        noise = np.random.normal(0, 0.01, degraded.shape)  # 大幅减小
        degraded = degraded + noise

        if self.crop_size:
            img_gt = self._crop_center(img_gt, self.crop_size)

        # => [1, H, W] → [1, 1, H, W]
        img_gt_batched = img_gt.unsqueeze(0)
        degraded_batched = degraded.unsqueeze(0)
        # 下采样 + 上采样（bicubic）
        lq = F.interpolate(
            degraded_batched, scale_factor=0.25, mode="bicubic", align_corners=False
        )

        lq_up = F.interpolate(
            lq, size=img_gt.shape[-2:], mode="bicubic", align_corners=False
        )
        # 去掉 batch 维度 → [1, H, W]
        lq_up = lq_up.squeeze(0)

        # 小波变换在 lq 图上（上采样前/后均可）
        wavelet = self._dwt_tensor(lq_up)  # shape: [4, H/2, W/2]

        return {
            "gt_image": img_gt,  # [1, H, W]
            "lq_image": lq_up,  # [1, H, W]
            "wavelet": wavelet,  # [4, H/2, W/2]
        }

    def __len__(self):
        return len(self.image_paths)
class OriginalDataset(Dataset):
    def __init__(self, gt_path=None, crop_size=None, **kwargs):
        """
        Args:
            gt_path (str or list): 图像路径根目录
            crop_size (int): 裁剪大小
            kwargs: 兼容 config 调用时传入的额外字段
        """

        # 如果是 config 字典，说明传的是整个 params
        if isinstance(gt_path, (DictConfig, dict)):
            params = gt_path
            gt_path = params.get("gt_path", None)
            crop_size = params.get("crop_size", crop_size)
            # wavelet = params.get("wavelet", wavelet)

        # 兼容 ListConfig（配置中是列表形式）
        if isinstance(gt_path, (ListConfig, list)):
            assert len(gt_path) > 0, "gt_path 不能是空列表"
            gt_path = gt_path[0]

        assert isinstance(gt_path, str), f"gt_path 应为字符串，但实际为: {type(gt_path)}"

        self.image_paths = sorted(glob(os.path.join(gt_path, "*.png")))
        self.crop_size = crop_size
        # self.wavelet = wavelet

    def _load_image(self, path):
        img = Image.open(path).convert("L")  # 转为单通道灰度图
        tensor = to_tensor(img)  # [1, H, W], range [0,1]
        return tensor

    def _crop_center(self, img, size):
        _, h, w = img.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return img[:, top : top + size, left : left + size]

    # def _dwt_tensor(self, img_tensor):
    #     # img_tensor: [1, H, W]
    #     img_np = img_tensor.squeeze(0).numpy()  # → [H, W]
    #     coeffs2 = pywt.dwt2(img_np, self.wavelet)
    #     LL, (LH, HL, HH) = coeffs2  # 每个 shape: [H/2, W/2]

    #     # 转成 tensor，并拼接
    #     dwt_tensor = torch.stack(
    #         [
    #             torch.from_numpy(LL),
    #             torch.from_numpy(LH),
    #             torch.from_numpy(HL),
    #             torch.from_numpy(HH),
    #         ],
    #         dim=0,
    #     ).float()  # [4, H/2, W/2]

    #     return dwt_tensor

    def __getitem__(self, index):
        path = self.image_paths[index]
        img_gt = self._load_image(path)  # [1, H, W]

        if self.crop_size:
            img_gt = self._crop_center(img_gt, self.crop_size)

        # => [1, H, W] → [1, 1, H, W]
        img_gt_batched = img_gt.unsqueeze(0)
        # 下采样 + 上采样（bicubic）
        lq = F.interpolate(
            img_gt_batched, scale_factor=0.25, mode="bicubic", align_corners=False
        )
        lq_up = F.interpolate(
            lq, size=img_gt.shape[-2:], mode="bicubic", align_corners=False
        )
        # 去掉 batch 维度 → [1, H, W]
        lq_up = lq_up.squeeze(0)

        # # 小波变换在 lq 图上（上采样前/后均可）
        # wavelet = self._dwt_tensor(lq_up)  # shape: [4, H/2, W/2]

        return {
            "gt_image": img_gt,  # [1, H, W]
            "lq_image": lq_up,  # [1, H, W]
            # "wavelet": wavelet,  # [4, H/2, W/2]
        }

    def __len__(self):
        return len(self.image_paths)