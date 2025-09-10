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
from bm3d import bm3d, BM3DStages
from scipy.ndimage import convolve


class WaveletSRDataset(Dataset):
    def __init__(self, opt, **kwargs):
        """
        Args:
            gt_path (str or list): 图像路径根目录
            crop_size (int): 裁剪大小
            wavelet (str): 小波基名称
            kwargs: 兼容 config 调用时传入的额外字段
        """

        if isinstance(opt, (DictConfig, dict)):
            params = opt
            gt_path = params["gt_path"]
            crop_size = params["crop_size"]
            wavelet = params.get("wavelet", "haar")
            image_type = params.get(
                "image_type", ["png", "jpg", "jpg", "bmp", "tif", "tiff"]
            )
        # else:
        #     raise ValueError("opt is not DictConfig")
        # 兼容 ListConfig（配置中是列表形式）
        self.image_paths = []
        if isinstance(gt_path, (ListConfig, list)):
            for p in gt_path:
                for ext in image_type:
                    self.image_paths.extend(sorted(glob(os.path.join(p, f"*.{ext}"))))
        elif isinstance(gt_path, str):
            for ext in image_type:
                self.image_paths.extend(sorted(glob(os.path.join(gt_path, f"*.{ext}"))))
        else:
            raise TypeError("gt_path 应为字符串或路径列表")

        self.crop_size = crop_size
        self.wavelet = wavelet

    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # shape: (H, W), dtype=uint8
        img = img.astype("float32") / 255.0  # shape: (H, W), float32
        tensor = torch.from_numpy(img).unsqueeze(0)  # C=1通道加上去

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
        tries = 0
        while tries < 100:
            path = self.image_paths[index]
            try:
                img_gt = self._load_image(path)  # [1, H, W]
                _, h, w = img_gt.shape

                # 判断尺寸是否满足要求
                if h < self.crop_size or w < self.crop_size:
                    # 跳过尺寸过小的图片
                    tries += 1
                    index = (index + 1) % len(self.image_paths)
                    continue

                # 裁剪中心区域
                if self.crop_size:
                    img_gt = self._crop_center(img_gt, self.crop_size)

                # 下采样 + 上采样 (bicubic)
                img_gt_batched = img_gt.unsqueeze(0)
                lq = F.interpolate(
                    img_gt_batched,
                    scale_factor=0.25,
                    mode="bicubic",
                    align_corners=False,
                )
                lq_up = F.interpolate(
                    lq, size=img_gt.shape[-2:], mode="bicubic", align_corners=False
                )
                lq_up = lq_up.squeeze(0)

                # 小波变换
                wavelet = self._dwt_tensor(lq_up)

                return {
                    "gt_image": img_gt,
                    "lq_image": lq_up,
                    "wavelet": wavelet,
                    "gt_path": path,
                }

            except Exception as e:
                # 读取异常或处理异常，跳过当前图片
                tries += 1
                index = (index + 1) % len(self.image_paths)

        raise RuntimeError(f"Too many bad images starting from index {index}")

    def __len__(self):
        return len(self.image_paths)


class GradientSRDataset(Dataset):
    def __init__(self, gt_path=None, crop_size=None, **kwargs):
        """
        Args:
            gt_path (str or list): 图像路径根目录
            crop_size (int): 裁剪大小
            kwargs: 兼容 config 调用时传入的额外字段
        """

        if isinstance(gt_path, (DictConfig, dict)):
            params = gt_path
            gt_path = params.get("gt_path", None)
            crop_size = params.get("crop_size", crop_size)
            image_type = params.get(
                "image_type", ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
            )
        self.image_paths = []
        if isinstance(gt_path, (ListConfig, list)):
            for p in gt_path:
                for ext in image_type:
                    self.image_paths.extend(sorted(glob(os.path.join(p, f"*.{ext}"))))
        elif isinstance(gt_path, str):
            for ext in image_type:
                self.image_paths.extend(sorted(glob(os.path.join(gt_path, f"*.{ext}"))))
        else:
            raise TypeError("gt_path 应为字符串或路径列表")
        # if isinstance(gt_path, (ListConfig, list)):
        #     assert len(gt_path) > 0, "gt_path 不能是空列表"
        #     gt_path = gt_path[0]
        #
        # assert isinstance(gt_path, str), (
        #     f"gt_path 应为字符串，但实际为: {type(gt_path)}"
        # )

        # self.image_paths = sorted(glob(os.path.join(gt_path, "*.png")))
        self.crop_size = crop_size

    def _load_image(self, path):
        img = Image.open(path).convert("L")  # 单通道灰度
        tensor = to_tensor(img)  # [1, H, W], [0, 1]
        return tensor

    def _crop_center(self, img, size):
        _, h, w = img.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return img[:, top : top + size, left : left + size]

    def _gradient_tensor(self, img_tensor):
        """
        输入: [1, H, W]
        输出: [4, H, W]：Gx, Gy, Gxx, Gyy
        """
        img_np = img_tensor.squeeze(0).numpy()  # → [H, W]

        Gx = cv2.Sobel(img_np, cv2.CV_32F, 1, 0, ksize=3)
        Gy = cv2.Sobel(img_np, cv2.CV_32F, 0, 1, ksize=3)
        Gxx = cv2.Sobel(Gx, cv2.CV_32F, 1, 0, ksize=3)
        Gyy = cv2.Sobel(Gy, cv2.CV_32F, 0, 1, ksize=3)

        grad_tensor = torch.from_numpy(
            np.stack([Gx, Gy, Gxx, Gyy], axis=0)
        ).float()  # [4, H, W]
        return grad_tensor

    def __getitem__(self, index):
        path = self.image_paths[index]
        img_gt = self._load_image(path)  # [1, H, W]

        if self.crop_size:
            img_gt = self._crop_center(img_gt, self.crop_size)

        img_gt_batched = img_gt.unsqueeze(0)  # [1, 1, H, W]
        lq = F.interpolate(
            img_gt_batched, scale_factor=0.25, mode="bicubic", align_corners=False
        )
        lq_up = F.interpolate(
            lq, size=img_gt.shape[-2:], mode="bicubic", align_corners=False
        ).squeeze(0)  # → [1, H, W]

        struct_cond = self._gradient_tensor(lq_up)  # [4, H, W]

        return {
            "gt_image": img_gt,  # [1, H, W]
            "lq_image": lq_up,  # [1, H, W]
            "grad": struct_cond,  # [4, H, W]
            "gt_path": path,
        }

    def __len__(self):
        return len(self.image_paths)


def motion_kernel(length: float, angle: float):
    if length <= 0:
        return None
    k = int(length * 2 + 1)
    kernel = np.zeros((k, k), np.float32)
    rad, cx = np.deg2rad(angle), k // 2
    dx, dy = np.cos(rad), np.sin(rad)
    for t in np.linspace(-length / 2, length / 2, int(length * 3)):
        x, y = int(cx + t * dx), int(cx + t * dy)
        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] = 1.0
    kernel /= kernel.sum() + 1e-8
    return kernel


def bm3d_denoise(img: np.ndarray, sigma: float) -> np.ndarray:
    return np.clip(
        bm3d(img, sigma_psd=sigma, stage_arg=BM3DStages.HARD_THRESHOLDING), 0, 1
    )


def simulate_degradation(
    img_arr: np.ndarray,
    scale: int = 4,
    blur_sigma: float = 0.5,
    speckle_scale: float = 0.25,
    motion_len: float = 0.5,
    motion_angle: float = 0.0,
    gaussian_std: float = 0.03,
    bm3d_sigma: float = 0.03,
):
    # 轻微高斯模糊 + speckle（Gamma 乘性噪声）
    deg = cv2.GaussianBlur(img_arr, (3, 3), blur_sigma)
    speckle = np.random.gamma(
        shape=1 / speckle_scale, scale=speckle_scale, size=deg.shape
    )
    deg *= speckle

    # 运动模糊
    if motion_len > 0:
        k = motion_kernel(motion_len, motion_angle)
        deg = convolve(deg, k, mode="reflect")

    # 加性高斯噪声
    if gaussian_std > 0:
        deg += np.random.normal(0, gaussian_std, deg.shape)

    deg = np.clip(deg, 0, 1)

    # 下采样
    h, w = deg.shape
    lr_deg = cv2.resize(deg, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)

    # 轻度去噪
    lr_den = bm3d_denoise(lr_deg, bm3d_sigma)

    return deg, lr_deg, lr_den


class WaveletSRDGDataset(Dataset):
    def __init__(self, opt, **kwargs):
        """
        Args:
            gt_path (str or list): 图像路径根目录
            crop_size (int): 裁剪大小
            wavelet (str): 小波基名称
            kwargs: 兼容 config 调用时传入的额外字段
        """

        if isinstance(opt, (DictConfig, dict)):
            params = opt
            gt_path = params["gt_path"]
            crop_size = params["crop_size"]
            wavelet = params.get("wavelet", "haar")
            image_type = params.get("image_type", ["png"])
        # else:
        #     raise ValueError("opt is not DictConfig")
        # 兼容 ListConfig（配置中是列表形式）
        self.image_paths = []
        if isinstance(gt_path, (ListConfig, list)):
            for p in gt_path:
                for ext in image_type:
                    self.image_paths.extend(sorted(glob(os.path.join(p, f"*.{ext}"))))
        elif isinstance(gt_path, str):
            for ext in image_type:
                self.image_paths.extend(sorted(glob(os.path.join(gt_path, f"*.{ext}"))))
        else:
            raise TypeError("gt_path 应为字符串或路径列表")

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
        tries = 0
        while tries < 100:
            path = self.image_paths[index]
            try:
                img_gt = self._load_image(path)  # [1, H, W]
                _, h, w = img_gt.shape

                # 判断尺寸是否满足要求
                if h < self.crop_size or w < self.crop_size:
                    # 跳过尺寸过小的图片
                    tries += 1
                    index = (index + 1) % len(self.image_paths)
                    continue

                # 裁剪中心区域
                if self.crop_size:
                    img_gt = self._crop_center(img_gt, self.crop_size)
                # --- Degrade GT to LR (numpy) ---
                gt_np = img_gt.squeeze(0).numpy().astype(np.float32)  # [H, W]
                _, lr_np, lr_denoised_np = simulate_degradation(
                    gt_np
                )  # lr_np ~ [H/scale, W/scale]
                # 选择去噪后或未去噪版本作为网络输入，这里用去噪后的：
                lr_np = lr_denoised_np

                # --- Back to torch tensor (NCHW) ---
                device = img_gt.device
                lr_t = (
                    torch.from_numpy(lr_np)
                    .to(device=device, dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )  # [1,1,h',w']

                # --- Upsample back to GT size with bicubic ---
                lq_up = F.interpolate(
                    lr_t, size=(h, w), mode="bicubic", align_corners=False
                ).squeeze(0)  # [1,H,W]
                lq_up = lq_up.clamp_(0, 1)

                # --- Wavelet on the upsampled LQ (2D numpy needed) ---
                wavelet = self._dwt_tensor(lq_up)  # returns [4, H/2, W/2]
                # # 下采样 + 上采样 (bicubic)
                # degraded = img_gt.clone()
                # degraded_np = degraded.squeeze(0).numpy().astype(np.float32)
                # _, _, degraded_np = simulate_degradation(degraded_np)
                # degraded_batched = degraded.unsqueeze(0)
                # lq_up = F.interpolate(
                #     degraded_np,
                #     size=img_gt.shape[-2:],
                #     mode="bicubic",
                #     align_corners=False,
                # )
                # lq_up = lq_up.squeeze(0)

                # 小波变换
                wavelet = self._dwt_tensor(lq_up)

                return {
                    "gt_image": img_gt,
                    "lq_image": lq_up,
                    "wavelet": wavelet,
                    "gt_path": path,
                }

            except Exception as e:
                # 读取异常或处理异常，跳过当前图片
                tries += 1
                index = (index + 1) % len(self.image_paths)

        raise RuntimeError(f"Too many bad images starting from index {index}")

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

        assert isinstance(gt_path, str), (
            f"gt_path 应为字符串，但实际为: {type(gt_path)}"
        )

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
            "gt_path": path,
            # "wavelet": wavelet,  # [4, H/2, W/2]
        }

    def __len__(self):
        return len(self.image_paths)
