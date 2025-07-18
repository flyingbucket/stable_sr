"""
批量 SAR 图像退化模拟（4× 下采样，轻中度退化）
------------------------------------------------
* 轻微高斯模糊 + 适度 speckle 乘性噪声
* 可选运动模糊 & 轻量级高斯加性噪声
* 下采样 → 轻度 BM3D 去噪
* 结果保存在   hr_original / hr_degraded / lr_degraded / lr_denoised   四个子目录
"""

import os
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from bm3d import bm3d, BM3DStages
from scipy.ndimage import convolve


# ---------- 工具函数 ---------- #
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def is_image_file(fname: str) -> bool:
    return os.path.splitext(fname.lower())[1] in IMG_EXTS


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


def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    mse = np.mean((gt - pred) ** 2)
    return float("inf") if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))


def simulate_degradation(
    img_arr: np.ndarray,
    scale: int,
    blur_sigma: float,
    speckle_scale: float,
    motion_len: float,
    motion_angle: float,
    gaussian_std: float,
    bm3d_sigma: float,
):
    # 轻微高斯模糊 + speckle（Gamma 乘性噪声）
    deg = cv2.GaussianBlur(img_arr, (3, 3), blur_sigma)
    speckle = np.random.gamma(shape=1 / speckle_scale, scale=speckle_scale, size=deg.shape)
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


# ---------- CLI ---------- #
def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input_path", type=str, required=True, help="输入单张图片或包含图片的文件夹")
    p.add_argument("--output_root", type=str, default="output_case4x", help="输出根目录")
    p.add_argument("--scale", type=int, default=4, help="下采样因子（固定 4×）")

    # ↓↓↓ 调整后的默认退化强度参数 ↓↓↓
    p.add_argument("--blur_sigma", type=float, default=0.5, help="高斯模糊 σ ≈ 系统 PSF 软化")
    p.add_argument("--speckle_scale", type=float, default=0.25, help="speckle 乘性噪声 scale≈0.04 → σ≈0.2")
    p.add_argument("--motion_len", type=float, default=0.5, help="运动模糊长度 (0=关闭)")
    p.add_argument("--motion_angle", type=float, default=0.0, help="运动模糊角度 (度)")
    p.add_argument("--gaussian_std", type=float, default=0.03, help="加性高斯噪声 σ")
    p.add_argument("--bm3d_sigma", type=float, default=0.03, help="BM3D 轻度去噪 σ")
    # ↑↑↑ ------------------------------------------------ ↑↑↑

    return p.parse_args()


def main():
    args = get_args()
