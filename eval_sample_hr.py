import os
import json
import shutil
import argparse
import contextlib
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips
from pytorch_fid import fid_score
import cv2


@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def bootstrap_ci(data, confidence=0.95, n_bootstrap=10000):
    data = np.array(data)
    boot_samples = np.random.choice(data, (n_bootstrap, len(data)), replace=True)
    boot_means = np.mean(boot_samples, axis=1)
    lower = np.percentile(boot_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(boot_means, (1 + confidence) / 2 * 100)
    return np.mean(data), lower, upper


def save_compare_img(
    gt: np.ndarray,
    lq: np.ndarray,
    pred: np.ndarray,
    metrics: Dict[str, Any],
    compare_dir: str,
    img_name: str,
    show_diff: bool = True,
    diff_overlay: bool = False,
    diff_percentile: float = 99.0,
) -> None:
    """
    参数
    ----
    show_diff : 是否显示 |GT-PRED| 面板（第三列）。
    diff_overlay : 若为 True，用热力图把 diff 叠加在 GT 上（替代第三列独立显示）。
    diff_percentile : 可视化时把 diff 按该百分位缩放（增强对比）。
    """
    assert gt.ndim == 2 and pred.ndim == 2, "gt/pred 必须是 2D 灰度图"

    out_path = os.path.join(compare_dir, img_name)

    # 误差可视化准备
    if show_diff or diff_overlay:
        diff = np.abs(pred - gt)
        # 百分位缩放（避免极少数异常点把对比拉平）
        vmax = np.percentile(diff, diff_percentile)
        if vmax <= 1e-12:
            vmax = 1.0
        diff_vis = np.clip(diff / vmax, 0, 1)
    else:
        diff_vis = None

    # 布局：GT | PRED | (DIFF 或 Overlay)
    ncols = 3 if (not show_diff and not diff_overlay) else 4
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), dpi=150)

    if ncols == 3:
        ax0, ax1, ax2 = axes
    else:
        ax0, ax1, ax2, ax3 = axes

    for ax in axes if isinstance(axes, (list, np.ndarray)) else [axes]:
        ax.set_axis_off()

    im0 = ax0.imshow(gt, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax0.set_title("GT")

    im1 = ax1.imshow(lq, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax1.set_title("LQ")

    im2 = ax2.imshow(pred, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax2.set_title("PRED")

    if ncols == 4:
        if diff_overlay:
            ax3.imshow(gt, cmap="gray", interpolation="nearest")
            ax3.imshow(
                diff_vis,
                cmap="inferno",
                alpha=0.6,
                interpolation="nearest",
            )
            ax3.set_title("GT + |GT-PRED| overlay")
        else:
            im2 = ax3.imshow(
                diff_vis, cmap="inferno", vmin=0, vmax=1, interpolation="nearest"
            )
            ax3.set_title("|GT - PRED|")
            cbar = fig.colorbar(im2, ax=ax3, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

    # 标题汇总
    line_parts = []
    if "mode" in metrics:
        line_parts.append(f"[{metrics['mode']}]")
    if "psnr" in metrics:
        line_parts.append(f"PSNR {metrics['psnr']:.3f} dB")
    if "ssim" in metrics:
        line_parts.append(f"SSIM {metrics['ssim']:.4f}")
    if "lpips" in metrics:
        line_parts.append(f"LPIPS {metrics['lpips']:.4f}")
    if "enl" in metrics:
        line_parts.append(f"ENL {metrics['enl']:.3f}")
    if "epi" in metrics:
        line_parts.append(f"EPI {metrics['epi']:.3f}")

    title_left = metrics.get("img", img_name)
    title_right = "  |  ".join(line_parts)
    fig.suptitle(f"{title_left}   {title_right}", y=0.98, fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    assert os.path.exists(out_path), "failed to save compare image"
    plt.close(fig)


def evaluate(sample_dir, gt_dir, lq_dir, result_dir, save_images=True):
    device = torch.device("cpu")
    # === LPIPS ===
    with suppress_stdout_stderr():
        lpips_fn = lpips.LPIPS(net="alex").to(device)

    compare_dir = os.path.join(result_dir, "compare")
    os.makedirs(compare_dir, exist_ok=True)
    df_path = os.path.join(result_dir, "eval_result.csv")
    # === FID 目录准备 ===
    fid_real = os.path.join(result_dir, "fid_real")
    fid_fake = os.path.join(result_dir, "fid_fake")
    os.makedirs(fid_real, exist_ok=True)
    os.makedirs(fid_fake, exist_ok=True)

    psnr_list, ssim_list, lpips_list, enl_list, epi_list = [], [], [], [], []
    img_count = 0

    def min_max_normalize(img):
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max - img_min < 1e-8:
            return np.zeros_like(img, dtype=np.float32)
        norm_img = (img - img_min) / (img_max - img_min)
        return norm_img.astype(np.float32)

    for img_name in tqdm(os.listdir(sample_dir), desc="Evaluating"):
        s_path = os.path.join(sample_dir, img_name)
        lq_path = os.path.join(lq_dir, img_name)
        gt_path = os.path.join(gt_dir, img_name)
        if os.path.exists(gt_path):
            gt = min_max_normalize(cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE))
        else:
            print(f"{img_name} not found in GT DIR {gt_dir}")
            continue
        if os.path.exists(lq_path):
            lq = min_max_normalize(cv2.imread(lq_path, cv2.IMREAD_GRAYSCALE))
        else:
            print(f"{img_name} not found in lq DIR {lq_dir}")
            continue
        pred = cv2.imread(s_path, cv2.IMREAD_GRAYSCALE)

        # 保存 FID 图片
        gt_img = (gt * 255).clip(0, 255).astype(np.uint8)
        pred_img = (pred * 255).clip(0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(fid_real, f"{img_count}.png"), gt_img)
        cv2.imwrite(os.path.join(fid_fake, f"{img_count}.png"), pred_img)

        img_count += 1

        # === PSNR & SSIM ===
        # print(f"gt shape:{gt.shape}\tpred shape: {pred.shape}")
        psnr = compare_psnr(gt, pred, data_range=1.0)
        ssim = compare_ssim(gt, pred, data_range=1.0, win_size=7)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

        # === LPIPS ===
        gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device).float()
        pred_tensor = (
            torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).to(device).float()
        )
        lp = lpips_fn(gt_tensor.repeat(1, 3, 1, 1), pred_tensor.repeat(1, 3, 1, 1))
        lpips_list.append(lp.item())

        # === ENL ===
        mean_pred = np.mean(pred)
        var_pred = np.var(pred)
        enl = (mean_pred**2) / (var_pred + 1e-8)
        enl_list.append(enl)

        # === EPI（Sobel边缘强度比）===
        gt_edges = cv2.Sobel(gt, cv2.CV_64F, 1, 1, ksize=3)
        pred_edges = cv2.Sobel(pred, cv2.CV_64F, 1, 1, ksize=3)
        epi = np.sum(np.abs(pred_edges)) / (np.sum(np.abs(gt_edges)) + 1e-8)
        epi_list.append(epi)

        img_path = gt_path
        img_name = os.path.basename(img_path)
        metrics_of_this_img = {
            "img": img_name,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lp.item(),
            "enl": enl,
            "epi": epi,
        }
        if save_images:
            os.makedirs(compare_dir, exist_ok=True)
            save_compare_img(
                gt,
                lq,
                pred,
                metrics_of_this_img,
                compare_dir,
                img_name,
                diff_overlay=False,
            )

        df_img = pd.DataFrame([metrics_of_this_img])
        df_img.to_csv(
            df_path,
            mode="a",
            header=not os.path.exists(df_path),
            index=False,
            encoding="utf-8",
        )

    # === FID ===
    fid = fid_score.calculate_fid_given_paths(
        [fid_real, fid_fake], batch_size=2, device=device, dims=2048
    )

    psnr_array = np.array(psnr_list)
    ssim_array = np.array(ssim_list)

    psnr, b_l_psnr, b_u_psnr = bootstrap_ci(psnr_array)
    ssim, b_l_ssim, b_u_ssim = bootstrap_ci(ssim_array)
    psnr_max = np.max(psnr_array)
    psnr_min = np.min(psnr_array)
    ssim_max = np.max(ssim_array)
    ssim_min = np.min(ssim_array)
    lpips_val = np.mean(lpips_list)
    enl = np.mean(enl_list)
    epi = np.mean(epi_list)

    print(f"Detailed results of each image is written to {df_path}")

    res_dict = {
        # PSNR
        "psnr": psnr,
        "psnr_max": psnr_max,
        "psnr_min": psnr_min,
        "psnr_ci_lower": b_l_psnr,
        "psnr_ci_upper": b_u_psnr,
        # SSIM
        "ssim": ssim,
        "ssim_max": ssim_max,
        "ssim_min": ssim_min,
        "ssim_ci_lower": b_l_ssim,
        "ssim_ci_upper": b_u_ssim,
        # LPIPS
        "lpips": lpips_val,
        # FID
        "fid": fid,
        # ENL & EPI
        "enl": enl,
        "epi": epi,
    }
    if not save_images:
        shutil.rmtree(fid_real)
        shutil.rmtree(fid_fake)
    return res_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, help="Path to gt dir")
    parser.add_argument("--result_dir", type=str, help="Path to result dir")
    parser.add_argument("--sample_dir", type=str, help="Path to sample dir")
    parser.add_argument("--lq_dir", type=str, help="Path to lq dir")
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Whether to save inference result images",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # prepare eval INFO
    gt_dir = args.gt_dir
    lq_dir = args.lq_dir
    s_dir = args.sample_dir
    result_dir = args.result_dir
    res_dict = evaluate(
        sample_dir=s_dir,
        gt_dir=gt_dir,
        lq_dir=lq_dir,
        result_dir=result_dir,
        save_images=args.save_images,
    )
    print("=" * 60)
    for key, item in res_dict.items():
        print(f"{key}:  {item}")
    print("=" * 60)

    result = {
        "gt_dir": gt_dir,
        "lq_dir": lq_dir,
        "sample_dir": s_dir,
    }
    result.update(res_dict)
    result_path = os.path.join(result_dir, "final_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
