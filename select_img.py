import os, re, json, glob
import math
import time
import warnings
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _safe_zscore(series: pd.Series) -> pd.Series:
    """对指标做 z-score 标准化，std 为 0 时退化为 0（不影响综合分）"""
    mu = series.mean()
    std = series.std(ddof=0)
    if std == 0 or math.isclose(std, 0.0):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - mu) / std


def _find_image(
    img_root: Path,
    name_in_csv: str,
    extra_exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"),
) -> Optional[Path]:
    """
    在 img_root 下寻找图片：
    1) 先直接用 name_in_csv 相对路径
    2) 如果找不到，则用不带扩展名的 basename 在全目录递归匹配（大小写不敏感、扩展名自适配）
    """
    p = img_root / name_in_csv
    if p.exists():
        return p

    # 允许 csv 中只有文件名，不含子目录；或扩展名对不上
    stem = Path(name_in_csv).stem.lower()
    for ext in extra_exts:
        cand = img_root / f"{stem}{ext}"
        if cand.exists():
            return cand

    # 全局递归搜索（可能慢，但更鲁棒）
    for ext in extra_exts:
        for fp in img_root.rglob(f"*{ext}"):
            if fp.stem.lower() == stem:
                return fp

    return None


def _make_info_panel(
    size: Tuple[int, int],
    text_lines: List[str],
    margin: int = 24,
    bg=(245, 245, 245),
    fg=(20, 20, 20),
) -> Image.Image:
    """
    生成右侧信息面板：根据目标 size 高度自适应字体大小
    """
    panel_w, panel_h = size
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)

    # 估算字体大小（尽量不依赖系统字体），高度按 18~24 行的容纳度来估计
    # 这里使用 PIL 默认字体，避免跨平台字体路径问题
    # 根据高度粗略选择字号（非严格）
    line_count = max(1, len(text_lines))
    approx_font_size = max(
        12, min(24, int((panel_h - 2 * margin) / (line_count * 1.3)))
    )
    try:
        # 若你有等宽字体路径，可自行替换
        font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # 逐行写入
    x = margin
    y = margin
    for line in text_lines:
        draw.text((x, y), line, fill=fg, font=font)
        # 行距：字号 * 1.3
        y += int(approx_font_size * 1.3)

    # 画个简单分割线
    draw.line([(0, 0), (0, panel_h)], fill=(200, 200, 200), width=2)
    return img


def _stack_side_by_side(
    left_img: Image.Image, right_img: Image.Image, bg=(255, 255, 255)
) -> Image.Image:
    """
    将左图和右图并排拼接；高度取二者最大，高度不一致时垂直居中。
    """
    h = max(left_img.height, right_img.height)
    w = left_img.width + right_img.width
    out = Image.new("RGB", (w, h), bg)

    # 垂直居中摆放
    ly = (h - left_img.height) // 2
    ry = (h - right_img.height) // 2
    out.paste(left_img, (0, ly))
    out.paste(right_img, (left_img.width, ry))
    return out


def _exp_name_from_logdir(logdir: str) -> str:
    """
    解析 logs/2025-08-15T22-56-24_W_KLP_SO_256 -> W_KLP_SO_256
    """
    base = os.path.basename(logdir.rstrip("/"))
    m = re.match(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}_(.*)", base)
    if not m:
        raise ValueError(f"无法从日志目录名 '{logdir}' 中提取实验名称")
    return m.group(1)

def _ckpt_stem(ckpt_name: str) -> str:
    """
    epoch=000015.ckpt -> epoch=000015 ; last.ckpt -> last
    """
    return os.path.splitext(os.path.basename(ckpt_name))[0]

def _safe_dataset_tail(dataset_import: Optional[str]) -> Optional[str]:
    """
    'ldm.data.wavelet.WaveletSRDataset' -> 'WaveletSRDataset'
    None -> None
    """
    if not dataset_import:
        return None
    return str(dataset_import).rsplit(".", 1)[-1]

def _find_csv_under_detail_dir(
    detail_dir: str,
    logdir: str,
    ckpt_name: str,
    step_value: Optional[int] = None,
    dataset_import: Optional[str] = None,
    mode: str = "DDPM",
) -> str:
    """
    根据规则 <detail_dir>/<exp_name>/<ckpt_stem>/<mode>_<steps>_<dataset>.csv
    自动定位 CSV。若信息不全，回退到通配搜索并做最优匹配。
    """
    exp_name = _exp_name_from_logdir(logdir)
    ckpt = _ckpt_stem(ckpt_name)
    root = Path(detail_dir) / exp_name / ckpt
    if not root.exists():
        raise FileNotFoundError(f"明细目录不存在: {root}")

    # 优选完全匹配：步数 + 数据集尾名都齐全
    ds_tail = _safe_dataset_tail(dataset_import)
    if step_value is not None and ds_tail is not None:
        exact = root / f"{mode}_{int(step_value)}_{ds_tail}.csv"
        if exact.exists():
            return str(exact)

    # 次优：只用步数匹配
    if step_value is not None:
        hits = list(root.glob(f"{mode}_{int(step_value)}_*.csv"))
        if len(hits) == 1:
            return str(hits[0])
        if len(hits) > 1 and ds_tail:
            # 从候选里挑含数据集尾名的
            for p in hits:
                if p.name.endswith(f"_{ds_tail}.csv"):
                    return str(p)
            # 否则取修改时间最新的
            hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(hits[0])

    # 兜底：该 ckpt 目录下任意 DDPM_*.csv，取最新
    hits = list(root.glob(f"{mode}_*.csv"))
    if not hits:
        raise FileNotFoundError(f"未在 {root} 下找到任何 {mode}_*.csv")
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(hits[0])

def resolve_img_and_csv_from_task(
    task: dict,
    ckpt_name: str,
    default_mode: str = "DDPM",
) -> Tuple[str, str]:
    """
    从单个 task 配置中解析出：
    - img_dir: 来自 task['gt_path']
    - csv_path: 根据 detail_dir/logdir/ckpt/step_value/dataset 推断
    """
    img_dir = task.get("gt_path")
    if not img_dir:
        raise ValueError("task 中缺少 gt_path，无法定位原始图片目录")

    detail_dir = task.get("detail_dir")
    logdir = task.get("logdir")
    if not detail_dir or not logdir:
        raise ValueError("task 中缺少 detail_dir 或 logdir，无法定位 CSV")

    step_value = task.get("step_value")  # 如 200
    dataset_import = task.get("dataset")  # 可能为 None
    csv_path = _find_csv_under_detail_dir(
        detail_dir=detail_dir,
        logdir=logdir,
        ckpt_name=ckpt_name,
        step_value=step_value,
        dataset_import=dataset_import,
        mode=default_mode,
    )
    return img_dir, csv_path

def select_and_save_top_images(
    img_dir: str,
    csv_path: str,
    out_dir: Optional[str] = None,
    k: int = 50,
    weight_psnr: float = 0.6,
    weight_ssim: float = 0.4,
    score_method: str = "z",  # "z" 或 "minmax"
    panel_width: int = 480,
    max_width: Optional[int] = None,  # 若需要整体图缩小到不超过某宽度，比如 2000
    extra_cols_to_show: Optional[List[str]] = None,
) -> str:
    """
    读取 CSV（需含 'img', 'psnr', 'ssim'），计算综合分排序，取前 k 并保存「原图 + 评测信息」拼图到 out_dir。
    还会生成 top_k.csv（附带综合分与 rank）。

    参数：
        img_dir:        数据集图片根目录
        csv_path:       验证结果 CSV 路径，至少包含列 ['img', 'psnr', 'ssim']，其他列可选
        out_dir:        输出目录，不给则自动生成
        k:              取前多少张
        weight_psnr:    PSNR 权重
        weight_ssim:    SSIM 权重
        score_method:   评分标准化方法：'z' 使用 z-score；'minmax' 使用 [0,1] 归一化
        panel_width:    右侧信息面板宽度
        max_width:      拼接后图像的最大宽度（超过则等比缩小）
        extra_cols_to_show: 在信息面板中额外显示的列名列表（例如 ['lpips','enl','epi','mode']）

    返回：
        out_dir: 实际的输出目录路径
    """
    img_root = Path(img_dir)
    if not img_root.exists():
        raise FileNotFoundError(f"Image root not found: {img_root}")

    df = pd.read_csv(csv_path)
    required = {"img", "psnr", "ssim"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {sorted(required)}; found {list(df.columns)}"
        )

    # 计算综合分
    if score_method == "z":
        psnr_s = _safe_zscore(df["psnr"])
        ssim_s = _safe_zscore(df["ssim"])
    elif score_method == "minmax":

        def mm(s):
            lo, hi = s.min(), s.max()
            return (
                (s - lo) / (hi - lo)
                if not math.isclose(hi, lo)
                else pd.Series([0.0] * len(s), index=s.index)
            )

        psnr_s = mm(df["psnr"])
        ssim_s = mm(df["ssim"])
    else:
        raise ValueError("score_method must be 'z' or 'minmax'.")

    df["_score"] = weight_psnr * psnr_s + weight_ssim * ssim_s
    df = df.sort_values("_score", ascending=False).reset_index(drop=True)
    topk = df.head(k).copy()
    topk["_rank"] = range(1, len(topk) + 1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 额外展示列
    if extra_cols_to_show is None:
        # 自动选择几个常见列（若存在）
        extras = [c for c in ["mode", "lpips", "enl", "epi"] if c in df.columns]
        extra_cols_to_show = extras

    # 保存 top_k.csv
    cols_to_save = ["_rank", "img", "psnr", "ssim", "_score"] + extra_cols_to_show
    (topk[cols_to_save]).to_csv(out_dir / "top_k.csv", index=False)

    # 逐张生成拼接图
    skipped = 0
    for _, row in topk.iterrows():
        name = str(row["img"])
        img_path = _find_image(img_root, name)
        if img_path is None:
            warnings.warn(f"[Skip] image not found for '{name}' under {img_root}")
            skipped += 1
            continue

        # 读原图
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            # 构建右侧信息面板
            lines = [
                f"Rank: {int(row['_rank'])}",
                f"Score: {row['_score']:.4f}",
                f"PSNR: {row['psnr']:.4f}",
                f"SSIM: {row['ssim']:.6f}",
            ]
            for c in extra_cols_to_show:
                val = row[c]
                if isinstance(val, float):
                    # 根据字段名微调小数位
                    if c.lower() in ("lpips",):
                        lines.append(f"{c.upper()}: {val:.6f}")
                    else:
                        lines.append(f"{c.upper()}: {val:.4f}")
                else:
                    lines.append(f"{c.upper()}: {val}")

            panel_h = im.height  # 让面板高度与原图一致
            info_panel = _make_info_panel(size=(panel_width, panel_h), text_lines=lines)

            merged = _stack_side_by_side(im, info_panel)

            # 控制最大宽度（可选）
            if max_width is not None and merged.width > max_width:
                scale = max_width / merged.width
                new_w = int(merged.width * scale)
                new_h = int(merged.height * scale)
                merged = merged.resize((new_w, new_h), resample=Image.BICUBIC)

            # 保存，带 rank 前缀便于排序对齐
            out_name = f"{int(row['_rank']):03d}__{Path(name).name}"
            merged.save(out_dir / out_name, quality=95)

    if skipped:
        print(f"Done with {skipped} images skipped (not found).")

    return str(out_dir)

def select_and_save_top_images_from_task(
    task: Dict,
    ckpt_name: str,
    out_dir_base: str,
    nickname: str,
    k: int = 50,
    weight_psnr: float = 0.6,
    weight_ssim: float = 0.4,
    score_method: str = "z",
    panel_width: int = 480,
    max_width: Optional[int] = None,
    extra_cols_to_show: Optional[List[str]] = None,
) -> str:
    """
    读取 task_json 中的单个 task（可传 task_id 或已解析的 task dict），
    自动解析 img_dir/csv_path 后调用你原有的 select_and_save_top_images。
    """

    img_dir, csv_path = resolve_img_and_csv_from_task(task, ckpt_name)

    # 输出目录
    stamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    out_dir = Path(f"{out_dir_base}") / f"{nickname}_top{k}_psnr_ssim_{stamp}"

    # 调用你原来的函数（保持原签名不变）
    return select_and_save_top_images(
        img_dir=img_dir,
        csv_path=csv_path,
        out_dir=out_dir,
        k=k,
        weight_psnr=weight_psnr,
        weight_ssim=weight_ssim,
        score_method=score_method,
        panel_width=panel_width,
        max_width=max_width,
        extra_cols_to_show=extra_cols_to_show,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_path",type=str,help="Path to json file of tasks")
    parser.add_argument("--out_dir_base",type=str,default=None,help="Path to dir that saves select imgs from different datasets")
    args=parser.parse_args()
    TASKS_PATH = args.tasks_path
    OUT_DIR_BASE = args.out_dir_base

    with open(TASKS_PATH, "r") as f:
        cfg = json.load(f)
    tasks=cfg['tasks']

    for id,task in tasks.items():
        print("-"*15)
        print(f"Taks ID:{id}")
        # print task
        for key,item in task.items():
            print(f"{key} : {item}")
        NICKNAME=os.path.basename(task['detail_dir'])
        # run select
        out = select_and_save_top_images_from_task(
            task=task,
            out_dir_base=OUT_DIR_BASE,
            nickname=NICKNAME,
            ckpt_name="last.ckpt",
            k=100,  # 取前100
        )
        print("Saved to:", out)
        print("\n")
