#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import os
from pathlib import Path
from typing import Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image, ImageOps
from tqdm.auto import tqdm

# Image.MAX_IMAGE_PIXELS = None  # 如需允许超大图可放开

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def process_one(
    img_path: str,
    out_dir: str,
    scale: int,
    jpeg_quality: int,
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    子进程：处理单张图片。
    返回: (status, in_name, out_path|None, reason|None)
      - status ∈ {"ok", "skip", "error"}
    """
    in_p = Path(img_path)
    out_d = Path(out_dir)

    try:
        with Image.open(in_p) as im:
            try:
                im = ImageOps.exif_transpose(im)
            except Exception:
                pass

            w, h = im.size
            if (w % scale != 0) or (h % scale != 0):
                return (
                    "skip",
                    in_p.name,
                    None,
                    f"尺寸 {w}x{h} 不能被倍率 {scale} 整除",
                )

            # 1) bicubic 下采样
            dw, dh = w // scale, h // scale
            im_down = im.resize((dw, dh), resample=Image.BICUBIC)

            # 2) 最近邻放回原分辨率（像素复制）
            im_up = im_down.resize((w, h), resample=Image.NEAREST)

            # 输出路径：文件名保持不变
            out_p = out_d / in_p.name

            save_im = im_up
            if out_p.suffix.lower() in {".jpg", ".jpeg"} and save_im.mode in (
                "RGBA",
                "LA",
            ):
                save_im = save_im.convert("RGB")

            save_params = {}
            if out_p.suffix.lower() in {".jpg", ".jpeg"}:
                save_params.update(
                    dict(quality=jpeg_quality, subsampling=1, optimize=True)
                )

            save_im.save(out_p, **save_params)
            return ("ok", in_p.name, str(out_p), None)

    except Exception as e:
        return ("error", in_p.name, None, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="并行：对文件夹内图片做 bicubic 下采样，再用像素复制上采样回原分辨率并保存（文件名不变）。"
    )
    parser.add_argument("-i", "--input", required=True, help="输入图片文件夹路径")
    parser.add_argument(
        "-s", "--scale", required=True, type=int, help="下采样倍率（2 的整数幂）"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="输出文件夹路径（文件名保持不变）"
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="并行进程数（默认=CPU核数）"
    )
    parser.add_argument(
        "--chunksize", type=int, default=32, help="任务分发批大小（默认=32）"
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=95, help="JPEG 质量（默认95，越低越快）"
    )

    args = parser.parse_args()

    in_dir = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    scale = args.scale
    workers = max(1, int(args.workers))
    chunksize = max(1, int(args.chunksize))
    jpeg_quality = int(args.jpeg_quality)

    if not in_dir.exists() or not in_dir.is_dir():
        print(f"输入目录不存在或不是文件夹：{in_dir}", file=sys.stderr)
        sys.exit(1)
    if not is_power_of_two(scale):
        print(f"倍率必须为 2 的整数幂，收到：{scale}", file=sys.stderr)
        sys.exit(1)
    if in_dir == out_dir:
        print(
            "输出目录与输入目录相同，将覆盖原图。如需原地覆盖，请显式指定不同目录。",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    images = [
        str(p)
        for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ]
    total = len(images)
    if total == 0:
        print(f"输入目录中未找到图片：{in_dir}", file=sys.stderr)
        sys.exit(1)

    tqdm.write(
        f"共找到 {total} 张图片，倍率 = {scale}，并行进程 = {workers}，chunksize = {chunksize}\n"
        f"输出目录：{out_dir}（文件名保持不变）"
    )

    ok_cnt = skip_cnt = err_cnt = 0

    def gen_args():
        for img in images:
            yield (img, str(out_dir), scale, jpeg_quality)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_one, *a) for a in gen_args()]
        for fut in tqdm(as_completed(futures), total=total, smoothing=0.05):
            status, in_name, out_path, reason = fut.result()
            if status == "ok":
                ok_cnt += 1
                tqdm.write(f"[完成] {in_name} -> {out_path}")
            elif status == "skip":
                skip_cnt += 1
                tqdm.write(f"[跳过] {in_name}: {reason}")
            else:
                err_cnt += 1
                tqdm.write(f"[错误] 处理 {in_name} 时出错：{reason}")

    tqdm.write(f"\n统计：完成 {ok_cnt} | 跳过 {skip_cnt} | 错误 {err_cnt}")


if __name__ == "__main__":
    main()
