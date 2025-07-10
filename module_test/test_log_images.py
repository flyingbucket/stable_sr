import sys
import os
sys.path.append(os.path.abspath("."))  # 加入项目根目录路径
print(f"[INFO] sys.path: {sys.path}")  # 打印当前 sys.path 以确认路径设置

import os
import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image
from ldm.util import instantiate_from_config
from torch.utils.data import DataLoader

# ✅ 推荐：你的 dataset 定义模块路径，比如 datasets/sar_wavelet_dataset.py
from basicsr.data.wavelet_dataset import WaveletSRDataset

@torch.no_grad()
def main():
    # === 1. 设置配置和 ckpt 路径 ===
    config_path = "../configs/stableSRNew/wavelet_sar_512.yaml"
    ckpt_path = "../logs/2025-07-10T16-25-18_w/checkpoints/epoch=000001.ckpt"  # 改成你的路径

    # === 2. 加载 config & 模型 ===
    config = OmegaConf.load(config_path)
    config.model.params.ckpt_path = ckpt_path
    model = instantiate_from_config(config.model).cuda().eval()

    # === 3. 加载一个样本 batch ===
    dataset = WaveletSRDataset(split="val")  # 👈 自己的数据集类
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
    batch = next(iter(dataloader))
    batch = {k: v.cuda() for k, v in batch.items()}

    # === 4. 调试 log_images ===
    log = model.log_images(batch, N=2, sample=True, plot_diffusion_rows=False, plot_progressive_rows=False)
    print(f"[INFO] log_images keys: {list(log.keys())}")

    os.makedirs("debug_out", exist_ok=True)

    def save(key):
        if key in log:
            save_image(log[key], f"debug_out/{key}.png", normalize=True, value_range=(-1, 1))
            print(f"[SAVE] {key}.png saved")

    # === 5. 保存图像结果 ===
    save("inputs")
    save("reconstruction")
    save("samples")
    save("samples_x0_quantized")
    save("samples_inpainting")
    save("samples_outpainting")
    save("diffusion_row")
    save("progressive_row")
    save("denoise_row")
    save("mask")

if __name__ == "__main__":
    main()
