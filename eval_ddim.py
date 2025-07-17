import os
import shutil
import argparse
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from torch.utils.data import DataLoader
from basicsr.data.wavelet_dataset import WaveletSRDataset
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips
from pytorch_fid import fid_score
import cv2
import copy
from einops import repeat

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    这个函数会从原始的1000个时间步中，均匀地选择指定数量的时间步。
    例如，如果要250步，会从0-999中均匀选择250个时间步。
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []

    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size

    return set(all_steps)

def load_config(logdir):
    """获取 configs 目录下的两个文件"""
    config_dir = os.path.join(logdir, "configs")
    files = os.listdir(config_dir)
    project_file = [f for f in files if 'project.yaml' in f][0]
    lightning_file = [f for f in files if 'lightning.yaml' in f][0]

    project_cfg = OmegaConf.load(os.path.join(config_dir, project_file))
    lightning_cfg = OmegaConf.load(os.path.join(config_dir, lightning_file))

    config = OmegaConf.merge(project_cfg, lightning_cfg)
    return config

def setup_timestep_compression(model, ddpm_steps, device):
    """
    设置时间步压缩，重新计算beta调度
    返回原始的sqrt_alphas_cumprod和sqrt_one_minus_alphas_cumprod以供后续使用
    """
    # 保存原始的调度参数
    sqrt_alphas_cumprod = copy.deepcopy(model.sqrt_alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = copy.deepcopy(model.sqrt_one_minus_alphas_cumprod)

    # 使用space_timesteps选择要使用的时间步
    use_timesteps = set(space_timesteps(1000, [ddpm_steps]))

    # 重新计算beta值
    last_alpha_cumprod = 1.0
    new_betas = []
    timestep_map = []

    for i, alpha_cumprod in enumerate(model.alphas_cumprod):
        if i in use_timesteps:
            new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
            last_alpha_cumprod = alpha_cumprod
            timestep_map.append(i)

    # 将beta值转换为numpy数组
    new_betas = [beta.data.cpu().numpy() for beta in new_betas]

    # 重新注册调度器
    model.register_schedule(given_betas=np.array(new_betas), timesteps=len(new_betas))

    # 保存原始时间步映射
    model.ori_timesteps = list(use_timesteps)
    model.ori_timesteps.sort()

    # 确保模型在正确的设备上
    model = model.to(device)

    print(f"时间步压缩完成：从1000步压缩到{len(new_betas)}步")
    print(f"使用的时间步索引：{model.ori_timesteps[:10]}...（显示前10个）")

    return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

def evaluate(logdir, ckpt_name, args):
    if args.gpu == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    # === 加载配置 ===
    config = load_config(logdir)

    if args.batch_size is not None:
        config.data.params.batch_size = args.batch_size
    if args.gt_path is not None:
        config.test_data.params.test.params.gt_path = args.gt_path
        config.data.params.validation.params.gt_path = args.gt_path

    # === 加载模型 ===
    ckpt_path = os.path.join(logdir, "checkpoints", ckpt_name)
    model = instantiate_from_config(config.model)
    model.init_from_ckpt(ckpt_path)
    model.to(device).eval()

    # === 设置时间步压缩（如果指定了ddpm_steps）===
    sqrt_alphas_cumprod = None
    sqrt_one_minus_alphas_cumprod = None

    if args.ddpm_steps is not None and args.ddpm_steps < 1000:
        print(f"\n设置时间步压缩：{args.ddpm_steps}步")
        # 首先注册完整的1000步调度
        model.register_schedule(
            given_betas=None, 
            beta_schedule="linear", 
            timesteps=1000,
            linear_start=0.00085, 
            linear_end=0.0120, 
            cosine_s=8e-3
        )
        model.num_timesteps = 1000

        # 然后进行时间步压缩
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = setup_timestep_compression(
            model, args.ddpm_steps, device
        )
    else:
        print("\n使用模型默认的时间步设置")

    # === 加载数据 ===
    data = instantiate_from_config(config.data)
    data.setup()
    dataloader = data.val_dataloader()

    # === LPIPS ===
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    # === FID 目录准备 ===
    fid_real = os.path.join(logdir, "fid_real")
    fid_fake = os.path.join(logdir, "fid_fake")
    os.makedirs(fid_real, exist_ok=True)
    os.makedirs(fid_fake, exist_ok=True)

    psnr_list, ssim_list, lpips_list, enl_list, epi_list = [], [], [], [], []
    img_count = 0

    def min_max_normalize(img):
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max - img_min < 1e-8:
            return np.zeros_like(img, dtype=np.uint8)
        norm_img = (img - img_min) / (img_max - img_min) * 255.0
        return norm_img.astype(np.uint8)

    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Processing batches", leave=True) as pbar:
            for batch in dataloader:   
                batch = {k: v.to(device) for k, v in batch.items()}

                # === 构建log_images的参数 ===
                log_kwargs = {
                    "N": batch['lq_image'].shape[0],
                    "sample": True,
                    "plot_diffusion_rows": False,
                    "plot_progressive_rows": False,
                }

                # 如果设置了时间步压缩，传递相关参数
                if args.ddpm_steps is not None and args.ddpm_steps < 1000:
                    # 某些模型可能需要这些参数
                    log_kwargs["custom_steps"] = args.ddpm_steps
                    log_kwargs["ddim_steps"] = args.ddpm_steps

                # 生成样本
                try:
                    log = model.log_images(batch, **log_kwargs)
                except Exception as e:
                    print(f"使用自定义参数失败: {e}")
                    print("尝试使用默认参数...")
                    log = model.log_images(batch, N=batch['lq_image'].shape[0], sample=True)

                input_hq = min_max_normalize(log["input_hq"].detach().cpu().numpy())  # [B,1,H,W]
                samples = min_max_normalize(log["samples"].detach().cpu().numpy())    # [B,1,H,W]

                B = input_hq.shape[0]

                for i in range(B):
                    gt = input_hq[i,0]
                    pred = samples[i,0]

                    # 保存 FID 图片
                    gt_img = (gt * 255).clip(0,255).astype(np.uint8)
                    pred_img = (pred * 255).clip(0,255).astype(np.uint8)

                    cv2.imwrite(os.path.join(fid_real, f"{img_count}.png"), gt_img)
                    cv2.imwrite(os.path.join(fid_fake, f"{img_count}.png"), pred_img)

                    img_count += 1

                    # === PSNR & SSIM ===
                    psnr = compare_psnr(gt, pred, data_range=1.0)
                    ssim = compare_ssim(gt, pred, data_range=1.0)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                    # === LPIPS ===
                    gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device).float()
                    pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).to(device).float()
                    lp = lpips_fn(gt_tensor.repeat(1,3,1,1), pred_tensor.repeat(1,3,1,1))
                    lpips_list.append(lp.item())

                    # === ENL ===
                    mean_pred = np.mean(pred)
                    var_pred = np.var(pred)
                    enl = (mean_pred ** 2) / (var_pred + 1e-8)
                    enl_list.append(enl)

                    # === EPI（Sobel边缘强度比）===
                    gt_edges = cv2.Sobel(gt, cv2.CV_64F, 1, 1, ksize=3)
                    pred_edges = cv2.Sobel(pred, cv2.CV_64F, 1, 1, ksize=3)
                    epi = np.sum(np.abs(pred_edges)) / (np.sum(np.abs(gt_edges)) + 1e-8)
                    epi_list.append(epi)

                pbar.update(1)

    # === FID ===
    fid = fid_score.calculate_fid_given_paths([fid_real, fid_fake], batch_size=2, device=device, dims=2048)

    # === 打印结果 ===
    print("\n==== 评估指标 ====")
    print(f"DDPM步数: {args.ddpm_steps if args.ddpm_steps else '默认'}")
    print(f"PSNR: {np.mean(psnr_list):.4f}")
    print(f"SSIM: {np.mean(ssim_list):.4f}")
    print(f"LPIPS: {np.mean(lpips_list):.4f}")
    print(f"FID: {fid:.4f}")
    print(f"ENL: {np.mean(enl_list):.4f}")
    print(f"EPI: {np.mean(epi_list):.4f}")

    # 保存结果
    results_file = os.path.join(
        logdir, 
        f"eval_results_{ckpt_name.replace('.ckpt', '')}_steps{args.ddpm_steps if args.ddpm_steps else 'default'}.txt"
    )
    with open(results_file, 'w') as f:
        f.write(f"评估结果\n")
        f.write(f"检查点: {ckpt_name}\n")
        f.write(f"DDPM步数: {args.ddpm_steps if args.ddpm_steps else '默认'}\n")
        f.write(f"==================\n")
        f.write(f"PSNR: {np.mean(psnr_list):.4f}\n")
        f.write(f"SSIM: {np.mean(ssim_list):.4f}\n")
        f.write(f"LPIPS: {np.mean(lpips_list):.4f}\n")
        f.write(f"FID: {fid:.4f}\n")
        f.write(f"ENL: {np.mean(enl_list):.4f}\n")
        f.write(f"EPI: {np.mean(epi_list):.4f}\n")

    print(f"\n结果已保存到: {results_file}")

    shutil.rmtree(fid_real)
    shutil.rmtree(fid_fake)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True, help="训练日志根目录")
    parser.add_argument("--ckpt_name", type=str, required=True, help="ckpt文件名，如 epoch=000015.ckpt")
    parser.add_argument("--gpu", type=int, default=0, help="GPU编号，如 0，1，2。若为-1，则使用CPU")
    parser.add_argument('--batch_size', type=int, default=None, help='推理时的 batch size，默认用配置文件')
    parser.add_argument('--gt_path', type=str, default=None, help='推理数据的 ground-truth 路径')
    parser.add_argument("--ddpm_steps", type=int, default=None, 
                       help="DDPM采样步数（如50, 100, 200, 250等）。不指定则使用模型默认值(1000步)")

    args = parser.parse_args()

    print(f"\n===== 评估配置 =====")
    print(f"日志目录: {args.logdir}")
    print(f"检查点: {args.ckpt_name}")
    print(f"GPU: {args.gpu}")
    print(f"DDPM步数: {args.ddpm_steps if args.ddpm_steps else '默认(1000)'}")
    print("===================\n")

    evaluate(args.logdir, args.ckpt_name, args)