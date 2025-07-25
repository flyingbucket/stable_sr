import os
import re
import shutil
import argparse
import contextlib
import sys
import torch
import numpy as np
import pandas as pd
from filelock import FileLock
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
            desired_count = int(section_counts[len("ddim") :])
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
    project_file = [f for f in files if "project.yaml" in f][0]
    lightning_file = [f for f in files if "lightning.yaml" in f][0]

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


def bootstrap_ci(data, confidence=0.95, n_bootstrap=10000):
    data = np.array(data)
    boot_samples = np.random.choice(data, (n_bootstrap, len(data)), replace=True)
    boot_means = np.mean(boot_samples, axis=1)
    lower = np.percentile(boot_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(boot_means, (1 + confidence) / 2 * 100)
    return np.mean(data), lower, upper


def evaluate(logdir, ckpt_name, args):
    if args.gpu == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    # === 加载配置 ===
    config = load_config(logdir)

    if args.batch_size is not None:
        config.data.params.batch_size = args.batch_size
    if args.dataset is not None:
        config.data.params.train.target = args.dataset
        config.data.params.validation.target = args.dataset
    if args.gt_path is not None:
        config.test_data.params.test.params.gt_path = args.gt_path
        config.data.params.validation.params.gt_path = args.gt_path

    # === 加载模型 ===
    ckpt_path = os.path.join(logdir, "checkpoints", ckpt_name)
    with suppress_stdout_stderr():
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
            cosine_s=8e-3,
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
    with suppress_stdout_stderr():
        lpips_fn = lpips.LPIPS(net="alex").to(device)

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
        return norm_img.astype(np.uint8) / 255.0

    # prepare saving df
    basename = os.path.basename(logdir)
    match = re.match(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}_(.*)", basename)
    if match:
        exp_name = match.group(1)
    else:
        raise ValueError(f"无法从日志目录名 '{args.logdir}' 中提取实验名称")
    dataset_name = (
        args.dataset if args.dataset else config.data.params.validation.target
    )
    dataset_name = str(dataset).rsplit(".", 1)[-1]
    ckpt_name_in_df_path = os.path.splitext(ckpt_name)[0]
    df_dir = os.path.join("eval_unet", exp_name, ckpt_name_in_df_path)
    os.makedirs(df_dir, exist_ok=True)
    df_name = f"{mode}_{args.ddpm_steps}_{dataset_name}.csv"
    df_path = os.path.join(df_dir, df_name)
    print("Total eval results of this experiment writing to \n", df_path)
    assert os.path.exists(df_dir), f"The df dir {df_dir} should be made!"

    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Processing batches", leave=True) as pbar:
            for batch in dataloader:
                batch = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }

                # === 构建log_images的参数 ===
                log_kwargs = {
                    "N": batch["lq_image"].shape[0],
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
                    log = model.log_images(
                        batch, N=batch["lq_image"].shape[0], sample=True
                    )

                input_hq = log["input_hq"].detach().cpu().numpy()  # [B,1,H,W]
                samples = log["samples"].detach().cpu().numpy()  # [B,1,H,W]

                B = input_hq.shape[0]

                metrics_of_this_batch = []
                for i in range(B):
                    gt = min_max_normalize(input_hq[i, 0])
                    pred = min_max_normalize(samples[i, 0])

                    # 保存 FID 图片
                    gt_img = (gt * 255).clip(0, 255).astype(np.uint8)
                    pred_img = (pred * 255).clip(0, 255).astype(np.uint8)

                    cv2.imwrite(os.path.join(fid_real, f"{img_count}.png"), gt_img)
                    cv2.imwrite(os.path.join(fid_fake, f"{img_count}.png"), pred_img)

                    img_count += 1

                    # === PSNR & SSIM ===
                    psnr = compare_psnr(gt, pred, data_range=1.0)
                    ssim = compare_ssim(gt, pred, data_range=1.0)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                    # === LPIPS ===
                    gt_tensor = (
                        torch.from_numpy(gt)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                        .float()
                    )
                    pred_tensor = (
                        torch.from_numpy(pred)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                        .float()
                    )
                    lp = lpips_fn(
                        gt_tensor.repeat(1, 3, 1, 1), pred_tensor.repeat(1, 3, 1, 1)
                    )
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

                    img_path = batch["gt_path"][i]
                    assert isinstance(img_path, str), (
                        f"img_path should be be a str,now it is a {type(img_path)}. Check your dataset and dataloader"
                    )
                    img_name = os.path.basename(img_path)
                    metrics_of_this_img = {
                        "img": img_name,
                        "psnr": psnr,
                        "ssim": ssim,
                        "lpips": lp.item(),
                        "enl": enl,
                        "epi": epi,
                    }
                    metrics_of_this_batch.append(metrics_of_this_img)

                df_batch = pd.DataFrame(metrics_of_this_batch)
                df_batch.to_csv(
                    df_path,
                    mode="a",
                    header=not os.path.exists(df_path),
                    index=False,
                    encoding="utf-8",
                )

                pbar.update(1)

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

    # 保存结果
    # 保存每个样本的指标到 npz 文件
    metric_save_path = os.path.join(
        logdir,
        f"metrics_{ckpt_name.replace('.ckpt', '')}_ddpm_steps{args.ddpm_steps if args.ddpm_steps else 'default'}.npz",
    )

    np.savez(
        metric_save_path,
        psnr=np.array(psnr_list),
        ssim=np.array(ssim_list),
        lpips=np.array(lpips_list),
        enl=np.array(enl_list),
        epi=np.array(epi_list),
    )

    print(f"详细指标已保存到: {metric_save_path}")

    res_dict = {
        "data_path": metric_save_path,  # 每个样本指标保存的 npz 文件路径
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

    shutil.rmtree(fid_real)
    shutil.rmtree(fid_fake)
    return res_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True, help="训练日志根目录")
    parser.add_argument(
        "--ckpt_name", type=str, required=True, help="ckpt文件名，如 epoch=000015.ckpt"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU编号，如 0，1，2。若为-1，则使用CPU"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="推理时的 batch size，默认用配置文件",
    )
    parser.add_argument(
        "--gt_path", type=str, default=None, help="推理数据的 ground-truth 路径"
    )
    parser.add_argument(
        "--ddpm_steps",
        type=int,
        default=1000,
        help="DDPM采样步数（如50, 100, 200, 250等）。不指定则使用模型默认值(1000步)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="输入dataloader对应的dataset的python导入路径来指定下采样方式，默认使用logdir中config的指定模型",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="eval_results.csv",
        help="指定评估指标数据表的写入路径",
    )
    args = parser.parse_args()

    # prepare eval INFO
    # get experiment name from logdir
    basename = os.path.basename(args.logdir)  # 获取最后一层目录名
    match = re.match(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}_(.*)", basename)
    if match:
        exp_name = match.group(1)
    else:
        raise ValueError(f"无法从日志目录名 '{args.logdir}' 中提取实验名称")

    mode = "DDPM"
    gt_path = os.path.basename(args.gt_path)
    config = load_config(args.logdir)
    dataset = args.dataset if args.dataset else config.data.params.validation.target
    dataset = str(dataset).rsplit(".", 1)[-1]
    # print config before eval
    print(f"\n===== 评估配置 =====")
    print(f"实验名称: {exp_name}")
    print(f"MODE: {mode}")
    print(f"日志目录: {args.logdir}")
    print(f"检查点: {args.ckpt_name}")
    print(f"数据模型: {dataset}")
    print(f"测试集: {args.gt_path}")
    print(f"GPU: {args.gpu}")
    print(f"DDPM步数: {args.ddpm_steps}")
    print(f"数据表写入位置: {args.save_path}")
    print("===================\n")
    print("Loading Model ...")

    res_dict = evaluate(args.logdir, args.ckpt_name, args)

    # write to database
    result = {
        "exp_name": exp_name,
        "ckpt_name": args.ckpt_name,
        "gt_path": gt_path,
        "mode": mode,
        "dataset": dataset,
        "ddpm_steps": args.ddpm_steps if args.ddpm_steps else 1000,  # 默认是1000步DDPM
        "ddim_steps": None,
        "eta": None,
    }
    result.update(res_dict)

    # 保存到 CSV
    save_path = args.save_path
    lock_path = save_path + ".lock"

    # 明确指定列顺序（支持旧指标+新指标）
    columns_order = [
        "exp_name",
        "ckpt_name",
        "dataset",
        "gt_path",
        "mode",
        "ddim_steps",
        "eta",
        "ddpm_steps",
        "psnr",
        "psnr_max",
        "psnr_min",
        "psnr_ci_lower",
        "psnr_ci_upper",
        "ssim",
        "ssim_max",
        "ssim_min",
        "ssim_ci_lower",
        "ssim_ci_upper",
        "enl",
        "epi",
        "fid",
        "lpips",
        "data_path",  # 每个样本的npz保存路径
    ]

    with FileLock(lock_path):
        new_df = pd.DataFrame([result])

        # 强制按指定列顺序 reindex（多余列丢弃，缺失列补NaN）
        new_df = new_df.reindex(columns=columns_order)

        if os.path.exists(save_path):
            old_df = pd.read_csv(save_path)

            # 对齐老数据，确保列顺序一致
            old_df = old_df.reindex(columns=columns_order)

            # 拼接
            df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            df = new_df

        # 保存
        df.to_csv(save_path, index=False)

    # === 打印结果 ===
    print("\n==== 评估指标 ====")
    print(f"DDPM步数: {args.ddpm_steps}")
    print(
        f"PSNR: {res_dict['psnr']:.4f} (min={res_dict['psnr_min']:.4f}, max={res_dict['psnr_max']:.4f}, CI=[{res_dict['psnr_ci_lower']:.4f}, {res_dict['psnr_ci_upper']:.4f}])"
    )
    print(
        f"SSIM: {res_dict['ssim']:.4f} (min={res_dict['ssim_min']:.4f}, max={res_dict['ssim_max']:.4f}, CI=[{res_dict['ssim_ci_lower']:.4f}, {res_dict['ssim_ci_upper']:.4f}])"
    )
    print(f"LPIPS: {res_dict['lpips']:.4f}")
    print(f"FID: {res_dict['fid']:.4f}")
    print(f"ENL: {res_dict['enl']:.4f}")
    print(f"EPI: {res_dict['epi']:.4f}")

