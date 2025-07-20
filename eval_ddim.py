import os,re,shutil,sys,contextlib
import argparse
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
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
def load_config(logdir):
    # 获取 configs 目录下的两个文件
    config_dir = os.path.join(logdir, "configs")
    files = os.listdir(config_dir)
    project_file = [f for f in files if 'project.yaml' in f][0]
    lightning_file = [f for f in files if 'lightning.yaml' in f][0]

    project_cfg = OmegaConf.load(os.path.join(config_dir, project_file))
    lightning_cfg = OmegaConf.load(os.path.join(config_dir, lightning_file))

    config = OmegaConf.merge(project_cfg, lightning_cfg)
    return config

def bootstrap_ci(data, confidence=0.95, n_bootstrap=10000):
    data = np.array(data)
    boot_samples = np.random.choice(data, (n_bootstrap, len(data)), replace=True)
    boot_means = np.mean(boot_samples, axis=1)
    lower = np.percentile(boot_means, (1-confidence)/2*100)
    upper = np.percentile(boot_means, (1+confidence)/2*100)
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
        config.data.params.train.target=args.dataset
        config.data.params.validation.target=args.dataset
    if args.gt_path is not None:
        config.test_data.params.test.params.gt_path = args.gt_path
        config.data.params.validation.params.gt_path = args.gt_path

    # === 加载模型 ===
    with suppress_stdout_stderr():
        ckpt_path = os.path.join(logdir, "checkpoints", ckpt_name)
        model = instantiate_from_config(config.model)
        model.init_from_ckpt(ckpt_path)
        model.to(device).eval()

    # === 设置DDIM采样器（如果需要）===
    ddim_sampler = None
    if args.use_ddim:
        print(f"初始化DDIM采样器...")
        ddim_sampler = DDIMSampler(model)
        print(f"DDIM采样器准备就绪，将使用 {args.ddim_steps} 步")

    # === 加载数据 ===
    data = instantiate_from_config(config.data)
    data.setup()
    dataloader = data.val_dataloader()

    # === LPIPS ===
    with suppress_stdout_stderr():
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
        return norm_img.astype(np.uint8)/255.0

    # prepare saving df
    match = re.match(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}_(.*)", basename)
    if match:
        exp_name = match.group(1)
    else:
        raise ValueError(f"无法从日志目录名 '{args.logdir}' 中提取实验名称")
    dataset_name=args.dataset if args.dataset else config.data.params.validation.target
    dataset_name=str(dataset).rsplit(".",1)[-1]
    ckpt_name_in_df_path = os.path.splitext(ckpt_name)[0]
    df_dir = os.path.join("eval_unet", exp_name, ckpt_name_in_df_path)
    os.makedirs(df_dir, exist_ok=True)
    df_name = f"{mode}_{args.ddim_steps}_{args.ddim_eta}_{dataset_name}.csv"
    df_path = os.path.join(df_dir, df_name)
    assert os.path.exists(df_dir) ,f"The df dir {df_dir} should be made!"

    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Processing batches", leave=True) as pbar:
            for batch in dataloader:   
                batch = {k: v.to(device) for k, v in batch.items()}

                # === 使用log_images方法，它会正确处理条件输入 ===
                if args.use_ddim and ddim_sampler is not None:
                    # 修改模型的采样方法以使用DDIM
                    # 保存原始的sample方法
                    original_sample = model.sample if hasattr(model, 'sample') else None
                    original_sample_log = model.sample_log if hasattr(model, 'sample_log') else None

                    # 临时替换为DDIM采样
                    def ddim_sample_wrapper(cond, batch_size, return_intermediates=False, x_T=None, 
                                          verbose=True, timesteps=None, unconditional_guidance_scale=1., 
                                          unconditional_conditioning=None, eta=0., **kwargs):
                        if timesteps is None:
                            timesteps = args.ddim_steps

                        # 获取形状
                        if hasattr(model, 'channels'):
                            c = model.channels
                        else:
                            c = 4  # 默认潜在空间通道数

                        if hasattr(model, 'image_size'):
                            h = w = model.image_size // 8  # 假设8倍下采样
                        else:
                            # 从输入推断
                            h = w = batch['lq_image'].shape[-1] // 8

                        shape = [c, h, w]

                        samples, intermediates = ddim_sampler.sample(
                            S=timesteps,
                            conditioning=cond,
                            batch_size=batch_size,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=unconditional_guidance_scale,
                            unconditional_conditioning=unconditional_conditioning,
                            eta=args.ddim_eta,
                            x_T=x_T,
                            **kwargs
                        )

                        if return_intermediates:
                            return samples, intermediates
                        return samples

                    def ddim_sample_log_wrapper(cond, batch_size, ddim, ddim_steps, **kwargs):
                        """包装sample_log方法以兼容DDIM采样"""
                        if ddim:
                            # 使用DDIM采样
                            samples = ddim_sample_wrapper(
                                cond=cond, 
                                batch_size=batch_size, 
                                timesteps=ddim_steps,
                                **kwargs
                            )
                            # 返回samples和None作为z_denoise_row
                            return samples, None
                        else:
                            # 使用原始的sample_log方法
                            if original_sample_log is not None:
                                return original_sample_log(cond=cond, batch_size=batch_size, 
                                                         ddim=ddim, ddim_steps=ddim_steps, **kwargs)
                            else:
                                # 如果没有原始方法，使用sample方法
                                samples = original_sample(cond=cond, batch_size=batch_size, **kwargs)
                                return samples, None

                    # 临时替换采样方法
                    if hasattr(model, 'sample'):
                        model.sample = ddim_sample_wrapper
                    if hasattr(model, 'sample_log'):
                        model.sample_log = ddim_sample_log_wrapper

                    try:
                        # 使用修改后的采样方法
                        log = model.log_images(batch, N=batch['lq_image'].shape[0], sample=True, 
                                             ddim=True, ddim_steps=args.ddim_steps,
                                             plot_diffusion_rows=False, plot_progressive_rows=False)
                    finally:
                        # 恢复原始方法
                        if original_sample is not None:
                            model.sample = original_sample
                        if original_sample_log is not None:
                            model.sample_log = original_sample_log
                else:
                    # 使用默认的DDPM采样
                    log = model.log_images(batch, N=batch['lq_image'].shape[0], sample=True,
                                         plot_diffusion_rows=False, plot_progressive_rows=False)

                input_hq = log["input_hq"].detach().cpu().numpy()  # [B,C,H,W]
                samples = log["samples"].detach().cpu().numpy()    # [B,C,H,W]

                B = input_hq.shape[0]

                metrics_of_this_batch=[]
                for i in range(B):
                    # 处理单通道或多通道
                    if input_hq.shape[1] == 1:
                        gt = min_max_normalize(input_hq[i,0])
                        pred = min_max_normalize(samples[i,0])
                    else:
                        # 如果是多通道，转换为灰度用于某些指标
                        gt = np.mean(input_hq[i], axis=0)
                        pred = np.mean(samples[i], axis=0)

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

                    img_path=batch['gt_path'][i]
                    assert isinstance(img_path,str),f"img_path should be be a str,now it is a {type(img_path)}. Check your dataset and dataloader"
                    img_name=os.path.basename(img_path)
                    metrics_of_this_img={
                        'img':img_name,
                        'psnr':psnr,
                        'ssim':ssim,
                        'lpips':lp.item(),
                        'enl':enl,
                        'epi':epi

                    }
                    metrics_of_this_batch.append(metrics_of_this_img)
               
                df_batch = pd.DataFrame(metrics_of_this_batch)
                df_batch.to_csv(
                    df_path,
                    mode='a', 
                    header=not os.path.exists(df_path),
                    index=False,
                    encoding='utf-8'
                )
                pbar.update(1)

    # === FID ===
    fid = fid_score.calculate_fid_given_paths([fid_real, fid_fake], batch_size=2, device=device, dims=2048)
    
    psnr_array=np.array(psnr_list)
    ssim_array=np.array(ssim_list)

    psnr,b_l_psnr,b_u_psnr=bootstrap_ci(psnr_array)
    ssim,b_l_ssim,b_u_ssim=bootstrap_ci(ssim_array)
    psnr_max=np.max(psnr_array)
    psnr_min=np.min(psnr_array)
    ssim_max=np.max(ssim_array)
    ssim_min=np.min(ssim_array)
    lpips_val=np.mean(lpips_list)
    enl=np.mean(enl_list)
    epi=np.mean(epi_list)

    # 保存结果
    # 保存每个样本的指标到 npz 文件
    metric_save_path = os.path.join(
        logdir, 
        f"metrics_{ckpt_name.replace('.ckpt', '')}_ddim_steps{args.ddpm_steps if args.ddpm_steps else 'default'}.npz"
    )

    np.savez(metric_save_path,
            psnr=np.array(psnr_list),
            ssim=np.array(ssim_list),
            lpips=np.array(lpips_list),
            enl=np.array(enl_list),
            epi=np.array(epi_list))
    print(f"详细指标已保存到: {metric_save_path}")

    res_dict = {
        "data_path": metric_save_path,   # 每个样本指标保存的 npz 文件路径

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
    # === 打印结果 ===
    print("\n==== 评估指标 ====")
    if args.use_ddim:
        print(f"采样方式: DDIM")
        print(f"步数: {args.ddim_steps}")
        print(f"eta: {args.ddim_eta}")
    else:
        print(f"采样方式: DDPM (默认)")
    print(f"PSNR: {np.mean(psnr_list):.4f}")
    print(f"SSIM: {np.mean(ssim_list):.4f}")
    print(f"LPIPS: {lpips_val:.4f}")
    print(f"FID: {fid:.4f}")
    print(f"ENL: {np.mean(enl_list):.4f}")
    print(f"EPI: {np.mean(epi_list):.4f}")

    shutil.rmtree(fid_real)
    shutil.rmtree(fid_fake)  

    return res_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True, help="训练日志根目录")
    parser.add_argument("--ckpt_name", type=str, required=True, help="ckpt文件名，如 epoch=000015.ckpt")
    parser.add_argument("--gpu", type=int, default=0, help="GPU编号，如 0，1，2。若为-1，则使用CPU")
    parser.add_argument('--batch_size', type=int, default=None, help='推理时的 batch size，默认用配置文件')
    parser.add_argument('--gt_path', type=str, default=None, help='推理数据的 ground-truth 路径')
    parser.add_argument("--dataset",type=str,default=None,help="输入dataloader对应的dataset的python导入路径来指定下采样方式，默认使用logdir中config的指定模型")

    # DDIM相关参数
    parser.add_argument("--use_ddim", action="store_true", help="使用DDIM采样器")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM采样步数")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM的eta参数（0.0=确定性）")
    parser.add_argument("--save_path",type=str,default="eval_results.csv",help="指定评估指标数据表的写入路径")
    args = parser.parse_args()

    # prepare eval INFO
    # get experiment name from logdir
    basename_log = os.path.basename(args.logdir)  # 获取最后一层目录名
    match = re.match(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}_(.*)", basename_log)
    if match:
        exp_name = match.group(1)
    else:
        raise ValueError(f"无法从日志目录名 '{args.logdir}' 中提取实验名称")

    mode="DDIM" if args.use_ddim else "DDPM"
    gt_path=os.path.basename(args.gt_path)
    config=load_config(args.logdir)
    dataset=args.dataset if args.dataset else config.data.params.validation.target
    dataset=str(dataset).rsplit(".",1)[-1]

    # print configs before eval
    print(f"\n===== 评估配置 =====")
    print(f"实验名称: {exp_name}")
    print(f"MODE: {mode}")
    print(f"日志目录: {args.logdir}")
    print(f"检查点: {args.ckpt_name}")
    print(f"数据模型: {dataset}")
    print(f"测试集: {args.gt_path}")
    print(f"GPU: {args.gpu}")
    print(f"采样器: {mode}")
    print(f"数据表写入位置: {args.save_path}")
    if args.use_ddim:
        print(f"采样器: DDIM")
        print(f"步数: {args.ddim_steps}")
        print(f"eta: {args.ddim_eta}")
    print("===================\n")

    # run eval
    res_dict= evaluate(args.logdir, args.ckpt_name, args)


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
        "exp_name","ckpt_name", "dataset", "gt_path","mode","ddim_steps", "eta", "ddpm_steps",
        "psnr", "psnr_max", "psnr_min", "psnr_ci_lower", "psnr_ci_upper",
        "ssim", "ssim_max", "ssim_min", "ssim_ci_lower", "ssim_ci_upper",
        "enl", "epi", "fid", "lpips",  
        "data_path"   # 每个样本的npz保存路径
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
    print(f"PSNR: {res_dict['psnr']:.4f} (min={res_dict['psnr_min']:.4f}, max={res_dict['psnr_max']:.4f}, CI=[{res_dict['psnr_ci_lower']:.4f}, {res_dict['psnr_ci_upper']:.4f}])")
    print(f"SSIM: {res_dict['ssim']:.4f} (min={res_dict['ssim_min']:.4f}, max={res_dict['ssim_max']:.4f}, CI=[{res_dict['ssim_ci_lower']:.4f}, {res_dict['ssim_ci_upper']:.4f}])")
    print(f"LPIPS: {res_dict['lpips']:.4f}")
    print(f"FID: {res_dict['fid']:.4f}")
    print(f"ENL: {res_dict['enl']:.4f}")
    print(f"EPI: {res_dict['epi']:.4f}")