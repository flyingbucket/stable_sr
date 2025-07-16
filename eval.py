import os
import shutil
import argparse
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
# from torchmetrics.image.fid import FrechetInceptionDistance
# from ldm.models.diffusion.ddpm_ori_so import LatentDiffusionOriSO
# from ldm.models.diffusion.ddpm_wavelet import LatentDiffusionWaveletCS
from torch.utils.data import DataLoader
from basicsr.data.wavelet_dataset import WaveletSRDataset
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips
from pytorch_fid import fid_score
import cv2

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

def evaluate(logdir, ckpt_name):
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
    # model = LatentDiffusionWaveletCS(**config.model.params)
    ckpt_path = os.path.join(logdir, "checkpoints", ckpt_name)
    model = instantiate_from_config(config.model)
    model.init_from_ckpt(ckpt_path)
    model.to(device).eval()

    # === 加载数据 ===
    # data_params = config.data.params.validation.params
    # dataset = instantiate_from_config(config.data)
    # dataset = WaveletSRDataset(data_params, split="val")
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
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
    # fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    psnr_list, ssim_list, lpips_list, enl_list, epi_list = [], [], [], [], []
    img_count = 0

    max_iter=10
    iter_count = 0
    with torch.no_grad():
        # for batch in dataloader:
        for batch in tqdm(dataloader, desc="Overall Progress", leave=True):   
            batch = {k: v.to(device) for k, v in batch.items()}
            log = model.log_images(batch, N=batch['lq_image'].shape[0], sample=True, plot_diffusion_rows=False, plot_progressive_rows=False)

            input_hq = log["input_hq"].detach().cpu().numpy()  # [B,1,H,W]
            samples = log["samples"].detach().cpu().numpy()    # [B,1,H,W]

            B = input_hq.shape[0]

            for i in range(B):
                gt = input_hq[i,0]
                pred = samples[i,0]

                # # === 转为 tensor ===
                # gt_tensor = torch.from_numpy(gt).unsqueeze(0).repeat(3,1,1).to(device).float()
                # pred_tensor = torch.from_numpy(pred).unsqueeze(0).repeat(3,1,1).to(device).float()

                # # torchmetrics FID 需要 [B, C, H, W]，输入范围 [0, 1]
                # gt_tensor = (gt_tensor + 1) / 2  # 如果数据是[-1,1]，归一化到[0,1]
                # pred_tensor = (pred_tensor + 1) / 2

                # fid_metric.update(gt_tensor.unsqueeze(0), real=True)
                # fid_metric.update(pred_tensor.unsqueeze(0), real=False)


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

            iter_count += 1
            if iter_count > max_iter:
                break
    # === FID ===
    fid = fid_score.calculate_fid_given_paths([fid_real, fid_fake], batch_size=2, device=device, dims=2048)
    # fid = fid_metric.compute().item()

    # === 打印结果 ===
    print("==== 评估指标 ====")
    print(f"PSNR: {np.mean(psnr_list):.4f}")
    print(f"SSIM: {np.mean(ssim_list):.4f}")
    print(f"LPIPS: {np.mean(lpips_list):.4f}")
    print(f"FID: {fid:.4f}")
    print(f"ENL: {np.mean(enl_list):.4f}")
    print(f"EPI: {np.mean(epi_list):.4f}")
    shutil.rmtree(fid_real)
    shutil.rmtree(fid_fake)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True, help="训练日志根目录")
    parser.add_argument("--ckpt_name", type=str, required=True, help="ckpt文件名，如 epoch=000015.ckpt")
    parser.add_argument("--gpu", type=int, default=0, help="GPU编号，如 0，1，2。若为-1，则使用CPU")
    parser.add_argument('--batch_size', type=int, default=None, help='推理时的 batch size，默认用配置文件')
    parser.add_argument('--gt_path', type=str, default=None, help='推理数据的 ground-truth 路径')
    
    args = parser.parse_args()


    evaluate(args.logdir, args.ckpt_name)
