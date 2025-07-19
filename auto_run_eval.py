from tqdm import tqdm
import subprocess

logdirs = [
    "logs/2025-07-14T18-45-43_W_KL_CS_2",
    "logs/2025-07-14T18-46-39_Ori_none_2",
    "logs/2025-07-15T16-18-12_Ori_SO",
    "logs/2025-07-17T16-05-07_W_KLP_SO",
    "logs/2025-07-14T18-43-47_W_KL_SO_2",
    "logs/2025-07-13T18-46-07_Ori_none",
]

gpu = 0
ckpt_name = "last.ckpt"
batch_size = 10
gt_path = "../DataStore/WHU_512_small"
ddpm_step = 200

pbar=tqdm(total=len(logdirs)*2,desc="total progress",dynamic_ncols=True)
for logdir in logdirs:
    cmd = [
        "python", "eval_ddpm.py",
        "--logdir", logdir,
        "--gpu", str(gpu),
        "--ckpt_name", ckpt_name,
        "--batch_size", str(batch_size),
        "--gt_path", gt_path,
        "--ddpm_step", str(ddpm_step)
    ]

    subprocess.run(cmd)
    pbar.update(1)
    cmd.extend(["--dataset","basicsr.data.wavelet_dataset.WaveletSRDGDataset"])
    subprocess.run(cmd)
    pbar.update(1)
pbar.close()