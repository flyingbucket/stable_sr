from tqdm import tqdm
from datetime import datetime
import subprocess

logdirs = [
    "logs/2025-07-14T18-43-47_W_KL_SO_2",
    "logs/2025-07-14T18-45-43_W_KL_CS_2",
    "logs/2025-07-17T16-05-07_W_KLP_SO",
    "logs/2025-07-18T15-40-30_W_KLP_SODG",
    "logs/2025-07-14T18-46-39_Ori_none_2",
    "logs/2025-07-15T16-18-12_Ori_SO",
]

gpu = 0
ckpt_name = "last.ckpt"
batch_size = 10
gt_path = "../DataStore/WHU_512_small"
ddpm_step = 200
ddim_step = 200
save_path="eval_results_new.csv"

success_tasks = []
failed_tasks = []

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def print_summary():
    print(f"\n[{timestamp()}] \033[1m\033[94m[Summary so far]\033[0m")
    if success_tasks:
        print("\033[92m[Succeeded:]\033[0m")
        for s in success_tasks:
            print(f"\033[92m✔ {s}\033[0m")
    if failed_tasks:
        print("\033[91m[Failed:]\033[0m")
        for f in failed_tasks:
            print(f"\033[91m✘ {f}\033[0m")
    print()

pbar = tqdm(total=len(logdirs)*4, desc="Total Progress", dynamic_ncols=True)

for logdir in logdirs:
    for mode, script, step_flag, step_value, dataset in [
        ("DDPM original", "eval_ddpm.py", "--ddpm_step", ddpm_step, None),
        ("DDPM degrade", "eval_ddpm.py", "--ddpm_step", ddpm_step, "basicsr.data.wavelet_dataset.WaveletSRDGDataset"),
        ("DDIM original", "eval_ddim.py", "--ddim_step", ddim_step, None),
        ("DDIM degrade", "eval_ddim.py", "--ddim_step", ddim_step, "basicsr.data.wavelet_dataset.WaveletSRDGDataset"),
    ]:
        cmd = [
            "python", script,
            "--logdir", logdir,
            "--gpu", str(gpu),
            "--ckpt_name", ckpt_name,
            "--batch_size", str(batch_size),
            "--gt_path", gt_path,
            step_flag, str(step_value),
            "--save_path", save_path
        ]
        if "ddim" in script:
            cmd += ["--ddim_eta", "0.5"]
        if dataset:
            cmd += ["--dataset", dataset]

        try:
            subprocess.run(cmd, check=True)
            msg = f"{mode} success: {logdir}"
            success_tasks.append(msg)
        except Exception as e:
            msg = f"{mode} failed: {logdir}"
            failed_tasks.append(msg)

        pbar.update(1)
        print_summary()

pbar.close()

# === 汇总 ===
print(f"\n\033[1m[{timestamp()}] \033[94mFINAL SUMMARY\033[0m")

if success_tasks:
    print("\033[92m[Succeeded:]\033[0m")
    for s in success_tasks:
        print(f"\033[92m✔ {s}\033[0m")

if failed_tasks:
    print("\033[91m[Failed:]\033[0m")
    for f in failed_tasks:
        print(f"\033[91m✘ {f}\033[0m")
else:
    print("\033[92mAll evaluations completed successfully!\033[0m")
