from tqdm import tqdm
from datetime import datetime
import subprocess
import json


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_summary(finished_tasks):
    print(f"\n[{timestamp()}] \033[1m\033[94m[Summary so far]\033[0m")
    if success_tasks:
        print("\033[92m[Succeeded:]\033[0m")
        for s in success_tasks:
            print(f"\033[92m✔ {s}\033[0m")
    if failed_tasks:
        print("\033[91m[Failed:]\033[0m")
        for f in failed_tasks:
            print(f"\033[91m✘ {f}\033[0m")
    print(f"\nSo far finished tasks {finished_tasks}")
    print()


# load tasks
with open("tasks.json", "r") as f:
    config = json.load(f)

tasks = config["tasks"]
skip = config.get("skip", [])
primary = config.get("primary", [])
gpu = 1
ckpt_name = "last.ckpt"
batch_size = 6
gt_path = "../DataStore/WHU_512_small"
ddpm_step = 200
ddim_step = 200
eta = 0.5
save_path = "eval_results_new.csv"

success_tasks = []
failed_tasks = []


pbar = tqdm(total=len(tasks), desc="Total Progress", dynamic_ncols=True)
finished_tasks = []
while tasks:
    if primary:
        task_id = str(primary.pop(0))
        task = tasks.pop(task_id)
    else:
        task_id, task = tasks.popitem()
    print(f"\033[94m[{timestamp()}] Evaluating task {int(task_id)}\033[0m")
    # print(f"evaluating task {int(task_id)}")

    mode = task["mode"]
    logdir = task["logdir"]
    script = task["script"]
    step_flag = task["step_flag"]
    step_value = task["step_value"]
    dataset = task["dataset"]

    # === 跳过逻辑（例）===
    if int(task_id) in skip:
        # print(f"Skiping task {int(task_id)}")
        print(f"\033[94mSkipping task {int(task_id)}\033[0m")
        continue

    cmd = [
        "python",
        script,
        "--logdir",
        logdir,
        "--gpu",
        str(gpu),
        "--ckpt_name",
        ckpt_name,
        "--batch_size",
        str(batch_size),
        "--gt_path",
        gt_path,
        step_flag,
        str(step_value),
        "--save_path",
        save_path,
        "--detail_dir",
        "eval_unet_new"
    ]

    if "ddim" in script:
        cmd += ["--ddim_eta", str(eta)]
        cmd += ["--use_ddim"]

    if dataset:
        cmd += ["--dataset", dataset]

    try:
        subprocess.run(cmd, check=True)
        msg = f"[{task_id}] {mode} success: {logdir}"
        success_tasks.append(msg)
    except Exception as e:
        msg = f"[{task_id}] {mode} failed: {logdir}"
        failed_tasks.append(msg)
    finished_tasks.append(int(task_id))
    finished_tasks.sort()
    pbar.update(1)
    print_summary(finished_tasks)

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
