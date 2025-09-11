from tqdm import tqdm
from datetime import datetime
import subprocess
import pandas as pd
import os
import re
import json
import argparse


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


if __name__ == "__main__":
    # add args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_path", type=str, help="Path of the json file that records tasks"
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        help="Path to test set image dir,could be covered in task.json",
    )
    parser.add_argument("--gpu", type=int, help="GPU id")
    parser.add_argument(
        "--batch_size", type=int, help="batch_size to use ,default 6", default=6
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to csv file that records final result of each task",
    )
    parser.add_argument(
        "--detail_dir",
        type=str,
        help="Path to the dir that saves detail results of this run of tasks,could be covered in task.json",
    )
    parser.add_argument(
        "--eta", type=float, help="ETA to use when in DDIM mode", default=0.5
    )
    parser.add_argument(
        "--max_batch", type=int, default=None, help="Max number of batches to eval"
    )
    parser.add_argument(
        "--save_images", action="store_true", help="Whether t save inferene images"
    )

    args = parser.parse_args()

    # load tasks
    task_path = args.task_path
    with open(task_path, "r") as f:
        config = json.load(f)

    tasks = config["tasks"]
    skip = config.get("skip", [])
    primary = config.get("primary", [])

    gpu = args.gpu
    batch_size = args.batch_size
    gt_path = args.gt_path
    eta = args.eta
    save_path = args.save_path
    detail_dir = args.detail_dir
    max_batch = args.max_batch
    save_images = args.save_images
    print(f"save_images: {save_images}")
    success_tasks = []
    failed_tasks = []
    # read extint csv
    # if os.path.exists(save_path):

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

        basename = os.path.basename(logdir)  # 获取最后一层目录名
        match = re.match(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}_(.*)", basename)
        if match:
            exp_name = match.group(1)
        else:
            raise ValueError(f"无法从日志目录名 '{args.logdir}' 中提取实验名称")
        script = task["script"]
        step_flag = task["step_flag"]
        step_value = task["step_value"]
        dataset = task["dataset"]
        ckpt_dir = os.path.join(logdir, "checkpoints")
        ckpts = []
        if os.path.exists(save_path):
            prev = pd.read_csv(save_path)
            finished_ckpts = prev[prev["exp_name"] == exp_name]["ckpt_name"].tolist()
        else:
            finished_ckpts = []
        for name in os.listdir(ckpt_dir):
            if not "v1" in name and not "last" in name:
                ckpts.append(name)
        ckpts.sort()
        print(f"\033[33mCKPTS:\n{ckpts}\033[0m")
        for ckpt_name in ckpts:
            if ckpt_name in finished_ckpts:
                continue
            gt_path = task.get("gt_path", gt_path)
            detail_dir = task.get("detail_dir", detail_dir)
            max_batch = task.get("max_batch", None)
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
                detail_dir,
            ]

            if max_batch:
                cmd += ["--max_batch", str(max_batch)]
            if save_images:
                cmd += ["--save_images"]

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
