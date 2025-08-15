from tqdm import tqdm
from datetime import datetime
import subprocess
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
        "--ckpt_name", type=str, help="The name of ckpt that you want to eval"
    )
    parser.add_argument("--gt_path", type=str, help="Path to test set image dir,could be covered in task.json")
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
        "--save_images",
        action="store_true",
        help="Whether t save inferene images"
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
    ckpt_name = args.ckpt_name
    batch_size = args.batch_size
    gt_path = args.gt_path
    eta = args.eta
    save_path = args.save_path
    detail_dir = args.detail_dir
    save_images = args.save_images
    print(f"save_images: {save_images}")
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
        ckpt_name = task.get("ckpt_name",ckpt_name)
        gt_path = task.get("gt_path",gt_path)
        detail_dir = task.get("detail_dir",detail_dir)
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
        
        if save_images:
            cmd+=["--save_images"]

        if "ddim" in script:
            cmd += ["--ddim_eta", str(eta)]
            cmd += ["--use_ddim"]

        if dataset:
            cmd += ["--dataset", dataset]
        # try:
        #     # 捕获stdout和stderr
        #     result = subprocess.run(
        #         cmd,
        #         check=True,
        #         stdout=subprocess.PIPE,  # 捕获标准输出
        #         stderr=subprocess.PIPE,  # 捕获标准错误
        #         text=True  # 以文本形式返回（Python 3.7+）
        #     )
        #     msg = f"[{task_id}] {mode} success: {logdir}"
        #     success_tasks.append(msg)
            
        # except subprocess.CalledProcessError as e:
        #     # 子进程返回非零状态码
        #     error_msg = f"""
        #     [{task_id}] {mode} failed: {logdir}
        #     Command: {e.cmd}
        #     Return code: {e.returncode}
        #     Output:
        #     {e.stdout}
        #     Error:
        #     {e.stderr}
        #     """
        #     failed_tasks.append(error_msg.strip())
        #     print(error_msg)  # 打印详细错误信息
            
        # except Exception as e:
        #     # 其他异常（如文件未找到等）
        #     error_msg = f"""
        #     [{task_id}] {mode} failed with unexpected error: {logdir}
        #     Error type: {type(e).__name__}
        #     Error message: {str(e)}
        #     """
        #     failed_tasks.append(error_msg.strip())
        #     print(error_msg)
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
