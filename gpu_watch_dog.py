#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpu_watch_dog.py — GPU 显存看门狗

持续监控机器上每张 NVIDIA GPU 的显存空余量，当某张卡的空余显存
连续若干次达到启动项要求的阈值时，自动在该卡上启动对应训练脚本。

启动项是一个列表，按顺序检测：每轮检查时优先满足排在前面的启动项。
每个启动项可配置 "remove_on_launch"：
  - True ：启动成功后从列表移除，无定时配置时只跑一次；
  - False：常驻列表，之后再有卡空闲还会再次启动。

requeue_cron：None 不定时重新入队；"@every 30m" 表示成功结束后等待 30 分钟；
标准五字段 cron（如 "*/30 * * * *"）按机器本地时间计算，需要安装 croniter。
配置了 requeue_cron 的任务不会并发重复启动，且看门狗不会自动退出。
相对间隔和普通任务仅依赖 Python 标准库 + nvidia-smi。
直接修改下方「配置区」后运行：python gpu_watch_dog.py
"""

import csv
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta

# ============================ 配置区 ============================

CHECK_INTERVAL = 3        # 每次检查的间隔（秒）
CHECK_TIMES = 3            # 空余显存需连续 N 次达标才触发，防止瞬时抖动
GPU_FILTER = None          # 只监控指定卡，如 [0, 2]；None 表示监控所有卡
LOG_DIR = "./train_logs"   # 训练输出日志目录
WATCH_LOG = None           # 看门狗自身日志文件路径；None 表示只输出到终端
FREE_GPU_DEVICES = None    # 每次启动前更新为本次选中的物理 GPU 编号
# cmd 使用普通字符串中的 {FREE_GPU_DEVICES} 占位符，不要加 f 前缀。
# 也可以省略 CUDA_VISIBLE_DEVICES=...，launch_training 会自动设置子进程环境。
# 启动项列表：按顺序检测，有空闲卡时排在前面的优先启动
JOBS = [
    # {
    #     "cmd": "python test_train.py",   # 要启动的训练命令
    #     "mem_free": 2048,                        # 空余显存达到此值（MiB）才启动
    #     "remove_on_launch": True,                 # 启动成功后是否移除该启动项
    # },
    {
        "cmd": 'python notify.py --subject "GPU 空闲通知" --body "空余显存已达到 40 GiB，可以开始实验。"',
        "mem_free": 40960,  # MiB，40 GiB
        "remove_on_launch": True,
        "requeue_cron": "@every 30m",  # 成功结束后 30 分钟重新入队
        # "requeue_cron": "*/30 * * * *",  # 标准 cron：每小时 0、30 分
        # "requeue_cron": None,  # 不定时重新入队
    },
    {
        "cmd": 'CUDA_VISIBLE_DEVICES={FREE_GPU_DEVICES} /home/yanmeng/xingqiaogen/miniconda3/envs/swift1/bin/python /home/public_space/yanmeng/xingqiaogen/retror1/train_grpo_trl_retror1_3b_processed_cot.py',
        "mem_free": 55*1024,  # MiB，按需修改
        "remove_on_launch": True,
        "requeue_cron": None,
    },
    # 可以再添加更多启动项，例如：
    # {
    #     "cmd": "python train_sft_trl_retror1.py --config exp2.yaml",
    #     "mem_free": 40960,
    #     "remove_on_launch": False,
    # },
]

# ===============================================================


def setup_logging(log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def find_nvidia_smi():
    """跨平台定位 nvidia-smi。"""
    path = shutil.which("nvidia-smi")
    if path:
        return path
    # Windows 常见安装路径
    win_path = r"C:\Windows\System32\nvidia-smi.exe"
    if os.path.exists(win_path):
        return win_path
    raise RuntimeError("找不到 nvidia-smi，请确认已安装 NVIDIA 驱动并加入 PATH。")


def query_gpus(smi_path):
    """
    返回每张卡的状态列表：
    [{"index": 0, "name": "...", "mem_used": 1024, "mem_total": 24576}, ...]
    mem 单位 MiB。
    """
    cmd = [
        smi_path,
        "--query-gpu=index,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi 执行失败: {out.stderr.strip()}")

    gpus = []
    reader = csv.reader(io.StringIO(out.stdout))
    for row in reader:
        if len(row) < 4:
            continue
        gpus.append({
            "index": int(row[0].strip()),
            "name": row[1].strip(),
            "mem_used": int(row[2].strip()),
            "mem_total": int(row[3].strip()),
        })
    return gpus


def next_requeue_time(expression, finished_at):
    """从检测到成功退出的时间算下次入队时间；cron 使用机器本地时间。"""
    if expression is None:
        return None
    if not isinstance(expression, str):
        raise ValueError("requeue_cron 必须为 None 或字符串")
    expression = expression.strip()
    interval = re.fullmatch(r"@every\s+([1-9]\d*)([smhd])", expression)
    if interval:
        count, unit = interval.groups()
        seconds = int(count) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        return finished_at + timedelta(seconds=seconds)
    if len(expression.split()) != 5:
        raise ValueError("requeue_cron 需为五字段 cron 或 @every 30m（支持 s/m/h/d）")
    try:
        from croniter import croniter
    except ImportError as exc:
        raise ValueError("标准 cron 需要安装依赖：python -m pip install croniter") from exc
    try:
        return croniter(expression, finished_at).get_next(datetime)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"无法计算 requeue_cron={expression!r}: {exc}") from exc


def prepare_jobs(jobs):
    """复制并提前检查配置，避免任务跑完后才发现表达式错误。"""
    prepared = []
    for i, original in enumerate(jobs):
        job = dict(original)
        if "cmd" not in job or "mem_free" not in job:
            raise ValueError(f"JOBS[{i}] 缺少 cmd 或 mem_free 字段")
        if not isinstance(job["cmd"], str) or not job["cmd"].strip():
            raise ValueError(f"JOBS[{i}].cmd 必须为非空字符串")
        if "CUDA_VISIBLE_DEVICES=None" in job["cmd"]:
            raise ValueError(f"JOBS[{i}].cmd 已提前得到 None，请移除 f 前缀，使用普通字符串占位符")
        job.setdefault("remove_on_launch", True)
        job.setdefault("requeue_cron", None)
        try:
            next_requeue_time(job["requeue_cron"], datetime.now())
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"JOBS[{i}]: {exc}") from exc
        prepared.append(job)
    return prepared


def launch_training(cmd, gpu_index, log_dir):
    """在指定 GPU 上启动训练脚本，stdout/stderr 写入日志文件。"""
    global FREE_GPU_DEVICES
    FREE_GPU_DEVICES = gpu_index
    # 仅替换指定占位符，避免 str.format 破坏命令里的其他花括号。
    cmd = cmd.replace("{FREE_GPU_DEVICES}", str(FREE_GPU_DEVICES))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = os.path.join(log_dir, f"train_gpu{gpu_index}_{ts}.log")

    logging.info(f"GPU {gpu_index} 满足条件，启动训练: {cmd}")
    logging.info(f"训练日志: {log_path}")

    with open(log_path, "a", encoding="utf-8", errors="replace") as log_f:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
    return proc, log_path


def main():
    setup_logging(WATCH_LOG)
    if not JOBS:
        logging.error("JOBS 列表为空，请先在脚本配置区添加启动项。")
        sys.exit(1)
    try:
        jobs = prepare_jobs(JOBS)
    except ValueError as exc:
        logging.error("任务配置错误：%s", exc)
        sys.exit(1)
    smi_path = find_nvidia_smi()

    pending = list(jobs)
    running = {}        # gpu_index -> (Popen, job)
    waiting = []        # (下次入队 datetime, job)，仅成功退出后添加
    keep_alive = any(job["requeue_cron"] is not None for job in jobs)
    idle_streak = {}
    streak_job = {}     # 用任务对象身份计数，相同命令的不同任务不会混淆

    logging.info("=" * 60)
    logging.info(f"GPU 看门狗已启动 | 间隔 {CHECK_INTERVAL}s | 连续 {CHECK_TIMES} 次触发")
    for i, job in enumerate(pending):
        logging.info(f"  启动项 {i}: 空余显存 >= {job['mem_free']} MiB -> {job['cmd']} "
                     f"(remove_on_launch={job['remove_on_launch']}, "
                     f"requeue_cron={job['requeue_cron']!r})")
    logging.info("=" * 60)

    try:
        while True:
            # 成功执行指子进程返回码为 0，而不是 Popen 启动成功。
            for idx in list(running):
                proc, job = running[idx]
                returncode = proc.poll()
                if returncode is None:
                    continue
                logging.info("GPU %s 上的任务已退出（返回码 %s）", idx, returncode)
                del running[idx]
                idle_streak[idx] = 0
                streak_job.pop(idx, None)
                if returncode == 0 and job["requeue_cron"] is not None:
                    due = next_requeue_time(job["requeue_cron"], datetime.now())
                    waiting.append((due, job))
                    logging.info("任务 [%s] 将于 %s 重新入队", job["cmd"], due)
                elif returncode != 0:
                    logging.warning("任务 [%s] 执行失败，不安排定时重新入队", job["cmd"])

            now = datetime.now()
            for due, job in waiting[:]:
                if now >= due:
                    pending.append(job)
                    waiting.remove((due, job))
                    logging.info("任务 [%s] 已到期，重新加入队列", job["cmd"])

            if not pending:
                if not running and not waiting and not keep_alive:
                    logging.info("所有启动项均已完成且没有在跑的任务，看门狗退出。")
                    break
                time.sleep(CHECK_INTERVAL)
                continue

            try:
                gpus = query_gpus(smi_path)
            except Exception as e:
                logging.warning(f"查询 GPU 状态失败: {e}，{CHECK_INTERVAL}s 后重试")
                time.sleep(CHECK_INTERVAL)
                continue

            for gpu in gpus:
                idx = gpu["index"]
                if GPU_FILTER is not None and idx not in GPU_FILTER:
                    continue
                if idx in running:
                    continue  # 这张卡已经被我们占用了
                if not pending:
                    break

                mem_free = gpu["mem_total"] - gpu["mem_used"]

                # 按顺序找到第一个该卡能满足的启动项
                target = None
                for job in pending:
                    if mem_free >= job["mem_free"]:
                        target = job
                        break

                if target is None:
                    if idle_streak.get(idx, 0) > 0:
                        logging.info(f"GPU {idx} 空余显存回落到 {mem_free} MiB，计数清零。")
                    idle_streak[idx] = 0
                    streak_job.pop(idx, None)
                    continue

                # 同一张卡若匹配的启动项变了（比如前面的项被移除），重新计数
                if streak_job.get(idx) != id(target):
                    idle_streak[idx] = 0
                    streak_job[idx] = id(target)

                idle_streak[idx] = idle_streak.get(idx, 0) + 1
                logging.info(
                    f"GPU {idx} [{gpu['name']}] 空余显存 {mem_free} MiB "
                    f"-> 匹配启动项 [{target['cmd']}] ({idle_streak[idx]}/{CHECK_TIMES})"
                )

                if idle_streak[idx] >= CHECK_TIMES:
                    try:
                        proc, _ = launch_training(target["cmd"], idx, LOG_DIR)
                        running[idx] = (proc, target)
                        idle_streak[idx] = 0
                        streak_job.pop(idx, None)
                        # 配置定时重入队时始终移出，避免在其他 GPU 上立即重复启动。
                        if target["remove_on_launch"] or target["requeue_cron"] is not None:
                            pending = [job for job in pending if job is not target]
                            # 清掉其他卡对同一任务的旧计数，重新入队后必须重新连续达标。
                            for other_idx in list(streak_job):
                                if streak_job[other_idx] == id(target):
                                    idle_streak[other_idx] = 0
                                    del streak_job[other_idx]
                            logging.info(f"启动项 [{target['cmd']}] 已启动并移出队列，"
                                         f"剩余 {len(pending)} 项。")
                    except Exception as e:
                        logging.error(f"GPU {idx} 启动训练失败: {e}")

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        logging.info("收到 Ctrl+C，看门狗退出。")
        for idx, (proc, job) in running.items():
            if proc.poll() is None:
                logging.info(f"GPU {idx} 上的训练进程（PID {proc.pid}）仍在运行，"
                             f"看门狗不会杀它；如需停止请自行 kill。")


if __name__ == "__main__":
    main()
