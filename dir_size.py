# -*- coding: utf-8 -*-
r"""分析指定目录下各文件夹的占用体积，递归到指定深度。python dir_size.py C:\Users\hp\Documents"""
import os
import sys
from datetime import datetime

MAX_DEPTH = 4


def human_size(size):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024


def dir_size(path):
    """递归计算目录总大小（含所有子目录，不受深度限制）。"""
    total = 0
    try:
        with os.scandir(path) as it:
            for entry in it:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                    elif entry.is_dir(follow_symlinks=False):
                        total += dir_size(entry.path)
                except (PermissionError, OSError):
                    pass
    except (PermissionError, OSError):
        pass
    return total


def walk(path, depth, results):
    """对每个目录计算总大小，递归到 MAX_DEPTH 层。"""
    if depth > MAX_DEPTH:
        return
    results.append((depth, path, dir_size(path)))
    if depth == MAX_DEPTH:
        return
    try:
        with os.scandir(path) as it:
            subs = [e.path for e in it if e.is_dir(follow_symlinks=False)]
    except (PermissionError, OSError):
        return
    for sub in subs:
        walk(sub, depth + 1, results)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    root = os.path.abspath(root)
    log_file = sys.argv[2] if len(sys.argv) > 2 else "dir_size.log"

    results = []
    walk(root, 1, results)
    results.sort(key=lambda x: x[2], reverse=True)

    lines = []
    lines.append(f"目录: {root}  (深度 {MAX_DEPTH} 层)")
    lines.append(f"时间: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("")
    lines.append(f"{'大小':>12}  {'深度':<4}  文件夹")
    lines.append("-" * 80)
    for depth, path, size in results:
        rel = os.path.relpath(path, root)
        lines.append(f"{human_size(size):>12}  {depth:<4}  {rel}")
    output = "\n".join(lines)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(output + "\n")

    print(output)
    print(f"\n结果已写入日志文件: {os.path.abspath(log_file)}")


if __name__ == "__main__":
    main()
