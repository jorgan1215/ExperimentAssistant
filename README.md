# GPU 空闲邮件通知

`notify.py` 保留原有的装饰器和上下文管理器，同时支持直接运行发送一封邮件。

## 1. 配置邮箱

Linux / macOS（在启动看门狗的同一个终端里执行）：

```bash
export QQ_SENDER="你的邮箱@qq.com"
export QQ_AUTH_CODE="你的SMTP授权码"
export QQ_RECEIVER="接收通知的邮箱"
```

Windows CMD 使用 `set QQ_SENDER=...` 等命令。未设置 `QQ_RECEIVER` 时默认发给自己。
授权码通过环境变量配置，不写入源码。

## 2. 直接发送

```bash
python notify.py --subject "GPU 空闲了" --body "空余显存已达到 20 GiB，可以开始实验。"
```

也可以修改 `notify.py` 中的 `DEFAULT_SUBJECT` 和 `DEFAULT_BODY`，然后执行 `python notify.py`。
正文会自动附上机器名、时间，以及看门狗传入的 GPU 编号。
发送成功退出码为 0，失败为 1。导入模块不会发送邮件。

## 3. 加入代码2的 JOBS

把 `notify.py` 放到看门狗的运行目录，在 `JOBS` 中加入：

```python
{
    "cmd": 'python notify.py --subject "GPU 空闲通知" --body "空余显存已达到 20 GiB，可以开始实验。"',
    "mem_free": 20480,  # MiB，按实际需求修改
    "remove_on_launch": True,
},
```

看门狗每隔 `CHECK_INTERVAL` 秒查询一次，连续 `CHECK_TIMES` 次达标后执行该命令。
通知项放在 `JOBS` 前面可以优先触发；看门狗会在通知进程结束后继续检查后续训练项。
如果运行目录不同，请在命令中为 `notify.py` 使用带引号的绝对路径。

未配置定时重新入队时，`remove_on_launch=True` 表示该通知项启动后就从队列移除，只触发一次（不是每张卡各一次）。
移除发生在进程启动时，即使邮件最终发送失败也不会重新入队，发送结果可在 `train_logs` 查看。
设为 `False` 会在显存持续空闲时反复触发并发送邮件。

## 4. 成功执行后的定时重新入队

`gpu_watch_dog.py` 的启动项新增 `requeue_cron`：

```python
{
    "cmd": 'python notify.py --subject "GPU 空闲通知" --body "空余显存已达到 40 GiB，可以开始实验。"',
    "mem_free": 40960,
    "remove_on_launch": True,
    "requeue_cron": "@every 30m",
},
```

- `"@every 30m"`：本程序扩展语法，从检测到任务成功结束开始等待 30 分钟后重新入队。支持 `s`、`m`、`h`、`d`，无需额外依赖。
- `"*/30 * * * *"`：标准五字段 cron，每小时第 0、30 分钟重新入队。例如 10:17 成功结束，10:30 入队；`@every 30m` 则为 10:47。标准 cron 使用运行机器的本地时间，需要先执行 `python -m pip install croniter`。
- `None`：不安排定时重新入队。搭配 `remove_on_launch=True` 执行一次后结束该任务；搭配 `False` 保留原先常驻重复执行的行为。

定时重新入队只在进程返回码为 `0` 时安排，失败不会安排下一次。
配置了定时重新入队的任务在运行、等待期间不在启动队列中，即使 `remove_on_launch=False` 也不会在多张 GPU 上立即重复启动。
到期后加入队尾，仍需 GPU 空余显存连续 `CHECK_TIMES` 次达标才执行，不保证到点立刻执行。
有任何任务配置了 `requeue_cron`，看门狗就保持运行，包括该任务执行失败的情况；可用 Ctrl+C 退出。
计时误差取决于轮询间隔和查询耗时。等待队列仅保存在当前进程内，重启后从初始 `JOBS` 开始。

## 5. 启动时指定空闲 GPU

每次执行任务前更新全局变量 `FREE_GPU_DEVICES`，并设置该子进程的 `CUDA_VISIBLE_DEVICES`。
命令中需要引用 GPU 编号时，用普通字符串占位符，**不要加 `f` 前缀**：

```python
{
    "cmd": 'CUDA_VISIBLE_DEVICES={FREE_GPU_DEVICES} /home/yanmeng/xingqiaogen/miniconda3/envs/swift1/bin/python /home/public_space/yanmeng/xingqiaogen/retror1/train_grpo_trl_retror1_3b_processed_cot.py',
    "mem_free": 55 * 1024,
    "remove_on_launch": True,
    "requeue_cron": None,
},
```

这种命令前置环境变量的语法用于 Linux / macOS；跨平台用法是直接省略 `CUDA_VISIBLE_DEVICES={FREE_GPU_DEVICES}` 前缀，程序已自动通过子进程环境指定对应卡。
全局变量表示最近一次启动使用的物理 GPU 编号，每个子进程获得独立环境，不受后续全局变量更新影响。
子进程只看到所选 GPU 时，训练框架内部通常使用逻辑编号 `cuda:0`。

验证：`python -m unittest -v test_gpu_watch_dog`。测试使用模拟显卡和时钟，不启动实际训练或发送邮件；未安装 croniter 时跳过标准 cron 测试。
