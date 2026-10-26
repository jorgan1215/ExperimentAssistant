# -*- coding: utf-8 -*-
"""
实验通知模块：训练完成 / 失败 / 被打断时，通过 QQ 邮箱发通知。

配置方式（二选一，推荐方式一）：
  方式一：环境变量
    set QQ_SENDER=你的邮箱@qq.com
    set QQ_AUTH_CODE=你的16位SMTP授权码
    set QQ_RECEIVER=接收通知的邮箱（可以和发件人相同，自己发自己）

  方式二：在调用时直接传参 sender / auth_code / receiver

用法示例：
    from notify import notify_on_finish, NotifyContext

    # 用法 1：装饰器，包住整个训练入口
    @notify_on_finish(exp_name="ResNet50-baseline", extra_info="lr=1e-3, bs=64")
    def main():
        ...

    # 用法 2：上下文管理器，包住某一段代码
    with NotifyContext(exp_name="消融实验-无注意力"):
        train()

授权码获取：QQ邮箱网页版 -> 设置 -> 账户 -> POP3/SMTP服务 -> 开启，按提示发短信获取16位授权码。
"""

import argparse
import os
import sys
import time
import smtplib
import socket
import traceback
import functools
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
from datetime import datetime


# ---------- 邮件发送（带超时与兜底，绝不影响训练程序本身） ----------

def send_qq_mail(subject, body, sender=None, auth_code=None, receiver=None,
                 timeout=15, max_retries=2):
    """
    通过 QQ 邮箱 SMTP (SSL 465) 发送纯文本邮件。
    发送失败只打印警告，不抛异常——不让通知功能搞崩训练。
    返回 True/False 表示是否发送成功。
    """
    sender = sender or "2624355457@qq.com"
    auth_code = auth_code or "cgunplcqoymodjga"
    receiver = receiver or "2624355457@qq.com"

    if not sender or not auth_code or not receiver:
        print("[notify] 警告：未配置 QQ_SENDER / QQ_AUTH_CODE / QQ_RECEIVER，跳过发送。")
        print(f"[notify] 原邮件内容 -> 主题: {subject}")
        return False

    msg = MIMEText(body, "plain", "utf-8")
    # QQ 邮箱严格校验 From 头：昵称和邮箱地址必须分开编码后用 formataddr 拼接
    msg["From"] = formataddr((str(Header("实验助手", "utf-8")), sender))
    msg["To"] = formataddr((str(Header("我", "utf-8")), receiver))
    msg["Subject"] = Header(subject, "utf-8")

    for attempt in range(1, max_retries + 1):
        try:
            with smtplib.SMTP_SSL("smtp.qq.com", 465, timeout=timeout) as smtp:
                smtp.login(sender, auth_code)
                smtp.sendmail(sender, [receiver], msg.as_string())
            print(f"[notify] 邮件已发送：{subject}")
            return True
        except Exception as e:
            print(f"[notify] 第 {attempt} 次发送失败：{e}")
            if attempt < max_retries:
                time.sleep(3)
    print("[notify] 多次重试仍失败，放弃发送（训练结果不受影响）。")
    return False


# ---------- 通知内容组装 ----------

def _build_report(exp_name, status, start_time, extra_info="", error_text=""):
    """组装邮件标题和正文。status 取值为 'success' / 'failed' / 'interrupted'。"""
    elapsed = time.time() - start_time
    hours, rem = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(rem, 60)

    status_map = {
        "success": ("✅ 完成", "实验跑完了"),
        "failed": ("❌ 失败", "实验挂了，需要处理"),
        "interrupted": ("⚠️ 被打断", "实验被手动打断（KeyboardInterrupt）"),
    }
    tag, desc = status_map[status]

    subject = f"[实验{tag}] {exp_name}"
    lines = [
        f"实验名称：{exp_name}",
        f"状态：{desc}",
        f"机器：{socket.gethostname()}",
        f"开始时间：{datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}",
        f"结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"耗时：{hours} 小时 {minutes} 分 {seconds} 秒",
    ]
    if extra_info:
        lines.append(f"实验参数：{extra_info}")
    if error_text:
        lines += ["", "—— 错误信息 ——", error_text]
    return subject, "\n".join(lines)


# ---------- 用法 1：装饰器 ----------

def notify_on_finish(exp_name="未命名实验", extra_info="", **mail_cfg):
    """
    装饰器：被装饰的函数跑完发成功邮件，抛异常发失败邮件（并继续抛出），
    Ctrl+C 打断发打断邮件（并继续抛出）。
    若被装饰的函数 return 一个非空字符串，会作为"结果报告"附在成功邮件正文末尾，
    适合评测脚本把指标汇总 return 出来、随邮件一起发。
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
            except KeyboardInterrupt:
                subject, body = _build_report(exp_name, "interrupted", start, extra_info)
                send_qq_mail(subject, body, **mail_cfg)
                raise
            except Exception:
                err = traceback.format_exc(limit=10)  # 只保留最后10层，邮件不至于太长
                subject, body = _build_report(exp_name, "failed", start, extra_info, err)
                send_qq_mail(subject, body, **mail_cfg)
                raise  # 异常继续抛出，不吞错误
            subject, body = _build_report(exp_name, "success", start, extra_info)
            # 函数返回的字符串（如评测指标报告）附在邮件正文末尾
            if isinstance(result, str) and result.strip():
                body += "\n\n—— 结果报告 ——\n" + result
            send_qq_mail(subject, body, **mail_cfg)
            return result
        return wrapper
    return decorator


# ---------- 用法 2：上下文管理器 ----------

class NotifyContext:
    """
    上下文管理器，包住任意代码块：
        with NotifyContext("消融实验"):
            train()
    """
    def __init__(self, exp_name="未命名实验", extra_info="", **mail_cfg):
        self.exp_name = exp_name
        self.extra_info = extra_info
        self.mail_cfg = mail_cfg

    def __enter__(self):
        self.start = time.time()
        print(f"[notify] 开始监控实验：{self.exp_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            status, err = "success", ""
        elif issubclass(exc_type, KeyboardInterrupt):
            status, err = "interrupted", ""
        else:
            status, err = "failed", "".join(
                traceback.format_exception(exc_type, exc_val, exc_tb, limit=10))
        subject, body = _build_report(self.exp_name, status, self.start,
                                      self.extra_info, err)
        send_qq_mail(subject, body, **self.mail_cfg)
        return False  # 不吞异常，照常向外抛


# ---------- 直接运行：发送自定义通知 ----------

# 可以直接在这里修改默认邮件内容，也可以运行时通过命令行参数覆盖。
DEFAULT_SUBJECT = "[GPU 空闲通知] 有 GPU 满足空余显存条件"
DEFAULT_BODY = "GPU 看门狗检测到有显卡的空余显存连续达标，可以安排实验了。"


def main(argv=None):
    """直接发送一封通知，成功返回 0，失败返回 1；导入模块时不会发送。"""
    parser = argparse.ArgumentParser(description="发送自定义 QQ 邮件通知")
    parser.add_argument("--subject", default=DEFAULT_SUBJECT, help="邮件标题")
    parser.add_argument("--body", default=DEFAULT_BODY, help="邮件正文")
    args = parser.parse_args(argv)

    lines = [
        args.body,
        "",
        f"机器：{socket.gethostname()}",
        f"通知时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    # gpu_watch_train.py 启动 job 时会设置此变量，对应触发通知的 GPU。
    gpu = os.getenv("CUDA_VISIBLE_DEVICES")
    if gpu:
        lines.append(f"触发通知的 GPU（CUDA_VISIBLE_DEVICES）：{gpu}")

    ok = send_qq_mail(args.subject, "\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
