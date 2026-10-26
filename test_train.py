import time
def main():
    print("GPU 监控训练启动，按 Ctrl+C 可退出。")
    time.sleep(3)
    from notify import notify_on_finish, NotifyContext, send_qq_mail
    send_qq_mail(
    f"训练完成!",
    f"训练完成!",
)



if __name__ == "__main__":
    main()