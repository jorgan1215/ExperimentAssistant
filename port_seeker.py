#!/usr/bin/env python3
import argparse
import glob
import os
import pwd
import socket
import struct

""" 
python port_info.py 53 --protocol udp
python port_info.py 1087 --protocol all
"""
def read_text(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def read_link(path):
    try:
        return os.readlink(path)
    except OSError:
        return "无权限或进程已退出"


def username(uid):
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def decode_address(value, ipv6):
    address, port = value.split(":")
    raw = bytes.fromhex(address)
    # /proc/net stores addresses as native-endian 32-bit words.
    raw = b"".join(
        struct.pack("!I", struct.unpack("=I", raw[i:i + 4])[0])
        for i in range(0, len(raw), 4)
    )
    family = socket.AF_INET6 if ipv6 else socket.AF_INET
    host = socket.inet_ntop(family, raw)
    return f"[{host}]:{int(port, 16)}" if ipv6 else f"{host}:{int(port, 16)}"


def find_sockets(port, protocol):
    results = []
    for suffix, ipv6 in (("", False), ("6", True)):
        path = f"/proc/net/{protocol}{suffix}"
        content = read_text(path)
        if content is None:
            continue

        for line in content.splitlines()[1:]:
            fields = line.split()
            if len(fields) < 10:
                continue
            if int(fields[1].split(":")[1], 16) != port:
                continue
            # TCP: only listeners. UDP has no LISTEN state.
            if protocol == "tcp" and fields[3] != "0A":
                continue
            results.append({
                "protocol": protocol.upper() + ("6" if ipv6 else ""),
                "address": decode_address(fields[1], ipv6),
                "uid": int(fields[7]),
                "inode": fields[9],
            })
    return results


def find_owners(inodes):
    owners = {inode: set() for inode in inodes}
    for proc in glob.iglob("/proc/[0-9]*"):
        pid = int(os.path.basename(proc))
        try:
            with os.scandir(f"{proc}/fd") as entries:
                for entry in entries:
                    try:
                        target = os.readlink(entry.path)
                    except OSError:
                        continue
                    if target.startswith("socket:[") and target.endswith("]"):
                        inode = target[8:-1]
                        if inode in owners:
                            owners[inode].add(pid)
        except OSError:
            continue
    return owners


def show_process(pid):
    root = f"/proc/{pid}"
    status = read_text(f"{root}/status") or ""
    fields = {}
    for line in status.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()

    try:
        with open(f"{root}/cmdline", "rb") as f:
            args = [
                part.decode("utf-8", errors="replace")
                for part in f.read().split(b"\0") if part
            ]
        import shlex
        command = shlex.join(args) if args else "不可用"
    except OSError:
        command = "无权限或进程已退出"

    uids = fields.get("Uid", "").split()
    effective_uid = int(uids[1]) if len(uids) > 1 else None
    user = username(effective_uid) if effective_uid is not None else "不可用"

    print(f"  PID        : {pid}")
    print(f"  父进程 PID : {fields.get('PPid', '不可用')}")
    print(f"  进程名     : {fields.get('Name', '不可用')}")
    print(f"  运行用户   : {user}")
    print(f"  进程状态   : {fields.get('State', '不可用')}")
    print(f"  启动命令   : {command}")
    print(f"  程序路径   : {read_link(f'{root}/exe')}")
    print(f"  工作目录   : {read_link(f'{root}/cwd')}")
    print()


def main():
    parser = argparse.ArgumentParser(description="查询 Linux 端口对应的进程信息")
    parser.add_argument("port", type=int, help="端口号，例如 1087")
    parser.add_argument(
        "--protocol", choices=["tcp", "udp", "all"], default="tcp",
        help="默认 tcp，仅查询监听端口；udp 查询绑定端口",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("端口必须在 1–65535 之间")

    protocols = ["tcp", "udp"] if args.protocol == "all" else [args.protocol]
    sockets = [
        item for protocol in protocols
        for item in find_sockets(args.port, protocol)
    ]
    if not sockets:
        print(f"当前网络命名空间中未找到端口 {args.port} 的匹配记录。")
        return

    owners = find_owners({item["inode"] for item in sockets})
    for item in sockets:
        print("=" * 70)
        print(f"协议       : {item['protocol']}")
        print(f"绑定地址   : {item['address']}")
        print(f"Socket 用户: {username(item['uid'])} (UID {item['uid']})")
        print(f"Socket inode: {item['inode']}")
        print()
        pids = owners[item["inode"]]
        if not pids:
            print("无法取得对应 PID：可能受权限限制，或进程已退出。\n")
        for pid in sorted(pids):
            show_process(pid)


if __name__ == "__main__":
    main()