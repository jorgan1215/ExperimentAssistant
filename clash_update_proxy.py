#!/usr/bin/env python3
"""
python clash_update_proxy.py \
  --config ~/.config/clash/config.yaml \
  --url-file ~/.config/clash/subscription.url \
  --dry-run
"""
import argparse
import copy
import datetime
import fcntl
import os
from pathlib import Path
import re
import socket
import ssl
import stat
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error

try:
    import yaml
except ImportError:
    sys.exit("缺少 PyYAML。请在当前环境执行：python -m pip install PyYAML；conda 环境也可执行 conda install pyyaml")

MAX_SIZE = 16 * 1024 * 1024
RESERVED = {"DIRECT", "REJECT", "REJECT-DROP", "PASS", "COMPATIBLE", "GLOBAL"}
INFO_NODE = re.compile(r"^(剩余流量|套餐到期|账号邮箱|官网地址)\s*[:：]")
STAGE = "启动"


class Invalid(ValueError):
    pass


class StrictLoader(yaml.SafeLoader):
    pass


def mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str) or key in result:
            raise Invalid("YAML 存在重复键或非文本键，请先整理配置。")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)


def parse(text):
    # Anchors can couple changed sections to settings that should stay untouched.
    if any(isinstance(e, yaml.AliasEvent) or getattr(e, "anchor", None)
           for e in yaml.parse(text)):
        raise Invalid("暂不支持 YAML 锚点/别名，请先展开它们。")
    data = yaml.load(text, Loader=StrictLoader)
    if not isinstance(data, dict):
        raise Invalid("需要 Clash/Mihomo YAML 订阅，不能是网页、Base64 或 vmess:// 链接列表。")
    return data, yaml.compose(text, Loader=StrictLoader)


def pair(node, key):
    for k, v in node.value:
        if k.value == key:
            return k, v
    raise Invalid("本地配置缺少字段：" + key)


def nodes(data):
    items = data.get("proxies")
    if not isinstance(items, list) or not items:
        raise Invalid("proxies 必须是非空节点列表；仅含 proxy-providers 的订阅不适用。")
    names = []
    for p in items:
        if not isinstance(p, dict) or not all(isinstance(p.get(k), str) and p[k]
                                             for k in ("name", "type", "server")):
            raise Invalid("订阅节点缺少有效的 name/type/server。")
        port = p.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise Invalid("订阅节点端口无效。")
        names.append(p["name"])
    if len(names) != len(set(names)) or set(names) & RESERVED:
        raise Invalid("节点名称重复或与内置名称冲突。")
    return names


def patch(text, key_node, value_node, value):
    column = key_node.start_mark.column
    start = key_node.start_mark.index - column
    end = value_node.end_mark.index
    # Keep indentation belonging to the following key outside this patch.
    if value_node.end_mark.column:
        line_start = text.rfind("\n", 0, end) + 1
        if not text[line_start:end].strip():
            end = line_start
    dumped = yaml.safe_dump({key_node.value: value}, allow_unicode=True,
                            sort_keys=False, width=4096)
    replacement = "".join(" " * column + line for line in dumped.splitlines(True))
    # Inline flow values may have another key on the same line. Refuse those layouts.
    if text[end:].split("\n", 1)[0].strip() and value_node.end_mark.column:
        raise Invalid("待更新字段使用了复杂行内布局，请改为普通多行 YAML。")
    return start, end, replacement


def merge(text, subscription):
    old, root = parse(text)
    incoming, _ = parse(subscription)
    old_names, new_names = nodes(old), nodes(incoming)
    old_set, new_set = set(old_names), set(new_names)
    groups = old.get("proxy-groups", [])
    if not isinstance(groups, list):
        raise Invalid("proxy-groups 必须是列表。")
    group_names = [g.get("name") for g in groups if isinstance(g, dict)]
    if (len(group_names) != len(groups) or any(not isinstance(n, str) for n in group_names)
            or len(set(group_names)) != len(group_names)
            or set(group_names) & (old_set | new_set)):
        raise Invalid("策略组结构无效，或组名与节点名冲突。")
    result = copy.deepcopy(old)
    result["proxies"] = incoming["proxies"]
    k, v = pair(root, "proxies")
    edits = [patch(text, k, v, incoming["proxies"])]
    group_root = pair(root, "proxy-groups")[1] if groups else None
    for index, group in enumerate(groups):
        refs = group.get("proxies", [])
        if not isinstance(refs, list) or any(not isinstance(x, str) for x in refs):
            raise Invalid("策略组的 proxies 必须是名称列表。")
        selected = old_set.intersection(refs)
        if not selected:
            continue
        if selected == old_set:
            replacements = new_names
        else:
            # Keep existing exclusions; add only genuinely new nodes.
            replacements = [n for n in refs if n in selected and n in new_set]
            replacements += [n for n in new_names if n not in old_set]
            if group.get("type") in {"url-test", "fallback", "load-balance"}:
                replacements = [n for n in replacements if not INFO_NODE.match(n)]
        updated, inserted = [], False
        for ref in refs:
            if ref in old_set:
                if not inserted:
                    updated.extend(replacements)
                    inserted = True
            else:
                updated.append(ref)
        if not updated and not group.get("use"):
            raise Invalid("更新后某个策略组为空，已取消写入。")
        if updated != refs:
            result["proxy-groups"][index]["proxies"] = updated
            k, v = pair(group_root.value[index], "proxies")
            edits.append(patch(text, k, v, updated))
    merged = text
    previous = len(text) + 1
    for start, end, replacement in sorted(edits, reverse=True):
        if end > previous:
            raise Invalid("更新范围重叠，已取消写入。")
        merged = merged[:start] + replacement + merged[end:]
        previous = start
    verified, _ = parse(merged)
    if verified != result:
        raise Invalid("合并校验失败，原配置未修改。")
    return merged, old_names, new_names


def fetch(args):
    if args.subscription_file:
        raw = Path(args.subscription_file).expanduser().read_bytes()
    else:
        url = Path(args.url_file).expanduser().read_text(encoding="utf-8").strip()
        if not url.startswith(("https://", "http://")) or any(c.isspace() for c in url):
            raise Invalid("链接文件应只包含一行 http(s) 订阅链接。")
        handlers = []
        if args.proxy:
            if not args.proxy.startswith(("http://", "https://")):
                raise Invalid("--proxy 请使用 HTTP 代理地址，例如 http://127.0.0.1:1087")
            handlers.append(urllib.request.ProxyHandler({"http": args.proxy, "https": args.proxy}))
        else:
            handlers.append(urllib.request.ProxyHandler({}))
        opener = urllib.request.build_opener(*handlers)
        request = urllib.request.Request(url, headers={"User-Agent": "clash.meta"})
        with opener.open(request, timeout=60) as response:
            raw = response.read(MAX_SIZE + 1)
    if len(raw) > MAX_SIZE:
        raise Invalid("订阅超过 16 MiB，已取消。")
    return raw.decode("utf-8-sig")


def main():
    global STAGE
    parser = argparse.ArgumentParser(description="只更新 Clash 节点与组内引用，保留其他设置")
    parser.add_argument("--config", required=True, help="本地 config.yml 路径")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url-file", help="只包含订阅链接的文件")
    source.add_argument("--subscription-file", help="读取已下载的 Clash YAML，适合离线验证")
    parser.add_argument("--proxy", help="下载代理，例如 http://127.0.0.1:1087；默认直连")
    parser.add_argument("--core", help="可选：Clash/Mihomo 可执行文件路径，写入前执行 -t 检查")
    parser.add_argument("--dry-run", action="store_true", help="仅验证并显示数量，不写文件")
    args = parser.parse_args()
    STAGE = "读取本地配置"
    config = Path(args.config).expanduser().resolve(strict=True)
    STAGE = "在配置目录创建锁文件"
    with open(str(config) + ".subscription.lock", "a") as lock:
        os.chmod(lock.name, 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        STAGE = "读取本地配置"
        original = config.read_bytes()
        text = original.decode("utf-8-sig")
        newline = "\r\n" if b"\r\n" in original else "\n"
        STAGE = "读取订阅文件" if args.subscription_file else "读取链接文件并下载订阅"
        subscription = fetch(args)
        STAGE = "解析 YAML 并合并节点"
        merged, old_names, new_names = merge(text.replace("\r\n", "\n"), subscription)
        content = merged.replace("\n", newline).encode("utf-8")
        if original.startswith(b"\xef\xbb\xbf"):
            content = b"\xef\xbb\xbf" + content
        print("节点：{} → {}；新增 {}，移除 {}。".format(
            len(old_names), len(new_names), len(set(new_names) - set(old_names)),
            len(set(old_names) - set(new_names))))
        if content == original:
            print("内容未变化，无需写入。")
            return
        STAGE = "在配置目录写入临时文件"
        fd, temp_name = tempfile.mkstemp(prefix=".subscription-", suffix=".yaml", dir=config.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            if args.core:
                STAGE = "执行内核配置检查"
                check = subprocess.run([args.core, "-t", "-d", str(config.parent), "-f", temp_name],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
                if check.returncode:
                    raise Invalid("内核配置检查失败，原配置未修改。可用 --subscription-file 在本机排查。")
            if args.dry_run:
                print("预览通过，未修改 config.yml；去掉 --dry-run 即可更新。")
                return
            STAGE = "核对原配置并创建备份"
            if config.read_bytes() != original:
                raise Invalid("配置被其他程序修改，请重试。")
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            backup = config.with_name(config.name + ".bak." + stamp)
            with backup.open("xb") as stream:
                os.chmod(backup, 0o600)
                stream.write(original)
            meta = config.stat()
            if os.geteuid() == 0:
                os.chown(temp_name, meta.st_uid, meta.st_gid)
            os.chmod(temp_name, stat.S_IMODE(meta.st_mode))
            STAGE = "替换配置文件"
            os.replace(temp_name, config)
            print("更新完成；备份：" + str(backup))
            print("请重新加载配置或重启你的 Clash/Mihomo 服务，使本次更新生效。")
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


def error_message(error):
    """Only expose classifications and numeric locations, never response text or URLs."""
    if isinstance(error, Invalid):
        return str(error)
    if isinstance(error, urllib.error.HTTPError):
        hints = {401: "订阅鉴权失败，请确认链接有效。", 403: "服务器拒绝访问，请确认链接或访问限制。",
                 404: "订阅地址不存在，请重新复制 Clash/Mihomo 订阅链接。",
                 429: "请求过于频繁，请稍后重试。"}
        return "订阅服务器返回 HTTP {}。{}".format(error.code, hints.get(error.code, "请检查订阅服务状态。"))
    reason = error.reason if isinstance(error, urllib.error.URLError) else error
    if isinstance(reason, ssl.SSLCertVerificationError):
        return "HTTPS 证书验证失败。请检查系统时间和当前 conda 环境的 CA 证书。"
    if isinstance(reason, ssl.SSLError):
        return "HTTPS/TLS 握手失败，请检查网络、代理或订阅服务。"
    if isinstance(reason, socket.gaierror):
        return "DNS 解析失败，无法解析订阅或代理服务器地址。"
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return "连接或读取超时。若订阅无法直连，可尝试 --proxy http://127.0.0.1:1087。"
    if isinstance(reason, ConnectionRefusedError):
        return "连接被拒绝。若使用 --proxy，请检查代理是否已启动以及端口是否正确。"
    if isinstance(error, urllib.error.URLError):
        code = getattr(reason, "errno", None)
        return "订阅下载连接失败{}。请检查网络；脚本默认直连，可用 --proxy 指定代理。".format(
            "（系统错误码 {}）".format(code) if isinstance(code, int) else "")
    if isinstance(error, PermissionError):
        return "权限不足。需要读取配置和链接文件，并在配置所在目录创建锁、临时文件及备份；仅配置文件可写还不够。"
    if isinstance(error, FileNotFoundError):
        return "文件或命令不存在。请检查当前阶段对应的 --config、--url-file、--subscription-file 或 --core 路径。"
    if isinstance(error, BlockingIOError):
        return "另一个更新任务正在运行，请等待其完成。"
    if isinstance(error, UnicodeDecodeError):
        return "文件或订阅不是 UTF-8 文本；请检查是否返回了压缩内容或其他编码。"
    if isinstance(error, UnicodeEncodeError):
        return "请求地址含无法编码的字符；请复制服务商提供的完整订阅链接。"
    if isinstance(error, yaml.YAMLError):
        mark = getattr(error, "problem_mark", None)
        where = "（第 {} 行，第 {} 列）".format(mark.line + 1, mark.column + 1) if mark else ""
        return "YAML 语法或类型错误{}。请确认订阅为 Clash/Mihomo YAML，且不是网页或链接列表。".format(where)
    if isinstance(error, subprocess.TimeoutExpired):
        return "内核配置检查超过 120 秒，可能正在等待规则或数据文件下载。"
    if isinstance(error, OSError):
        return "系统操作失败（错误码 {}）。请检查磁盘空间、文件类型和目录权限。".format(error.errno)
    return "参数或数据格式无效；错误类型 {}。".format(type(error).__name__)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, yaml.YAMLError, subprocess.SubprocessError) as error:
        print("更新失败；阶段：{}。{}\n原配置未被覆盖。".format(STAGE, error_message(error)), file=sys.stderr)
        sys.exit(1)
