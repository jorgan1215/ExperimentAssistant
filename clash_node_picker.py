#!/usr/bin/env python3
"""Interactive Clash/Mihomo latency test and node selection. Standard library only."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request


class Error(Exception):
    pass


def clean(value):
    return ''.join(c if c.isprintable() else '?' for c in str(value))


class API:
    def __init__(self, address, secret=''):
        self.address = address.rstrip('/')
        self.secret = secret

    def call(self, path, method='GET', data=None, timeout=10):
        headers = {'Accept': 'application/json'}
        if self.secret:
            headers['Authorization'] = 'Bearer ' + self.secret
        body = None
        if data is not None:
            headers['Content-Type'] = 'application/json'
            body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        req = urllib.request.Request(self.address + path, data=body, headers=headers, method=method)
        # A local control request must not follow terminal proxy settings or redirects.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        try:
            with opener.open(req, timeout=timeout) as response:
                raw = response.read(16 * 1024 * 1024 + 1)
            if len(raw) > 16 * 1024 * 1024:
                raise Error('控制接口响应过大。')
            return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise Error('控制接口鉴权失败：请通过 --secret-file 或 CLASH_SECRET 提供配置中的 secret。') from None
            raise Error('接口返回 HTTP {}。'.format(exc.code)) from None
        except (TimeoutError, socket.timeout):
            raise Error('请求超时') from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise Error('请求超时') from None
            raise Error('无法连接控制接口；请确认 Clash 已启动且 --api 地址正确。') from None
        except (ValueError, UnicodeError):
            raise Error('接口未返回有效 JSON；请确认连接的是 external-controller，而非代理端口。') from None

    def snapshot(self):
        configs = self.call('/configs')
        proxies = self.call('/proxies').get('proxies')
        if not isinstance(proxies, dict):
            raise Error('控制接口返回的 proxies 结构无效。')
        return configs.get('mode', '').lower(), proxies


def endpoint(name):
    return '/proxies/' + urllib.parse.quote(name, safe='')


def is_selector(proxy):
    return str(proxy.get('type', '')).lower() == 'selector'


def physical_nodes(proxies):
    builtins = {'direct', 'reject', 'rejectdrop', 'reject-drop', 'pass', 'compatible', 'dns'}
    return [name for name, value in proxies.items()
            if isinstance(value, dict) and 'all' not in value
            and str(value.get('type', '')).lower() not in builtins]


def route(proxies, group, target):
    """Find a route through manual selectors, without pinning automatic groups."""
    queue = deque([(group, [])])
    visited = set()
    while queue:
        name, steps = queue.popleft()
        if name in visited:
            continue
        visited.add(name)
        entry = proxies.get(name, {})
        if not is_selector(entry):
            continue
        members = entry.get('all', [])
        if target in members:
            return steps + [(name, target)]
        for child in members:
            if is_selector(proxies.get(child, {})):
                queue.append((child, steps + [(name, child)]))
    return None


def effective_node(proxies, group):
    seen = set()
    while group not in seen and group in proxies:
        seen.add(group)
        item = proxies[group]
        if 'all' not in item:
            return group
        group = item.get('now')
    return None


def select_group(mode, proxies, requested):
    if mode == 'direct':
        raise Error('当前为 direct 直连模式，切换节点不会生效。请先在 Clash 中改为 global 或 rule 模式。')
    if requested:
        group = requested
    elif mode == 'global':
        group = 'GLOBAL'
    else:
        candidates = [n for n, p in proxies.items() if n != 'GLOBAL' and is_selector(p)]
        preferred = [n for n in candidates if '节点选择' in n]
        if len(preferred) == 1:
            group = preferred[0]
        elif len(candidates) == 1:
            group = candidates[0]
        else:
            print('可用的手动策略组：')
            for i, name in enumerate(candidates, 1):
                print('  {}. {}'.format(i, clean(name)))
            if not candidates:
                raise Error('没有可手动切换的策略组。')
            answer = input('输入策略组序号或完整名称（q 退出）：').strip()
            if answer.lower() == 'q':
                raise EOFError
            group = answer if answer in candidates else (
                candidates[int(answer) - 1] if answer.isdecimal() and 1 <= int(answer) <= len(candidates) else '')
    if not is_selector(proxies.get(group, {})):
        raise Error('指定的策略组不存在或不是手动选择组；可用 --group 指定。')
    if mode == 'global' and group != 'GLOBAL':
        raise Error('global 模式请使用 GLOBAL 组，使选择直接作用于全局流量。')
    return group


def measure(api, names, args):
    def one(name):
        query = urllib.parse.urlencode({'url': args.test_url, 'timeout': args.timeout})
        try:
            result = api.call(endpoint(name) + '/delay?' + query, timeout=args.timeout / 1000 + 3)
            delay = result.get('delay')
            if isinstance(delay, bool) or not isinstance(delay, (int, float)) or delay < 0:
                return None, '测试失败'
            return delay, ''
        except Error as exc:
            return None, str(exc)
    results = {}
    print('正在测试 {} 个节点，并发 {}，每个节点超时 {} ms…'.format(len(names), args.workers, args.timeout), flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(one, name): name for name in names}
        for future in as_completed(jobs):
            results[jobs[future]] = future.result()
            print('\r已完成 {}/{}'.format(len(results), len(names)), end='', flush=True)
    print()
    return sorted(names, key=lambda n: (results[n][0] is None,
                                       results[n][0] if results[n][0] is not None else float('inf'), n)), results


def display(names, results, proxies, group):
    current = effective_node(proxies, group)
    print('\n序号   延迟        节点名称（* 当前节点；[不可选] 表示此组无法切换到该节点）')
    print('-' * 75)
    for i, name in enumerate(names, 1):
        delay, reason = results[name]
        latency = '{} ms'.format(delay) if delay is not None else '失败/超时'
        flag = ' *' if name == current else ''
        if route(proxies, group, name) is None:
            flag += ' [不可选]'
        print('{:>4}   {:<11} {}{}'.format(i, latency, clean(name), flag))
    print('\n测速为到测试网址的 HTTP 延迟，不是下载速度。')
    failed = sum(results[n][0] is None for n in names)
    if failed:
        print('{} 个节点测试失败；可输入 e 查看失败原因。'.format(failed))


def switch(api, group, name, expected_mode):
    mode, proxies = api.snapshot()
    if mode != expected_mode:
        raise Error('运行模式已变化，请输入 r 重新读取并测速后再选。')
    steps = route(proxies, group, name)
    if steps is None:
        raise Error('该节点已不存在，或不属于此组可手动选择的路径；请输入 r 刷新。')
    changed = []
    try:
        for selector, member in reversed(steps):
            api.call(endpoint(selector), method='PUT', data={'name': member})
            changed.append(selector)
            if api.call(endpoint(selector)).get('now') != member:
                raise Error('切换后的实际选中项与请求不一致。')
        _, after = api.snapshot()
        if effective_node(after, group) != name:
            raise Error('最终节点核对失败，请刷新查看实际状态。')
    except Error as exc:
        if changed:
            raise Error('{} 已对以下组发送切换请求：{}；请输入 r 刷新实际状态。'.format(
                exc, '、'.join(clean(n) for n in changed))) from None
        raise


def main():
    p = argparse.ArgumentParser(description='Clash/Mihomo 所有节点测速与交互切换，无需额外依赖')
    p.add_argument('--api', default='http://127.0.0.1:9087', help='external-controller 地址，默认 http://127.0.0.1:9087')
    p.add_argument('--group', help='rule 模式中需要切换的手动策略组，默认优先选择“节点选择”组')
    p.add_argument('--secret-file', help='保存控制接口 secret 的文本文件；也可使用环境变量 CLASH_SECRET')
    p.add_argument('--test-url', default='http://www.gstatic.com/generate_204', help='用于延迟测试的网址')
    p.add_argument('--timeout', type=int, default=5000, help='单节点测试超时，毫秒，默认 5000')
    p.add_argument('--workers', type=int, default=8, help='并发测试数，默认 8')
    args = p.parse_args()
    if not 1 <= args.workers <= 64 or not 100 <= args.timeout <= 120000:
        p.error('--workers 应在 1–64 之间，--timeout 应在 100–120000 之间。')
    if not args.api.startswith(('http://', 'https://')) or not args.test_url.startswith(('http://', 'https://')):
        p.error('--api 和 --test-url 必须是 http(s) 地址。')
    secret = (Path(args.secret_file).expanduser().read_text(encoding='utf-8').strip()
              if args.secret_file else os.environ.get('CLASH_SECRET', ''))
    api = API(args.api, secret)
    while True:
        mode, proxies = api.snapshot()
        group = select_group(mode, proxies, args.group)
        names = physical_nodes(proxies)
        if not names:
            raise Error('服务中没有实际代理节点，请先加载含节点的配置。')
        print('\n当前模式：{}；切换组：{}；当前选项：{}'.format(
            clean(mode), clean(group), clean(proxies[group].get('now', '未知'))))
        if mode == 'rule':
            print('rule 模式仅影响使用该组的流量；其他规则和独立策略组保持原样。')
        names, results = measure(api, names, args)
        display(names, results, proxies, group)
        while True:
            answer = input('\n输入序号或完整节点名切换；r 重新测速；l 重列；e 失败原因；q 退出：').strip()
            if answer.lower() == 'q':
                return
            if answer.lower() == 'r':
                break
            if answer.lower() == 'e':
                for name in names:
                    if results[name][0] is None:
                        print('{}：{}'.format(clean(name), clean(results[name][1])))
                continue
            if answer.lower() == 'l':
                _, proxies = api.snapshot()
                display(names, results, proxies, group)
                continue
            name = answer if answer in names else (
                names[int(answer) - 1] if answer.isdecimal() and 1 <= int(answer) <= len(names) else None)
            if name is None:
                print('未找到该名称或序号，请重试。')
                continue
            try:
                switch(api, group, name, mode)
                print('已切换并核对：{} → {}'.format(clean(group), clean(name)))
                if results[name][0] is None:
                    print('该节点本轮测速失败；已按你的选择切换。')
                print('已有连接可能继续使用原节点；新连接采用当前选择。')
            except Error as exc:
                print('切换失败：' + str(exc))


if __name__ == '__main__':
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print('\n已退出。')
    except (Error, OSError) as exc:
        if isinstance(exc, Error):
            print('错误：' + str(exc), file=sys.stderr)
        else:
            print('错误：无法读取密钥文件或执行系统操作，请检查路径和权限。', file=sys.stderr)
        sys.exit(1)
