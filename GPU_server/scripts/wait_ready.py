#!/usr/bin/env python3
"""Đợi worker nạp xong model rồi mới thoát. Hiện thanh tiến trình trên terminal."""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

STEPS = {'tracking': 1, 'tracking_seg': 2, 'full': 4}


def alive(pidfile):
    try:
        os.kill(int(open(pidfile).read().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def state(port, token=''):
    try:
        req = urllib.request.Request(f'http://127.0.0.1:{port}/internal/health')
        if token:
            req.add_header('X-GPU-Token', token)
        with urllib.request.urlopen(req, timeout=3) as r:
            return json.load(r)
    except Exception:  # noqa: BLE001
        return None


def log_steps(path):
    try:
        txt = open(path, encoding='utf-8', errors='replace').read()
    except OSError:
        return 0, False
    marks = re.findall(r'^\[nạp\] (.*)$', txt, re.M)
    return sum(1 for m in marks if not m.startswith('XONG')), any(m.startswith('XONG') for m in marks)


def bar(frac, width=30):
    n = int(round(width * max(0.0, min(1.0, frac))))
    return '█' * n + '░' * (width - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8001)
    ap.add_argument('--pid-file', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--mode', default='full')
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('--token', default='')
    a = ap.parse_args()
    total = STEPS.get(a.mode, 4)
    tty = sys.stdout.isatty()
    t0, last_line = time.time(), ''
    while True:
        el = int(time.time() - t0)
        if not alive(a.pid_file):
            print('\n[chờ] server đã dừng trước khi nạp xong. 20 dòng log cuối:', file=sys.stderr)
            try:
                print(''.join(open(a.log, encoding='utf-8', errors='replace').readlines()[-20:]), file=sys.stderr)
            except OSError:
                pass
            return 1
        st = state(a.port, token=a.token)
        done_steps, done_marker = log_steps(a.log)
        ms = (st or {}).get('models') or {}
        if ms.get('state') == 'error':
            print(f'\n[chờ] nạp model lỗi: {ms.get("msg")}', file=sys.stderr)
            return 1
        if ms.get('state') == 'ready' or (st and done_marker):
            line = f'[nạp] {bar(1.0)} 100%  sẵn sàng ({el}s)'
            print(('\r' if tty else '') + line + ' ' * 20)
            return 0
        if st is None:
            frac, what = 0.02, 'khởi động server…'
        else:
            frac = min(0.95, 0.05 + 0.9 * done_steps / total)
            what = ms.get('msg') or 'đang nạp model…'
        line = f'[nạp] {bar(frac)} {int(frac * 100):3d}%  {what[:60]}  ({el}s)'
        if tty:
            print('\r' + line + ' ' * 6, end='', flush=True)
        elif line != last_line and (el % 10 == 0 or what != last_line):
            print(line, flush=True)
        last_line = line if tty else what
        if el > a.timeout:
            print(f'\n[chờ] quá {int(a.timeout)}s mà chưa nạp xong.', file=sys.stderr)
            return 1
        time.sleep(1)


if __name__ == '__main__':
    sys.exit(main())
