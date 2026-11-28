"""Nguồn video không phải file: camera RTSP/RTMP và webcam của trình duyệt (đẩy từng khung JPEG lên server)."""
import json
import os
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np

STREAM_EXT = '.efstream'
RTSP_OPTS = 'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|stimeout;8000000'

PUSH = {}
INFO = {}


def is_stream(path):
    return str(path).lower().endswith(STREAM_EXT)


def write_stream_file(path, **spec):
    Path(path).write_text(json.dumps(spec), encoding='utf-8')
    return Path(path)


def _spec(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class PushBuffer:
    """Khung JPEG mới nhất do trình duyệt đẩy lên."""

    def __init__(self):
        self.cv, self.jpg, self.seq, self.closed = threading.Condition(), None, 0, False

    def put(self, jpg):
        with self.cv:
            self.jpg, self.seq = jpg, self.seq + 1
            self.cv.notify_all()

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()

    def wait_new(self, last_seq, timeout):
        with self.cv:
            self.cv.wait_for(lambda: self.seq != last_seq or self.closed, timeout)
            if self.seq == last_seq:
                return None, last_seq
            return self.jpg, self.seq


class _LiveBase:
    """Giao diện giống cv2.VideoCapture đủ cho engine: isOpened / read / grab / retrieve / get / release."""
    first_timeout, frame_timeout = 20.0, 10.0

    def __init__(self):
        self._last, self._wh, self._frame, self._fps = 0, None, None, 25.0

    def isOpened(self):
        return True

    def _next(self, timeout):
        raise NotImplementedError

    def read(self):
        f = self._next(self.frame_timeout if self._wh else self.first_timeout)
        if f is None:
            return False, None
        if self._wh is None:
            self._wh = (f.shape[1], f.shape[0])
        return True, f

    def grab(self):
        ok, self._frame = self.read()
        return ok

    def retrieve(self):
        return (self._frame is not None), self._frame

    def get(self, prop):
        if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            if self._wh is None:
                ok, f = self.read()
                self._pending = f if ok else None
            if self._wh is None:
                return 0
            return self._wh[0] if prop == cv2.CAP_PROP_FRAME_WIDTH else self._wh[1]
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 0
        return 0

    def release(self):
        pass


class PushCapture(_LiveBase):
    def __init__(self, spec):
        super().__init__()
        self.buf = PUSH.get(spec['push'])
        if self.buf is None:
            raise RuntimeError('Phiên webcam không còn (trình duyệt đã ngắt?)')
        self._fps = float(spec.get('fps') or 15)

    def _next(self, timeout):
        jpg, self._last = self.buf.wait_new(self._last, timeout)
        if jpg is None:
            return None
        return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)


def _open_url(url):
    if not str(url).lower().startswith(('rtsp://', 'rtsps://')):
        return cv2.VideoCapture(str(url), cv2.CAP_FFMPEG)
    key, old = 'OPENCV_FFMPEG_CAPTURE_OPTIONS', os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS')
    os.environ[key] = RTSP_OPTS
    try:
        return cv2.VideoCapture(str(url), cv2.CAP_FFMPEG)
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


class LiveCapture(_LiveBase):
    """RTSP / RTMP: luồng nền đọc liên tục, chỉ giữ khung mới nhất."""

    def __init__(self, url, warmup=3):
        super().__init__()
        self.cap = _open_url(url)
        self._opened = self.cap.isOpened()
        if not self._opened:
            raise RuntimeError(f'Không mở được camera: {_mask(url)} (sai URL / tài khoản, hoặc server không tới được camera)')
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0
        self._fps = float(fps) if 1 < fps <= 120 else 25.0
        self.cv, self.frame, self.seq, self.dead, self._stop = threading.Condition(), None, 0, False, False
        self._skip = warmup
        self._pace = None if re.match(r'^[a-z]+://', str(url), re.I) else 1.0 / self._fps
        self.th = threading.Thread(target=self._loop, daemon=True, name='rtsp-reader')
        self.th.start()

    def isOpened(self):
        return self._opened

    def _loop(self):
        bad = 0
        while not self._stop:
            ok, f = self.cap.read()
            if not ok:
                bad += 1
                if bad > 50:
                    break
                time.sleep(0.05)
                continue
            bad = 0
            if self._pace:
                time.sleep(self._pace)
            if self._skip > 0:
                self._skip -= 1
                continue
            with self.cv:
                self.frame, self.seq = f, self.seq + 1
                self.cv.notify_all()
        with self.cv:
            self.dead = True
            self.cv.notify_all()

    def _next(self, timeout):
        with self.cv:
            self.cv.wait_for(lambda: self.seq != self._last or self.dead, timeout)
            if self.seq == self._last:
                return None
            self._last = self.seq
            return self.frame

    def release(self):
        self._stop = True
        try:
            self.th.join(2)
            self.cap.release()
        except Exception:
            pass


def _mask(url):
    return re.sub(r'(://[^:/@]+:)[^@]*@', r'\1***@', str(url))


def open_capture(path):
    """cv2.VideoCapture cho file, hoặc đối tượng tương đương cho camera (.efstream)."""
    if not is_stream(path):
        return cv2.VideoCapture(str(path))
    spec = _spec(path)
    cap = PushCapture(spec) if spec.get('push') else LiveCapture(spec['url'])
    return _Remember(cap, str(path))


class _Remember:
    """Bọc cap: ghi kích thước / fps đã đo vào INFO[path] để web dùng."""

    def __init__(self, cap, path):
        self._cap, self._path = cap, path
        self._pending = None

    def __getattr__(self, k):
        return getattr(self._cap, k)

    def get(self, prop):
        v = self._cap.get(prop)
        w, h = self._cap._wh or (0, 0)
        if w:
            INFO[self._path] = dict(width=int(w), height=int(h), fps=float(self._cap._fps))
        return v

    def read(self):
        p = getattr(self._cap, '_pending', None)
        if p is not None:
            self._cap._pending = None
            return True, p
        return self._cap.read()

    def grab(self):
        ok, self._cap._frame = self.read()
        return ok
