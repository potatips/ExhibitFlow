"""Nhiều camera chạy SONG SONG: mỗi camera là một TIẾN TRÌNH CON riêng (`python -m exhibitflow_gpu.cam_worker <socket>`).

Mỗi camera giữ một PipelineRunner (hoặc DemoRunner khi demo) và trả lời lệnh:
    start (cấu hình) / snap (trạng thái + cảnh báo) / frame (khung JPEG mới) / stop / quit.
VLM dùng chung một `llama-server` duy nhất (qua `cfg['extra']['models']['vlm']['server_url']` hoặc `EF_VLM_URL` / `SharedVLMServer`).
"""
import atexit
import copy
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
_ALIVE, _ALIVE_LOCK = set(), threading.Lock()


def free_port():
    """Cổng TCP trống trên máy."""
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def view_of(snap, cancelled=False):
    """Trạng thái job con (cùng quy tắc với view() cho job đơn)."""
    st, alive = snap['st'], snap['alive']
    if cancelled and not alive:
        return dict(status='cancelled', stage=4, progress=0, error=None)
    if st == 'error':
        return dict(status='error', stage=0, progress=0, error=snap.get('msg') or 'Pipeline lỗi')
    if st == 'done' and not alive:
        return dict(status='done', stage=4, progress=1.0, error=None)
    if st in ('idle', 'loading'):
        return dict(status='running', stage=1 if 'map' in (snap.get('phase') or '').lower() else 0, progress=0.0, error=None)
    if st == 'running':
        fi, nf = snap.get('fi') or 0, snap.get('nf') or 0
        return dict(status='running', stage=3, progress=round(fi / nf, 3) if nf else 0.0, error=None)
    return dict(status='running', stage=4, progress=0.5, error=None)


def close_all():
    """Đóng mọi tiến trình camera còn sống (khi server tắt)."""
    with _ALIVE_LOCK:
        procs = list(_ALIVE)
    for p in procs:
        p.close(grace=5)


atexit.register(close_all)


class CamProc:
    """Phía cha: một tiến trình con chạy một camera. Mọi lệnh đều đồng bộ (gửi -> nhận), có khoá."""

    def __init__(self, cfg, startup_timeout=120):
        from multiprocessing.connection import Listener
        cfg = copy.deepcopy(cfg)
        # Tự động tái sử dụng SharedVLMServer URL nếu parent đã khởi động và cfg chưa truyền server_url
        vlm_extra = ((cfg.get('extra') or {}).get('models') or {}).get('vlm') or {}
        if not cfg.get('demo') and (cfg.get('settings') or {}).get('vlm_backend') != 'dry_run':
            if not vlm_extra.get('server_url') and not os.environ.get('EF_VLM_URL'):
                from .vlm_server import SharedVLMServer
                st_vlm = SharedVLMServer.instance().status()
                if st_vlm.get('state') == 'ready' and st_vlm.get('url'):
                    cfg.setdefault('extra', {}).setdefault('models', {}).setdefault('vlm', {})['server_url'] = st_vlm['url']

        self._lock, self._closed = threading.Lock(), False
        self._dir = tempfile.mkdtemp(prefix='ef_cam_')
        key = secrets.token_bytes(16)
        addr = os.path.join(self._dir, 'c.sock')
        listener = Listener(addr, family='AF_UNIX', authkey=key)
        env = dict(os.environ, EF_CAM_KEY=key.hex(),
                   PYTHONPATH=os.pathsep.join([str(PROJ), os.environ.get('PYTHONPATH', '')]))
        self.proc = subprocess.Popen([sys.executable, '-m', 'exhibitflow_gpu.cam_worker', addr], env=env, start_new_session=True)
        box = {}
        t = threading.Thread(target=lambda: box.update(conn=listener.accept()), daemon=True)
        t.start()
        t0 = time.time()
        while t.is_alive() and self.proc.poll() is None and time.time() - t0 < startup_timeout:
            t.join(0.2)
        listener.close()
        if 'conn' not in box:
            self._kill()
            shutil.rmtree(self._dir, ignore_errors=True)
            raise RuntimeError(f'Tiến trình camera không khởi động được (mã {self.proc.poll()})')
        self.conn = box['conn']
        with _ALIVE_LOCK:
            _ALIVE.add(self)
        try:
            self._call(('start', cfg), timeout=60)
        except Exception:
            self.close(grace=5)
            raise

    @property
    def pid(self):
        return self.proc.pid

    def _kill(self):
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            self.proc.wait(5)
        except Exception:
            pass

    def _call(self, msg, timeout=30):
        with self._lock:
            if self._closed:
                raise RuntimeError('Tiến trình camera đã đóng')
            self.conn.send(msg)
            t0 = time.time()
            while not self.conn.poll(0.5):
                if self.proc.poll() is not None:
                    raise RuntimeError(f'Tiến trình camera đã thoát (mã {self.proc.returncode})')
                if time.time() - t0 > timeout:
                    raise TimeoutError(f'Tiến trình camera không trả lời sau {timeout}s')
            status, val = self.conn.recv()
        if status == 'err':
            raise RuntimeError('Lỗi trong tiến trình camera:\n' + val)
        return val

    def snap(self):
        """Trạng thái + cảnh báo đang có của camera."""
        return self._call(('snap',), timeout=15)

    def frame(self, last):
        """Khung JPEG mới nhất nếu khác `last` (seq), kèm trạng thái."""
        return self._call(('frame', last), timeout=15)

    def stop(self):
        """Yêu cầu dừng: xử lý nốt rồi có kết quả."""
        self._call(('stop',), timeout=15)

    def close(self, grace=20):
        """Kết thúc tiến trình con (quit, rồi diệt nhóm nếu quá lâu). Gọi nhiều lần cũng được."""
        with _ALIVE_LOCK:
            _ALIVE.discard(self)
        if self._closed:
            return
        try:
            self._call(('quit',), timeout=grace)
        except Exception:
            pass
        with self._lock:
            self._closed = True
        try:
            self.conn.close()
        except Exception:
            pass
        try:
            self.proc.wait(grace)
        except subprocess.TimeoutExpired:
            self._kill()
        shutil.rmtree(self._dir, ignore_errors=True)


# ------------------------------------------------------------------ phía con
class _Stopped(Exception):
    pass


def _deep_merge(dst, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


class _Cam:
    def __init__(self):
        self.state = self.runner = self.thread = self.video = None
        self.stop_flag = False

    def start(self, cfg):
        import importlib
        import exhibitflow_gpu.runner as R
        from exhibitflow_gpu import settings as efs
        for mod in cfg.get('preload') or []:
            importlib.import_module(mod)
        for sp in cfg.get('setup') or []:
            if sp.get('path') and sp['path'] not in sys.path:
                sys.path.insert(0, sp['path'])
            importlib.import_module(sp['module']).setup(sp.get('args') or {}, self)
        root = Path(cfg['root'])
        root.mkdir(parents=True, exist_ok=True)
        (root / 'settings.json').write_text(json.dumps(cfg['settings'], indent=2), encoding='utf-8')
        if cfg.get('extra'):
            orig = R.overrides_from_settings
            extra = cfg['extra']

            def patched(s):
                ov = orig(s)
                _deep_merge(ov, extra)
                return ov
            R.overrides_from_settings = patched
        if cfg.get('stream_max_sec'):
            efs.RT['stream_max_sec'] = float(cfg['stream_max_sec'])
        self.state = R.State(root)
        self.runner = (R.DemoRunner if cfg['demo'] else R.PipelineRunner)(self.state, local=bool(cfg.get('local')))
        self.video = cfg['video']
        with self.state.lock:
            self.state.status, self.state.phase = 'loading', 'Chuẩn bị'
        self.thread = threading.Thread(target=self._run, args=(cfg['flock'],), daemon=True, name='cam-start')
        self.thread.start()

    def _run(self, flock):
        import fcntl
        st, runner = self.state, self.runner
        try:
            with open(flock, 'a+') as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    if self.stop_flag:
                        raise _Stopped()
                    runner.preload()
                    if runner.loader is not None:
                        runner.loader.join()
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)
            if self.stop_flag:
                raise _Stopped()
            if st.models_state == 'error':
                raise RuntimeError(st.models_msg or 'Nạp model lỗi')
            runner.start(self.video)
            if self.stop_flag:
                runner.stop()
        except _Stopped:
            with st.lock:
                st.status, st.phase = 'done', 'Đã dừng trước khi chạy'
        except BaseException as e:
            with st.lock:
                st.status, st.message = 'error', f'{type(e).__name__}: {e}'
            print(traceback.format_exc(), flush=True)

    def snap(self):
        if self.state is None:
            return dict(st='idle', phase='', msg='', fi=0, nf=0, alive=False, alerts=[], cam=None)
        st = self.state
        with st.lock:
            s, ph, msg, fi, nf = st.status, st.phase, st.message, st.frame_idx, st.n_frames
            alerts, cam = list(st.alerts), st.cam_name
        return dict(st=s, phase=ph, msg=msg, fi=fi, nf=nf, alive=self.runner.running(), alerts=alerts, cam=cam)

    def frame(self, last):
        if self.state is None:
            return dict(seq=last, jpg=None, st='idle', alive=False)
        st = self.state
        with st.lock:
            seq, jpg, s = st.frame_seq, st.frame_jpeg, st.status
        alive = self.runner.running()
        if jpg is not None and seq != last:
            return dict(seq=seq, jpg=jpg, st=s, alive=alive)
        return dict(seq=last, jpg=None, st=s, alive=alive)

    def stop(self):
        self.stop_flag = True
        if self.runner is not None:
            self.runner.stop()

    def shutdown(self):
        """Dừng pipeline, chờ luồng kết thúc, giải phóng tài nguyên cục bộ (shared VLM server vẫn giữ nguyên)."""
        if self.runner is None:
            return
        self.stop()
        if self.runner.thread is not None:
            self.runner.thread.join(60)
        eng = getattr(self.runner, 'engine', None)
        if eng is not None:
            try:
                eng.close()
            except Exception:
                pass


def _child_main(addr):
    from multiprocessing.connection import Client
    conn = Client(addr, family='AF_UNIX', authkey=bytes.fromhex(os.environ['EF_CAM_KEY']))
    cam = _Cam()
    try:
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                break
            op = msg[0]
            try:
                if op == 'start':
                    cam.start(msg[1])
                    val = None
                elif op == 'snap':
                    val = cam.snap()
                elif op == 'frame':
                    val = cam.frame(msg[1])
                elif op == 'stop':
                    cam.stop()
                    val = None
                elif op == 'quit':
                    conn.send(('ok', None))
                    break
                else:
                    raise ValueError(f'lệnh không hợp lệ: {op}')
                conn.send(('ok', val))
            except Exception:
                try:
                    conn.send(('err', traceback.format_exc()))
                except OSError:
                    break
    finally:
        cam.shutdown()


if __name__ == '__main__':
    _child_main(sys.argv[1])
    sys.stdout.flush()
    os._exit(0)
