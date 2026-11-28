"""Shared single VLM server manager (`llama-server` for Qwen3-VL-8B-Instruct-GGUF).

Spawns and owns at most ONE `llama-server` process across runs, engine reloads,
and parallel `cam_worker` subprocesses.
"""
import atexit
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from . import settings

DEFAULT_VLM_PORT = 8089
DEFAULT_VLM_NP = 2


class SharedVLMServer:
    """Thread-safe singleton managing a single shared `llama-server` process."""

    _instance = None
    _cls_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._log = None
        self._log_path = None
        self._url = None
        self._model_id = None
        self._state = 'stopped'   # stopped | starting | ready | error
        self._error = None
        self._port = None
        self._n_parallel = DEFAULT_VLM_NP
        self._gpus = []
        self._external = False
        atexit.register(self.close)

    @classmethod
    def instance(cls) -> 'SharedVLMServer':
        with cls._cls_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @staticmethod
    def _is_healthy(url: str, timeout: float = 3.0) -> bool:
        try:
            with urllib.request.urlopen(f'{url.rstrip("/")}/health', timeout=timeout) as r:
                return r.status == 200
        except Exception:
            return False

    def _log_tail(self, n: int = 3000) -> str:
        try:
            if self._log is not None and not self._log.closed:
                self._log.flush()
            if self._log_path is not None and Path(self._log_path).is_file():
                return Path(self._log_path).read_text(errors='replace')[-n:]
        except Exception:
            pass
        return ''

    def _cleanup_proc_locked(self):
        p = self._proc
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                p.kill()
                try:
                    p.wait(5)
                except Exception:
                    pass
        self._proc = None
        if self._log is not None and not self._log.closed:
            try:
                self._log.close()
            except Exception:
                pass
        self._log = None

    def ensure_started(self, spec: dict | None = None) -> str:
        """Ensure the shared llama-server is running and healthy; return its base URL."""
        spec = dict(spec or settings.MODELS.get('vlm') or {})
        ext_url = spec.get('server_url') or os.environ.get('EF_VLM_URL')
        startup_timeout = float(spec.get('startup_timeout', 900))

        with self._lock:
            # Reuse already-running managed process if still alive and healthy
            if self._proc is not None and self._proc.poll() is None and self._url:
                if self._is_healthy(self._url, timeout=3.0):
                    self._state = 'ready'
                    return self._url
                # Process alive but unhealthy -> restart cleanly
                self._cleanup_proc_locked()

            if self._proc is not None and self._proc.poll() is not None:
                self._cleanup_proc_locked()

            # External server URL provided (e.g. EF_VLM_URL or spec['server_url'])
            if ext_url:
                url = str(ext_url).rstrip('/')
                self._state = 'starting'
                self._error = None
                self._external = True
                self._url = url
                self._model_id = spec.get('repo_id') or str(spec.get('model_path') or 'external')
                t0 = time.perf_counter()
                while not self._is_healthy(url, timeout=3.0):
                    if time.perf_counter() - t0 > min(startup_timeout, 60.0):
                        self._state = 'error'
                        self._error = f'Shared VLM server at {url} did not respond to /health'
                        raise RuntimeError(self._error)
                    time.sleep(0.3)
                self._state = 'ready'
                return self._url

            # Spawn one managed llama-server process
            from .vlm import ensure_llama_server, fetch_gguf

            self._external = False
            self._state = 'starting'
            self._error = None
            port = int(spec.get('port') or os.environ.get('EF_VLM_PORT') or DEFAULT_VLM_PORT)
            n_parallel = max(1, int(spec.get('n_parallel') or os.environ.get('EF_VLM_NP') or DEFAULT_VLM_NP))
            ctx = int(spec.get('ctx', 16384))
            n_gpu_layers = int(spec.get('n_gpu_layers', 99))
            device = spec.get('device', settings.GPU_B)
            flash_attn = spec.get('flash_attn', 'on')
            extra_args = tuple(spec.get('extra_args') or ())

            try:
                exe, xenv = ensure_llama_server()
                if spec.get('model_path'):
                    model = str(spec['model_path'])
                    proj = str(spec.get('mmproj_path') or '')
                    mname = Path(model).name
                    pname = Path(proj).name if proj else ''
                    model_id = f'{Path(model).parent}/{mname}' + (f' + {pname}' if pname else '')
                else:
                    repo_id = spec.get('repo_id') or 'unsloth/Qwen3-VL-8B-Instruct-GGUF'
                    quant_file = spec.get('quant_file') or 'Q4_K_M'
                    mmproj = spec.get('mmproj') or 'F16'
                    model, proj, mname, pname = fetch_gguf(repo_id, quant_file, mmproj)
                    model_id = f'{repo_id}/{mname} + {pname}'

                env = {**os.environ, **xenv}
                vis = [v for v in (settings.CUDA_VISIBLE_DEVICES_0 or ','.join(map(str, range(settings.N_GPU)))).split(',') if v]
                dev = str(device)
                if dev.startswith('cuda:') and vis:
                    idx = int(dev.split(':')[1])
                    gpus = [vis[idx]] if idx < len(vis) else vis[:1]
                elif dev == 'auto' and settings.N_GPU and vis:
                    gpus = vis[:settings.N_GPU]
                else:
                    gpus, n_gpu_layers = [], 0
                env['CUDA_VISIBLE_DEVICES'] = ','.join(gpus)

                settings.WORK.mkdir(parents=True, exist_ok=True)
                self._log_path = settings.WORK / f'llama_server_{port}.log'
                cmd = [exe, '-m', model]
                if proj:
                    cmd += ['--mmproj', proj]
                cmd += ['-ngl', str(n_gpu_layers), '-c', str(ctx), '-np', str(n_parallel),
                        '--host', '127.0.0.1', '--port', str(port), *map(str, extra_args)]
                if flash_attn and not {'-fa', '--flash-attn'} & set(map(str, extra_args)):
                    cmd += ['-fa', str(flash_attn)]

                self._port = port
                self._n_parallel = n_parallel
                self._gpus = gpus
                self._model_id = model_id
                self._url = f'http://127.0.0.1:{port}'
                self._log = open(self._log_path, 'w')
                self._proc = subprocess.Popen(cmd, env=env, stdout=self._log, stderr=subprocess.STDOUT)

                t0 = time.perf_counter()
                while True:
                    if self._proc.poll() is not None:
                        tail = self._log_tail()
                        raise RuntimeError(f'llama-server dừng (mã {self._proc.returncode}):\n{tail}')
                    if time.perf_counter() - t0 > startup_timeout:
                        tail = self._log_tail()
                        self._cleanup_proc_locked()
                        raise RuntimeError(f'llama-server không sẵn sàng sau {startup_timeout}s:\n{tail}')
                    if self._is_healthy(self._url, timeout=5.0):
                        break
                    time.sleep(0.4)

                self._state = 'ready'
                print(f'    [SharedVLM] {model_id} | {self._url} | GPU {gpus or "cpu"} | -np {n_parallel} ctx={ctx}',
                      flush=True)
                return self._url
            except Exception as e:
                self._cleanup_proc_locked()
                self._url = None
                self._state = 'error'
                self._error = f'{type(e).__name__}: {e}'
                raise

    def status(self) -> dict:
        with self._lock:
            state = self._state
            if not self._external and self._proc is not None and self._proc.poll() is not None and state == 'ready':
                state = 'stopped'
                self._state = 'stopped'
                self._url = None
            return {
                'state': state,
                'url': self._url,
                'model': self._model_id,
                'shared': True,
                'n_parallel': self._n_parallel,
                'error': self._error,
            }

    def close(self):
        with self._lock:
            self._cleanup_proc_locked()
            self._url = None
            self._state = 'stopped'
            self._external = False
