# ExhibitFlow — Museum Visitor Behavior & Journey Analytics

Hệ thống phân tích video camera bảo tàng theo kiến trúc **3 tầng tách biệt**: phát hiện và theo dõi khách tham quan (Visitor ID), phát hiện hành vi tiếp xúc / cầm / nghi lấy cắp hiện vật bằng Vision-Language Model (**Qwen3-VL-8B**), dựng bản đồ 2D (**Floorplan & Heatmap / Luồng đi**), và hỗ trợ nhiều người dùng xem lại lịch sử / duyệt cảnh báo độc lập ngay cả khi máy GPU đã tắt.

```
[Web Static (web/)]  ──(HTTPS + Bearer BACKEND_API_KEY + ?st=)──▶  [CPU Backend (backend/ :8000)]
  HTML/CSS/JS thuần                                                  SQLite WAL / Redis + ArtifactStore
  Vercel CDN                                                         FIFO Queue + Prometheus /metrics
                                                                                │
                                                           (HTTPS + X-GPU-Token /internal/*)
                                                                                ▼
                                                                     [GPU Worker (GPU_server/ :8001)]
                                                                       YOLOX + ByteTrack-DMA-LTC + Pose/Hand
                                                                       MM-Grounding-DINO + SAM 2.1 + SharedVLMServer
                                                                       FloorplanService (single & multi-cam)
```

---

## 1. Cấu trúc thư mục

| Thành phần | Vai trò | Tài liệu chi tiết |
|---|---|---|
| `web/` | Giao diện Dashboard tĩnh thuần HTML/CSS/ES6+ (không build step), sẵn sàng cho Vercel Static Hosting (`web/vercel.json`) | [web/README.md](web/README.md) |
| `backend/` | FastAPI thuần CPU (không cài `torch`/`cv2`/`ultralytics`), quản lý hàng đợi FIFO, SQLite WAL / Redis, lưu trữ artifact cục bộ / Vercel Blob, auth `BACKEND_API_KEY` + `?st=`, và endpoint Prometheus `/metrics` | [backend/README.md](backend/README.md) |
| `GPU_server/` | Gói suy luận AI độc lập (Perception + Segmenter + `SharedVLMServer` dùng chung + `FloorplanService`), có chế độ `--demo` thuần CPU và script triển khai 1 lệnh trên Vast.ai / Kaggle | [GPU_server/README.md](GPU_server/README.md) |
| `monitoring/` | Cấu hình Prometheus + Grafana Dashboard dựng sẵn (`ExhibitFlow 3-Tier Overview`) | [docker-compose.yml](docker-compose.yml) |
| `scripts/` | Script Load Test tái lập được (`scripts/load_test.py`) | [docs/benchmarks/load_test_report.md](docs/benchmarks/load_test_report.md) |
| `docs/` | Hướng dẫn vận hành 3 tầng, báo cáo kiểm chứng & kết quả benchmark | [docs/HUONG_DAN_VAN_HANH.md](docs/HUONG_DAN_VAN_HANH.md) |

---

## 2. Chạy nhanh bằng 1 lệnh (`docker compose`)

> Chạy được trên mọi máy sạch có Docker, **không cần NVIDIA GPU** (mặc định dùng `gpu-demo`).

```bash
cp .env.example .env   # tuỳ chọn chỉnh BACKEND_API_KEY, GPU_TOKEN
docker compose up --build
```

Sau khi container khỏe, mở trực tiếp **http://localhost:8000** (Backend tự phục vụ `web/` và nối sẵn tới `gpu-demo:8001`).

### Các profile mở rộng

```bash
# Kèm Prometheus (:9090) + Grafana (:3000, user/pass mặc định: admin/admin)
docker compose --profile monitoring up --build

# Kèm Redis StateStore (:6379) — đặt REDIS_URL=redis://redis:6379/0 trong .env
docker compose --profile redis --profile monitoring up --build

# Chạy GPU Worker thật trên máy có NVIDIA GPU + Container Toolkit
HF_TOKEN=hf_xxx GPU_WORKER_URL=http://gpu-worker:8001 docker compose --profile gpu up --build
```

---

## 3. Chạy thủ công / Triển khai Hybrid (Vercel + Backend CPU + Vast.ai)

### A. Test nhanh không cần Docker & không cần GPU (2 terminal)

```bash
# Cài phụ thuộc CPU nhẹ
python3 -m venv .venv_be
.venv_be/bin/pip install -r backend/requirements.txt

# Terminal 1: GPU Worker ở chế độ --demo
.venv_be/bin/python GPU_server/launch.py --demo --port 8001 --no-tunnel

# Terminal 2: CPU Backend (tự phục vụ web/ tại http://127.0.0.1:8000)
.venv_be/bin/python backend/launch.py --no-tunnel --port 8000
```

### B. Triển khai thực tế (Web Vercel + Backend trên máy bất kỳ + GPU Vast.ai)

Xem hướng dẫn chi tiết từng bước tại [docs/HUONG_DAN_VAN_HANH.md](docs/HUONG_DAN_VAN_HANH.md):
1. **Web (`web/`):** deploy lên Vercel (`Root Directory = web`, không chứa secret).
2. **Backend (`backend/`):** chạy `CORS_ORIGINS=https://<app>.vercel.app python3 backend/launch.py --tunnel quick` (hoặc `--tunnel named`).
3. **GPU Worker (`GPU_server/`):** trên Vast.ai chạy `cd GPU_server && HF_TOKEN=hf_xxx bash start_vast.sh` (hoặc dùng [exhibitflow_kaggle_worker.ipynb](GPU_server/notebooks/exhibitflow_kaggle_worker.ipynb) trên Kaggle).

---

## 4. Kiểm thử tự động & CI/CD

Workflow GitHub Actions nằm tại [.github/workflows/ci.yml](.github/workflows/ci.yml), tự động chạy cho mỗi push / PR vào `main` và `Khang`:

```bash
# 1. Backend tests (unit, stores, artifacts, auth, metrics, E2E với GPU --demo)
PYTHONPATH=. pytest backend/tests -v

# 2. GPU_server CPU-safe tests (worker demo, shared VLM, floorplan calibration, launcher)
PYTHONPATH=GPU_server pytest GPU_server/tests -v

# 3. Frontend syntax & analytics unit tests
node --check web/js/app.js
node --test web/tests/heatmap.test.js web/tests/museum_analytics.test.js
```

---

## 5. Benchmark & Load Test (Tái lập được)

### A. Load Test CPU Backend (đã đo thực tế)

Chạy lại bằng 1 lệnh (tự dựng `GPU_server --demo` + `backend` SQLite WAL trên 1 process Uvicorn):

```bash
python3 scripts/load_test.py --self-hosted --levels 10,25,50,100 --duration-s 5 --submit-concurrency 15 --out-dir docs/benchmarks
```

Kết quả đo thực tế trên `Darwin 27.0.0 (arm64)`, 8 logical cores, 8.0 GB RAM ([báo cáo đầy đủ](docs/benchmarks/load_test_report.md)):

| Kịch bản | Concurrent Users | Tổng Request | RPS | P50 (ms) | P95 (ms) | Failure Rate (%) |
|---|---:|---:|---:|---:|---:|---:|
| `read_replay_mix` | 10 | 1501 | **298.77** | 23.10 | 88.83 | 0.0% |
| `read_replay_mix` | 25 | 533 | **96.38** | 142.58 | 841.21 | 0.0% |
| `read_replay_mix` | 50 | 107 | **10.08** | 3048.98 | 9493.09 | 1.87% |
| `read_replay_mix` | 100 | 243 | **35.11** | 2453.48 | 4827.97 | 0.0% |
| `concurrent_job_submit` | 15 | 15 | **72.31** | 193.24 | 204.93 | 0.0% |

> **Ghi chú kỹ thuật:** Các route đọc/xem lại thuần SQLite đạt tới **298.77 RPS** (`0.0%` lỗi ở 10/25/100 VU). Tại mức 50 VU xuất hiện 1.87% timeout do `/api/system` trong hỗn hợp 7 route thực hiện proxy trực tiếp tới GPU worker và ghi cache SQLite trên mỗi lượt đọc; 15 lượt upload đồng thời vào `POST /api/jobs` đều được xếp hàng FIFO an toàn (`0%` lỗi).

### B. GPU Optimization Benchmark (chạy trên Vast.ai / Kaggle CUDA)

```bash
# Kiểm tra cú pháp / tạo template trên CPU:
python3 GPU_server/scripts/benchmark_gpu.py --dry-run --out docs/benchmarks

# Đo thực tế Baseline vs Optimized (FP16, Pose Gate 1.5, VLM Q4_K_M 8f_512_short) trên máy GPU:
python3 GPU_server/scripts/benchmark_gpu.py --video /path/to/test.mp4 --runs 3 --out docs/benchmarks
```
