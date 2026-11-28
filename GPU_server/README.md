# ExhibitFlow Standalone GPU Worker (`GPU_server/`)

GPU inference worker độc lập hoàn toàn cho kiến trúc 3 tầng (`web/` -> `backend/` -> `GPU_server/`). Cung cấp API `/internal/*` trên cổng `8001` với xác thực `X-GPU-Token` và không import bất kỳ module nào từ `exhibitflow_pipeline/`, `backend/`, hay `web/`.

## Khởi chạy nhanh (One-Command)

### 1. Vast.ai / Máy có NVIDIA GPU (1 lệnh duy nhất)

```bash
cd GPU_server
HF_TOKEN=hf_xxx bash start_vast.sh
# hoặc:
HF_TOKEN=hf_xxx make vast
```

Lệnh trên tự động:
1. Kiểm tra GPU CUDA và cài đặt idempotent (`deps`, `external/`, `weights/`, `llama-server`, `mapmodels`).
2. Cache HuggingFace tại `/workspace/.hf_home` (bền vững trên Vast.ai).
3. Khởi chạy `launch.py` ở foreground, đợi `/internal/health` và các model sẵn sàng (`models.state == "ready"`).
4. Mở Cloudflare Tunnel HTTPS, kiểm tra sức khoẻ từ bên ngoài (xác nhận `200` khi có `X-GPU-Token` và `401` khi thiếu token), rồi in `STATUS: READY (SẴN SÀNG)` kèm `Public HTTPS` và `X-GPU-Token` **một lần duy nhất trên terminal** (không ghi secret vào `.make/worker.log` hay `worker.log`).

### 2. Chế độ Demo thuần CPU (không cần GPU / weights)

```bash
cd GPU_server
python3 launch.py --demo --port 8001
# hoặc không mở tunnel:
python3 launch.py --demo --port 8001 --no-tunnel
```

### 3. Docker

```bash
docker build -t exhibitflow-gpu-worker GPU_server/
docker run --gpus all -p 8001:8001 -e HF_TOKEN=hf_xxx -e GPU_TOKEN=my-secret exhibitflow-gpu-worker
```

## Cloudflare Quick Tunnel vs Named Tunnel

| Chế độ | Cờ lệnh | Yêu cầu | URL |
| :--- | :--- | :--- | :--- |
| **Quick Tunnel** (mặc định) | `--tunnel quick` | Không cần tài khoản Cloudflare | URL tạm thời `https://*.trycloudflare.com` (đổi sau mỗi lần khởi động lại) |
| **Named Tunnel** (cố định) | `--tunnel named --tunnel-token <TOKEN> --public-url https://gpu.example.com` | **Bắt buộc** có tài khoản Cloudflare + tên miền riêng (custom domain) + Zero Trust Tunnel Token (`CLOUDFLARE_TUNNEL_TOKEN`) | URL HTTPS cố định trên tên miền của bạn |
| **Không dùng tunnel** | `--tunnel none` (hoặc `--no-tunnel`) | Mạng nội bộ / Docker / Vast.ai direct IP:port | Chỉ truy cập cục bộ hoặc IP trực tiếp |

> [!WARNING]
> **Bảo mật log:** `GPU_TOKEN`, `HF_TOKEN`, và `CLOUDFLARE_TUNNEL_TOKEN` **không bao giờ** được ghi vào `--log-file` (`worker.log` / `.make/worker.log`). Khi tiến trình tunnel ngắt bất ngờ, `launch.py` cảnh báo ngay trên stdout.

## Đánh giá khả thi chạy trên Kaggle (Experimental Only)

Notebook thử nghiệm nằm tại `GPU_server/notebooks/exhibitflow_kaggle_worker.ipynb`. Kaggle **KHÔNG** tương đương Vast.ai và chỉ phù hợp để demo/thử nghiệm ngắn:

- **Phần cứng miễn phí:** GPU T4 x2 (16 GB × 2 VRAM, ~29 GB RAM, 4 vCPU; P100 đã ngừng cung cấp).
- **Giới hạn phiên:** tối đa **12 giờ/phiên**, tự ngắt nếu idle tương tác **20 phút**, hạn mức GPU khoảng **30 giờ/tuần**.
- **Lưu trữ:** `/kaggle/working` tối đa **20 GB** và bị xoá sạch sau khi hết phiên (chỉ `/kaggle/input` datasets là bền vững).
- **Điều khoản & mạng:** Kaggle ToS chỉ cho phép thử nghiệm/nghiên cứu cá nhân phi thương mại (không dùng làm API production); Quick Tunnel URL thay đổi mỗi phiên và `llama-server` có thể cần fallback sang `MODE=tracking_seg` nếu môi trường không hỗ trợ build CUDA.

## Kiến trúc & Internal API (`/internal/*`)

- `exhibitflow_gpu/` — ML pipeline độc lập (`engine`, `pipeline`, `vlm_server.SharedVLMServer`, `floorplan.FloorplanService`, `runner`, `cam_worker`, `multicam`).
- `worker/` — FastAPI worker (`auth.py`, `demo.py`, `runtime.py` lazy-loaded khi không `--demo`, `app.py`).
- `launch.py` — Launcher + strict readiness gate (`evaluate_readiness`, `probe_external_health`) + tunnel monitor.

Kiểm thử CPU:

```bash
PYTHONPATH=GPU_server pytest GPU_server/tests -v
```
