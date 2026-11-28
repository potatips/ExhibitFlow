# ExhibitFlow CPU Backend

Lightweight CPU-only FastAPI orchestration, persistence, authentication, and offline-replay backend for ExhibitFlow.

---

## Khởi chạy nhanh (Tiếng Việt) / Quick Start (English)

### 1. Chạy cục bộ với Cloudflare Quick Tunnel (Khuyên dùng khi nối với Vercel Web)

Chạy một lệnh tại máy của bạn (lưu SQLite + video + kết quả trực tiếp trong `backend_data/`):

```bash
cd backend
python3 launch.py --tunnel quick
```

Hoặc dùng `Makefile`:

```bash
cd backend
make tunnel
```

Trình khởi chạy sẽ tự động:
1. Mở Backend CPU FastAPI trên `127.0.0.1:8000` (tự đổi cổng nếu đang bận).
2. Tự tạo `Backend Access Key` ngẫu nhiên (hoặc dùng `BACKEND_API_KEY` / `--api-key`).
3. Tải `cloudflared` nếu chưa có và mở Cloudflare Quick Tunnel (`https://*.trycloudflare.com`).
4. Kiểm tra kết nối cả nội bộ lẫn từ URL HTTPS công khai, sau đó in ra `STATUS: READY (SẴN SÀNG)` kèm URL và `Backend Access Key` **một lần duy nhất** trên terminal.

> **Lưu ý quan trọng:**
> - `Backend Access Key` dùng để xác thực từ Trình duyệt / Vercel Web tới Backend (`Authorization: Bearer <key>`), **hoàn toàn tách biệt** với `X-GPU-Token` (dùng giữa Backend và `GPU_server`).
> - Không dùng `--no-auth` khi mở tunnel ra Internet!
> - **Quick Tunnel vs Named Tunnel**:
>   - **Quick Tunnel (`--tunnel quick`)**: Không cần tài khoản Cloudflare; mỗi lần khởi động lại sẽ tạo một URL `https://*.trycloudflare.com` mới (nếu đổi URL, hãy dán URL mới vào Settings của Web).
>   - **Named Tunnel (`--tunnel named`)**: URL HTTPS cố định, **yêu cầu** tài khoản Cloudflare + tên miền riêng (custom domain trên Cloudflare DNS) + Cloudflare Zero Trust Tunnel Token (`--tunnel-token` hoặc `CLOUDFLARE_TUNNEL_TOKEN` và `--public-url` hoặc `PUBLIC_BACKEND_URL`).

```bash
# Chạy với Named Tunnel (URL cố định)
CLOUDFLARE_TUNNEL_TOKEN=<token> PUBLIC_BACKEND_URL=https://api.example.com \
  CORS_ORIGINS=https://exhibitflow.vercel.app \
  python3 launch.py --tunnel named
```

### 2. Chạy nội bộ không cần Tunnel (Local dev)

```bash
cd backend
make run-local
# hoặc với xác thực:
python3 launch.py --no-tunnel
```

---

## Security & Authentication Contract

When `BACKEND_API_KEY` is non-empty:
- **Header-only API routes** (`Authorization: Bearer <key>` or `X-Backend-Key: <key>`; `?st=` rejected with `401`, `X-GPU-Token` rejected with `401`):
  - `POST /api/auth/verify`, `POST /api/auth/stream-token`
  - `/api/system`, `/api/config/gpu`, `/api/tuning/schema`, `/api/sites`
  - `/api/jobs`, `/api/jobs/{id}`, `/api/jobs/{id}/cancel`, `/api/jobs/{id}/finish`, `/api/jobs/{id}/live/events`, `/api/jobs/{id}/result`, `/api/jobs/{id}/map/data`
  - `/api/alerts`, `PATCH /api/alerts/{id}`
  - `/api/site-jobs`, `/api/site-jobs/{id}`, `/api/site-jobs/{id}/result`
- **Media / Stream / Download routes** (header OR short-lived HMAC-SHA256 `?st=<exp>.<sig>` ticket from `POST /api/auth/stream-token`):
  - `GET /api/events`, `GET /api/jobs/{id}/events`
  - `GET /api/jobs/{id}/live`, `GET /api/jobs/{id}/live/stream`
  - `GET /api/jobs/{id}/video`, `GET /api/jobs/{id}/map`, `GET /api/jobs/{id}/files/{path}`
  - `GET /api/site-jobs/{id}/map`, `GET /api/site-jobs/{id}/video/{cam}`
  - `GET /api/alerts/export.csv`
- **`GET /api/health`**: Public. When unauthenticated, returns only `{ok, version, auth_required: true, authenticated: false, store, artifact_store}` without leaking `gpu`, `gpu.url`, or `queue_length`.

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `BACKEND_API_KEY` | `""` | Shared secret required from browser clients (`Authorization: Bearer` / `X-Backend-Key`). |
| `STREAM_TOKEN_TTL_S` | `600` | Lifetime (seconds) of `?st=` HMAC stream tickets. |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins (e.g. `https://exhibitflow.vercel.app`). |
| `CLOUDFLARE_TUNNEL_TOKEN` | `""` | Zero Trust token for `--tunnel named`. |
| `PUBLIC_BACKEND_URL` | `""` | Public HTTPS URL for `--tunnel named` readiness probe. |
| `GPU_WORKER_URL` | `""` | Base URL of the `GPU_server` worker (runtime configurable via `PUT /api/config/gpu`). |
| `GPU_TOKEN` | `""` | Shared secret sent as `X-GPU-Token` on `/internal/*` requests to `GPU_server`. |
| `REDIS_URL` | `""` | Optional Redis URL. Defaults to SQLite WAL when empty. |
| `DATA_DIR` | `./backend_data` | Root directory for SQLite DB, logs, and local artifact storage. |
| `SQLITE_PATH` | `<DATA_DIR>/state.db` | Optional explicit path to the SQLite database file. |
| `ARTIFACT_STORE` | `local` | Artifact backend: `local` or `vercel_blob`. |
| `BLOB_READ_WRITE_TOKEN` | `""` | Vercel Blob API read/write token when `ARTIFACT_STORE=vercel_blob`. |
| `WEB_DIR` | auto (`<repo>/web`) | Directory of static frontend files served at `/`. |
| `MAX_UPLOAD_MB` | `600` | Maximum uploaded video size in MB. |
| `POLL_INTERVAL_S` | `0.5` | Dispatcher polling interval in seconds. |
| `HEALTH_INTERVAL_S` | `2.0` | Background GPU health probe interval in seconds. |
| `PORT` | `8000` | HTTP listen port. |

## Docker

```bash
docker build -t exhibitflow-backend ./backend
docker run -p 8000:8000 \
  -e BACKEND_API_KEY=your-backend-key \
  -e GPU_WORKER_URL=https://your-gpu.trycloudflare.com \
  -e GPU_TOKEN=your-gpu-token \
  exhibitflow-backend
```
