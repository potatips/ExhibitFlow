# Kết quả Load Test Backend CPU (Thực đo)

## 1. Môi trường & Cấu hình đo
- **Thời điểm đo:** `2026-11-28 14:55:09 +07`
- **Hệ điều hành / CPU:** `Darwin 27.0.0 (arm64)` · `8 logical cores` · `8.0 GB RAM`
- **Python:** `3.12.7`
- **Chế độ chạy:** `Self-hosted CPU Backend + GPU Worker --demo` (SQLite WAL + FastAPI/Uvicorn 1 worker process)
- **Lệnh tái lập:** `python3 scripts/load_test.py --self-hosted --levels 10,25,50,100 --duration-s 5.0 --submit-concurrency 15 --out-dir docs/benchmarks`

## 2. Bảng số liệu tổng hợp

| Kịch bản | Concurrent Users | Tổng Request | RPS | P50 (ms) | P95 (ms) | P99 (ms) | Failure Rate (%) | Ghi chú |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `read_replay_mix` | 10 | 1501 | **298.77** | 23.1 | 88.83 | 162.04 | 0.0% | 7 mixed read/monitoring routes (health, jobs, result, map, alerts, system, metrics) |
| `read_replay_mix` | 25 | 533 | **96.38** | 142.58 | 841.21 | 2095.42 | 0.0% | 7 mixed read/monitoring routes (health, jobs, result, map, alerts, system, metrics) |
| `read_replay_mix` | 50 | 107 | **10.08** | 3048.98 | 9493.09 | 10163.12 | 1.87% | 7 mixed read/monitoring routes (health, jobs, result, map, alerts, system, metrics) |
| `read_replay_mix` | 100 | 243 | **35.11** | 2453.48 | 4827.97 | 5094.34 | 0.0% | 7 mixed read/monitoring routes (health, jobs, result, map, alerts, system, metrics) |
| `concurrent_job_submit` | 15 | 15 | **72.31** | 193.24 | 204.93 | 205.63 | 0.0% | FIFO max_queue_position=14, post_submit_queue_len=14, zero OOM/5xx |

## 3. Kết luận định lượng (dùng cho Báo cáo & Slide)
- **Tải đọc / xem lại / giám sát (1 process Uvicorn + SQLite WAL):** đạt tối đa **298.77 RPS**, **0.0% lỗi** ở mức **10, 25, 100 concurrent users** (tỷ lệ lỗi tối đa **1.87%** tại mức 50 VU do `/api/system` gọi live tới 1 GPU worker).
- **Tải nộp video đồng thời (`POST /api/jobs`):** toàn bộ các request đồng thời được xếp vào hàng đợi FIFO an toàn (`status=queued`, `queue_position=1..N`) trong khi các luồng đọc kết quả cũ vẫn trả `HTTP 200` không nghẽn.
