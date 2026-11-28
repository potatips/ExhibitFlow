#!/usr/bin/env bash
# Ensure llama-server (VLM) is available:
#   1. Already in PATH                           -> done
#   2. glibc >= 2.38 (Ubuntu 24.04)              -> prebuilt download handled at first run
#   3. Older glibc (Ubuntu 22.04) + nvcc present -> build from source into $PREFIX
#   4. Otherwise                                 -> exit 2 (run MODE=tracking_seg without VLM)
set -euo pipefail
PREFIX="${PREFIX:-/usr/local}"

if command -v llama-server >/dev/null 2>&1; then
    echo "[llama] llama-server already installed: $(command -v llama-server)"
    exit 0
fi

glibc="$(ldd --version | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
if [ -n "$glibc" ] && [ "$(printf '%s\n2.38\n' "$glibc" | sort -V | head -1)" = "2.38" ]; then
    echo "[llama] glibc $glibc >= 2.38: prebuilt llama-server will auto-download on first VLM start"
    exit 0
fi

NVCC="$(command -v nvcc || true)"
[ -z "$NVCC" ] && [ -x /usr/local/cuda/bin/nvcc ] && NVCC=/usr/local/cuda/bin/nvcc
if [ -z "$NVCC" ]; then
    echo "[llama] ERROR: glibc ${glibc:-unknown} < 2.38 and nvcc not found." >&2
    echo "[llama] Use a CUDA '-devel' image, Ubuntu 24.04+, or run MODE=tracking_seg." >&2
    exit 2
fi
export PATH="$(dirname "$NVCC"):$PATH"

command -v cmake >/dev/null 2>&1 || { echo "[llama] installing cmake"; python3 -m pip install -q cmake; }
command -v git >/dev/null 2>&1 || { echo "[llama] missing git"; exit 2; }

arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '.' | sort -u | paste -sd';' || true)"
[ -z "$arch" ] && arch="75;80;86;89;90"
echo "[llama] Building llama.cpp (CUDA arch: $arch)..."

src="$(mktemp -d)/llama.cpp"
git clone --depth 1 https://github.com/ggml-org/llama.cpp "$src"
cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="$arch" -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$src/build" --config Release --target llama-server -j"$(nproc)"

mkdir -p "$PREFIX/bin" "$PREFIX/lib"
cp "$src/build/bin/llama-server" "$PREFIX/bin/"
cp -a "$src"/build/bin/*.so* "$PREFIX/lib/" 2>/dev/null || true
ldconfig 2>/dev/null || true
rm -rf "$(dirname "$src")"
echo "[llama] Done: $PREFIX/bin/llama-server"
