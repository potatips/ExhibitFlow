# Standalone ExhibitFlow GPU Worker image (Vast.ai / Docker CUDA >= 12.8).
# Build context: GPU_server/
#   docker build -t exhibitflow-gpu-worker GPU_server/
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential cmake ffmpeg libgl1 libglib2.0-0 openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Build llama-server with CUDA support for Qwen3-VL-8B GGUF
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp \
    && cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    && cmake --build /tmp/llama.cpp/build --config Release --target llama-server -j"$(nproc)" \
    && cp /tmp/llama.cpp/build/bin/llama-server /usr/local/bin/ \
    && (cp -a /tmp/llama.cpp/build/bin/*.so* /usr/local/lib/ 2>/dev/null || true) \
    && ldconfig \
    && rm -rf /tmp/llama.cpp

WORKDIR /opt/exhibitflow-gpu
COPY . /opt/exhibitflow-gpu

RUN mkdir -p /opt/exhibitflow-gpu/external \
    && git clone --depth 1 https://github.com/khangkaka066/ByteTrack-DMA-LTC-Motion-Tracker.git /opt/exhibitflow-gpu/external/ByteTrack-DMA-LTC-Motion-Tracker \
    && git clone --depth 1 https://github.com/potatips/multicam-2d-floorplan-tracker.git /opt/exhibitflow-gpu/external/multicam-2d-floorplan-tracker

RUN python3 scripts/install_deps.py && pip install -U "brotli>=1.2"
RUN python3 /opt/exhibitflow-gpu/external/multicam-2d-floorplan-tracker/scripts/download_structural_models.py

ENV EF_WEIGHTS_DIR=/opt/exhibitflow-gpu/weights \
    EXHIBITFLOW_WORK=/workspace/work \
    HF_HOME=/workspace/hf \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PORT=8001 \
    MODE=full

EXPOSE 8001

CMD ["sh", "-c", "python3 launch.py --host 0.0.0.0 --port ${PORT:-8001} --mode ${MODE:-full} --data-dir ${DATA_DIR:-/workspace/gpu_data}"]
