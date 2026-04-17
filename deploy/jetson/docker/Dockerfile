# Build on Jetson AGX Orin with the NVIDIA L4T PyTorch base image.
# The base ships with CUDA 12.2 / cuDNN / TensorRT preinstalled for aarch64.
#
#   cd <repo-root>
#   docker build -f deploy/jetson/docker/Dockerfile -t tradebot:jetson .
#   docker run --runtime nvidia --gpus all -d --restart unless-stopped \
#     --env-file .env -p 127.0.0.1:8000:8000 \
#     -v $(pwd)/logs:/app/logs -v $(pwd)/data_cache:/app/data_cache \
#     tradebot:jetson
FROM nvcr.io/nvidia/l4t-pytorch:r36.3.0-pth2.2-py3

# System deps for building llama.cpp
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git cmake build-essential tzdata curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV TZ=America/New_York \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Build llama-cpp-python with CUDA offload (same flags as install_llama_cpp.sh).
RUN CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_F16=on -DCMAKE_CUDA_ARCHITECTURES=87" \
    FORCE_CMAKE=1 \
    pip install --no-cache-dir llama-cpp-python

COPY . .

# Container entrypoint runs the paper loop. Override with a different
# `docker run` command to run the dashboard or backtest.
CMD ["python", "scripts/run_paper.py"]
