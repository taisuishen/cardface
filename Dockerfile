# RunPod GPU Pod 用镜像。
# 基础镜像自带 CUDA 12 + cuDNN 9，onnxruntime-gpu 才能用上 CUDAExecutionProvider。
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        libgl1 libglib2.0-0 \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY server/requirements-gpu.txt /app/server/requirements-gpu.txt
RUN pip install --no-cache-dir -r server/requirements-gpu.txt

COPY server/ /app/server/
COPY web/    /app/web/
COPY models/ /app/models/
COPY cardpose.onnx /app/cardpose.onnx

ENV CARD_MODEL=/app/cardpose.onnx \
    MODEL_DIR=/app/models \
    PORT=8000

EXPOSE 8000

# RunPod 的 HTTP 代理已经帮你做了 TLS（https://<podid>-8000.proxy.runpod.net），
# 所以容器内部跑普通 HTTP 就行，不需要在这里配证书。
CMD ["sh", "-c", "python server/app.py --host 0.0.0.0 --port ${PORT}"]
