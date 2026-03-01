FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3-pip git && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

ENV VLLM_BASE_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ \
    GRPC_HOST=0.0.0.0 \
    GRPC_PORT=50051 \
    GATEWAY_HOST=0.0.0.0 \
    GATEWAY_PORT=8000

EXPOSE 50051 8000

CMD ["python", "-m", "server.run_grpc_server"]

