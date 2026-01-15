# Use PyTorch image with CUDA 12.8 for RTX 5090 (Blackwell, sm_120) support
# This image already includes PyTorch compiled with CUDA 12.8 kernels
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}/app \
    DEBIAN_FRONTEND=noninteractive

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

ENV DEVICE=cuda

EXPOSE 8000

CMD ["python", "inference.py"]