FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}/app \
    DEBIAN_FRONTEND=noninteractive

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DEVICE=cuda

EXPOSE 8000

CMD ["python", "inference.py"]