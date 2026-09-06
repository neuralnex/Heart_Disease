FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir huggingface-hub sentencepiece accelerate fasttext

ENV HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    HUGGINGFACE_HUB_CACHE=/models/huggingface \
    HF_HUB_CACHE=/models/huggingface

RUN mkdir -p /models/huggingface && chmod -R 777 /models/huggingface

COPY . .

EXPOSE 7860

ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "30"]
