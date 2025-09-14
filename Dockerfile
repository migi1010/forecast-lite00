# 基礎 image
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 避免 pip 緩存和 timeout 問題
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=100

# 安裝系統套件
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 套件（舊版 TensorFlow/Keras 避免 batch_shape 問題）
RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    scipy==1.10.1 \
    tensorflow-cpu==2.9.3 \
    keras==2.9.0 \
    pandas==2.1.1 \
    matplotlib==3.8.0 \
    yfinance==0.2.33 \
    scikit-learn==1.3.1

# 複製程式碼
COPY . /app

# 指定啟動指令
CMD ["python", "main.py"]
