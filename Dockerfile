# 使用官方 Python 3.10 slim 影像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 複製程式碼與模型
COPY . /app

# 安裝系統依賴（若有需要）
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 套件（降版本 TensorFlow/Keras 避免 InputLayer 問題）
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    scipy==1.10.1 \
    tensorflow-cpu==2.9.3 \
    keras==2.9.0 \
    pandas==2.1.1 \
    matplotlib==3.8.0 \
    yfinance==0.2.33 \
    scikit-learn==1.3.1

# 設定環境變數（避免 TensorFlow 輸出大量 warning，可選）
ENV TF_CPP_MIN_LOG_LEVEL=2

# 啟動程式
CMD ["python", "main.py"]
