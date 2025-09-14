# --------- 基礎映像 ---------
FROM python:3.10-slim

# --------- 環境變數 ---------
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0  # 關閉 oneDNN 自訂操作，減少浮點數差異

# --------- 更新系統並安裝必要工具 ---------
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# --------- 升級 pip ---------
RUN pip install --upgrade pip setuptools wheel

# --------- 安裝 Python 套件（分步 + 超時 + 降版本） ---------
# Step 1: numpy
RUN pip install --default-timeout=600 --no-cache-dir numpy==1.23.5

# Step 2: scipy
RUN pip install --default-timeout=600 --no-cache-dir scipy==1.10.1

# Step 3: TensorFlow / Keras / h5py
RUN pip install --default-timeout=600 --no-cache-dir \
    tensorflow-cpu==2.9.3 \
    keras==2.9.0 \
    h5py==3.1.0

# Step 4: 其他科學計算套件
RUN pip install --default-timeout=600 --no-cache-dir \
    pandas==2.1.1 \
    matplotlib==3.8.0 \
    yfinance==0.2.33 \
    scikit-learn==1.3.1

# --------- 建立工作目錄 ---------
WORKDIR /app

# --------- 複製專案文件 ---------
COPY . /app

# --------- 執行程式 ---------
CMD ["python", "main.py"]
