# ===== 基礎映像 =====
FROM python:3.10-slim

# ===== 系統依賴 =====
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# ===== 設定 pip 超時與使用清華源 =====
RUN pip install --upgrade pip --timeout=600 \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# ===== 安裝 Python 套件（降版本，固定版本） =====
RUN pip install --no-cache-dir --default-timeout=600 \
    numpy==1.23.5 \
    scipy==1.10.1 \
    tensorflow-cpu==2.9.3 \
    keras==2.9.0 \
    h5py==3.7.0 \
    pandas==2.1.1 \
    matplotlib==3.8.0 \
    scikit-learn==1.3.1 \
    yfinance==0.2.33

# ===== 設定工作目錄 =====
WORKDIR /app

# ===== 複製程式碼 =====
COPY . /app

# ===== 預設啟動 =====
CMD ["python", "main.py"]
