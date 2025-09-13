# 使用官方 Python 3.11.4 slim 映像
FROM python:3.11.4-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴（避免 numpy/pandas build 失敗）
RUN apt-get update && \
    apt-get install -y build-essential libffi-dev libssl-dev curl git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 複製需求檔
COPY requirements.txt .

# 升級 pip、setuptools、wheel
RUN python -m pip install --upgrade pip setuptools wheel

# 安裝 Python 套件
RUN python -m pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 啟動命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
