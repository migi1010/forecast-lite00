# 使用官方 Python 3.10 slim 映像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 避免 Python 緩存影響
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 更新系統套件並安裝必要工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 升級 pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# 安裝固定版本套件以避免 backtracking
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    jax==0.3.15 \
    jaxlib==0.4.31 \
    tensorflow-cpu==2.12.0 \
    keras==2.12.0

# 複製你的專案需求檔
COPY requirements.txt .

# 安裝其他需求套件
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案程式碼
COPY . .

# 設定執行程式
CMD ["python", "app.py"]
