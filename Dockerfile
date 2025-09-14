# 使用 Slim 版本 Python 作為基礎映像
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴（輕量化）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt
COPY requirements.txt .

# 修改 requirements.txt 以使用 CPU 版本的 TensorFlow
# 例如：tensorflow-cpu==2.12.0
# 建議你在本地先修改好 requirements.txt 或直接用下面 RUN 指令
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir tensorflow-cpu==2.12.0 keras==2.12.0 \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /root/.cache/pip

# 複製專案程式碼
COPY . .

# 如果是 FastAPI，暴露埠號
EXPOSE 8000

# 預設啟動命令（可依你的程式修改）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
