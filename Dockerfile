FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 複製 requirements 並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案程式碼
COPY . .

# 啟動 FastAPI / Flask
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
