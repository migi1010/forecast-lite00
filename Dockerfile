# 使用官方 Python 3.10
FROM python:3.10-slim

WORKDIR /app

# 複製 requirements 並安裝
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# 複製專案程式碼
COPY . .

# 開放端口
EXPOSE 10000

# 啟動 uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
