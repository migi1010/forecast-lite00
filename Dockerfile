# 使用官方 Python 3.10 slim 映像檔
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt 並安裝套件
COPY requirements.txt .

# 避免 numpy / tensorflow 安裝出錯
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案檔案
COPY . .

# 將 5000 port 對外開放
EXPOSE 5000

# 啟動命令
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:5000"]
