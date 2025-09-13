# 使用官方 Python 3.11 映像
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt
COPY requirements.txt .

# 安裝依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 暴露 Render 會用到的 PORT
ENV PORT=10000
EXPOSE 10000

# 啟動指令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
