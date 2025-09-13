# 使用官方 Python 3.11 slim 映像
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt
COPY requirements.txt .

# 安裝依賴套件
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 複製程式碼與模型
COPY . .

# 環境變數，FastAPI/Flask 偵聽所有 IP
ENV HOST=0.0.0.0
ENV PORT=8000

# 對外開放 8000 port
EXPOSE 8000

# 啟動指令
# 假設你使用 FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# 如果你使用 Flask，可以改成：
# CMD ["python", "main.py"]
