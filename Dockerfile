# 使用官方 Python 3.11 slim 版本
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt
COPY requirements.txt .

# 升級 pip 並安裝依賴
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案程式碼
COPY . .

# 設定 uvicorn 啟動指令 (假設 main.py 是入口)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
