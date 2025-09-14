# 使用 Python 3.10 官方輕量映像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt
COPY requirements.txt .

# 更新 pip 並安裝依賴套件
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 複製專案所有檔案
COPY . .

# 設定環境變數
ENV PYTHONUNBUFFERED=1

# 服務啟動命令（依你的 main.py 或 uvicorn 指令調整）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
