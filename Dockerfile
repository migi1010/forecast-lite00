# 基礎映像
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 複製 requirements
COPY requirements.txt .

# 升級 pip
RUN pip install --no-cache-dir --upgrade pip

# 安裝 tensorflow 與其他依賴
# 假設模型是用 TF 2.12 訓練的
RUN pip install --no-cache-dir tensorflow==2.12.0

# 安裝其他套件，不鎖版本的話會自動解決依賴
RUN pip install --no-cache-dir -r requirements.txt --ignore-installed tensorflow

# 複製程式碼
COPY . .

# 啟動服務
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
