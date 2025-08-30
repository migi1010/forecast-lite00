# 使用官方 TensorFlow 映像，避免依賴衝突
FROM tensorflow/tensorflow:2.12.0

WORKDIR /app

# 複製 requirements.txt 並安裝非 TensorFlow 套件
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# 複製程式碼與模型
COPY . .

# Cloud Run 預設使用 8080
EXPOSE 8080

# 啟動 FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
