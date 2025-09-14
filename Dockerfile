# 使用官方 Python 3.10 slim 映像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 複製程式碼到容器
COPY . /app

# 升級 pip
RUN pip install --no-cache-dir --upgrade pip

# 安裝 Python 套件，增加 timeout 與 retries 避免下載大檔案超時
RUN pip install --no-cache-dir \
    --default-timeout=600 \
    --retries=10 \
    numpy==1.23.5 \
    scipy==1.10.1 \
    tensorflow-cpu==2.9.3 \
    keras==2.9.0 \
    pandas==2.1.1 \
    matplotlib==3.8.0 \
    yfinance==0.2.33 \
    scikit-learn==1.3.1

# 暴露服務 port（如果需要）
EXPOSE 8000

# 設定容器啟動命令
CMD ["python", "main.py"]
