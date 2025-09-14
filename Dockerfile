# 選擇 Python 3.10 官方映像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 更新 pip
RUN pip install --upgrade pip

# 安裝必要套件（穩定版本）
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    scipy==1.10.1 \
    tensorflow-cpu==2.12.0 \
    keras==2.12.0 \
    pandas==2.1.1 \
    matplotlib==3.8.0 \
    yfinance==0.2.33 \
    scikit-learn==1.3.1

# 複製專案檔案到容器
COPY . /app

# 預設啟動命令（可依需求修改）
CMD ["python", "main.py"]
