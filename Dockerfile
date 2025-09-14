# 基礎映像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 更新 pip
RUN pip install --upgrade pip

# 安裝必要套件，避免版本衝突
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    scipy==1.10.1 \
    pandas==2.1.1 \
    tensorflow-cpu==2.12.0 \
    keras==2.12.0 \
    matplotlib==3.8.0 \
    yfinance==0.2.28

# 複製專案檔案到容器
COPY . /app

# 預設指令（可依專案修改）
CMD ["python", "main.py"]
