# 使用 Python 官方 slim 映像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 複製程式碼與模型
COPY . /app

# 升級 pip
RUN pip install --upgrade pip

# 安裝套件（降版本以匹配模型）
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    scipy==1.10.1 \
    tensorflow-cpu==2.9.3 \
    keras==2.9.0 \
    h5py==3.1.0 \
    pandas==2.1.1 \
    matplotlib==3.8.0 \
    yfinance==0.2.33 \
    scikit-learn==1.3.1

# 執行程式
CMD ["python", "main.py"]
