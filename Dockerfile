# 使用官方 Python 基礎映像
FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 複製專案檔案
COPY . /app

# 升級 pip
RUN pip install --no-cache-dir --upgrade pip

# 安裝依賴套件，順序按照依賴關係，避免衝突
RUN pip install --no-cache-dir \
    numpy==1.25.2 \
    scipy==1.11.1 \
    jax==0.3.15 \
    jaxlib==0.4.31 \
    absl-py==2.3.1 \
    opt_einsum==3.4.0 \
    etils==1.13.0 \
    typing_extensions==4.15.0 \
    tensorflow-cpu==2.12.0 \
    keras==2.12.0

# 如果有其他 requirements.txt，可以放最後安裝
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# 設定容器啟動指令（依你的專案修改）
CMD ["python", "main.py"]
