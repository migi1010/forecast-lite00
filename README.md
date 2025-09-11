# Forecast Lite — Clean Rebuild (2025-09-11)
- 只含三個模型：model_user_rf / model_user_lstm / model_user_inception
- 英鎊（GBPUSD）暫不支援：前端禁用、後端 /predict 回 501
- 前端：Chart.js，Y 軸價格刻度清楚；Netlify-ready（config.js、_redirects.example、支援 ?api=）
- LSTM：若存在 `models/weights/lstm_model.h5` 且環境有 TensorFlow，即用真模型；否則 fallback

## 部署
- Render（Web Service）：
  - Build Command: `pip install -r requirements.txt`
  - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
  - Root Directory: 空白
- Netlify（前端）
  - 發佈資料夾：`static`
  - 首次用：`https://你的站名.netlify.app/?api=https://你的後端.onrender.com` 或使用 `_redirects` 代理
