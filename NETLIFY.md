# Netlify 前端部署
- 發佈資料夾：`static`
- 方案 A：把 `_redirects.example` 改名為 `_redirects`，替換 YOUR-BACKEND-URL → 同網域代理，免 CORS。
- 方案 B：不改檔，首次用 `?api=` 指向後端，例如：
  https://你的站名.netlify.app/?api=https://你的後端.onrender.com
