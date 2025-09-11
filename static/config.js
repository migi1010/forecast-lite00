// Netlify/Runtime API config: ?api=https://your-backend
(function(){
  try {
    const qs = new URLSearchParams(location.search);
    const q = qs.get("api");
    if (q) localStorage.setItem("API_BASE", q.replace(/\/$/, ""));
  } catch {}
  const saved = (typeof localStorage !== 'undefined') ? localStorage.getItem("API_BASE") : "";
  if (saved) window.API_BASE = saved;
  else if (typeof window.API_BASE === "string") window.API_BASE = window.API_BASE.replace(/\/$/, "");
  else window.API_BASE = ""; // use same origin + _redirects proxy
})();