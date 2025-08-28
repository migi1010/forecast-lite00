let CHART = null;

// --- 只顯示你的三個模型 + 自訂名稱 ---
const MODEL_WHITELIST = ["model_user_rf","model_user_lstm","model_user_inception"];
const MODEL_LABELS = {
  "model_user_rf": "隨機森林",
  "model_user_lstm": "LSTM",
  "model_user_inception": "Inception Time"
};

async function fetchModels() {
  const res = await fetch('/models');
  const data = await res.json();
  const holder = document.getElementById('modelRadios');
  holder.innerHTML = '';
  const available = Array.isArray(data.models) ? data.models : (Array.isArray(data) ? data : []);
  const ids = MODEL_WHITELIST.filter(id => available.includes(id));
  ids.forEach((id, i) => {
    const label = MODEL_LABELS[id] || id;
    holder.insertAdjacentHTML('beforeend', `<label><input type="radio" name="model" value="${id}" ${i===0?'checked':''}> ${label}</label>`);
  });
}

function getSelected(name) {
  const nodes = document.querySelectorAll(`input[name="${name}"]`);
  for (const n of nodes) if (n.checked) return n.value;
  return null;
}

function fmt(n) { return new Intl.NumberFormat('en-US', { maximumFractionDigits: 6 }).format(n); }

function buildChart(points, symbol) {
  const ctx = document.getElementById('chart').getContext('2d');
  const labels = points.map(p => (new Date(p.t)).toLocaleDateString());
  const data = points.map(p => p.price);
  if (CHART) { CHART.destroy(); }
  CHART = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: [{ label: `${symbol} 預測價格`, data, fill: false }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'nearest', intersect: false },
      plugins: { legend: { display: true } },
      scales: {
        x: { grid: { display: true } },
        y: { grid: { display: true }, ticks: { callback: (v) => fmt(v) } }
      }
    }
  });
}

async function runOnce() {
  const btn = document.getElementById('run'); btn.disabled = true;
  const symbol = getSelected('symbol');
  const sample = parseInt(getSelected('sample'), 10);
  const model = getSelected('model');
  const payload = { symbol, sample_size: sample, horizon_days: 7, model };
  const res = await fetch('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  const data = await res.json();
  if (data && data.points) {
    buildChart(data.points, data.symbol);
    const s = data.stats || {};
    document.getElementById('stats').textContent = `最小: ${fmt(s.min)} ｜ 最大: ${fmt(s.max)} ｜ 起: ${fmt(s.start)} ｜ 迄: ${fmt(s.end)}`;
  } else { alert('沒有取得預測結果'); }
  btn.disabled = false;
}

document.getElementById('run').addEventListener('click', runOnce);
fetchModels();
