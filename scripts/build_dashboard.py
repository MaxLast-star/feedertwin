"""Сборка интерактивного what-if дашборда (этап 5).

Запуск:  python scripts/build_dashboard.py

Предрассчитывает две сетки параметров (Monte Carlo для контура контроля,
численная модель для вибротранспортирования) и собирает автономный
dashboard/index.html: один файл без сервера, пригодный для GitHub Pages.
Расчёт занимает ~5–8 минут.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feedertwin.flowsim import FlowParams, run_replications  # noqa: E402
from feedertwin.strategies import (  # noqa: E402
    AdaptiveVibration,
    BaselineThreshold,
    MeteringGate,
)
from feedertwin.transport import TrayParams, mean_velocity  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"

# Сетка контура контроля
RATES = [round(r, 1) for r in np.arange(3.0, 8.01, 0.5)]
THRESHOLDS = [round(t, 2) for t in np.arange(0.40, 0.951, 0.05)]
STRATEGIES = [
    ("none", "Без контроля", lambda: BaselineThreshold(n_confirm_frames=10**9)),
    ("baseline", "Базовая пороговая (НИР)", lambda: None),
    ("adaptive", "Адаптивная вибрация", AdaptiveVibration),
    ("metering", "Дозирующий выпуск", MeteringGate),
]
N_REPS = 6
SIM_TIME = 300.0

# Сетка вибротранспортирования
FREQS = list(range(10, 46))          # Гц
ALPHAS = list(range(0, 16))          # °
AMPS_MM = [0.25, 0.275, 0.30]        # мм


def build_control_grid() -> dict:
    """Сетка метрик контура: стратегия × интенсивность × порог."""
    grid: dict[str, list] = {s[0]: [] for s in STRATEGIES}
    total = len(STRATEGIES) * len(RATES) * len(THRESHOLDS)
    done = 0
    for key, _, factory in STRATEGIES:
        for r in RATES:
            row = []
            for thr in THRESHOLDS:
                s = run_replications(
                    FlowParams(sim_time_s=SIM_TIME, seed=900,
                               arrival_rate_hz=float(r), threshold=float(thr)),
                    n_reps=N_REPS, strategy=factory(),
                )
                lat = s.mean_latency_s[0]
                row.append([
                    round(s.escaped_per_1000[0], 2),
                    round(s.throughput_hz[0], 3),
                    round(s.flap_duty[0] * 100, 2),
                    round(s.false_engagements[0], 1),
                    round(lat * 1000, 1) if np.isfinite(lat) else None,
                ])
                done += 1
            grid[key].append(row)
            print(f"\r  контур: {done}/{total}", end="", flush=True)
    print()
    return {
        "rates": RATES, "thresholds": THRESHOLDS,
        "strategies": [{"key": k, "name": n} for k, n, _ in STRATEGIES],
        "metrics": ["escaped", "throughput", "duty", "false", "latency"],
        "grid": grid,
    }


def build_transport_grid() -> dict:
    """Сетка скоростей подачи: амплитуда × частота × угол."""
    out = []
    total = len(AMPS_MM) * len(FREQS) * len(ALPHAS)
    done = 0
    for a_mm in AMPS_MM:
        plane = []
        for f in FREQS:
            row = []
            for al in ALPHAS:
                p = TrayParams(amplitude_m=a_mm * 1e-3, freq_hz=float(f), alpha_deg=float(al))
                v = mean_velocity(p, n_cycles=120, steps_per_cycle=1000) * 1000.0
                row.append(round(max(v, 0.0), 3))
                done += 1
            plane.append(row)
            print(f"\r  транспорт: {done}/{total}", end="", flush=True)
        out.append(plane)
    print()
    return {"amps_mm": AMPS_MM, "freqs": FREQS, "alphas": ALPHAS, "v_mm_s": out}


TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FeederTwin — интерактивная панель what-if</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --paper:#f6f7f4; --ink:#1d3a5f; --ink-soft:#41608a; --line:#c8d0d9;
    --signal:#d96a2b; --ok:#2c7a4b; --card:#ffffff; --mono:'IBM Plex Mono',ui-monospace,monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 'Inter',system-ui,sans-serif;
    background-image:linear-gradient(var(--line) .5px,transparent .5px),linear-gradient(90deg,var(--line) .5px,transparent .5px);
    background-size:24px 24px;background-position:-.5px -.5px}
  .frame{max-width:1180px;margin:24px auto;border:2px solid var(--ink);background:var(--card);
    box-shadow:6px 6px 0 rgba(29,58,95,.12)}
  header{display:flex;justify-content:space-between;align-items:baseline;gap:16px;flex-wrap:wrap;
    border-bottom:2px solid var(--ink);padding:18px 26px;background:repeating-linear-gradient(135deg,#fff 0 14px,#f3f5f8 14px 16px)}
  header h1{margin:0;font:600 21px var(--mono);letter-spacing:.04em}
  header .formula{font:500 13px var(--mono);color:var(--ink-soft)}
  nav{display:flex;border-bottom:1.5px solid var(--ink)}
  nav button{flex:1;padding:11px 8px;border:0;background:#eef1f5;cursor:pointer;
    font:500 13px var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-soft)}
  nav button+button{border-left:1.5px solid var(--ink)}
  nav button.on{background:var(--card);color:var(--ink);box-shadow:inset 0 -3px 0 var(--signal)}
  nav button:focus-visible{outline:2px solid var(--signal);outline-offset:-2px}
  .tab{display:none;padding:22px 26px}
  .tab.on{display:grid;grid-template-columns:280px 1fr;gap:22px}
  @media(max-width:880px){.tab.on{grid-template-columns:1fr}}
  .rail .field{margin-bottom:18px}
  .rail label{display:block;font:500 11px var(--mono);letter-spacing:.08em;text-transform:uppercase;
    color:var(--ink-soft);margin-bottom:6px}
  .rail label b{color:var(--ink);font-weight:600}
  select,input[type=range]{width:100%}
  select{padding:8px;border:1.5px solid var(--ink);background:#fff;font:500 13px var(--mono);color:var(--ink)}
  input[type=range]{accent-color:var(--signal);height:28px}
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:0;border:1.5px solid var(--ink);margin-bottom:18px}
  .kpi{padding:10px 14px;border-right:1.5px solid var(--ink)}
  .kpi:last-child{border-right:0}
  .kpi .v{font:600 24px var(--mono)}
  .kpi .u{font:400 11px var(--mono);color:var(--ink-soft)}
  .kpi.alert .v{color:var(--signal)}
  .kpi.good .v{color:var(--ok)}
  .chart{border:1.5px solid var(--line);margin-bottom:18px;background:#fff}
  .note{font-size:12.5px;color:var(--ink-soft);margin:4px 0 0}
  footer.stamp{display:grid;grid-template-columns:2fr 1fr 1fr 1fr;border-top:2px solid var(--ink);font:400 11px var(--mono)}
  footer.stamp div{padding:8px 12px;border-right:1.5px solid var(--ink)}
  footer.stamp div:last-child{border-right:0}
  footer.stamp b{display:block;font-size:9px;letter-spacing:.1em;color:var(--ink-soft);text-transform:uppercase;margin-bottom:2px}
  a{color:var(--ink)}
  @media (prefers-reduced-motion:no-preference){nav button{transition:background .15s}}
</style>
</head>
<body>
<div class="frame">
  <header>
    <h1>FeederTwin · панель what-if</h1>
    <span class="formula">Γ = A·ω²/g &nbsp;·&nbsp; бюджет коррекции 66+40+80 мс &nbsp;·&nbsp; цель ≥ 5 дет/с</span>
  </header>
  <nav>
    <button id="bt0" class="on" onclick="show(0)">Контур контроля</button>
    <button id="bt1" onclick="show(1)">Вибротранспортирование</button>
  </nav>

  <section class="tab on" id="tab0">
    <div class="rail">
      <div class="field"><label>Стратегия управления</label>
        <select id="strat"></select></div>
      <div class="field"><label>Интенсивность потока — <b id="rateV"></b> дет/с</label>
        <input type="range" id="rate" min="0"></div>
      <div class="field"><label>Порог детектора — <b id="thrV"></b></label>
        <input type="range" id="thr" min="0"></div>
      <p class="note">Каждая точка — Monte Carlo: 6 репликаций × 300 с модельного
      времени дискретно-событийной модели (SimPy) на временны́х параметрах НИР.</p>
    </div>
    <div>
      <div class="kpis">
        <div class="kpi alert"><b class="u">Дефекты подачи</b><div class="v" id="kEsc">—</div><div class="u">на 1000 деталей</div></div>
        <div class="kpi good"><b class="u">Производительность</b><div class="v" id="kThr">—</div><div class="u">дет/с</div></div>
        <div class="kpi"><b class="u">Заслонка введена</b><div class="v" id="kDuty">—</div><div class="u">% времени</div></div>
        <div class="kpi"><b class="u">Латентность коррекции</b><div class="v" id="kLat">—</div><div class="u">мс</div></div>
      </div>
      <div class="chart" id="cRate"></div>
      <div class="chart" id="cThr"></div>
    </div>
  </section>

  <section class="tab" id="tab1">
    <div class="rail">
      <div class="field"><label>Амплитуда колебаний</label>
        <select id="amp"></select></div>
      <div class="field"><label>Частота — <b id="freqV"></b> Гц</label>
        <input type="range" id="freq" min="0"></div>
      <div class="field"><label>Угол наклона лотка — <b id="alV"></b>°</label>
        <input type="range" id="al" min="0"></div>
      <p class="note">Численная модель материальной точки на вибрирующем наклонном
      лотке (μ = 0,3). Рабочая точка НИР: 16,7 Гц (1000 об/мин), A = 0,25–0,3 мм.</p>
    </div>
    <div>
      <div class="kpis">
        <div class="kpi"><b class="u">Коэффициент режима Γ</b><div class="v" id="kGam">—</div><div class="u" id="kReg">—</div></div>
        <div class="kpi good"><b class="u">Скорость подачи</b><div class="v" id="kVel">—</div><div class="u">мм/с</div></div>
        <div class="kpi"><b class="u">Порог подбрасывания</b><div class="v" id="kFcr">—</div><div class="u">Гц при выбранной A</div></div>
      </div>
      <div class="chart" id="cHeat"></div>
    </div>
  </section>

  <footer class="stamp">
    <div><b>Проект</b>FeederTwin — имитационная модель системы поштучной подачи</div>
    <div><b>Разработал</b>Ластовенко М.А.</div>
    <div><b>Основание</b>НИР, НИТУ МИСИС</div>
    <div><b>Источник</b><a href="https://github.com/MaxLast-star/feedertwin">github.com/MaxLast-star/feedertwin</a></div>
  </footer>
</div>

<script>
const C = __CONTROL_DATA__;
const T = __TRANSPORT_DATA__;
const FONT = {family:"'IBM Plex Mono',monospace", size:11, color:"#1d3a5f"};
const LAYOUT = {margin:{l:58,r:16,t:42,b:44}, paper_bgcolor:"#fff", plot_bgcolor:"#fff",
  font:FONT, height:300};
const CFG = {displayModeBar:false, responsive:true};
const COLORS = {none:"#8a93a0", baseline:"#2c7a4b", adaptive:"#d96a2b", metering:"#1d3a5f"};

function show(i){
  for(const k of [0,1]){
    document.getElementById("tab"+k).classList.toggle("on", k===i);
    document.getElementById("bt"+k).classList.toggle("on", k===i);
  }
  if(i===1) drawHeat();
}

/* ---------- Контур контроля ---------- */
const stratSel = document.getElementById("strat");
C.strategies.forEach(s=>{const o=document.createElement("option");o.value=s.key;o.textContent=s.name;stratSel.appendChild(o);});
stratSel.value = "baseline";
const rate = document.getElementById("rate"), thr = document.getElementById("thr");
rate.max = C.rates.length-1; rate.value = C.rates.indexOf(5);
thr.max = C.thresholds.length-1; thr.value = C.thresholds.indexOf(0.75);

function cell(key,ri,ti){ return C.grid[key][ri][ti]; }

function updControl(){
  const ri=+rate.value, ti=+thr.value, key=stratSel.value;
  document.getElementById("rateV").textContent = C.rates[ri];
  document.getElementById("thrV").textContent = C.thresholds[ti].toFixed(2);
  const m = cell(key,ri,ti);
  document.getElementById("kEsc").textContent = m[0].toFixed(1);
  document.getElementById("kThr").textContent = m[1].toFixed(2);
  document.getElementById("kDuty").textContent = m[2].toFixed(1);
  document.getElementById("kLat").textContent = m[4]==null ? "—" : m[4].toFixed(0);

  const traces = C.strategies.map(s=>({
    x:C.rates, y:C.rates.map((_,i)=>C.grid[s.key][i][ti][0]),
    name:s.name, mode:"lines+markers", line:{color:COLORS[s.key],width:2}, marker:{size:5}
  }));
  traces.push({x:[C.rates[ri]], y:[m[0]], mode:"markers", showlegend:false,
    marker:{size:13,symbol:"circle-open",color:"#d96a2b",line:{width:2.5}}});
  Plotly.react("cRate", traces, {...LAYOUT,
    title:{text:"Дефекты подачи от интенсивности (порог "+C.thresholds[ti].toFixed(2)+")",font:{...FONT,size:13}},
    xaxis:{title:"дет/с",gridcolor:"#e3e8ee"}, yaxis:{title:"дефекты /1000",gridcolor:"#e3e8ee"},
    legend:{orientation:"h",y:-0.28}, height:330}, CFG);

  const esc = C.thresholds.map((_,i)=>cell(key,ri,i)[0]);
  const fls = C.thresholds.map((_,i)=>cell(key,ri,i)[3]);
  Plotly.react("cThr", [
    {x:C.thresholds, y:esc, name:"дефекты /1000", mode:"lines+markers", line:{color:"#d96a2b",width:2}, marker:{size:5}},
    {x:C.thresholds, y:fls, name:"ложные срабатывания", yaxis:"y2", mode:"lines+markers", line:{color:"#1d3a5f",width:2,dash:"dot"}, marker:{size:5}},
    {x:[C.thresholds[ti]], y:[esc[ti]], mode:"markers", showlegend:false,
     marker:{size:13,symbol:"circle-open",color:"#d96a2b",line:{width:2.5}}}
  ], {...LAYOUT,
    title:{text:"Компромисс порога ("+stratName(key)+", "+C.rates[ri]+" дет/с)",font:{...FONT,size:13}},
    xaxis:{title:"порог уверенности",gridcolor:"#e3e8ee"},
    yaxis:{title:"дефекты /1000",gridcolor:"#e3e8ee"},
    yaxis2:{title:"ложные за 300 с",overlaying:"y",side:"right"},
    legend:{orientation:"h",y:-0.28}}, CFG);
}
function stratName(k){return C.strategies.find(s=>s.key===k).name;}
[stratSel,rate,thr].forEach(el=>el.addEventListener("input",updControl));

/* ---------- Вибротранспортирование ---------- */
const ampSel=document.getElementById("amp");
T.amps_mm.forEach((a,i)=>{const o=document.createElement("option");o.value=i;o.textContent="A = "+a+" мм";ampSel.appendChild(o);});
ampSel.value = 1;
const freq=document.getElementById("freq"), al=document.getElementById("al");
freq.max=T.freqs.length-1; freq.value=T.freqs.indexOf(17); /* ближайшее к 16,7 */
al.max=T.alphas.length-1; al.value=T.alphas.indexOf(6);
let heatDrawn=false;

function updTransport(){
  const ai=+ampSel.value, fi=+freq.value, li=+al.value;
  const A=T.amps_mm[ai]*1e-3, f=T.freqs[fi], alpha=T.alphas[li];
  document.getElementById("freqV").textContent=f;
  document.getElementById("alV").textContent=alpha;
  const w=2*Math.PI*f, g=9.81, gam=A*w*w/g;
  document.getElementById("kGam").textContent=gam.toFixed(2);
  document.getElementById("kReg").textContent = gam>1 ? "режим подбрасывания" : "безотрывный режим";
  document.getElementById("kVel").textContent = T.v_mm_s[ai][fi][li].toFixed(2);
  document.getElementById("kFcr").textContent = (Math.sqrt(g/A)/(2*Math.PI)).toFixed(1);
  if(heatDrawn) drawHeat();
}
function drawHeat(){
  heatDrawn=true;
  const ai=+ampSel.value, fi=+freq.value, li=+al.value;
  const z=T.freqs.map((_,i)=>T.alphas.map((_,j)=>T.v_mm_s[ai][i][j]));
  const fcr=Math.sqrt(9.81/(T.amps_mm[ai]*1e-3))/(2*Math.PI);
  Plotly.react("cHeat", [
    {z:z, x:T.alphas, y:T.freqs, type:"heatmap", colorscale:"YlGnBu",
     colorbar:{title:{text:"мм/с",font:FONT},tickfont:FONT}},
    {x:[T.alphas[li]], y:[T.freqs[fi]], mode:"markers", showlegend:false,
     marker:{size:13,symbol:"x",color:"#d96a2b",line:{width:2}}}
  ], {...LAYOUT, height:430,
    title:{text:"Скорость подачи v(α, f); линия — порог Γ = 1 ("+fcr.toFixed(1)+" Гц)",font:{...FONT,size:13}},
    xaxis:{title:"угол наклона α, °"}, yaxis:{title:"частота f, Гц"},
    shapes:[{type:"line",x0:T.alphas[0],x1:T.alphas[T.alphas.length-1],y0:fcr,y1:fcr,
      line:{color:"#d96a2b",width:2,dash:"dash"}},
      {type:"line",x0:T.alphas[0],x1:T.alphas[T.alphas.length-1],y0:16.7,y1:16.7,
      line:{color:"#1d3a5f",width:1.5,dash:"dot"}}],
    annotations:[{x:T.alphas[2],y:16.7,text:"рабочая точка НИР 16,7 Гц",showarrow:false,
      yshift:10,font:{...FONT,size:10,color:"#1d3a5f"}}]}, CFG);
}
[ampSel,freq,al].forEach(el=>el.addEventListener("input",updTransport));

updControl(); updTransport();
</script>
</body>
</html>
"""


def main() -> None:
    print("Сборка дашборда FeederTwin")
    print("Расчёт сетки контура контроля (~3 мин)...")
    control = build_control_grid()
    print("Расчёт сетки вибротранспортирования (~3 мин)...")
    transport = build_transport_grid()
    html = TEMPLATE.replace("__CONTROL_DATA__", json.dumps(control, ensure_ascii=False))
    html = html.replace("__TRANSPORT_DATA__", json.dumps(transport))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Готово: {OUT} ({OUT.stat().st_size // 1024} КБ)")


if __name__ == "__main__":
    main()
