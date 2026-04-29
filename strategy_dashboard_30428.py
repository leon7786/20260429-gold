from flask import Flask, jsonify, request, Response
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd

BASE = Path('/root/project/20260428-10gold')
RES = BASE / 'backtest_results'

ASSETS = {
    'gold_spot_proxy': BASE / 'data_long_history' / 'gold_daily_proxy_gld_yahoo.csv',
    'brent': BASE / 'data' / 'brent.csv',
    'btc': BASE / 'data' / 'btc.csv',
    'copper_core': BASE / 'data' / 'copper.csv',
    'natgas_core': BASE / 'data' / 'natgas.csv',
    'eurusd': BASE / 'data' / 'eurusd.csv',
    'usdjpy': BASE / 'data' / 'usdjpy.csv',
    'sp500': BASE / 'data' / 'sp500.csv',
    'us10y_yield': BASE / 'data' / 'us10y_yield.csv',
    'usd_dxy': BASE / 'data' / 'usd_dxy.csv',
    'nasdaq100': BASE / 'data_extra_1990' / 'nasdaq100.csv',
    'eth': BASE / 'data_extra_1990' / 'eth.csv',
    'ttf_gas_europe': BASE / 'data_extra_1990' / 'ttf_gas_europe.csv',
    'silver': BASE / 'data_extra_1990' / 'silver.csv',
    'msci_world_proxy': BASE / 'data_extra_1990' / 'msci_world_proxy.csv',
    'msci_em_proxy': BASE / 'data_extra_1990' / 'msci_em_proxy.csv',
    'chinext_proxy_etf_159915': BASE / 'data_extra_1990' / 'chinext_proxy_etf_159915.csv',
    'nikkei225': BASE / 'data_extra_1990' / 'nikkei225.csv',
    'topix_proxy_etf': BASE / 'data_extra_1990' / 'topix_proxy_etf.csv',
}

DISPLAY_NAME = {
    'gold_spot_proxy': '黄金现货',
    'brent': '布伦特原油',
    'btc': '比特币',
    'copper_core': '铜(核心)',
    'natgas_core': '天然气(核心)',
    'eurusd': '欧元/美元',
    'usdjpy': '美元/日元',
    'sp500': '标普500',
    'us10y_yield': '美债10Y收益率',
    'usd_dxy': '美元指数',
    'nasdaq100': '纳斯达克100',
    'eth': '以太坊',
    'ttf_gas_europe': '欧洲TTF天然气',
    'silver': '白银',
    'msci_world_proxy': 'MSCI全球',
    'msci_em_proxy': 'MSCI新兴市场',
    'chinext_proxy_etf_159915': '创业板ETF(159915)',
    'nikkei225': '日经225',
    'topix_proxy_etf': 'TOPIX ETF',
}

STRATEGIES = pd.read_csv(RES / 'best_ma_strategies_19_assets_2_700.csv')[['asset', 'best_buy_ma', 'best_sell_ma']]
STRATEGY_MAP = {r['asset']: (int(r['best_buy_ma']), int(r['best_sell_ma'])) for _, r in STRATEGIES.iterrows()}

CROSS_FILE = RES / 'cross_test_19_strategies_x_19_assets.csv'
CROSS_DF = pd.read_csv(CROSS_FILE) if CROSS_FILE.exists() else pd.DataFrame()


def load_close(path: Path):
    df = pd.read_csv(path)
    colmap = {c.lower().strip(): c for c in df.columns}
    date_col = colmap.get('date') or colmap.get('timestamp') or 'Date'
    close_col = colmap.get('close') or colmap.get('adj close') or colmap.get('adj_close') or 'Close'
    out = pd.DataFrame({
        'date': pd.to_datetime(df[date_col], errors='coerce'),
        'close': pd.to_numeric(df[close_col], errors='coerce')
    }).dropna().sort_values('date').reset_index(drop=True)

    # 数据源偶发“单位切换/拆分未复权”导致的断层（如 TOPIX 2026-03-30 附近 ~1/10）
    # 仅在出现极端跳变且比例接近 2/5/10 时，对跳变后的序列做缩放修正。
    c = out['close'].astype(float)
    r = c.pct_change()
    bad_idx = r.index[(r < -0.65) | (r > 1.8)]
    if len(bad_idx) > 0:
        i = int(bad_idx[0])
        if i > 0 and c.iat[i] != 0:
            ratio = c.iat[i - 1] / c.iat[i]
            candidates = [2, 5, 10]
            best = min(candidates, key=lambda x: abs(ratio - x))
            if abs(ratio - best) / best < 0.12:
                out.loc[i:, 'close'] = out.loc[i:, 'close'] * best

    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    return out


CLOSE_CACHE = {k: load_close(v) for k, v in ASSETS.items()}


def run_strategy(close: pd.Series, buy_ma: int, sell_ma: int):
    c = close.reset_index(drop=True)
    ma_buy = c.rolling(buy_ma).mean()
    ma_sell = c.rolling(sell_ma).mean()
    buy = (c.shift(1) <= ma_buy.shift(1)) & (c > ma_buy)
    sell = (c.shift(1) >= ma_sell.shift(1)) & (c < ma_sell)

    pos = 0
    entry = None
    eq = 1.0
    prev = None
    trades = 0
    wins = 0

    for i in range(len(c)):
        px = float(c.iat[i])
        if prev is not None and pos == 1:
            eq *= px / prev
        if pos == 0 and bool(buy.iat[i]):
            pos = 1
            entry = px
        elif pos == 1 and bool(sell.iat[i]):
            r = px / entry - 1.0
            trades += 1
            if r > 0:
                wins += 1
            pos = 0
            entry = None
        prev = px

    if pos == 1 and entry is not None:
        r = float(c.iat[-1]) / entry - 1.0
        trades += 1
        if r > 0:
            wins += 1

    return (eq - 1.0) * 100.0, trades, (wins / trades * 100.0 if trades else 0.0)


def run_strategy_equity_curve(close: pd.Series, buy_ma: int, sell_ma: int):
    c = close.reset_index(drop=True)
    ma_buy = c.rolling(buy_ma).mean()
    ma_sell = c.rolling(sell_ma).mean()
    buy = (c.shift(1) <= ma_buy.shift(1)) & (c > ma_buy)
    sell = (c.shift(1) >= ma_sell.shift(1)) & (c < ma_sell)

    pos = 0
    eq = 1.0
    prev = None
    curve = []

    for i in range(len(c)):
        px = float(c.iat[i])
        if prev is not None and pos == 1:
            eq *= px / prev
        if pos == 0 and bool(buy.iat[i]):
            pos = 1
        elif pos == 1 and bool(sell.iat[i]):
            pos = 0
        prev = px
        curve.append(eq * 100.0)

    return curve


@lru_cache(maxsize=64)
def build_performance_for_strategy(strategy_key: str):
    buy_ma, sell_ma = STRATEGY_MAP[strategy_key]
    perf = []
    for a, df in CLOSE_CACHE.items():
        c = df['close']
        bh = (float(c.iloc[-1]) / float(c.iloc[0]) - 1.0) * 100.0
        sr, tr, wr = run_strategy(c, buy_ma, sell_ma)
        perf.append({
            'asset': a,
            'strategy_return_pct': round(sr, 6),
            'buy_hold_return_pct': round(bh, 6),
            'excess_pct': round(sr - bh, 6),
            'trades': tr,
            'win_rate_pct': round(wr, 4)
        })
    return sorted(perf, key=lambda x: x['excess_pct'], reverse=True)


# 预加载 long/short 对比表（用于顶部一键查看）
LONG_FILE = RES / 'all_assets_best_ma_2_700_vs_2_700.csv'
SHORT_FILE = RES / 'all_assets_best_short_only_ma_2_700_vs_2_700_summary.csv'
if LONG_FILE.exists() and SHORT_FILE.exists():
    _ldf = pd.read_csv(LONG_FILE)
    _sdf = pd.read_csv(SHORT_FILE)
    LONG_SHORT_DF = _ldf[['asset', 'best_buy_ma', 'best_sell_ma', 'best_return_pct']].merge(
        _sdf[['asset', 'best_entry_ma', 'best_cover_ma', 'best_short_return_pct']],
        on='asset', how='inner'
    )
    LONG_SHORT_DF['better_side'] = np.where(
        LONG_SHORT_DF['best_return_pct'] >= LONG_SHORT_DF['best_short_return_pct'],
        'LONG', 'SHORT'
    )
    LONG_SHORT_DF['ret_gap_pct'] = LONG_SHORT_DF['best_return_pct'] - LONG_SHORT_DF['best_short_return_pct']
else:
    LONG_SHORT_DF = pd.DataFrame()

app = Flask(__name__)


@app.get('/api/meta')
def api_meta():
    return jsonify({
        'assets': [{'key': k, 'label': DISPLAY_NAME.get(k, k)} for k in ASSETS.keys()],
        'strategies': [
            {'key': k, 'label': DISPLAY_NAME.get(k, k), 'buy_ma': v[0], 'sell_ma': v[1]}
            for k, v in STRATEGY_MAP.items()
        ]
    })


@app.get('/api/cross_excess')
def api_cross_excess():
    if CROSS_DF.empty:
        return jsonify({'error': 'cross test file not found'}), 404

    x = CROSS_DF.copy()
    if 'buy_hold_return_pct' not in x.columns:
        bh_map = {}
        for a, df in CLOSE_CACHE.items():
            c = df['close']
            bh_map[a] = (float(c.iloc[-1]) / float(c.iloc[0]) - 1.0) * 100.0
        x['buy_hold_return_pct'] = x['asset_tested'].map(bh_map)

    x['excess_pct'] = x['total_return_pct'] - x['buy_hold_return_pct']
    mat = x.pivot(index='strategy_from', columns='asset_tested', values='excess_pct').round(4)

    return jsonify({'strategies': mat.index.tolist(), 'assets': mat.columns.tolist(), 'matrix': mat.values.tolist()})


@app.get('/api/long_short_best')
def api_long_short_best():
    if LONG_SHORT_DF.empty:
        return jsonify({'error': 'long/short summary file not found'}), 404
    x = LONG_SHORT_DF.copy()
    x['asset_label'] = x['asset'].map(lambda a: DISPLAY_NAME.get(a, a))
    x = x.sort_values('asset_label')
    rows = []
    for _, r in x.iterrows():
        rows.append({
            'asset': r['asset'],
            'asset_label': r['asset_label'],
            'long_buy_ma': int(r['best_buy_ma']),
            'long_sell_ma': int(r['best_sell_ma']),
            'long_return_pct': float(r['best_return_pct']),
            'short_entry_ma': int(r['best_entry_ma']),
            'short_cover_ma': int(r['best_cover_ma']),
            'short_return_pct': float(r['best_short_return_pct']),
            'better_side': r['better_side'],
            'ret_gap_pct': float(r['ret_gap_pct'])
        })
    return jsonify({'rows': rows})


@app.get('/api/data')
def api_data():
    strategy = request.args.get('strategy', 'gold_spot_proxy')
    assets = request.args.get('assets', '')
    selected = [a for a in assets.split(',') if a in ASSETS] if assets else ['gold_spot_proxy']

    if strategy not in STRATEGY_MAP:
        return jsonify({'error': 'invalid strategy'}), 400

    buy_ma, sell_ma = STRATEGY_MAP[strategy]

    chart = []
    for a in selected:
        df = CLOSE_CACHE[a]
        c = df['close']
        base = float(c.iloc[0])
        norm = (c / base) * 100.0
        eq_curve = run_strategy_equity_curve(c, buy_ma, sell_ma)
        chart.append({
            'asset': a,
            'asset_label': DISPLAY_NAME.get(a, a),
            'dates': df['date'].tolist(),
            'price_norm100': norm.round(4).tolist(),
            'strategy_eq100': [round(v, 4) for v in eq_curve]
        })

    return jsonify({
        'strategy': strategy,
        'buy_ma': buy_ma,
        'sell_ma': sell_ma,
        'chart': chart,
        'performance': build_performance_for_strategy(strategy)
    })


@app.get('/')
def index():
    html = """
<!doctype html><html><head><meta charset='utf-8'><title>策略对比看板</title>
<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
<style>
body{font-family:Arial;margin:4px;background:#0f172a;color:#e2e8f0}
.card{background:#111827;padding:4px;border-radius:6px}
#chart{height:720px}
table{width:100%;border-collapse:collapse}
td,th{border-bottom:1px solid #334155;padding:3px;font-size:10px}
.grid10{display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:4px}
.btn{background:#1f2937;color:#e5e7eb;border:1px solid #374151;padding:3px 4px;border-radius:5px;cursor:pointer;font-size:9px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.1}
.btn.active{background:#2563eb;border-color:#60a5fa}
.toggle{display:flex;align-items:center;gap:4px;margin-bottom:4px;font-size:9px}
h2{font-size:10px;margin:0 0 3px 0;font-weight:600;color:#cbd5e1}
h3{font-size:9px;margin:0 0 3px 0;font-weight:600;color:#cbd5e1}
label,strong,#strategyInfo{font-size:9px}
</style>
</head><body>
<h2>19策略 × 全品种对比面板</h2>
<div class='card' style='margin-top:6px'>
  <div class='toggle'>
    <strong>品种按钮（19个，10个一排）</strong>
    <label><input type='checkbox' id='multiMode'/> 多选重叠模式</label>
  </div>
  <div id='assetButtons' class='grid10'></div>
</div>
<div class='card'>
  <div class='toggle'>
    <strong>策略按钮（19个，10个一排）</strong>
    <button id='toggleLongShort' class='btn' style='margin-left:8px'>一键查看：多空最优对照</button>
  </div>
  <div id='strategyButtons' class='grid10'></div>
  <div id='strategyInfo' style='margin-top:4px;color:#93c5fd'></div>
</div>
<div id='longShortPanel' class='card' style='margin-top:6px;display:none'>
  <h3>各品种 Long/Short 最优策略对照</h3>
  <div style='overflow:auto;max-height:260px'><table id='longShortTable'></table></div>
</div>
<div class='card' style='margin-top:6px'><div id='chart'></div></div>
<div class='card' style='margin-top:6px'>
  <h3>相对买入持有的超额收益（按超额降序）</h3>
  <table id='perf'><thead><tr><th>品种</th><th>策略收益%</th><th>买入持有%</th><th>超额%</th><th>交易次数</th><th>胜率%</th></tr></thead><tbody></tbody></table>
</div>
<div class='card' style='margin-top:6px'>
  <h3>19策略 × 19品种 超额收益矩阵（%）</h3>
  <div style='overflow:auto;max-height:340px'><table id='crossTable'></table></div>
</div>
<script>
let meta=null;
let currentStrategy='gold_spot_proxy';
let selectedAssets=['gold_spot_proxy'];
let crossData=null;
let longShortData=null;
let longShortVisible=false;

function makeBtn(text, active=false){
  const b=document.createElement('button');
  b.className='btn'+(active?' active':'');
  b.textContent=text;
  return b;
}

function renderStrategyButtons(){
  const box=document.getElementById('strategyButtons');
  box.innerHTML='';
  meta.strategies.forEach(s=>{
    const b=makeBtn(`${s.label} (MA${s.buy_ma}/${s.sell_ma})`, s.key===currentStrategy);
    b.onclick=()=>{currentStrategy=s.key;renderStrategyButtons();reloadAll();};
    box.appendChild(b);
  });
}

function renderAssetButtons(){
  const box=document.getElementById('assetButtons');
  box.innerHTML='';
  meta.assets.forEach(a=>{
    const b=makeBtn(a.label, selectedAssets.includes(a.key));
    b.onclick=()=>{
      const multi=document.getElementById('multiMode').checked;
      if(!multi){
        selectedAssets=[a.key];
      }else{
        if(selectedAssets.includes(a.key)){
          selectedAssets=selectedAssets.filter(x=>x!==a.key);
          if(selectedAssets.length===0) selectedAssets=[a.key];
        }else{
          selectedAssets.push(a.key);
        }
      }
      renderAssetButtons();
      reloadAll();
    };
    box.appendChild(b);
  });
}

function renderLongShortTable(d){
  const tbl=document.getElementById('longShortTable');
  if(!d || !d.rows){ tbl.innerHTML=''; return; }
  let h='<thead><tr><th>品种</th><th>LONG最优MA(买/卖)</th><th>LONG收益%</th><th>SHORT最优MA(开/平)</th><th>SHORT收益%</th><th>推荐</th></tr></thead><tbody>';
  d.rows.forEach(r=>{
    const rec = r.better_side==='LONG' ? '做多' : '做空';
    const recColor = r.better_side==='LONG' ? '#86efac' : '#fca5a5';
    h += `<tr><td>${r.asset_label}</td><td>${r.long_buy_ma}/${r.long_sell_ma}</td><td>${r.long_return_pct.toFixed(2)}</td><td>${r.short_entry_ma}/${r.short_cover_ma}</td><td>${r.short_return_pct.toFixed(2)}</td><td style="color:${recColor}">${rec}</td></tr>`;
  });
  h += '</tbody>';
  tbl.innerHTML = h;
}

function renderCrossTable(c){
  const tbl=document.getElementById('crossTable');
  let h='<thead><tr><th>策略\\品种</th>';
  c.assets.forEach(a=>h+=`<th>${meta.assets.find(x=>x.key===a)?.label || a}</th>`);
  h+='</tr></thead><tbody>';
  for(let i=0;i<c.strategies.length;i++){
    const sKey=c.strategies[i];
    const sLabel=meta.strategies.find(x=>x.key===sKey)?.label || sKey;
    h+=`<tr><td>${sLabel}</td>`;
    for(let j=0;j<c.assets.length;j++){
      const v=c.matrix[i][j];
      const color=(v>=0)?'#86efac':'#fca5a5';
      h+=`<td style="color:${color}">${Number(v).toFixed(2)}</td>`;
    }
    h+='</tr>';
  }
  h+='</tbody>';
  tbl.innerHTML=h;
}

async function reloadAll(){
  const assets=selectedAssets.join(',');
  const d=await (await fetch(`/api/data?strategy=${currentStrategy}&assets=${assets}`)).json();
  const cs = meta.strategies.find(s=>s.key===d.strategy);
  const strategyLabel = cs ? cs.label : d.strategy;
  document.getElementById('strategyInfo').textContent=`当前策略: ${strategyLabel} | 买入上穿MA${d.buy_ma}，卖出下穿MA${d.sell_ma}`;

  const traces=[];
  d.chart.forEach(x=>{
    const label = x.asset_label || x.asset;
    traces.push({x:x.dates,y:x.price_norm100,mode:'lines',name:`${label}-价格(100基准)`,line:{width:1,dash:'dot'}});
    traces.push({x:x.dates,y:x.strategy_eq100,mode:'lines',name:`${label}-资金曲线`,line:{width:2}});
  });

  Plotly.react('chart',traces,{
    paper_bgcolor:'#111827', plot_bgcolor:'#111827', font:{color:'#e2e8f0',size:10},
    title:{text:'行情 + 策略资金曲线（100基准）',font:{size:11}}, margin:{l:36,r:14,t:28,b:22}, legend:{font:{size:9}}
  }, {responsive:true});

  const tb=document.querySelector('#perf tbody');
  tb.innerHTML='';
  d.performance.forEach(r=>{
    const tr=document.createElement('tr');
    const label = (meta.assets.find(x=>x.key===r.asset)?.label) || r.asset;
    tr.innerHTML=`<td>${label}</td><td>${r.strategy_return_pct.toFixed(2)}</td><td>${r.buy_hold_return_pct.toFixed(2)}</td><td>${r.excess_pct.toFixed(2)}</td><td>${r.trades}</td><td>${r.win_rate_pct.toFixed(2)}</td>`;
    tb.appendChild(tr);
  });

  if(crossData && !crossData.error) renderCrossTable(crossData);
}

async function init(){
  const [m, c, ls] = await Promise.all([
    fetch('/api/meta').then(r=>r.json()),
    fetch('/api/cross_excess').then(r=>r.json()),
    fetch('/api/long_short_best').then(r=>r.json()).catch(()=>({rows:[]}))
  ]);
  meta = m;
  crossData = c;
  longShortData = ls;

  const panel=document.getElementById('longShortPanel');
  const btn=document.getElementById('toggleLongShort');
  btn.onclick=()=>{
    longShortVisible = !longShortVisible;
    panel.style.display = longShortVisible ? 'block' : 'none';
    if(longShortVisible) renderLongShortTable(longShortData);
  };

  renderStrategyButtons();
  renderAssetButtons();
  reloadAll();
}
init();
</script></body></html>
"""
    return Response(html, mimetype='text/html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30428)
