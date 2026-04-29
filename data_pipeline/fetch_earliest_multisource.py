#!/usr/bin/env python3
import os
import json
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

OUT_DIR = '/root/project/20260428-10gold'
DATA_DIR = os.path.join(OUT_DIR, 'data_multisource')
CHART_DIR = os.path.join(OUT_DIR, 'charts_multisource')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

START = '1990-01-01'
END = datetime.utcnow().strftime('%Y-%m-%d')

ASSETS = {
    'usd_dxy': {'name': 'US Dollar Index (DXY)', 'candidates': [('fred', 'DTWEXBGS'), ('yahoo', 'DX-Y.NYB')]},
    'us10y_yield': {'name': 'US 10Y Treasury Yield', 'candidates': [('fred', 'DGS10'), ('yahoo', '^TNX')]},
    'sp500': {'name': 'S&P 500 Index', 'candidates': [('fred', 'SP500'), ('yahoo', '^GSPC')]},
    'brent': {'name': 'Brent Crude', 'candidates': [('fred', 'DCOILBRENTEU'), ('yahoo', 'BZ=F')]},
    'gold': {'name': 'Gold', 'candidates': [('fred', 'GOLDAMGBD228NLBM'), ('yahoo', 'GC=F')]},
    'eurusd': {'name': 'EUR/USD', 'candidates': [('fred', 'DEXUSEU'), ('yahoo', 'EURUSD=X')]},
    'usdjpy': {'name': 'USD/JPY', 'candidates': [('fred', 'DEXJPUS'), ('yahoo', 'JPY=X')]},
    'natgas': {'name': 'Natural Gas (Henry Hub proxy)', 'candidates': [('fred', 'DHHNGSP'), ('yahoo', 'NG=F')]},
    'copper': {'name': 'Copper', 'candidates': [('fred', 'PCOPPUSDM'), ('yahoo', 'HG=F')]},
    'btc': {'name': 'Bitcoin USD', 'candidates': [('yahoo', 'BTC-USD')]},
}


def fetch_fred(series: str) -> pd.DataFrame:
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}'
    df = pd.read_csv(url)
    if df.empty or 'DATE' not in df.columns or series not in df.columns:
        return pd.DataFrame()
    df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    df[series] = pd.to_numeric(df[series], errors='coerce')
    df = df.dropna(subset=['DATE', series])
    df = df[(df['DATE'] >= pd.to_datetime(START)) & (df['DATE'] <= pd.to_datetime(END))]
    if df.empty:
        return pd.DataFrame()
    s = df.set_index('DATE')[series].rename('Close')
    out = pd.DataFrame(index=s.index)
    out['Open'] = s
    out['High'] = s
    out['Low'] = s
    out['Close'] = s
    out['Adj Close'] = s
    out['Volume'] = pd.NA
    out.index.name = 'Date'
    return out


def fetch_yahoo(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start=START, end=END, interval='1d', auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    keep_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] if c in df.columns]
    out = df[keep_cols].copy()
    if 'Adj Close' not in out.columns and 'Close' in out.columns:
        out['Adj Close'] = out['Close']
    if 'Volume' not in out.columns:
        out['Volume'] = pd.NA
    out.index.name = 'Date'
    return out.dropna(subset=['Close'])

summary = {'requested_start': START, 'requested_end': END, 'generated_at_utc': datetime.utcnow().isoformat() + 'Z', 'assets': []}

for key, cfg in ASSETS.items():
    best = None
    tried = []
    for src, code in cfg['candidates']:
        try:
            df = fetch_fred(code) if src == 'fred' else fetch_yahoo(code)
            if df.empty:
                tried.append({'source': src, 'symbol': code, 'status': 'empty'})
                continue
            rec = {
                'source': src,
                'symbol': code,
                'df': df,
                'rows': int(len(df)),
                'start': str(pd.to_datetime(df.index.min()).date()),
                'end': str(pd.to_datetime(df.index.max()).date()),
            }
            tried.append({'source': src, 'symbol': code, 'status': 'ok', 'rows': rec['rows'], 'start': rec['start'], 'end': rec['end']})
            if best is None or rec['start'] < best['start'] or (rec['start'] == best['start'] and rec['rows'] > best['rows']):
                best = rec
        except Exception as e:
            tried.append({'source': src, 'symbol': code, 'status': f'error: {type(e).__name__}: {e}'})

    if best is None:
        summary['assets'].append({'asset': key, 'name': cfg['name'], 'status': 'failed', 'tried': tried})
        continue

    df = best.pop('df')
    csv_path = os.path.join(DATA_DIR, f'{key}.csv')
    df.to_csv(csv_path)

    close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df.index, df[close_col], linewidth=1.0)
    ax.set_title(f"{cfg['name']} [{best['source']}:{best['symbol']}] {best['start']} → {best['end']}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    png_path = os.path.join(CHART_DIR, f'{key}.png')
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    summary['assets'].append({
        'asset': key,
        'name': cfg['name'],
        'selected_source': best['source'],
        'selected_symbol': best['symbol'],
        'rows': best['rows'],
        'start': best['start'],
        'end': best['end'],
        'csv': csv_path,
        'chart': png_path,
        'tried': tried,
    })

with open(os.path.join(OUT_DIR, 'download_summary_multisource.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
