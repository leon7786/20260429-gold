#!/usr/bin/env python3
import os
import json
from datetime import datetime

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

OUT_DIR = '/root/project/20260428-10gold'
DATA_DIR = os.path.join(OUT_DIR, 'data')
CHART_DIR = os.path.join(OUT_DIR, 'charts')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

# 10 assets mapped to liquid proxies/tickers suitable for daily backtests
ASSETS = [
    ('usd_dxy', 'DX-Y.NYB', 'US Dollar Index (DXY)'),
    ('us10y_yield', '^TNX', 'US 10Y Treasury Yield'),
    ('sp500', '^GSPC', 'S&P 500 Index'),
    ('brent', 'BZ=F', 'Brent Crude Futures'),
    ('gold', 'GC=F', 'Gold Futures'),
    ('eurusd', 'EURUSD=X', 'EUR/USD FX'),
    ('usdjpy', 'JPY=X', 'USD/JPY FX'),
    ('natgas', 'NG=F', 'Henry Hub Natural Gas Futures'),
    ('copper', 'HG=F', 'Copper Futures'),
    ('btc', 'BTC-USD', 'Bitcoin USD'),
]

START = '1990-01-01'
END = datetime.utcnow().strftime('%Y-%m-%d')

summary = []

for key, ticker, name in ASSETS:
    df = yf.download(ticker, start=START, end=END, interval='1d', auto_adjust=False, progress=False)

    if df.empty:
        summary.append({
            'asset': key,
            'ticker': ticker,
            'name': name,
            'status': 'empty',
        })
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    keep_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] if c in df.columns]
    df = df[keep_cols].copy()
    df.index.name = 'Date'

    csv_path = os.path.join(DATA_DIR, f'{key}.csv')
    df.to_csv(csv_path)

    # quick daily close chart
    close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df.index, df[close_col], linewidth=1.0)
    ax.set_title(f'{name} ({ticker}) Daily {close_col} from {df.index.min().date()} to {df.index.max().date()}')
    ax.set_xlabel('Date')
    ax.set_ylabel(close_col)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    png_path = os.path.join(CHART_DIR, f'{key}.png')
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    summary.append({
        'asset': key,
        'ticker': ticker,
        'name': name,
        'status': 'ok',
        'rows': int(len(df)),
        'start': str(df.index.min().date()),
        'end': str(df.index.max().date()),
        'csv': csv_path,
        'chart': png_path,
    })

with open(os.path.join(OUT_DIR, 'download_summary.json'), 'w', encoding='utf-8') as f:
    json.dump({'requested_start': START, 'requested_end': END, 'generated_at_utc': datetime.utcnow().isoformat() + 'Z', 'assets': summary}, f, ensure_ascii=False, indent=2)

print(json.dumps({'out_dir': OUT_DIR, 'assets': summary}, ensure_ascii=False, indent=2))
