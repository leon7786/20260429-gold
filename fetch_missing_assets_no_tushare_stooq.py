from pathlib import Path
import json
import requests
import pandas as pd
import yfinance as yf

BASE = Path('/root/project/20260428-10gold')
OUT = BASE / 'data_extra_1990'
OUT.mkdir(parents=True, exist_ok=True)

results = []

# 1) TOPIX proxy without Stooq: use Yahoo JP TOPIX ETFs
#   1306.T (TOPIX ETF) has long history and stable availability
for sym in ['1306.T', '1348.T']:
    try:
        df = yf.Ticker(sym).history(period='max', auto_adjust=False)
        if df is None or df.empty:
            continue
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.reset_index().rename(columns={'Date': 'date', 'Open':'open','High':'high','Low':'low','Close':'close','Adj Close':'adj_close','Volume':'volume'})
        keep = [c for c in ['date','open','high','low','close','adj_close','volume'] if c in df.columns]
        df = df[keep]
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        out = OUT / 'topix_proxy_etf.csv'
        df.to_csv(out, index=False)
        results.append({
            'asset':'topix_proxy_etf','status':'ok','source':'yahoo','symbol':sym,
            'start':df['date'].iloc[0],'end':df['date'].iloc[-1],'rows':int(len(df)),'file':str(out)
        })
        break
    except Exception:
        pass
else:
    results.append({'asset':'topix_proxy_etf','status':'error','source':'yahoo','error':'1306.T/1348.T unavailable'})

# 2) Chinext index without Tushare: Eastmoney kline API (public)
try:
    url = 'https://push2his.eastmoney.com/api/qt/stock/kline/get'
    params = {
        'secid': '0.399006',
        'klt': '101',    # daily
        'fqt': '1',
        'lmt': '100000',
        'end': '20500101',
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58',
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get('data', {})
    klines = data.get('klines', [])
    rows = []
    for k in klines:
        # date,open,close,high,low,volume,amount,amplitude
        p = k.split(',')
        if len(p) < 7:
            continue
        rows.append({
            'date': p[0],
            'open': float(p[1]),
            'high': float(p[3]),
            'low': float(p[4]),
            'close': float(p[2]),
            'volume': float(p[5]),
            'amount': float(p[6]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        results.append({'asset':'chinext_index','status':'empty','source':'eastmoney'})
    else:
        df = df.sort_values('date').reset_index(drop=True)
        out = OUT / 'chinext_index.csv'
        df.to_csv(out, index=False)
        results.append({
            'asset':'chinext_index','status':'ok','source':'eastmoney','symbol':'399006',
            'start':df['date'].iloc[0],'end':df['date'].iloc[-1],'rows':int(len(df)),'file':str(out)
        })
except Exception as e:
    results.append({'asset':'chinext_index','status':'error','source':'eastmoney','error':str(e)})

summary = OUT / 'download_summary_extra_assets_missing_fill_no_tushare_stooq.json'
summary.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(json.dumps(results, ensure_ascii=False, indent=2))
print('saved', summary)
