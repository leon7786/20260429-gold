#!/usr/bin/env python3
import os, json
from datetime import datetime
import pandas as pd

OUT='/root/project/20260428-10gold/alt_sources'
os.makedirs(OUT, exist_ok=True)

summary={"generated_at": datetime.utcnow().isoformat()+"Z", "files": []}

# 1) Brent spot from FRED (EIA sourced)
brent_url='https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU'
brent=pd.read_csv(brent_url)
date_col = 'DATE' if 'DATE' in brent.columns else 'observation_date'
brent[date_col]=pd.to_datetime(brent[date_col], errors='coerce')
brent['DCOILBRENTEU']=pd.to_numeric(brent['DCOILBRENTEU'], errors='coerce')
brent=brent.dropna(subset=[date_col,'DCOILBRENTEU']).rename(columns={date_col:'Date','DCOILBRENTEU':'Close'})
brent['Open']=brent['Close']; brent['High']=brent['Close']; brent['Low']=brent['Close']; brent['Adj Close']=brent['Close']; brent['Volume']=pd.NA
brent=brent[['Date','Open','High','Low','Close','Adj Close','Volume']]
brent_path=f"{OUT}/brent_fred_dcoilbrenteu.csv"
brent.to_csv(brent_path, index=False)
summary['files'].append({"name":"brent_fred_dcoilbrenteu","path":brent_path,"start":str(brent['Date'].min().date()),"end":str(brent['Date'].max().date()),"rows":int(len(brent))})

# 2) Bitcoin daily from CoinGecko market_chart (max)
import requests
cg='https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=max&interval=daily'
js=requests.get(cg, timeout=60).json()
prices=js.get('prices',[])
vols={int(t):v for t,v in js.get('total_volumes',[])}
rows=[]
for t,p in prices:
    dt=pd.to_datetime(int(t), unit='ms', utc=True).tz_convert(None).date()
    rows.append((dt,float(p),vols.get(int(t))))
btc=pd.DataFrame(rows, columns=['Date','Close','Volume']).drop_duplicates('Date').sort_values('Date')
btc['Open']=btc['Close']; btc['High']=btc['Close']; btc['Low']=btc['Close']; btc['Adj Close']=btc['Close']
btc=btc[['Date','Open','High','Low','Close','Adj Close','Volume']]
btc_path=f"{OUT}/btc_coingecko_daily.csv"
btc.to_csv(btc_path, index=False)
summary['files'].append({"name":"btc_coingecko_daily","path":btc_path,"start":str(btc['Date'].min()),"end":str(btc['Date'].max()),"rows":int(len(btc))})

# 3) Kraken BTCUSD via Yahoo proxy as quick exchange-specific placeholder
import yfinance as yf
k=yf.download('BTC-USD', start='2010-01-01', interval='1d', auto_adjust=False, progress=False)
if not k.empty:
    if isinstance(k.columns, pd.MultiIndex):
        k.columns=[c[0] if isinstance(c, tuple) else c for c in k.columns]
    k=k.reset_index().rename(columns={'Adj Close':'Adj Close'})
    cols=[c for c in ['Date','Open','High','Low','Close','Adj Close','Volume'] if c in k.columns]
    k=k[cols]
    kpath=f"{OUT}/btc_yahoo_btcusd_reference.csv"
    k.to_csv(kpath, index=False)
    summary['files'].append({"name":"btc_yahoo_btcusd_reference","path":kpath,"start":str(pd.to_datetime(k['Date']).min().date()),"end":str(pd.to_datetime(k['Date']).max().date()),"rows":int(len(k))})

with open(f"{OUT}/summary.json",'w') as f:
    json.dump(summary,f,indent=2,ensure_ascii=False)
print(json.dumps(summary,indent=2,ensure_ascii=False))
