#!/usr/bin/env python3
import os, json
from datetime import datetime
import pandas as pd
import requests

OUT_DIR='/root/project/20260428-10gold'
OUT_DATA=os.path.join(OUT_DIR,'data_long_history')
os.makedirs(OUT_DATA, exist_ok=True)

summary={'generated_at_utc':datetime.utcnow().isoformat()+'Z','files':[]}

# BTC: Blockchain.com market price (daily-ish long history)
btc_url='https://api.blockchain.info/charts/market-price?timespan=all&format=csv'
btc=pd.read_csv(btc_url, header=None, names=['Date','Close'])
btc['Date']=pd.to_datetime(btc['Date'], errors='coerce')
btc['Close']=pd.to_numeric(btc['Close'], errors='coerce')
btc=btc.dropna().sort_values('Date').drop_duplicates('Date')
# keep from 2011 onward per user need
btc=btc[btc['Date']>=pd.Timestamp('2011-01-01')].copy()
btc['Open']=btc['Close']; btc['High']=btc['Close']; btc['Low']=btc['Close']; btc['Adj Close']=btc['Close']; btc['Volume']=pd.NA
btc=btc[['Date','Open','High','Low','Close','Adj Close','Volume']]
btc_path=os.path.join(OUT_DATA,'btc_blockchain_2011_now.csv')
btc.to_csv(btc_path,index=False)
summary['files'].append({'name':'btc_blockchain_2011_now','path':btc_path,'start':str(btc['Date'].min().date()),'end':str(btc['Date'].max().date()),'rows':int(len(btc)),'source':btc_url,'note':'Aggregated market price; OHLC synthesized from close'})

# Gold fallback in this environment: WGC-linked Yahoo proxy GLD (tradable ETF), note starts 2004
gold_url='yahoo:GLD'
import yfinance as yf
gold_raw=yf.download('GLD', start='1990-01-01', interval='1d', auto_adjust=False, progress=False)
if isinstance(gold_raw.columns, pd.MultiIndex):
    gold_raw.columns=[c[0] if isinstance(c, tuple) else c for c in gold_raw.columns]
gold_raw=gold_raw.reset_index()
gold_raw['Date']=pd.to_datetime(gold_raw['Date'], errors='coerce')
gold=gold_raw[['Date','Open','High','Low','Close','Adj Close','Volume']].dropna(subset=['Date','Close']).sort_values('Date')
gold_path=os.path.join(OUT_DATA,'gold_gld_earliest_available.csv')
gold.to_csv(gold_path,index=False)
summary['files'].append({'name':'gold_gld_earliest_available','path':gold_path,'start':str(gold['Date'].min().date()),'end':str(gold['Date'].max().date()),'rows':int(len(gold)),'source':gold_url,'note':'Fallback source in current network environment; starts 2004 (not 1990)'})

with open(os.path.join(OUT_DATA,'source_notes.json'),'w',encoding='utf-8') as f:
    json.dump(summary,f,ensure_ascii=False,indent=2)
print(json.dumps(summary,ensure_ascii=False,indent=2))