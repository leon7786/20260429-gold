from pathlib import Path
import json
import pandas as pd

BASE = Path('/root/project/20260428-10gold')
OUT = BASE / 'data_extra_1990'
OUT.mkdir(parents=True, exist_ok=True)

results = []

# ---------- TOPIX: try Stooq via pandas_datareader ----------
try:
    from pandas_datareader import data as pdr

    topix_candidates = ['^TPX', 'TPX', 'TOPX']
    got = None
    for sym in topix_candidates:
        try:
            df = pdr.DataReader(sym, 'stooq')
            if df is not None and not df.empty:
                got = (sym, df)
                break
        except Exception:
            pass

    if got is None:
        results.append({'asset': 'topix', 'status': 'error', 'source': 'stooq', 'error': 'all symbols failed'})
    else:
        sym, df = got
        df = df.sort_index().reset_index().rename(columns={'Date': 'date'})
        rename_map = {
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        }
        df = df.rename(columns=rename_map)
        keep = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume'] if c in df.columns]
        df = df[keep]
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        out = OUT / 'topix.csv'
        df.to_csv(out, index=False)
        results.append({
            'asset': 'topix', 'status': 'ok', 'source': 'stooq', 'symbol': sym,
            'start': df['date'].iloc[0], 'end': df['date'].iloc[-1], 'rows': int(len(df)), 'file': str(out)
        })
except Exception as e:
    results.append({'asset': 'topix', 'status': 'error', 'source': 'stooq', 'error': str(e)})

# ---------- Chinext index: try AkShare ----------
try:
    import akshare as ak

    # 创业板指代码 399006
    df = ak.index_zh_a_hist(symbol='399006', period='daily', start_date='20100101', end_date='20500101')
    if df is None or df.empty:
        results.append({'asset': 'chinext_index', 'status': 'empty', 'source': 'akshare'})
    else:
        # columns usually: 日期 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅 涨跌额 换手率
        rename = {
            '日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'
        }
        df = df.rename(columns=rename)
        keep = [c for c in ['date', 'open', 'high', 'low', 'close', 'volume'] if c in df.columns]
        df = df[keep].copy()
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        for c in ['open', 'high', 'low', 'close', 'volume']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['date', 'close']).sort_values('date').reset_index(drop=True)
        out = OUT / 'chinext_index.csv'
        df.to_csv(out, index=False)
        results.append({
            'asset': 'chinext_index', 'status': 'ok', 'source': 'akshare',
            'start': df['date'].iloc[0], 'end': df['date'].iloc[-1], 'rows': int(len(df)), 'file': str(out)
        })
except Exception as e:
    results.append({'asset': 'chinext_index', 'status': 'error', 'source': 'akshare', 'error': str(e)})

# ---------- write summary ----------
out_summary = OUT / 'download_summary_extra_assets_missing_fill.json'
out_summary.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(json.dumps(results, ensure_ascii=False, indent=2))
print('saved', out_summary)
