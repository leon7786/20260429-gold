from pathlib import Path
import json
import pandas as pd
import yfinance as yf

BASE = Path('/root/project/20260428-10gold')
OUT = BASE / 'data_extra_1990'
OUT.mkdir(parents=True, exist_ok=True)

# 以“可落地回测”为优先，优先用 Yahoo 可下载 ticker
ASSETS = [
    ("nasdaq100", "^NDX", "纳斯达克100指数"),
    ("eth", "ETH-USD", "以太坊"),
    ("copper", "HG=F", "铜期货"),
    ("natgas_henry_hub", "NG=F", "Henry Hub 天然气"),
    ("ttf_gas_europe", "TTF=F", "欧洲天然气TTF期货"),
    ("silver", "SI=F", "白银期货"),
    ("msci_world_proxy", "URTH", "MSCI World 代理ETF"),
    ("msci_em_proxy", "EEM", "MSCI EM 代理ETF"),
    ("chinext_index", "399006.SZ", "创业板指"),
    ("nikkei225", "^N225", "日经225"),
    ("topix", "^TOPX", "TOPIX"),
]


def fetch_one(name: str, ticker: str, label: str):
    try:
        df = yf.Ticker(ticker).history(period='max', auto_adjust=False)
        if df.empty:
            return {
                'asset': name,
                'ticker': ticker,
                'label': label,
                'status': 'empty',
            }

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.reset_index().rename(columns={'Date': 'date'})
        # 统一列名
        rename_map = {
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'
        }
        df = df.rename(columns=rename_map)
        keep_cols = [c for c in ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume'] if c in df.columns]
        df = df[keep_cols]
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        out_file = OUT / f'{name}.csv'
        df.to_csv(out_file, index=False)

        return {
            'asset': name,
            'ticker': ticker,
            'label': label,
            'status': 'ok',
            'rows': int(len(df)),
            'start': str(df['date'].iloc[0]),
            'end': str(df['date'].iloc[-1]),
            'file': str(out_file),
        }
    except Exception as e:
        return {
            'asset': name,
            'ticker': ticker,
            'label': label,
            'status': 'error',
            'error': str(e),
        }


def main():
    results = []
    for name, ticker, label in ASSETS:
        r = fetch_one(name, ticker, label)
        results.append(r)
        print(name, ticker, r.get('status'), r.get('start', ''), r.get('end', ''), r.get('rows', ''))

    summary_file = OUT / 'download_summary_extra_assets_1990.json'
    summary_file.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print('saved', summary_file)


if __name__ == '__main__':
    main()
