import os
import math
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from numba import njit

BASE = Path('/root/project/20260428-10gold')
OUT_DIR = BASE / 'backtest_results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BUY_MIN, BUY_MAX = 2, 700
SELL_MIN, SELL_MAX = 2, 700

ASSETS = {
    # core
    'brent': BASE / 'data' / 'brent.csv',
    'btc': BASE / 'data' / 'btc.csv',
    'copper_core': BASE / 'data' / 'copper.csv',
    'eurusd': BASE / 'data' / 'eurusd.csv',
    'natgas_core': BASE / 'data' / 'natgas.csv',
    'sp500': BASE / 'data' / 'sp500.csv',
    'us10y_yield': BASE / 'data' / 'us10y_yield.csv',
    'usd_dxy': BASE / 'data' / 'usd_dxy.csv',
    'usdjpy': BASE / 'data' / 'usdjpy.csv',

    # extra
    'gold_spot_proxy': BASE / 'data_long_history' / 'gold_daily_proxy_gld_yahoo.csv',
    'chinext_proxy_etf_159915': BASE / 'data_extra_1990' / 'chinext_proxy_etf_159915.csv',
    'copper_extra_hg': BASE / 'data_extra_1990' / 'copper.csv',
    'eth': BASE / 'data_extra_1990' / 'eth.csv',
    'msci_em_proxy': BASE / 'data_extra_1990' / 'msci_em_proxy.csv',
    'msci_world_proxy': BASE / 'data_extra_1990' / 'msci_world_proxy.csv',
    'nasdaq100': BASE / 'data_extra_1990' / 'nasdaq100.csv',
    'natgas_henry_hub': BASE / 'data_extra_1990' / 'natgas_henry_hub.csv',
    'nikkei225': BASE / 'data_extra_1990' / 'nikkei225.csv',
    'silver': BASE / 'data_extra_1990' / 'silver.csv',
    'topix_proxy_etf': BASE / 'data_extra_1990' / 'topix_proxy_etf.csv',
    'ttf_gas_europe': BASE / 'data_extra_1990' / 'ttf_gas_europe.csv',
}

G_CLOSE = None
G_PREV_CLOSE = None
G_MA = None


def rolling_mean_np(arr: np.ndarray, window: int) -> np.ndarray:
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 1:
        out[:] = arr
        return out
    if window > n:
        return out
    c = np.cumsum(arr, dtype=np.float64)
    out[window - 1] = c[window - 1] / window
    if n > window:
        out[window:] = (c[window:] - c[:-window]) / window
    return out


@njit(cache=True)
def sim(close, prev_close, buy_signal, sell_signal):
    position = 0
    entry = 0.0
    equity = 1.0
    trades = 0
    wins = 0

    for i in range(close.shape[0]):
        c = close[i]
        pc = prev_close[i]
        if position == 1 and not math.isnan(pc):
            equity *= c / pc

        if position == 0 and buy_signal[i] == 1:
            position = 1
            entry = c
        elif position == 1 and sell_signal[i] == 1:
            ret = c / entry - 1.0
            trades += 1
            if ret > 0.0:
                wins += 1
            position = 0

    if position == 1:
        ret = close[-1] / entry - 1.0
        trades += 1
        if ret > 0.0:
            wins += 1

    return equity, trades, wins


def init_worker(close, prev_close, ma_matrix):
    global G_CLOSE, G_PREV_CLOSE, G_MA
    G_CLOSE = close
    G_PREV_CLOSE = prev_close
    G_MA = ma_matrix


def worker_one_buy_ma(buy_ma: int):
    close = G_CLOSE
    prev_close = G_PREV_CLOSE
    ma = G_MA

    out = []
    ma_buy = ma[buy_ma]
    prev_ma_buy = np.roll(ma_buy, 1)
    prev_ma_buy[0] = np.nan
    buy_signal = (prev_close <= prev_ma_buy) & (close > ma_buy)
    buy_signal = np.nan_to_num(buy_signal, nan=False).astype(np.int8)

    for sell_ma in range(SELL_MIN, SELL_MAX + 1):
        ma_sell = ma[sell_ma]
        prev_ma_sell = np.roll(ma_sell, 1)
        prev_ma_sell[0] = np.nan
        sell_signal = (prev_close >= prev_ma_sell) & (close < ma_sell)
        sell_signal = np.nan_to_num(sell_signal, nan=False).astype(np.int8)

        eq, trades, wins = sim(close, prev_close, buy_signal, sell_signal)
        out.append((buy_ma, sell_ma, (eq - 1.0) * 100.0, int(trades), (wins / trades * 100.0) if trades else 0.0))

    return out


def load_close_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    colmap = {c.lower().strip(): c for c in df.columns}
    date_col = colmap.get('date') or colmap.get('timestamp') or 'Date'
    close_col = colmap.get('close') or colmap.get('adj close') or colmap.get('adj_close') or 'Close'

    out = pd.DataFrame({
        'date': pd.to_datetime(df[date_col], errors='coerce'),
        'close': pd.to_numeric(df[close_col], errors='coerce'),
    }).dropna()
    return out.sort_values('date').reset_index(drop=True)


def run_asset(asset: str, path: Path):
    raw = load_close_series(path)
    close = raw['close'].to_numpy(dtype=np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan

    ma_matrix = np.empty((SELL_MAX + 1, close.shape[0]), dtype=np.float64)
    ma_matrix[:] = np.nan
    for w in range(BUY_MIN, SELL_MAX + 1):
        ma_matrix[w] = rolling_mean_np(close, w)

    _ = sim(close[:50], prev_close[:50], np.zeros(50, dtype=np.int8), np.zeros(50, dtype=np.int8))

    buy_list = list(range(BUY_MIN, BUY_MAX + 1))
    workers = min(os.cpu_count() or 1, len(buy_list))

    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(close, prev_close, ma_matrix)) as ex:
        futs = [ex.submit(worker_one_buy_ma, b) for b in buy_list]
        done = 0
        for f in as_completed(futs):
            rows.extend(f.result())
            done += 1
            if done % 100 == 0 or done == len(buy_list):
                print(f'  {asset}: {done}/{len(buy_list)} buy_ma done')

    res = pd.DataFrame(rows, columns=['buy_ma', 'sell_ma', 'total_return_pct', 'trades', 'win_rate_pct'])
    res = res.sort_values('total_return_pct', ascending=False).reset_index(drop=True)

    out_all = OUT_DIR / f'{asset}_ma_gridsearch_2_700_vs_2_700.csv'
    res.to_csv(out_all, index=False)

    best = res.iloc[0]
    bh = (close[-1] / close[0] - 1.0) * 100.0
    return {
        'asset': asset,
        'start_date': raw['date'].iloc[0].strftime('%Y-%m-%d'),
        'end_date': raw['date'].iloc[-1].strftime('%Y-%m-%d'),
        'bars': int(len(raw)),
        'best_buy_ma': int(best['buy_ma']),
        'best_sell_ma': int(best['sell_ma']),
        'best_return_pct': float(best['total_return_pct']),
        'best_trades': int(best['trades']),
        'best_win_rate_pct': float(best['win_rate_pct']),
        'buy_hold_return_pct': float(bh),
        'excess_vs_buy_hold_pct': float(best['total_return_pct'] - bh),
        'detail_file': str(out_all),
    }


def main():
    t0 = time.time()
    summaries = []

    for i, (asset, path) in enumerate(ASSETS.items(), start=1):
        if not path.exists():
            print(f'[{i}/{len(ASSETS)}] SKIP {asset}: file not found {path}')
            continue
        print(f'[{i}/{len(ASSETS)}] running {asset} ...')
        summaries.append(run_asset(asset, path))

    summary_df = pd.DataFrame(summaries).sort_values('best_return_pct', ascending=False).reset_index(drop=True)
    summary_path = OUT_DIR / 'all_assets_best_ma_2_700_vs_2_700.csv'
    summary_df.to_csv(summary_path, index=False)

    print('\nDONE')
    print(summary_df[['asset','best_buy_ma','best_sell_ma','best_return_pct','buy_hold_return_pct','excess_vs_buy_hold_pct']].to_string(index=False))
    print('saved', summary_path)
    print('elapsed_sec', round(time.time() - t0, 2))


if __name__ == '__main__':
    main()
