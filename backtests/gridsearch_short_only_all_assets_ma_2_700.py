import math
import os
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
    'brent': BASE / 'data' / 'brent.csv',
    'btc': BASE / 'data' / 'btc.csv',
    'copper_core': BASE / 'data' / 'copper.csv',
    'eurusd': BASE / 'data' / 'eurusd.csv',
    'natgas_core': BASE / 'data' / 'natgas.csv',
    'sp500': BASE / 'data' / 'sp500.csv',
    'us10y_yield': BASE / 'data' / 'us10y_yield.csv',
    'usd_dxy': BASE / 'data' / 'usd_dxy.csv',
    'usdjpy': BASE / 'data' / 'usdjpy.csv',
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
    if window > n:
        return out
    c = np.cumsum(arr, dtype=np.float64)
    out[window - 1] = c[window - 1] / window
    if n > window:
        out[window:] = (c[window:] - c[:-window]) / window
    return out


@njit(cache=True)
def sim_short_only(close, prev_close, short_entry_signal, short_cover_signal):
    # position: 0 = flat, -1 = short
    position = 0
    entry = 0.0
    equity = 1.0
    trades = 0
    wins = 0

    for i in range(close.shape[0]):
        c = close[i]
        pc = prev_close[i]

        if position == -1 and not math.isnan(pc):
            equity *= pc / c

        if position == 0 and short_entry_signal[i] == 1:
            position = -1
            entry = c
        elif position == -1 and short_cover_signal[i] == 1:
            ret = entry / c - 1.0
            trades += 1
            if ret > 0.0:
                wins += 1
            position = 0

    if position == -1:
        ret = entry / close[-1] - 1.0
        trades += 1
        if ret > 0.0:
            wins += 1

    return equity, trades, wins


def init_worker(close, prev_close, ma_matrix):
    global G_CLOSE, G_PREV_CLOSE, G_MA
    G_CLOSE = close
    G_PREV_CLOSE = prev_close
    G_MA = ma_matrix


def worker_one_entry_ma(entry_ma: int):
    close = G_CLOSE
    prev_close = G_PREV_CLOSE
    ma = G_MA

    out = []
    ma_entry = ma[entry_ma]
    prev_ma_entry = np.roll(ma_entry, 1)
    prev_ma_entry[0] = np.nan

    # short entry: price down-crosses MA
    short_entry_signal = (prev_close >= prev_ma_entry) & (close < ma_entry)
    short_entry_signal = np.nan_to_num(short_entry_signal, nan=False).astype(np.int8)

    for cover_ma in range(SELL_MIN, SELL_MAX + 1):
        ma_cover = ma[cover_ma]
        prev_ma_cover = np.roll(ma_cover, 1)
        prev_ma_cover[0] = np.nan

        # short cover: price up-crosses MA
        short_cover_signal = (prev_close <= prev_ma_cover) & (close > ma_cover)
        short_cover_signal = np.nan_to_num(short_cover_signal, nan=False).astype(np.int8)

        eq, trades, wins = sim_short_only(close, prev_close, short_entry_signal, short_cover_signal)
        out.append((entry_ma, cover_ma, (eq - 1.0) * 100.0, int(trades), (wins / trades * 100.0) if trades else 0.0))

    return out


def load_close_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    colmap = {c.lower().strip(): c for c in df.columns}
    date_col = colmap.get('date') or colmap.get('timestamp') or 'Date'
    close_col = colmap.get('close') or colmap.get('adj close') or colmap.get('adj_close') or 'Close'

    out = pd.DataFrame({
        'date': pd.to_datetime(df[date_col], errors='coerce'),
        'close': pd.to_numeric(df[close_col], errors='coerce')
    }).dropna().sort_values('date').reset_index(drop=True)

    return out


def gridsearch_short_only(asset: str, csv_path: Path):
    t0 = time.time()
    px = load_close_series(csv_path)
    close = px['close'].to_numpy(dtype=np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan

    ma = {}
    for w in range(BUY_MIN, BUY_MAX + 1):
        ma[w] = rolling_mean_np(close, w)

    workers = max(1, (os.cpu_count() or 2) - 1)
    rows = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(close, prev_close, ma)
    ) as ex:
        futs = [ex.submit(worker_one_entry_ma, e) for e in range(BUY_MIN, BUY_MAX + 1)]
        done = 0
        total = len(futs)
        for f in as_completed(futs):
            rows.extend(f.result())
            done += 1
            if done % 50 == 0 or done == total:
                print(f"[{asset}] progress {done}/{total}", flush=True)

    res = pd.DataFrame(rows, columns=['entry_ma', 'cover_ma', 'short_return_pct', 'trades', 'win_rate_pct'])

    start_date = px['date'].iloc[0].strftime('%Y-%m-%d')
    end_date = px['date'].iloc[-1].strftime('%Y-%m-%d')

    # for reference: buy&hold short (pure short from first to last): first/last - 1
    short_hold_return = (close[0] / close[-1] - 1.0) * 100.0

    best = res.sort_values('short_return_pct', ascending=False).iloc[0]

    detail_file = OUT_DIR / f"{asset}_short_only_ma_gridsearch_2_700_vs_2_700.csv"
    res.to_csv(detail_file, index=False)

    summary = {
        'asset': asset,
        'start_date': start_date,
        'end_date': end_date,
        'bars': len(px),
        'best_entry_ma': int(best['entry_ma']),
        'best_cover_ma': int(best['cover_ma']),
        'best_short_return_pct': float(best['short_return_pct']),
        'best_trades': int(best['trades']),
        'best_win_rate_pct': float(best['win_rate_pct']),
        'short_hold_return_pct': float(short_hold_return),
        'excess_vs_short_hold_pct': float(best['short_return_pct'] - short_hold_return),
        'elapsed_sec': round(time.time() - t0, 2),
        'detail_file': str(detail_file)
    }

    summary_file = OUT_DIR / f"{asset}_best_short_only_ma_2_700_vs_2_700_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_file, index=False)
    return summary


def main():
    import os
    all_rows = []
    total_start = time.time()
    for i, (asset, path) in enumerate(ASSETS.items(), start=1):
        print(f"\n=== [{i}/{len(ASSETS)}] {asset} ===", flush=True)
        if not path.exists():
            print(f"skip missing file: {path}", flush=True)
            continue
        try:
            s = gridsearch_short_only(asset, path)
            all_rows.append(s)
            print(
                f"done {asset}: entry/cover=({s['best_entry_ma']},{s['best_cover_ma']}), "
                f"short_ret={s['best_short_return_pct']:.6f}%",
                flush=True
            )
        except Exception as e:
            print(f"FAILED {asset}: {e}", flush=True)

    if all_rows:
        out = pd.DataFrame(all_rows).sort_values('best_short_return_pct', ascending=False)
        out_file = OUT_DIR / 'all_assets_best_short_only_ma_2_700_vs_2_700_summary.csv'
        out.to_csv(out_file, index=False)
        print(f"\nSaved summary: {out_file}")

    print(f"Total elapsed: {time.time()-total_start:.2f}s")


if __name__ == '__main__':
    main()
