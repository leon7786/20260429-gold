import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from numba import njit

BASE = Path('/root/project/20260428-10gold')
OUT = BASE / 'backtest_results'
OUT.mkdir(parents=True, exist_ok=True)

# 大宗+加密（双边）
ASSETS = {
    'brent': BASE / 'data' / 'brent.csv',
    'copper_core': BASE / 'data' / 'copper.csv',
    'natgas_core': BASE / 'data' / 'natgas.csv',
    'gold_spot_proxy': BASE / 'data_long_history' / 'gold_daily_proxy_gld_yahoo.csv',
    'copper_extra_hg': BASE / 'data_extra_1990' / 'copper.csv',
    'natgas_henry_hub': BASE / 'data_extra_1990' / 'natgas_henry_hub.csv',
    'silver': BASE / 'data_extra_1990' / 'silver.csv',
    'ttf_gas_europe': BASE / 'data_extra_1990' / 'ttf_gas_europe.csv',
    'btc': BASE / 'data' / 'btc.csv',
    'eth': BASE / 'data_extra_1990' / 'eth.csv',
}

MA_MIN, MA_MAX = 1, 700
G_CLOSE = None
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
def sim_dual_side_t1(close, ma_fast, ma_slow):
    """
    T+1 执行：
    - t 日收盘产生信号
    - t+1 日收盘执行持仓切换
    """
    n = close.shape[0]
    pos = 0
    equity = 1.0
    prev = close[0]

    long_trades = 0
    short_trades = 0
    long_wins = 0
    short_wins = 0

    entry_px = 0.0
    entry_side = 0
    flat_bars = 0

    pending_target = 0  # 将在下一根K执行的目标仓位

    for i in range(1, n):
        c = close[i]

        # 1) 先按旧仓位记当日收益
        if pos == 1:
            equity *= c / prev
        elif pos == -1:
            equity *= prev / c
        else:
            flat_bars += 1

        # 2) 执行“昨日收盘形成、今日执行”的挂单
        if pending_target != pos:
            # 先平旧仓
            if pos == 1 and entry_side == 1:
                r = c / entry_px - 1.0
                long_trades += 1
                if r > 0:
                    long_wins += 1
            elif pos == -1 and entry_side == -1:
                r = entry_px / c - 1.0
                short_trades += 1
                if r > 0:
                    short_wins += 1

            # 再开新仓
            pos = pending_target
            if pos == 1:
                entry_side = 1
                entry_px = c
            elif pos == -1:
                entry_side = -1
                entry_px = c
            else:
                entry_side = 0
                entry_px = 0.0

        # 3) 读取今日信号，决定“明日执行”的目标仓位
        f0 = ma_fast[i - 1]
        s0 = ma_slow[i - 1]
        f1 = ma_fast[i]
        s1 = ma_slow[i]

        if not (np.isnan(f0) or np.isnan(s0) or np.isnan(f1) or np.isnan(s1)):
            cross_up = (f0 <= s0) and (f1 > s1)
            cross_dn = (f0 >= s0) and (f1 < s1)

            # 双边：下一日目标仓位
            if cross_up:
                pending_target = 1
            elif cross_dn:
                pending_target = -1

        prev = c

    # 最后强平
    if pos == 1 and entry_side == 1:
        r = close[-1] / entry_px - 1.0
        long_trades += 1
        if r > 0:
            long_wins += 1
    elif pos == -1 and entry_side == -1:
        r = entry_px / close[-1] - 1.0
        short_trades += 1
        if r > 0:
            short_wins += 1

    total_trades = long_trades + short_trades
    flat_ratio = flat_bars / (n - 1) * 100.0 if n > 1 else 0.0
    ret_pct = (equity - 1.0) * 100.0
    long_win = (long_wins / long_trades * 100.0) if long_trades > 0 else 0.0
    short_win = (short_wins / short_trades * 100.0) if short_trades > 0 else 0.0
    return ret_pct, total_trades, flat_ratio, long_trades, short_trades, long_win, short_win


def init_worker(close, ma_dict):
    global G_CLOSE, G_MA
    G_CLOSE = close
    G_MA = ma_dict


def worker_one_fast(fast_ma: int):
    close = G_CLOSE
    ma = G_MA
    out = []
    m_fast = ma[fast_ma]
    for slow_ma in range(MA_MIN, MA_MAX + 1):
        if slow_ma == fast_ma:
            continue
        m_slow = ma[slow_ma]
        ret, trades, flat_ratio, ltr, strd, lwr, swr = sim_dual_side_t1(close, m_fast, m_slow)
        out.append((fast_ma, slow_ma, ret, trades, flat_ratio, ret, ltr, strd, lwr, swr))
    return out


def load_close(path: Path):
    df = pd.read_csv(path)
    colmap = {c.lower().strip(): c for c in df.columns}
    dcol = colmap.get('date') or colmap.get('timestamp') or 'Date'
    ccol = colmap.get('close') or colmap.get('adj close') or colmap.get('adj_close') or 'Close'
    out = pd.DataFrame({
        'date': pd.to_datetime(df[dcol], errors='coerce'),
        'close': pd.to_numeric(df[ccol], errors='coerce')
    }).dropna().sort_values('date').reset_index(drop=True)
    return out


def run_one(asset, path):
    t0 = time.time()
    px = load_close(path)
    close = px['close'].to_numpy(dtype=np.float64)

    ma = {w: rolling_mean_np(close, w) for w in range(MA_MIN, MA_MAX + 1)}

    workers = max(1, (os.cpu_count() or 2) - 1)
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(close, ma)) as ex:
        futs = [ex.submit(worker_one_fast, f) for f in range(MA_MIN, MA_MAX + 1)]
        done = 0
        total = len(futs)
        for f in as_completed(futs):
            rows.extend(f.result())
            done += 1
            if done % 50 == 0 or done == total:
                print(f"[{asset}] progress {done}/{total}", flush=True)

    res = pd.DataFrame(rows, columns=[
        'fast_ma', 'slow_ma', 'dual_return_pct', 'trades', 'flat_ratio_pct', 'score',
        'long_trades', 'short_trades', 'long_win_rate_pct', 'short_win_rate_pct'
    ])

    best = res.sort_values('score', ascending=False).iloc[0]
    detail_file = OUT / f"{asset}_dual_side_ma_1_700_t1_grid.csv"
    res.to_csv(detail_file, index=False)

    summary = {
        'asset': asset,
        'start_date': px['date'].iloc[0].strftime('%Y-%m-%d'),
        'end_date': px['date'].iloc[-1].strftime('%Y-%m-%d'),
        'bars': len(px),
        'best_fast_ma': int(best['fast_ma']),
        'best_slow_ma': int(best['slow_ma']),
        'best_dual_return_pct': float(best['dual_return_pct']),
        'best_flat_ratio_pct': float(best['flat_ratio_pct']),
        'best_trades': int(best['trades']),
        'best_long_trades': int(best['long_trades']),
        'best_short_trades': int(best['short_trades']),
        'best_long_win_rate_pct': float(best['long_win_rate_pct']),
        'best_short_win_rate_pct': float(best['short_win_rate_pct']),
        'execution_model': 'T+1(close signal -> next close execute)',
        'elapsed_sec': round(time.time() - t0, 2),
        'detail_file': str(detail_file),
    }

    pd.DataFrame([summary]).to_csv(OUT / f"{asset}_best_dual_side_ma_1_700_t1_summary.csv", index=False)
    return summary


def main():
    all_rows = []
    t0 = time.time()
    for i, (asset, path) in enumerate(ASSETS.items(), start=1):
        if not path.exists():
            print(f"skip {asset}: missing {path}", flush=True)
            continue
        print(f"\n=== [{i}/{len(ASSETS)}] {asset} ===", flush=True)
        s = run_one(asset, path)
        all_rows.append(s)
        print(
            f"done {asset}: fast/slow=({s['best_fast_ma']},{s['best_slow_ma']}), "
            f"ret={s['best_dual_return_pct']:.4f}%, flat={s['best_flat_ratio_pct']:.2f}%",
            flush=True
        )

    if all_rows:
        out = pd.DataFrame(all_rows).sort_values('best_dual_return_pct', ascending=False)
        out_file = OUT / 'commodities_plus_crypto_best_dual_side_ma_1_700_t1_summary.csv'
        out.to_csv(out_file, index=False)
        print(f"saved {out_file}")
    print(f"elapsed_sec {time.time()-t0:.2f}")


if __name__ == '__main__':
    main()
