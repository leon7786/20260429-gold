import time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

BASE = Path('/root/project/20260428-10gold')
OUT = BASE / 'backtest_results'
OUT.mkdir(parents=True, exist_ok=True)

BTC_FILE = BASE / 'data' / 'btc.csv'

# realistic assumptions
FEE_BPS = 5.0
SLIPPAGE_BPS = 8.0
SPREAD_BPS = 4.0
SHORT_COST_BPS_DAY = 2.0
MAX_GROSS = 1.0

ONE_WAY_TRADE_COST = (FEE_BPS + SLIPPAGE_BPS + SPREAD_BPS * 0.5) / 10000.0
SHORT_DAILY_COST = SHORT_COST_BPS_DAY / 10000.0

FAST_LIST = [1, 2, 3, 5, 8, 10, 13, 16, 20]
MID_LIST = [15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]
SLOW_LIST = [80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 300]


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
def sim_long_only_t1(close, mf, mm, ms, one_way_cost, max_gross):
    n = close.shape[0]
    pos = 0
    pending = 0
    prev = close[0]
    eq = 1.0
    flat = 0
    trades = 0
    wins = 0
    entry = 0.0

    for i in range(1, n):
        c = close[i]

        if pos == 1:
            eq *= 1.0 + max_gross * (c / prev - 1.0)
        else:
            flat += 1

        # execute pending
        if pending != pos:
            if pos != 0:
                eq *= (1.0 - one_way_cost * max_gross)  # close
                r = c / entry - 1.0
                trades += 1
                if r > 0:
                    wins += 1
            pos = pending
            if pos != 0:
                eq *= (1.0 - one_way_cost * max_gross)  # open
                entry = c

        # signal at t, execute at t+1
        f0, f1 = mf[i - 1], mf[i]
        m0, m1 = mm[i - 1], mm[i]
        s1 = ms[i]
        if not (np.isnan(f0) or np.isnan(f1) or np.isnan(m0) or np.isnan(m1) or np.isnan(s1)):
            trend_up = m1 > s1
            cross_up = (f0 <= m0) and (f1 > m1)
            cross_dn = (f0 >= m0) and (f1 < m1)
            if pos == 0:
                if trend_up and cross_up:
                    pending = 1
            else:
                if cross_dn or not trend_up:
                    pending = 0

        prev = c

    if pos == 1:
        eq *= (1.0 - one_way_cost * max_gross)
        r = close[-1] / entry - 1.0
        trades += 1
        if r > 0:
            wins += 1

    ret = (eq - 1.0) * 100.0
    flat_ratio = flat / (n - 1) * 100.0 if n > 1 else 0.0
    win = (wins / trades * 100.0) if trades > 0 else 0.0
    return ret, trades, flat_ratio, win


@njit(cache=True)
def sim_short_only_t1(close, mf, mm, ms, one_way_cost, short_daily_cost, max_gross):
    n = close.shape[0]
    pos = 0
    pending = 0
    prev = close[0]
    eq = 1.0
    flat = 0
    trades = 0
    wins = 0
    entry = 0.0

    for i in range(1, n):
        c = close[i]

        if pos == -1:
            eq *= 1.0 + max_gross * (prev / c - 1.0)
            eq *= (1.0 - short_daily_cost * max_gross)
        else:
            flat += 1

        # execute pending
        if pending != pos:
            if pos != 0:
                eq *= (1.0 - one_way_cost * max_gross)  # close
                r = entry / c - 1.0
                trades += 1
                if r > 0:
                    wins += 1
            pos = pending
            if pos != 0:
                eq *= (1.0 - one_way_cost * max_gross)  # open
                entry = c

        # signal at t, execute at t+1
        f0, f1 = mf[i - 1], mf[i]
        m0, m1 = mm[i - 1], mm[i]
        s1 = ms[i]
        if not (np.isnan(f0) or np.isnan(f1) or np.isnan(m0) or np.isnan(m1) or np.isnan(s1)):
            trend_dn = m1 < s1
            cross_dn = (f0 >= m0) and (f1 < m1)
            cross_up = (f0 <= m0) and (f1 > m1)
            if pos == 0:
                if trend_dn and cross_dn:
                    pending = -1
            else:
                if cross_up or not trend_dn:
                    pending = 0

        prev = c

    if pos == -1:
        eq *= (1.0 - one_way_cost * max_gross)
        r = entry / close[-1] - 1.0
        trades += 1
        if r > 0:
            wins += 1

    ret = (eq - 1.0) * 100.0
    flat_ratio = flat / (n - 1) * 100.0 if n > 1 else 0.0
    win = (wins / trades * 100.0) if trades > 0 else 0.0
    return ret, trades, flat_ratio, win


def main():
    t0 = time.time()
    px = load_close(BTC_FILE)
    close = px['close'].to_numpy(dtype=np.float64)

    max_w = max(SLOW_LIST)
    ma = {w: rolling_mean_np(close, w) for w in range(1, max_w + 1)}

    rows_long = []
    rows_short = []

    combos = []
    for f in FAST_LIST:
        for m in MID_LIST:
            for s in SLOW_LIST:
                if f < m < s:
                    combos.append((f, m, s))

    total = len(combos)
    for i, (f, m, s) in enumerate(combos, start=1):
        mf, mm, ms = ma[f], ma[m], ma[s]
        lr, lt, lflat, lwin = sim_long_only_t1(close, mf, mm, ms, ONE_WAY_TRADE_COST, MAX_GROSS)
        sr, st, sflat, swin = sim_short_only_t1(close, mf, mm, ms, ONE_WAY_TRADE_COST, SHORT_DAILY_COST, MAX_GROSS)
        rows_long.append((f, m, s, lr, lt, lflat, lwin))
        rows_short.append((f, m, s, sr, st, sflat, swin))
        if i % 100 == 0 or i == total:
            print(f"[btc-multi-ma] progress {i}/{total}", flush=True)

    long_df = pd.DataFrame(rows_long, columns=['fast_ma', 'mid_ma', 'slow_ma', 'return_pct', 'trades', 'flat_ratio_pct', 'win_rate_pct'])
    short_df = pd.DataFrame(rows_short, columns=['fast_ma', 'mid_ma', 'slow_ma', 'return_pct', 'trades', 'flat_ratio_pct', 'win_rate_pct'])

    long_detail = OUT / 'btc_multi_ma_long_only_t1_realistic_grid.csv'
    short_detail = OUT / 'btc_multi_ma_short_only_t1_realistic_grid.csv'
    long_df.to_csv(long_detail, index=False)
    short_df.to_csv(short_detail, index=False)

    best_long = long_df.sort_values('return_pct', ascending=False).iloc[0]
    best_short = short_df.sort_values('return_pct', ascending=False).iloc[0]

    summary = pd.DataFrame([{
        'asset': 'btc',
        'start_date': px['date'].iloc[0].strftime('%Y-%m-%d'),
        'end_date': px['date'].iloc[-1].strftime('%Y-%m-%d'),
        'bars': len(px),
        'combos_tested': total,
        'best_long_fast_ma': int(best_long['fast_ma']),
        'best_long_mid_ma': int(best_long['mid_ma']),
        'best_long_slow_ma': int(best_long['slow_ma']),
        'best_long_return_pct': float(best_long['return_pct']),
        'best_long_trades': int(best_long['trades']),
        'best_long_flat_ratio_pct': float(best_long['flat_ratio_pct']),
        'best_long_win_rate_pct': float(best_long['win_rate_pct']),
        'best_short_fast_ma': int(best_short['fast_ma']),
        'best_short_mid_ma': int(best_short['mid_ma']),
        'best_short_slow_ma': int(best_short['slow_ma']),
        'best_short_return_pct': float(best_short['return_pct']),
        'best_short_trades': int(best_short['trades']),
        'best_short_flat_ratio_pct': float(best_short['flat_ratio_pct']),
        'best_short_win_rate_pct': float(best_short['win_rate_pct']),
        'execution_model': 'T+1(close signal -> next close execute)',
        'fee_bps': FEE_BPS,
        'slippage_bps': SLIPPAGE_BPS,
        'spread_bps': SPREAD_BPS,
        'short_cost_bps_day': SHORT_COST_BPS_DAY,
        'max_gross': MAX_GROSS,
        'elapsed_sec': round(time.time() - t0, 2),
        'long_detail_file': str(long_detail),
        'short_detail_file': str(short_detail)
    }])

    summary_file = OUT / 'btc_multi_ma_long_short_t1_realistic_summary.csv'
    summary.to_csv(summary_file, index=False)
    print(f"saved {summary_file}")
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
