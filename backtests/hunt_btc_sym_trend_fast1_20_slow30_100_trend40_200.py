import time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

BASE = Path('/root/project/20260428-10gold')
OUT = BASE / 'backtest_results'
OUT.mkdir(parents=True, exist_ok=True)
BTC_FILE = BASE / 'data' / 'btc.csv'

# benchmark to beat
TARGET = 377173.186194

# realistic 6 variables (same as prior runs)
FEE_BPS = 5.0
SLIPPAGE_BPS = 8.0
SPREAD_BPS = 4.0
SHORT_COST_BPS_DAY = 2.0
ONE_WAY = (FEE_BPS + SLIPPAGE_BPS + SPREAD_BPS * 0.5) / 10000.0
SHORT_DAY = SHORT_COST_BPS_DAY / 10000.0

FAST_RANGE = range(1, 21)
SLOW_RANGE = range(30, 101)
TREND_RANGE = range(40, 201)


def load_close(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cm = {c.lower().strip(): c for c in df.columns}
    dcol = cm.get('date') or cm.get('timestamp') or 'Date'
    ccol = cm.get('close') or cm.get('adj close') or cm.get('adj_close') or 'Close'
    out = pd.DataFrame({
        'date': pd.to_datetime(df[dcol], errors='coerce'),
        'close': pd.to_numeric(df[ccol], errors='coerce')
    }).dropna()
    return out.sort_values('date').reset_index(drop=True)


def rolling_mean_np(a: np.ndarray, w: int) -> np.ndarray:
    n = a.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if w > n:
        return out
    c = np.cumsum(a, dtype=np.float64)
    out[w - 1] = c[w - 1] / w
    if n > w:
        out[w:] = (c[w:] - c[:-w]) / w
    return out


@njit(cache=True)
def eval_combo(close, mf, ms, mt, one_way, short_day):
    n = close.shape[0]
    pos = 0
    pending = 0
    trades = 0
    flat = 0
    eq = 1.0
    prev = close[0]

    for i in range(1, n):
        cur = close[i]

        # mark to market
        if pos == 1:
            eq *= (cur / prev)
        elif pos == -1:
            eq *= (prev / cur)
            eq *= (1.0 - short_day)
        else:
            flat += 1

        # execute pending at today's close (T+1)
        if pending != pos:
            if pos != 0:
                eq *= (1.0 - one_way)
                trades += 1
            pos = pending
            if pos != 0:
                eq *= (1.0 - one_way)

        # today's signal decides next pending
        f0 = mf[i - 1]
        f1 = mf[i]
        s0 = ms[i - 1]
        s1 = ms[i]
        t1 = mt[i]

        if not (np.isnan(f0) or np.isnan(f1) or np.isnan(s0) or np.isnan(s1) or np.isnan(t1)):
            if f0 <= s0 and f1 > s1 and cur > t1:
                pending = 1
            elif f0 >= s0 and f1 < s1 and cur < t1:
                pending = -1

        prev = cur

    if pos != 0:
        eq *= (1.0 - one_way)
        trades += 1

    ret = (eq - 1.0) * 100.0
    flat_ratio = flat / (n - 1) * 100.0
    return ret, trades, flat_ratio


def main():
    t0 = time.time()
    df = load_close(BTC_FILE)
    close = df['close'].to_numpy(np.float64)

    # Precompute MA windows once
    max_w = max(max(FAST_RANGE), max(SLOW_RANGE), max(TREND_RANGE))
    ma = {}
    for w in range(1, max_w + 1):
        ma[w] = rolling_mean_np(close, w)

    # warmup numba
    _ = eval_combo(close, ma[1], ma[30], ma[40], ONE_WAY, SHORT_DAY)

    total = 0
    for f in FAST_RANGE:
        for s in SLOW_RANGE:
            if f < s:
                total += len(TREND_RANGE)

    rows = []
    done = 0
    checkpoint_every = 20000

    for f in FAST_RANGE:
        mf = ma[f]
        for s in SLOW_RANGE:
            if f >= s:
                continue
            ms = ma[s]
            for t in TREND_RANGE:
                mt = ma[t]
                ret, trades, flat = eval_combo(close, mf, ms, mt, ONE_WAY, SHORT_DAY)
                rows.append((f, s, t, ret, trades, flat))
                done += 1
                if done % checkpoint_every == 0:
                    elapsed = time.time() - t0
                    print(f"progress {done}/{total} ({done/total:.2%}) elapsed={elapsed:.1f}s")

    out = pd.DataFrame(rows, columns=['fast_ma', 'slow_ma', 'trend_ma', 'return_pct', 'trades', 'flat_ratio_pct'])
    out = out.sort_values('return_pct', ascending=False).reset_index(drop=True)

    detail_file = OUT / 'btc_sym_trend_fast1_20_slow30_100_trend40_200_all.csv'
    out.to_csv(detail_file, index=False)

    best = out.iloc[0]
    beat = out[out['return_pct'] > TARGET]

    summary = pd.DataFrame([{
        'tested_total': int(len(out)),
        'best_fast': int(best['fast_ma']),
        'best_slow': int(best['slow_ma']),
        'best_trend': int(best['trend_ma']),
        'best_return_pct': float(best['return_pct']),
        'best_trades': int(best['trades']),
        'best_flat_ratio_pct': float(best['flat_ratio_pct']),
        'target_to_beat_pct': TARGET,
        'num_beating_target': int(len(beat)),
        'elapsed_sec': round(time.time() - t0, 2),
        'detail_file': str(detail_file),
    }])
    summary_file = OUT / 'btc_sym_trend_fast1_20_slow30_100_trend40_200_summary.csv'
    summary.to_csv(summary_file, index=False)

    print('\nSUMMARY:')
    print(summary.to_string(index=False))
    print('\nTOP 20:')
    print(out.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
