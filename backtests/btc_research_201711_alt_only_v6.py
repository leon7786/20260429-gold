import time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

BASE = Path('/root/project/20260428-10gold')
BTC_FILE = BASE / 'data' / 'btc.csv'
OUT_DIR = BASE / 'backtest_results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# User constraints
START_DATE = '2017-11-01'
INITIAL_CAPITAL = 1000.0

# Keep realistic 6-variable execution assumptions
FEE_BPS = 5.0
SLIPPAGE_BPS = 8.0
SPREAD_BPS = 4.0
SHORT_COST_BPS_DAY = 2.0
ONE_WAY = (FEE_BPS + SLIPPAGE_BPS + SPREAD_BPS * 0.5) / 10000.0
SHORT_DAY = SHORT_COST_BPS_DAY / 10000.0

FAST_RANGE = range(1, 31)
SLOW_RANGE = range(20, 181)
TREND_RANGE = range(30, 261)


def load_close(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cm = {c.lower().strip(): c for c in df.columns}
    dcol = cm.get('date') or cm.get('timestamp') or 'Date'
    ccol = cm.get('close') or cm.get('adj close') or cm.get('adj_close') or 'Close'
    out = pd.DataFrame({
        'date': pd.to_datetime(df[dcol], errors='coerce'),
        'close': pd.to_numeric(df[ccol], errors='coerce')
    }).dropna()
    out = out.sort_values('date')
    out = out[out['date'] >= pd.Timestamp(START_DATE)].reset_index(drop=True)
    return out


def rolling_mean_np(a: np.ndarray, w: int) -> np.ndarray:
    n = len(a)
    out = np.full(n, np.nan, dtype=np.float64)
    if w > n:
        return out
    c = np.cumsum(a, dtype=np.float64)
    out[w - 1] = c[w - 1] / w
    if n > w:
        out[w:] = (c[w:] - c[:-w]) / w
    return out


@njit(cache=True)
def eval_combo(close, mf, ms, mt, one_way, short_day, init_capital):
    n = close.shape[0]
    pos = 0            # -1 short, 0 flat, +1 long
    pending = 0
    last_open_dir = 0  # enforce alternation: cannot open same non-zero direction consecutively

    equity = init_capital
    prev = close[0]

    trades_close = 0
    open_count = 0
    flat_days = 0

    long_entries = 0
    short_entries = 0

    for i in range(1, n):
        cur = close[i]

        # MTM for current day before executing new target (T+1)
        if pos == 1:
            equity *= (cur / prev)
        elif pos == -1:
            equity *= (prev / cur)
            equity *= (1.0 - short_day)
        else:
            flat_days += 1

        # Execute pending at today's close.
        # Flip is allowed same day, but must be close then open.
        if pending != pos:
            if pos != 0:
                equity *= (1.0 - one_way)
                trades_close += 1
                pos = 0

            # Open new side only if alternation rule allows
            if pending != 0 and pending != last_open_dir:
                pos = pending
                equity *= (1.0 - one_way)
                open_count += 1
                last_open_dir = pending
                if pending == 1:
                    long_entries += 1
                else:
                    short_entries += 1

        # Generate next pending from today's signal
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

    # Force close at end
    if pos != 0:
        equity *= (1.0 - one_way)
        trades_close += 1

    ret_pct = (equity / init_capital - 1.0) * 100.0
    flat_ratio = flat_days / max(1, (n - 1)) * 100.0
    total_turns = trades_close
    return ret_pct, equity, total_turns, open_count, flat_ratio, long_entries, short_entries


def main():
    t0 = time.time()
    px = load_close(BTC_FILE)
    close = px['close'].to_numpy(np.float64)

    max_w = max(max(FAST_RANGE), max(SLOW_RANGE), max(TREND_RANGE))
    ma = {w: rolling_mean_np(close, w) for w in range(1, max_w + 1)}

    # warmup
    _ = eval_combo(close, ma[1], ma[20], ma[30], ONE_WAY, SHORT_DAY, INITIAL_CAPITAL)

    rows = []
    total = 0
    for f in FAST_RANGE:
        for s in SLOW_RANGE:
            if f < s:
                total += len(TREND_RANGE)

    done = 0
    for f in FAST_RANGE:
        mf = ma[f]
        for s in SLOW_RANGE:
            if f >= s:
                continue
            ms = ma[s]
            for t in TREND_RANGE:
                mt = ma[t]
                ret, eq, turns, opens, flat_ratio, le, se = eval_combo(
                    close, mf, ms, mt, ONE_WAY, SHORT_DAY, INITIAL_CAPITAL
                )
                rows.append((f, s, t, ret, eq, turns, opens, flat_ratio, le, se))
                done += 1
                if done % 50000 == 0:
                    print(f'progress {done}/{total} ({done/total:.2%})')

    out = pd.DataFrame(rows, columns=[
        'fast_ma', 'slow_ma', 'trend_ma', 'return_pct', 'final_equity_usd',
        'turns_close_count', 'open_count', 'flat_ratio_pct', 'long_entries', 'short_entries'
    ]).sort_values('return_pct', ascending=False).reset_index(drop=True)

    detail = OUT_DIR / 'btc_201711_alt_only_sym_trend_grid.csv'
    out.to_csv(detail, index=False)

    best = out.iloc[0]
    summary = pd.DataFrame([{
        'asset': 'btc',
        'start_date': str(px.date.iloc[0].date()) if len(px) else START_DATE,
        'end_date': str(px.date.iloc[-1].date()) if len(px) else None,
        'bars': int(len(px)),
        'initial_capital_usd': INITIAL_CAPITAL,
        'rule_position_states': 'long/full, flat, short/full',
        'rule_no_consecutive_same_side': True,
        'rule_flip_must_close_then_open': True,
        'best_fast': int(best['fast_ma']),
        'best_slow': int(best['slow_ma']),
        'best_trend': int(best['trend_ma']),
        'best_return_pct': float(best['return_pct']),
        'best_final_equity_usd': float(best['final_equity_usd']),
        'best_turns_close_count': int(best['turns_close_count']),
        'best_open_count': int(best['open_count']),
        'best_long_entries': int(best['long_entries']),
        'best_short_entries': int(best['short_entries']),
        'best_flat_ratio_pct': float(best['flat_ratio_pct']),
        'fee_bps': FEE_BPS,
        'slippage_bps': SLIPPAGE_BPS,
        'spread_bps': SPREAD_BPS,
        'short_cost_bps_day': SHORT_COST_BPS_DAY,
        'elapsed_sec': round(time.time() - t0, 2),
        'tested_total': int(len(out)),
        'detail_file': str(detail),
    }])
    summary_file = OUT_DIR / 'btc_201711_alt_only_sym_trend_summary.csv'
    summary.to_csv(summary_file, index=False)

    print('\nSUMMARY')
    print(summary.to_string(index=False))
    print('\nTOP 15')
    print(out.head(15).to_string(index=False))


if __name__ == '__main__':
    main()
