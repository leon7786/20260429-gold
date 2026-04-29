import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

BASE = Path('/root/project/20260428-10gold')
OUT = BASE / 'backtest_results'
OUT.mkdir(parents=True, exist_ok=True)

BTC_FILE = BASE / 'data' / 'btc.csv'
MA_MIN, MA_MAX = 1, 700

# 6个关键实盘变量
EXEC_MODEL = 'T+1(close signal -> next close execute)'
FEE_BPS = 5.0           # 单边手续费 5 bps
SLIPPAGE_BPS = 8.0      # 单边滑点 8 bps
SPREAD_BPS = 4.0        # 双边价差折半到单边 2 bps
SHORT_COST_BPS_DAY = 2.0  # 空头日持仓成本 2 bps/day
MAX_GROSS = 1.0         # 仓位上限（1x）

# 单边交易总摩擦（按成交额）
ONE_WAY_TRADE_COST = (FEE_BPS + SLIPPAGE_BPS + SPREAD_BPS * 0.5) / 10000.0
SHORT_DAILY_COST = SHORT_COST_BPS_DAY / 10000.0


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
def sim_t1_realistic(close, ma_fast, ma_slow, one_way_cost, short_daily_cost, max_gross):
    n = close.shape[0]
    pos = 0  # -1 short, 0 flat, 1 long
    prev = close[0]
    equity = 1.0

    pending_target = 0
    flat_bars = 0
    long_trades = 0
    short_trades = 0
    long_wins = 0
    short_wins = 0
    entry_px = 0.0
    entry_side = 0

    for i in range(1, n):
        c = close[i]

        # mark-to-market（旧仓）
        if pos == 1:
            equity *= 1.0 + max_gross * (c / prev - 1.0)
        elif pos == -1:
            equity *= 1.0 + max_gross * (prev / c - 1.0)
            # 空头持仓日成本
            equity *= (1.0 - short_daily_cost * max_gross)
        else:
            flat_bars += 1

        # 执行昨日信号 -> 今日收盘成交（T+1）
        if pending_target != pos:
            # 平旧仓（若有）
            if pos != 0:
                # 平仓交易成本
                equity *= (1.0 - one_way_cost * max_gross)
                if entry_side == 1:
                    r = c / entry_px - 1.0
                    long_trades += 1
                    if r > 0:
                        long_wins += 1
                elif entry_side == -1:
                    r = entry_px / c - 1.0
                    short_trades += 1
                    if r > 0:
                        short_wins += 1

            # 开新仓（若有）
            pos = pending_target
            if pos != 0:
                equity *= (1.0 - one_way_cost * max_gross)
                entry_side = pos
                entry_px = c
            else:
                entry_side = 0
                entry_px = 0.0

        # 当日生成信号（明日执行）
        f0 = ma_fast[i - 1]
        s0 = ma_slow[i - 1]
        f1 = ma_fast[i]
        s1 = ma_slow[i]

        if not (np.isnan(f0) or np.isnan(s0) or np.isnan(f1) or np.isnan(s1)):
            cross_up = (f0 <= s0) and (f1 > s1)
            cross_dn = (f0 >= s0) and (f1 < s1)
            if cross_up:
                pending_target = 1
            elif cross_dn:
                pending_target = -1

        prev = c

    # 收尾强平
    if pos != 0:
        equity *= (1.0 - one_way_cost * max_gross)
        if entry_side == 1:
            r = close[-1] / entry_px - 1.0
            long_trades += 1
            if r > 0:
                long_wins += 1
        else:
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


def main():
    px = load_close(BTC_FILE)
    close = px['close'].to_numpy(dtype=np.float64)
    ma = {w: rolling_mean_np(close, w) for w in range(MA_MIN, MA_MAX + 1)}

    t0 = time.time()
    rows = []
    total = (MA_MAX - MA_MIN + 1)
    done = 0
    for f in range(MA_MIN, MA_MAX + 1):
        mf = ma[f]
        for s in range(MA_MIN, MA_MAX + 1):
            if s == f:
                continue
            ms = ma[s]
            ret, tr, fr, ltr, strd, lwr, swr = sim_t1_realistic(
                close, mf, ms, ONE_WAY_TRADE_COST, SHORT_DAILY_COST, MAX_GROSS
            )
            rows.append((f, s, ret, tr, fr, ltr, strd, lwr, swr))
        done += 1
        if done % 50 == 0 or done == total:
            print(f"[btc-realistic] progress {done}/{total}", flush=True)

    res = pd.DataFrame(rows, columns=[
        'fast_ma', 'slow_ma', 'dual_return_pct', 'trades', 'flat_ratio_pct',
        'long_trades', 'short_trades', 'long_win_rate_pct', 'short_win_rate_pct'
    ])
    best = res.sort_values('dual_return_pct', ascending=False).iloc[0]

    detail_file = OUT / 'btc_dual_side_ma_1_700_t1_realistic_grid.csv'
    res.to_csv(detail_file, index=False)

    summary = pd.DataFrame([{
        'asset': 'btc',
        'start_date': px['date'].iloc[0].strftime('%Y-%m-%d'),
        'end_date': px['date'].iloc[-1].strftime('%Y-%m-%d'),
        'bars': len(px),
        'best_fast_ma': int(best['fast_ma']),
        'best_slow_ma': int(best['slow_ma']),
        'best_dual_return_pct': float(best['dual_return_pct']),
        'best_trades': int(best['trades']),
        'best_flat_ratio_pct': float(best['flat_ratio_pct']),
        'best_long_trades': int(best['long_trades']),
        'best_short_trades': int(best['short_trades']),
        'best_long_win_rate_pct': float(best['long_win_rate_pct']),
        'best_short_win_rate_pct': float(best['short_win_rate_pct']),
        'execution_model': EXEC_MODEL,
        'fee_bps': FEE_BPS,
        'slippage_bps': SLIPPAGE_BPS,
        'spread_bps': SPREAD_BPS,
        'short_cost_bps_day': SHORT_COST_BPS_DAY,
        'max_gross': MAX_GROSS,
        'elapsed_sec': round(time.time() - t0, 2),
        'detail_file': str(detail_file)
    }])
    out_file = OUT / 'btc_best_dual_side_ma_1_700_t1_realistic_summary.csv'
    summary.to_csv(out_file, index=False)
    print(f"saved {out_file}")
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
