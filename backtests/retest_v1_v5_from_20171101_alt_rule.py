import csv
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path('/root/project/20260428-10gold')
BTC = BASE / 'data' / 'btc.csv'
OUT = BASE / 'backtest_results' / 'btc_retest_v1_v5_20171101_alt_rule.csv'

START_DATE = '2017-11-01'
INIT = 1000.0

FEE_BPS = 5.0
SLIPPAGE_BPS = 8.0
SPREAD_BPS = 4.0
SHORT_COST_BPS_DAY = 2.0
ONE_WAY = (FEE_BPS + SLIPPAGE_BPS + SPREAD_BPS * 0.5) / 10000.0
SHORT_DAY = SHORT_COST_BPS_DAY / 10000.0

# v1..v5 fixed params
STRATS = {
    'v1': dict(type='sym', fast=1, slow=39, trend=None),
    'v2': dict(type='sym_trend', fast=5, slow=60, trend=100),
    'v3': dict(type='sym_trend', fast=3, slow=55, trend=80),
    'v4': dict(type='sym_trend', fast=8, slow=60, trend=90),
    'v5': dict(type='sym_trend', fast=14, slow=42, trend=93),
}

def load_close(path):
    df = pd.read_csv(path)
    cm = {c.lower().strip(): c for c in df.columns}
    dcol = cm.get('date') or cm.get('timestamp') or 'Date'
    ccol = cm.get('close') or cm.get('adj close') or cm.get('adj_close') or 'Close'
    out = pd.DataFrame({'date': pd.to_datetime(df[dcol], errors='coerce'), 'close': pd.to_numeric(df[ccol], errors='coerce')}).dropna()
    out = out.sort_values('date')
    out = out[out['date'] >= pd.Timestamp(START_DATE)].reset_index(drop=True)
    return out

def ma_np(a, w):
    n = len(a)
    out = np.full(n, np.nan, dtype=float)
    if w > n:
        return out
    c = np.cumsum(a, dtype=float)
    out[w-1] = c[w-1] / w
    if n > w:
        out[w:] = (c[w:] - c[:-w]) / w
    return out

def run_strategy(close, cfg, ma_cache):
    mf = ma_cache[cfg['fast']]
    ms = ma_cache[cfg['slow']]
    mt = ma_cache[cfg['trend']] if cfg['trend'] else None

    pos = 0
    pending = 0
    last_open_dir = 0
    eq = INIT
    prev = close[0]

    open_count = close_count = 0
    long_entries = short_entries = 0
    flat_days = 0

    for i in range(1, len(close)):
        cur = close[i]

        if pos == 1:
            eq *= (cur / prev)
        elif pos == -1:
            eq *= (prev / cur)
            eq *= (1.0 - SHORT_DAY)
        else:
            flat_days += 1

        if pending != pos:
            if pos != 0:
                eq *= (1.0 - ONE_WAY)
                close_count += 1
                pos = 0
            if pending != 0 and pending != last_open_dir:
                pos = pending
                eq *= (1.0 - ONE_WAY)
                open_count += 1
                last_open_dir = pending
                if pos == 1:
                    long_entries += 1
                else:
                    short_entries += 1

        f0, f1 = mf[i-1], mf[i]
        s0, s1 = ms[i-1], ms[i]
        if not (np.isnan(f0) or np.isnan(f1) or np.isnan(s0) or np.isnan(s1)):
            bull = (f0 <= s0 and f1 > s1)
            bear = (f0 >= s0 and f1 < s1)
            long_ok = bull
            short_ok = bear
            if cfg['type'] == 'sym_trend':
                t1 = mt[i]
                if not np.isnan(t1):
                    long_ok = long_ok and (cur > t1)
                    short_ok = short_ok and (cur < t1)
                else:
                    long_ok = short_ok = False
            if long_ok:
                pending = 1
            elif short_ok:
                pending = -1

        prev = cur

    if pos != 0:
        eq *= (1.0 - ONE_WAY)
        close_count += 1

    ret = (eq / INIT - 1.0) * 100.0
    return dict(return_pct=ret, final_equity_usd=eq, open_count=open_count, close_count=close_count,
                long_entries=long_entries, short_entries=short_entries,
                flat_ratio_pct=flat_days / max(1, (len(close)-1)) * 100.0)


def main():
    px = load_close(BTC)
    close = px['close'].to_numpy(float)
    max_w = max([max(v['fast'], v['slow'], v['trend'] or 1) for v in STRATS.values()])
    ma_cache = {w: ma_np(close, w) for w in range(1, max_w+1)}

    rows = []
    for name, cfg in STRATS.items():
        m = run_strategy(close, cfg, ma_cache)
        rows.append({
            'strategy': name,
            'start_date': str(px.date.iloc[0].date()),
            'end_date': str(px.date.iloc[-1].date()),
            'bars': int(len(px)),
            'fast': cfg['fast'],
            'slow': cfg['slow'],
            'trend': cfg['trend'],
            **m
        })

    df = pd.DataFrame(rows).sort_values('strategy')
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print('saved', OUT)

if __name__ == '__main__':
    main()
