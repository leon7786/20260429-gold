import time
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path('/root/project/20260428-10gold')
OUT = BASE / 'backtest_results'
OUT.mkdir(parents=True, exist_ok=True)
BTC_FILE = BASE / 'data' / 'btc.csv'
TARGET = 314828.563113  # beat MA(5/60)+trend100

FEE_BPS = 5.0
SLIPPAGE_BPS = 8.0
SPREAD_BPS = 4.0
SHORT_COST_BPS_DAY = 2.0
ONE_WAY = (FEE_BPS + SLIPPAGE_BPS + SPREAD_BPS * 0.5) / 10000.0
SHORT_DAY = SHORT_COST_BPS_DAY / 10000.0


def load_close(path):
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    dcol = cols.get('date') or 'Date'
    ccol = cols.get('close') or cols.get('adj close') or 'Close'
    out = pd.DataFrame({'date': pd.to_datetime(df[dcol], errors='coerce'), 'close': pd.to_numeric(df[ccol], errors='coerce')}).dropna()
    return out.sort_values('date').reset_index(drop=True)


def ma(a, w):
    return pd.Series(a).rolling(w).mean().to_numpy()


def sim(close, target):
    pos, pend, trades, flat = 0, 0, 0, 0
    eq = 1.0
    prev = close[0]
    for i in range(1, len(close)):
        cur = close[i]
        if pos == 1:
            eq *= (cur/prev)
        elif pos == -1:
            eq *= (prev/cur)
            eq *= (1 - SHORT_DAY)
        else:
            flat += 1

        if pend != pos:
            if pos != 0:
                eq *= (1 - ONE_WAY)
                trades += 1
            pos = pend
            if pos != 0:
                eq *= (1 - ONE_WAY)

        t = int(target[i])
        if t != 0:
            pend = t
        prev = cur

    if pos != 0:
        eq *= (1 - ONE_WAY)
        trades += 1

    return (eq - 1) * 100.0, trades, flat/(len(close)-1)*100


def main():
    t0 = time.time()
    close = load_close(BTC_FILE)['close'].to_numpy(float)

    # focused neighborhood + stricter trend regimes
    fasts = [1,2,3,4,5,6,7,8,9,10,12]
    slows = [30,35,39,44,50,55,60,65,70,80,90,100]
    trends = [80,90,100,110,120,140,160,180,200]

    ma_windows = sorted(set(fasts + slows + trends))
    m = {w: ma(close, w) for w in ma_windows}

    rows = []

    # Family 1: crossover + trend filter (previous winner family)
    for f in fasts:
        for s in slows:
            if f >= s:
                continue
            for tw in trends:
                mf, ms, mt = m[f], m[s], m[tw]
                tgt = np.zeros(len(close), dtype=np.int8)
                for i in range(1, len(close)):
                    if np.isnan(mf[i-1]) or np.isnan(ms[i-1]) or np.isnan(mf[i]) or np.isnan(ms[i]) or np.isnan(mt[i]):
                        continue
                    if mf[i-1] <= ms[i-1] and mf[i] > ms[i] and close[i] > mt[i]:
                        tgt[i] = 1
                    elif mf[i-1] >= ms[i-1] and mf[i] < ms[i] and close[i] < mt[i]:
                        tgt[i] = -1
                r, t, fr = sim(close, tgt)
                rows.append(dict(family='sym_trend', p1=f, p2=s, p3=tw, p4=None, ret=r, trades=t, flat=fr))

    # Family 2: crossover + trend slope filter
    slope_lags = [3,5,8,10,13]
    for f in fasts:
        for s in slows:
            if f >= s:
                continue
            for tw in trends:
                for lag in slope_lags:
                    mf, ms, mt = m[f], m[s], m[tw]
                    tgt = np.zeros(len(close), dtype=np.int8)
                    for i in range(max(1,lag), len(close)):
                        if np.isnan(mf[i-1]) or np.isnan(ms[i-1]) or np.isnan(mf[i]) or np.isnan(ms[i]) or np.isnan(mt[i]) or np.isnan(mt[i-lag]):
                            continue
                        slope_up = mt[i] > mt[i-lag]
                        slope_dn = mt[i] < mt[i-lag]
                        if mf[i-1] <= ms[i-1] and mf[i] > ms[i] and close[i] > mt[i] and slope_up:
                            tgt[i] = 1
                        elif mf[i-1] >= ms[i-1] and mf[i] < ms[i] and close[i] < mt[i] and slope_dn:
                            tgt[i] = -1
                    r, t, fr = sim(close, tgt)
                    rows.append(dict(family='sym_trend_slope', p1=f, p2=s, p3=tw, p4=lag, ret=r, trades=t, flat=fr))

    # Family 3: dual trend filter (fast trend + slow trend)
    tf = [50,60,80,100]
    ts = [120,140,160,180,200]
    for f in fasts:
        for s in slows:
            if f >= s:
                continue
            for t1 in tf:
                for t2 in ts:
                    if t1 >= t2:
                        continue
                    mf, ms, mt1, mt2 = m[f], m[s], m[t1], m[t2]
                    tgt = np.zeros(len(close), dtype=np.int8)
                    for i in range(1, len(close)):
                        if np.isnan(mf[i-1]) or np.isnan(ms[i-1]) or np.isnan(mf[i]) or np.isnan(ms[i]) or np.isnan(mt1[i]) or np.isnan(mt2[i]):
                            continue
                        bull = (close[i] > mt1[i] > mt2[i])
                        bear = (close[i] < mt1[i] < mt2[i])
                        if mf[i-1] <= ms[i-1] and mf[i] > ms[i] and bull:
                            tgt[i] = 1
                        elif mf[i-1] >= ms[i-1] and mf[i] < ms[i] and bear:
                            tgt[i] = -1
                    r, t, fr = sim(close, tgt)
                    rows.append(dict(family='sym_dual_trend', p1=f, p2=s, p3=t1, p4=t2, ret=r, trades=t, flat=fr))

    df = pd.DataFrame(rows).sort_values('ret', ascending=False).reset_index(drop=True)
    out = OUT / 'btc_strategy_hunt_realistic_t1_v3_all.csv'
    df.to_csv(out, index=False)

    best = df.iloc[0]
    beat = df[df['ret'] > TARGET]

    summary = pd.DataFrame([{
        'tested_total': int(len(df)),
        'best_family': best['family'],
        'best_p1': best['p1'],
        'best_p2': best['p2'],
        'best_p3': best['p3'],
        'best_p4': best['p4'],
        'best_return_pct': float(best['ret']),
        'best_trades': int(best['trades']),
        'num_beating_target': int(len(beat)),
        'target': TARGET,
        'elapsed_sec': round(time.time()-t0,2),
        'detail_file': str(out)
    }])
    sfile = OUT / 'btc_strategy_hunt_realistic_t1_v3_summary.csv'
    summary.to_csv(sfile, index=False)

    print(summary.to_string(index=False))
    print('\nTOP 15:')
    print(df.head(15).to_string(index=False))

if __name__ == '__main__':
    main()
