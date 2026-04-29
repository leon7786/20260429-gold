import time
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path('/root/project/20260428-10gold')
OUT = BASE / 'backtest_results'
OUT.mkdir(parents=True, exist_ok=True)
BTC_FILE = BASE / 'data' / 'btc.csv'
TARGET = 214435.94152

FEE_BPS = 5.0
SLIPPAGE_BPS = 8.0
SPREAD_BPS = 4.0
SHORT_COST_BPS_DAY = 2.0
MAX_GROSS = 1.0
ONE_WAY = (FEE_BPS + SLIPPAGE_BPS + SPREAD_BPS * 0.5) / 10000.0
SHORT_DAY = SHORT_COST_BPS_DAY / 10000.0


def load_close(path: Path):
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    dcol = cols.get('date') or cols.get('timestamp') or 'Date'
    ccol = cols.get('close') or cols.get('adj close') or cols.get('adj_close') or 'Close'
    out = pd.DataFrame({'date': pd.to_datetime(df[dcol], errors='coerce'), 'close': pd.to_numeric(df[ccol], errors='coerce')}).dropna()
    out = out.sort_values('date').reset_index(drop=True)
    return out


def ma(a, w):
    return pd.Series(a).rolling(w).mean().to_numpy()


def ema(a, w):
    return pd.Series(a).ewm(span=w, adjust=False).mean().to_numpy()


def rsi(close, w=14):
    s = pd.Series(close)
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/w, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/w, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.to_numpy()


def atr_like(close, w=14):
    ret = pd.Series(close).pct_change().abs()
    return ret.rolling(w).mean().to_numpy()


def sim(close, target):
    pos = 0
    pend = 0
    eq = 1.0
    prev = close[0]
    trades = 0
    flat = 0
    for i in range(1, len(close)):
        cur = close[i]
        if pos == 1:
            eq *= 1 + (cur/prev - 1)
        elif pos == -1:
            eq *= 1 + (prev/cur - 1)
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

    return (eq-1)*100, trades, flat/(len(close)-1)*100


def run(close):
    rows = []

    # 1) sym + trend filter (only allow long if price>trend, short if price<trend)
    fasts = [1,2,3,4,5,6,8,10]
    slows = [30,35,39,44,50,60,80,100,120,160,200]
    trends = [100,150,200,250,300]
    ma_cache = {w: ma(close, w) for w in sorted(set(fasts+slows+trends))}
    for f in fasts:
        mf = ma_cache[f]
        for s in slows:
            if f >= s: continue
            ms = ma_cache[s]
            for tw in trends:
                tr = ma_cache[tw]
                tgt = np.zeros(len(close), dtype=np.int8)
                for i in range(1, len(close)):
                    if np.isnan(mf[i-1]) or np.isnan(ms[i-1]) or np.isnan(mf[i]) or np.isnan(ms[i]) or np.isnan(tr[i]):
                        continue
                    if mf[i-1] <= ms[i-1] and mf[i] > ms[i] and close[i] > tr[i]:
                        tgt[i] = 1
                    elif mf[i-1] >= ms[i-1] and mf[i] < ms[i] and close[i] < tr[i]:
                        tgt[i] = -1
                r,t,fr = sim(close,tgt)
                rows.append(dict(family='sym_trend_filter',p1=f,p2=s,p3=tw,p4=None,return_pct=r,trades=t,flat_ratio_pct=fr))

    # 2) EMA cross + RSI state filter
    ef = [2,3,5,8,10,13]
    es = [30,39,44,50,60,80,100]
    rw = [7,14,21]
    up_th = [52,55,58,60]
    dn_th = [48,45,42,40]
    ema_cache = {w: ema(close, w) for w in sorted(set(ef+es))}
    rsi_cache = {w: rsi(close, w) for w in rw}
    for f in ef:
        e1 = ema_cache[f]
        for s in es:
            if f >= s: continue
            e2 = ema_cache[s]
            for w in rw:
                rr = rsi_cache[w]
                for uth in up_th:
                    for dth in dn_th:
                        tgt = np.zeros(len(close), dtype=np.int8)
                        for i in range(1, len(close)):
                            if np.isnan(e1[i-1]) or np.isnan(e2[i-1]) or np.isnan(e1[i]) or np.isnan(e2[i]) or np.isnan(rr[i]):
                                continue
                            if e1[i-1] <= e2[i-1] and e1[i] > e2[i] and rr[i] >= uth:
                                tgt[i] = 1
                            elif e1[i-1] >= e2[i-1] and e1[i] < e2[i] and rr[i] <= dth:
                                tgt[i] = -1
                        r,t,fr = sim(close,tgt)
                        rows.append(dict(family='ema_rsi_filter',p1=f,p2=s,p3=w,p4=float(f'{uth}/{dth}'.split('/')[0]),return_pct=r,trades=t,flat_ratio_pct=fr,extra=f'{uth}/{dth}'))

    # 3) price breakout + vol filter
    ns = [10,20,30,39,44,50,60,80,100]
    vw = [10,14,20]
    vth = [0.015,0.02,0.025,0.03,0.04]
    roll_max = {n: pd.Series(close).rolling(n).max().to_numpy() for n in ns}
    roll_min = {n: pd.Series(close).rolling(n).min().to_numpy() for n in ns}
    atrc = {w: atr_like(close,w) for w in vw}
    for n in ns:
        hh = roll_max[n]
        ll = roll_min[n]
        for w in vw:
            av = atrc[w]
            for th in vth:
                tgt = np.zeros(len(close), dtype=np.int8)
                for i in range(1, len(close)):
                    if np.isnan(hh[i-1]) or np.isnan(ll[i-1]) or np.isnan(av[i]):
                        continue
                    if av[i] > th:
                        continue
                    if close[i] > hh[i-1]:
                        tgt[i] = 1
                    elif close[i] < ll[i-1]:
                        tgt[i] = -1
                r,t,fr = sim(close,tgt)
                rows.append(dict(family='donchian_vol_filter',p1=n,p2=w,p3=th,p4=None,return_pct=r,trades=t,flat_ratio_pct=fr))

    return rows


def main():
    t0=time.time()
    close = load_close(BTC_FILE)['close'].to_numpy(float)
    rows = run(close)
    df = pd.DataFrame(rows).sort_values('return_pct', ascending=False).reset_index(drop=True)
    out = OUT / 'btc_strategy_hunt_realistic_t1_v2_all.csv'
    df.to_csv(out, index=False)
    best = df.iloc[0]
    beat = df[df['return_pct'] > TARGET]

    summary = {
        'tested_total': int(len(df)),
        'best_family': best['family'],
        'best_p1': best['p1'],
        'best_p2': best['p2'],
        'best_p3': best.get('p3', None),
        'best_p4': best.get('p4', None),
        'best_return_pct': float(best['return_pct']),
        'best_trades': int(best['trades']),
        'num_beating_target': int(len(beat)),
        'target': TARGET,
        'elapsed_sec': round(time.time()-t0,2),
        'detail_file': str(out)
    }
    sfile = OUT / 'btc_strategy_hunt_realistic_t1_v2_summary.csv'
    pd.DataFrame([summary]).to_csv(sfile, index=False)
    print(pd.DataFrame([summary]).to_string(index=False))
    print('\nTOP 20:')
    print(df.head(20).to_string(index=False))

if __name__ == '__main__':
    main()
