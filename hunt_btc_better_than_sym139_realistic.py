import time
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path('/root/project/20260428-10gold')
OUT = BASE / 'backtest_results'
OUT.mkdir(parents=True, exist_ok=True)

TARGET_BEAT = 214435.94
BTC_FILE = BASE / 'data' / 'btc.csv'

# realistic 6 variables
FEE_BPS = 5.0
SLIPPAGE_BPS = 8.0
SPREAD_BPS = 4.0
SHORT_COST_BPS_DAY = 2.0
MAX_GROSS = 1.0
ONE_WAY = (FEE_BPS + SLIPPAGE_BPS + SPREAD_BPS * 0.5) / 10000.0
SHORT_DAY = SHORT_COST_BPS_DAY / 10000.0


def load_close(path: Path):
    df = pd.read_csv(path)
    cm = {c.lower().strip(): c for c in df.columns}
    dcol = cm.get('date') or cm.get('timestamp') or 'Date'
    ccol = cm.get('close') or cm.get('adj close') or cm.get('adj_close') or 'Close'
    out = pd.DataFrame({'date': pd.to_datetime(df[dcol], errors='coerce'), 'close': pd.to_numeric(df[ccol], errors='coerce')})
    out = out.dropna().sort_values('date').reset_index(drop=True)
    return out


def rolling_mean_np(a, w):
    n = len(a)
    out = np.full(n, np.nan)
    if w > n:
        return out
    c = np.cumsum(a, dtype=float)
    out[w-1] = c[w-1] / w
    if n > w:
        out[w:] = (c[w:] - c[:-w]) / w
    return out


def rolling_std_np(a, w):
    s = pd.Series(a)
    return s.rolling(w).std(ddof=0).to_numpy()


def rolling_max_np(a, w):
    return pd.Series(a).rolling(w).max().to_numpy()


def rolling_min_np(a, w):
    return pd.Series(a).rolling(w).min().to_numpy()


def ema_np(a, span):
    return pd.Series(a).ewm(span=span, adjust=False).mean().to_numpy()


def simulate_from_target(close, target):
    pos = 0
    pending = 0
    eq = 1.0
    prev = close[0]
    trades = 0
    flat = 0
    for i in range(1, len(close)):
        cur = close[i]
        if pos == 1:
            eq *= 1.0 + MAX_GROSS * (cur/prev - 1.0)
        elif pos == -1:
            eq *= 1.0 + MAX_GROSS * (prev/cur - 1.0)
            eq *= (1.0 - SHORT_DAY * MAX_GROSS)
        else:
            flat += 1

        # execute pending at today's close
        if pending != pos:
            if pos != 0:
                eq *= (1.0 - ONE_WAY * MAX_GROSS)
                trades += 1
            pos = pending
            if pos != 0:
                eq *= (1.0 - ONE_WAY * MAX_GROSS)

        # today signal for tomorrow
        t = target[i]
        if t != 0:
            pending = int(t)

        prev = cur

    if pos != 0:
        eq *= (1.0 - ONE_WAY * MAX_GROSS)
        trades += 1

    ret = (eq - 1.0) * 100.0
    flat_ratio = flat / (len(close)-1) * 100.0
    return ret, trades, flat_ratio


def family_sym_ma(close):
    # Expanded around aggressive region + long trend region
    fasts = [1,2,3,4,5,6,8,10,13,16,21]
    slows = [20,26,30,35,39,44,50,60,80,100,120,160,200,260,320]
    rows = []
    ma = {w: rolling_mean_np(close, w) for w in sorted(set(fasts+slows))}
    for f in fasts:
        mf = ma[f]
        for s in slows:
            if f >= s:
                continue
            ms = ma[s]
            tgt = np.zeros(len(close), dtype=np.int8)
            for i in range(1, len(close)):
                f0, f1 = mf[i-1], mf[i]
                s0, s1 = ms[i-1], ms[i]
                if np.isnan(f0) or np.isnan(f1) or np.isnan(s0) or np.isnan(s1):
                    continue
                if f0 <= s0 and f1 > s1:
                    tgt[i] = 1
                elif f0 >= s0 and f1 < s1:
                    tgt[i] = -1
            ret, tr, fr = simulate_from_target(close, tgt)
            rows.append({'family':'sym_ma','p1':f,'p2':s,'p3':None,'p4':None,'return_pct':ret,'trades':tr,'flat_ratio_pct':fr})
    return rows


def family_donchian(close):
    # breakout / reversal variants
    rows = []
    ns = [5,8,10,13,16,20,26,30,35,39,44,50,60,80,100,120]
    for n in ns:
        hh = rolling_max_np(close, n)
        ll = rolling_min_np(close, n)
        tgt = np.zeros(len(close), dtype=np.int8)
        for i in range(1, len(close)):
            if np.isnan(hh[i-1]) or np.isnan(ll[i-1]):
                continue
            # strict breakout uses previous channel to avoid same-bar look
            if close[i] > hh[i-1]:
                tgt[i] = 1
            elif close[i] < ll[i-1]:
                tgt[i] = -1
        ret, tr, fr = simulate_from_target(close, tgt)
        rows.append({'family':'donchian_break','p1':n,'p2':None,'p3':None,'p4':None,'return_pct':ret,'trades':tr,'flat_ratio_pct':fr})

    # asymmetric entry/exit channels
    ent = [10,20,30,40,50,60]
    ex = [5,8,10,13,16,20]
    for ne in ent:
        hhe = rolling_max_np(close, ne)
        lle = rolling_min_np(close, ne)
        for nx in ex:
            hhx = rolling_max_np(close, nx)
            llx = rolling_min_np(close, nx)
            tgt = np.zeros(len(close), dtype=np.int8)
            pos_sig = 0
            for i in range(1, len(close)):
                if np.isnan(hhe[i-1]) or np.isnan(lle[i-1]) or np.isnan(hhx[i-1]) or np.isnan(llx[i-1]):
                    continue
                if pos_sig == 0:
                    if close[i] > hhe[i-1]:
                        pos_sig = 1
                    elif close[i] < lle[i-1]:
                        pos_sig = -1
                elif pos_sig == 1:
                    if close[i] < llx[i-1]:
                        pos_sig = -1
                elif pos_sig == -1:
                    if close[i] > hhx[i-1]:
                        pos_sig = 1
                tgt[i] = pos_sig
            ret, tr, fr = simulate_from_target(close, tgt)
            rows.append({'family':'donchian_dual','p1':ne,'p2':nx,'p3':None,'p4':None,'return_pct':ret,'trades':tr,'flat_ratio_pct':fr})
    return rows


def family_zscore(close):
    rows = []
    ws = [20,30,40,50,60,80,100]
    ths = [0.5,0.8,1.0,1.2,1.5,2.0]
    for w in ws:
        m = rolling_mean_np(close, w)
        sd = rolling_std_np(close, w)
        for th in ths:
            tgt = np.zeros(len(close), dtype=np.int8)
            for i in range(1, len(close)):
                if np.isnan(m[i]) or np.isnan(sd[i]) or sd[i] <= 1e-12:
                    continue
                z = (close[i] - m[i]) / sd[i]
                if z > th:
                    tgt[i] = 1
                elif z < -th:
                    tgt[i] = -1
            ret, tr, fr = simulate_from_target(close, tgt)
            rows.append({'family':'zscore_momo','p1':w,'p2':th,'p3':None,'p4':None,'return_pct':ret,'trades':tr,'flat_ratio_pct':fr})

            # mean-reversion inverse
            tgt2 = np.zeros(len(close), dtype=np.int8)
            for i in range(1, len(close)):
                if np.isnan(m[i]) or np.isnan(sd[i]) or sd[i] <= 1e-12:
                    continue
                z = (close[i] - m[i]) / sd[i]
                if z > th:
                    tgt2[i] = -1
                elif z < -th:
                    tgt2[i] = 1
            ret2, tr2, fr2 = simulate_from_target(close, tgt2)
            rows.append({'family':'zscore_revert','p1':w,'p2':th,'p3':None,'p4':None,'return_pct':ret2,'trades':tr2,'flat_ratio_pct':fr2})
    return rows


def family_ema_gap(close):
    rows = []
    fasts = [2,3,5,8,10,13,16]
    slows = [20,30,39,50,60,80,100,120]
    ths = [0.000,0.001,0.002,0.003,0.005]
    emas = {w: ema_np(close, w) for w in sorted(set(fasts+slows))}
    for f in fasts:
        ef = emas[f]
        for s in slows:
            if f >= s:
                continue
            es = emas[s]
            for th in ths:
                tgt = np.zeros(len(close), dtype=np.int8)
                for i in range(1, len(close)):
                    if np.isnan(ef[i]) or np.isnan(es[i]) or es[i] == 0:
                        continue
                    gap = (ef[i]-es[i]) / es[i]
                    if gap > th:
                        tgt[i] = 1
                    elif gap < -th:
                        tgt[i] = -1
                ret, tr, fr = simulate_from_target(close, tgt)
                rows.append({'family':'ema_gap','p1':f,'p2':s,'p3':th,'p4':None,'return_pct':ret,'trades':tr,'flat_ratio_pct':fr})
    return rows


def main():
    t0 = time.time()
    px = load_close(BTC_FILE)
    close = px['close'].to_numpy(float)

    all_rows = []
    for func in [family_sym_ma, family_donchian, family_zscore, family_ema_gap]:
        st = time.time()
        rows = func(close)
        all_rows.extend(rows)
        print(f"[{func.__name__}] tested={len(rows)} elapsed={time.time()-st:.2f}s")

    df = pd.DataFrame(all_rows).sort_values('return_pct', ascending=False).reset_index(drop=True)
    detail = OUT / 'btc_strategy_hunt_realistic_t1_all_families.csv'
    df.to_csv(detail, index=False)

    best = df.iloc[0]
    beat = df[df['return_pct'] > TARGET_BEAT]

    summary = {
        'tested_total': int(len(df)),
        'best_family': best['family'],
        'best_p1': best['p1'],
        'best_p2': best['p2'],
        'best_p3': best['p3'],
        'best_p4': best['p4'],
        'best_return_pct': float(best['return_pct']),
        'best_trades': int(best['trades']),
        'best_flat_ratio_pct': float(best['flat_ratio_pct']),
        'target_to_beat_pct': TARGET_BEAT,
        'num_strategies_beating_target': int(len(beat)),
        'elapsed_sec': round(time.time()-t0, 2),
        'detail_file': str(detail)
    }

    pd.DataFrame([summary]).to_csv(OUT / 'btc_strategy_hunt_realistic_t1_summary.csv', index=False)
    print('saved', OUT / 'btc_strategy_hunt_realistic_t1_summary.csv')
    print(pd.DataFrame([summary]).to_string(index=False))
    print('\nTOP 20:')
    print(df.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
