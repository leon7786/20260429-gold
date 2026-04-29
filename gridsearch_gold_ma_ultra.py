import os
import time
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from numba import njit

BASE = Path('/root/project/20260428-10gold')
PRICE_PATH = BASE / 'data_long_history' / 'gold_daily_proxy_gld_yahoo.csv'
OUT_DIR = BASE / 'backtest_results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BUY_MIN, BUY_MAX = 1, 400
SELL_MIN, SELL_MAX = 1, 4000
MIN_TRADES_FILTER = 30

G_CLOSE = None
G_PREV_CLOSE = None
G_MA = None


def rolling_mean_np(arr: np.ndarray, window: int) -> np.ndarray:
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 1:
        out[:] = arr
        return out
    if window > n:
        return out
    c = np.cumsum(arr, dtype=np.float64)
    out[window - 1] = c[window - 1] / window
    if n > window:
        out[window:] = (c[window:] - c[:-window]) / window
    return out


@njit(cache=True)
def simulate(close, prev_close, buy_signal, sell_signal):
    position = 0
    entry = 0.0
    equity = 1.0
    trades = 0
    wins = 0

    for i in range(close.shape[0]):
        c = close[i]
        pc = prev_close[i]
        if position == 1 and not math.isnan(pc):
            equity *= c / pc

        if position == 0 and buy_signal[i] == 1:
            position = 1
            entry = c
        elif position == 1 and sell_signal[i] == 1:
            ret = c / entry - 1.0
            trades += 1
            if ret > 0.0:
                wins += 1
            position = 0

    if position == 1:
        ret = close[-1] / entry - 1.0
        trades += 1
        if ret > 0.0:
            wins += 1

    return equity, trades, wins


def init_worker(close, prev_close, ma_matrix):
    global G_CLOSE, G_PREV_CLOSE, G_MA
    G_CLOSE = close
    G_PREV_CLOSE = prev_close
    G_MA = ma_matrix


def worker_one_buy_ma(buy_ma: int):
    close = G_CLOSE
    prev_close = G_PREV_CLOSE
    ma = G_MA

    out = []
    ma_buy = ma[buy_ma]
    prev_ma_buy = np.roll(ma_buy, 1)
    prev_ma_buy[0] = np.nan
    buy_signal = (prev_close <= prev_ma_buy) & (close > ma_buy)
    buy_signal = np.nan_to_num(buy_signal, nan=False).astype(np.int8)

    for sell_ma in range(SELL_MIN, SELL_MAX + 1):
        ma_sell = ma[sell_ma]
        prev_ma_sell = np.roll(ma_sell, 1)
        prev_ma_sell[0] = np.nan
        sell_signal = (prev_close >= prev_ma_sell) & (close < ma_sell)
        sell_signal = np.nan_to_num(sell_signal, nan=False).astype(np.int8)

        eq, trades, wins = simulate(close, prev_close, buy_signal, sell_signal)
        out.append((buy_ma, sell_ma, (eq - 1.0) * 100.0, int(trades), (wins / trades * 100.0) if trades else 0.0))

    return out


def main():
    t0 = time.time()
    raw = pd.read_csv(PRICE_PATH)
    raw['date'] = pd.to_datetime(raw['date'])
    raw = raw.sort_values('date').reset_index(drop=True)
    raw['close'] = pd.to_numeric(raw['close'], errors='coerce')
    raw = raw.dropna(subset=['close'])

    close = raw['close'].to_numpy(dtype=np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan

    max_ma = SELL_MAX
    ma_matrix = np.empty((max_ma + 1, close.shape[0]), dtype=np.float64)
    ma_matrix[:] = np.nan
    for w in range(1, max_ma + 1):
        ma_matrix[w] = rolling_mean_np(close, w)

    # numba warmup
    _ = simulate(close[:50], prev_close[:50], np.zeros(50, dtype=np.int8), np.zeros(50, dtype=np.int8))

    buy_list = list(range(BUY_MIN, BUY_MAX + 1))
    workers = min(os.cpu_count() or 1, len(buy_list))

    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(close, prev_close, ma_matrix)) as ex:
        futs = [ex.submit(worker_one_buy_ma, b) for b in buy_list]
        done = 0
        for f in as_completed(futs):
            results.extend(f.result())
            done += 1
            if done % 20 == 0 or done == len(futs):
                print(f'progress: {done}/{len(futs)} buy_ma finished')

    res = pd.DataFrame(results, columns=['buy_ma', 'sell_ma', 'total_return_pct', 'trades', 'win_rate_pct'])
    res = res.sort_values('total_return_pct', ascending=False).reset_index(drop=True)

    res_path = OUT_DIR / 'gold_ma_gridsearch_ultra_1_400_vs_1_4000.csv'
    res.to_csv(res_path, index=False)

    robust_top = res[res['trades'] >= MIN_TRADES_FILTER].head(50).copy()
    robust_path = OUT_DIR / 'gold_ma_gridsearch_ultra_top50_trades_ge30.csv'
    robust_top.to_csv(robust_path, index=False)

    print('\nrows_tested', len(res))
    print('workers', workers)
    print('best_overall')
    print(res.head(10).to_string(index=False))
    print('\nbest_trades_ge30')
    print(robust_top.head(10).to_string(index=False))
    print('\nsaved', res_path)
    print('saved', robust_path)
    print('elapsed_sec', round(time.time() - t0, 2))


if __name__ == '__main__':
    main()
