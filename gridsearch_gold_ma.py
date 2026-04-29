import pandas as pd
import numpy as np
from pathlib import Path

base = Path('/root/project/20260428-10gold')
price_path = base / 'data_long_history' / 'gold_daily_proxy_gld_yahoo.csv'
out_dir = base / 'backtest_results'
out_dir.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(price_path)
raw['date'] = pd.to_datetime(raw['date'])
raw = raw.sort_values('date').reset_index(drop=True)
raw['close'] = pd.to_numeric(raw['close'], errors='coerce')
raw = raw.dropna(subset=['close'])

close = raw['close']
prev_close_series = close.shift(1)

results = []

for buy_ma in range(2, 81):
    ma_buy = close.rolling(buy_ma).mean()
    prev_ma_buy = ma_buy.shift(1)
    buy_signal = np.logical_and(prev_close_series <= prev_ma_buy, close > ma_buy)

    for sell_ma in range(3, 201):
        if sell_ma <= buy_ma:
            continue

        ma_sell = close.rolling(sell_ma).mean()
        prev_ma_sell = ma_sell.shift(1)
        sell_signal = np.logical_and(prev_close_series >= prev_ma_sell, close < ma_sell)

        position = 0
        entry = None
        equity = 1.0
        prev_close = None
        trades = 0
        wins = 0

        for i in range(len(raw)):
            c = close.iat[i]
            if prev_close is not None and position == 1:
                equity *= c / prev_close

            if position == 0 and bool(buy_signal[i]):
                position = 1
                entry = c
            elif position == 1 and bool(sell_signal[i]):
                ret = c / entry - 1
                trades += 1
                if ret > 0:
                    wins += 1
                position = 0
                entry = None

            prev_close = c

        if position == 1 and entry is not None:
            ret = close.iat[-1] / entry - 1
            trades += 1
            if ret > 0:
                wins += 1

        results.append({
            'buy_ma': buy_ma,
            'sell_ma': sell_ma,
            'total_return_pct': (equity - 1) * 100,
            'trades': trades,
            'win_rate_pct': (wins / trades * 100) if trades else 0,
        })

res = pd.DataFrame(results).sort_values('total_return_pct', ascending=False).reset_index(drop=True)
res_path = out_dir / 'gold_ma_gridsearch_2_80_vs_3_200.csv'
res.to_csv(res_path, index=False)

res_robust = res[res['trades'] >= 30].copy()
robust_top = res_robust.head(20)
robust_path = out_dir / 'gold_ma_gridsearch_top20_trades_ge30.csv'
robust_top.to_csv(robust_path, index=False)

print('rows_tested', len(res))
print('best_overall')
print(res.head(10).to_string(index=False))
print('best_trades_ge30')
print(robust_top.head(10).to_string(index=False))
print('saved', res_path)
print('saved', robust_path)
