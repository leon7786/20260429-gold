import pandas as pd
from pathlib import Path

BASE = Path('/root/project/20260428-10gold')
OUT_DIR = BASE / 'backtest_results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = {
    'gold': BASE / 'data_long_history' / 'gold_daily_proxy_gld_yahoo.csv',
    'oil': BASE / 'data' / 'brent.csv',
}


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    colmap = {c.lower().strip(): c for c in df.columns}

    date_col = colmap.get('date') or colmap.get('timestamp') or 'Date'
    close_col = colmap.get('close') or colmap.get('adj close') or colmap.get('adj_close') or 'Close'

    out = pd.DataFrame({
        'date': pd.to_datetime(df[date_col], errors='coerce'),
        'close': pd.to_numeric(df[close_col], errors='coerce'),
    }).dropna()
    out = out.sort_values('date').reset_index(drop=True)
    return out


def run_backtest(df: pd.DataFrame):
    d = df.copy()
    d['ma9'] = d['close'].rolling(9).mean()
    d['ma10'] = d['close'].rolling(10).mean()

    d['prev_close'] = d['close'].shift(1)
    d['prev_ma9'] = d['ma9'].shift(1)
    d['prev_ma10'] = d['ma10'].shift(1)

    d['buy_signal'] = (d['prev_close'] <= d['prev_ma9']) & (d['close'] > d['ma9'])
    d['sell_signal'] = (d['prev_close'] >= d['prev_ma10']) & (d['close'] < d['ma10'])

    position = 0
    entry_price = None
    entry_date = None
    trades = []
    equity = 1.0
    prev_close = None
    equity_curve = []

    for _, row in d.iterrows():
        date = row['date']
        close = row['close']

        if prev_close is not None and position == 1:
            equity *= close / prev_close
        equity_curve.append((date, equity))

        if position == 0 and bool(row['buy_signal']):
            position = 1
            entry_price = close
            entry_date = date
        elif position == 1 and bool(row['sell_signal']):
            ret = close / entry_price - 1.0
            trades.append({
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'exit_date': date.strftime('%Y-%m-%d'),
                'entry_price': float(entry_price),
                'exit_price': float(close),
                'return_pct': float(ret * 100.0),
            })
            position = 0
            entry_price = None
            entry_date = None

        prev_close = close

    if position == 1 and entry_price is not None:
        last = d.iloc[-1]
        ret = last['close'] / entry_price - 1.0
        trades.append({
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'exit_date': last['date'].strftime('%Y-%m-%d'),
            'entry_price': float(entry_price),
            'exit_price': float(last['close']),
            'return_pct': float(ret * 100.0),
        })

    eq = pd.DataFrame(equity_curve, columns=['date', 'equity'])
    eq['cum_max'] = eq['equity'].cummax()
    eq['drawdown'] = eq['equity'] / eq['cum_max'] - 1.0

    n = len(trades)
    wins = sum(1 for t in trades if t['return_pct'] > 0)

    summary = {
        'start_date': d['date'].iloc[0].strftime('%Y-%m-%d'),
        'end_date': d['date'].iloc[-1].strftime('%Y-%m-%d'),
        'bars': int(len(d)),
        'trades': int(n),
        'wins': int(wins),
        'win_rate_pct': float((wins / n) * 100.0 if n else 0.0),
        'total_return_pct': float((eq['equity'].iloc[-1] - 1.0) * 100.0),
        'max_drawdown_pct': float(eq['drawdown'].min() * 100.0),
    }
    return summary, trades, eq


all_summary = {}
for name, path in ASSETS.items():
    df = load_df(path)
    summary, trades, eq = run_backtest(df)
    all_summary[name] = summary

    pd.DataFrame(trades).to_csv(OUT_DIR / f'{name}_ma9up_ma10down_trades.csv', index=False)
    eq.to_csv(OUT_DIR / f'{name}_ma9up_ma10down_equity.csv', index=False)

summary_df = pd.DataFrame(all_summary).T.reset_index().rename(columns={'index': 'asset'})
summary_path = OUT_DIR / 'ma9up_ma10down_summary.csv'
summary_df.to_csv(summary_path, index=False)

# compare buy&hold
rows = []
for name, path in ASSETS.items():
    raw = pd.read_csv(path)
    cm = {c.lower(): c for c in raw.columns}
    dc = cm.get('date') or cm.get('timestamp') or 'Date'
    cc = cm.get('close') or cm.get('adj close') or cm.get('adj_close') or 'Close'
    d = pd.to_datetime(raw[dc], errors='coerce')
    c = pd.to_numeric(raw[cc], errors='coerce')
    m = d.notna() & c.notna()
    d = d[m]
    c = c[m]
    rows.append({
        'asset': name,
        'buy_hold_return_pct': float((c.iloc[-1] / c.iloc[0] - 1.0) * 100.0),
    })

bh = pd.DataFrame(rows)
cmp_df = summary_df.merge(bh, on='asset', how='left')
cmp_df['excess_vs_buy_hold_pct'] = cmp_df['total_return_pct'] - cmp_df['buy_hold_return_pct']
cmp_path = OUT_DIR / 'ma9up_ma10down_vs_buyhold.csv'
cmp_df.to_csv(cmp_path, index=False)

print('saved', summary_path)
print(summary_df.to_string(index=False))
print('\ncompare_vs_buy_hold')
print(cmp_df[['asset', 'total_return_pct', 'buy_hold_return_pct', 'excess_vs_buy_hold_pct']].to_string(index=False))
print('saved', cmp_path)
