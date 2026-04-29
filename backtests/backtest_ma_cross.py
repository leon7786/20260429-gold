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
    # normalize columns
    colmap = {c.lower().strip(): c for c in df.columns}

    date_col = None
    for c in ['date', 'timestamp']:
        if c in colmap:
            date_col = colmap[c]
            break
    if date_col is None:
        raise ValueError(f'No date column in {path}')

    close_col = None
    for c in ['close', 'adj close', 'adj_close']:
        if c in colmap:
            close_col = colmap[c]
            break
    if close_col is None:
        raise ValueError(f'No close column in {path}')

    out = pd.DataFrame({
        'date': pd.to_datetime(df[date_col]),
        'close': pd.to_numeric(df[close_col], errors='coerce')
    }).dropna()
    out = out.sort_values('date').reset_index(drop=True)
    return out


def run_backtest(df: pd.DataFrame):
    d = df.copy()
    d['ma5'] = d['close'].rolling(5).mean()
    d['ma10'] = d['close'].rolling(10).mean()

    d['prev_close'] = d['close'].shift(1)
    d['prev_ma5'] = d['ma5'].shift(1)
    d['prev_ma10'] = d['ma10'].shift(1)

    # signals
    d['buy_signal'] = (d['prev_close'] <= d['prev_ma5']) & (d['close'] > d['ma5'])
    d['sell_signal'] = (d['prev_close'] >= d['prev_ma10']) & (d['close'] < d['ma10'])

    position = 0
    entry_price = None
    entry_date = None
    trades = []
    equity = 1.0

    equity_curve = []
    prev_close = None

    for _, row in d.iterrows():
        date = row['date']
        close = row['close']

        # mark-to-market daily
        if prev_close is not None and position == 1:
            equity *= close / prev_close
        equity_curve.append((date, equity))

        if position == 0 and bool(row['buy_signal']):
            position = 1
            entry_price = close
            entry_date = date
        elif position == 1 and bool(row['sell_signal']):
            exit_price = close
            exit_date = date
            ret = exit_price / entry_price - 1.0
            trades.append({
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'entry_price': float(entry_price),
                'exit_price': float(exit_price),
                'return_pct': float(ret * 100.0),
            })
            position = 0
            entry_price = None
            entry_date = None

        prev_close = close

    # close final open trade at last close
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
    win_rate = wins / n if n else 0.0
    total_return = eq['equity'].iloc[-1] - 1.0
    max_dd = eq['drawdown'].min()

    summary = {
        'start_date': d['date'].iloc[0].strftime('%Y-%m-%d'),
        'end_date': d['date'].iloc[-1].strftime('%Y-%m-%d'),
        'bars': int(len(d)),
        'trades': int(n),
        'wins': int(wins),
        'win_rate_pct': float(win_rate * 100.0),
        'total_return_pct': float(total_return * 100.0),
        'max_drawdown_pct': float(max_dd * 100.0),
    }
    return summary, trades, eq


all_summary = {}
for name, path in ASSETS.items():
    df = load_df(path)
    summary, trades, eq = run_backtest(df)
    all_summary[name] = summary

    pd.DataFrame(trades).to_csv(OUT_DIR / f'{name}_ma5up_ma10down_trades.csv', index=False)
    eq.to_csv(OUT_DIR / f'{name}_ma5up_ma10down_equity.csv', index=False)

summary_df = pd.DataFrame(all_summary).T.reset_index().rename(columns={'index': 'asset'})
summary_path = OUT_DIR / 'ma5up_ma10down_summary.csv'
summary_df.to_csv(summary_path, index=False)

print('saved', summary_path)
print(summary_df.to_string(index=False))
