import pandas as pd
from pathlib import Path

BASE = Path('/root/project/20260428-10gold')
RES = BASE / 'backtest_results'

assets = {
    'gold_spot_proxy': BASE / 'data_long_history' / 'gold_daily_proxy_gld_yahoo.csv',
    'brent': BASE / 'data' / 'brent.csv',
    'btc': BASE / 'data' / 'btc.csv',
    'copper_core': BASE / 'data' / 'copper.csv',
    'natgas_core': BASE / 'data' / 'natgas.csv',
    'eurusd': BASE / 'data' / 'eurusd.csv',
    'usdjpy': BASE / 'data' / 'usdjpy.csv',
    'sp500': BASE / 'data' / 'sp500.csv',
    'us10y_yield': BASE / 'data' / 'us10y_yield.csv',
    'usd_dxy': BASE / 'data' / 'usd_dxy.csv',
    'nasdaq100': BASE / 'data_extra_1990' / 'nasdaq100.csv',
    'eth': BASE / 'data_extra_1990' / 'eth.csv',
    'ttf_gas_europe': BASE / 'data_extra_1990' / 'ttf_gas_europe.csv',
    'silver': BASE / 'data_extra_1990' / 'silver.csv',
    'msci_world_proxy': BASE / 'data_extra_1990' / 'msci_world_proxy.csv',
    'msci_em_proxy': BASE / 'data_extra_1990' / 'msci_em_proxy.csv',
    'chinext_proxy_etf_159915': BASE / 'data_extra_1990' / 'chinext_proxy_etf_159915.csv',
    'nikkei225': BASE / 'data_extra_1990' / 'nikkei225.csv',
    'topix_proxy_etf': BASE / 'data_extra_1990' / 'topix_proxy_etf.csv',
}


def load_close(path: Path):
    df = pd.read_csv(path)
    colmap = {c.lower().strip(): c for c in df.columns}
    date_col = colmap.get('date') or colmap.get('timestamp') or 'Date'
    close_col = colmap.get('close') or colmap.get('adj close') or colmap.get('adj_close') or 'Close'
    out = pd.DataFrame({
        'date': pd.to_datetime(df[date_col], errors='coerce'),
        'close': pd.to_numeric(df[close_col], errors='coerce')
    }).dropna().sort_values('date').reset_index(drop=True)
    return out


def run_strategy(close: pd.Series, buy_ma: int, sell_ma: int):
    c = close.reset_index(drop=True)
    ma_buy = c.rolling(buy_ma).mean()
    ma_sell = c.rolling(sell_ma).mean()
    buy = (c.shift(1) <= ma_buy.shift(1)) & (c > ma_buy)
    sell = (c.shift(1) >= ma_sell.shift(1)) & (c < ma_sell)

    pos = 0
    entry = None
    eq = 1.0
    prev = None
    trades = 0
    wins = 0

    for i in range(len(c)):
        px = float(c.iat[i])
        if prev is not None and pos == 1:
            eq *= px / prev

        if pos == 0 and bool(buy.iat[i]):
            pos = 1
            entry = px
        elif pos == 1 and bool(sell.iat[i]):
            r = px / entry - 1.0
            trades += 1
            if r > 0:
                wins += 1
            pos = 0
            entry = None
        prev = px

    if pos == 1 and entry is not None:
        r = float(c.iat[-1]) / entry - 1.0
        trades += 1
        if r > 0:
            wins += 1

    return {
        'total_return_pct': (eq - 1.0) * 100.0,
        'trades': trades,
        'win_rate_pct': (wins / trades * 100.0) if trades else 0.0,
    }


def main():
    strat_df = pd.read_csv(RES / 'best_ma_strategies_19_assets_2_700.csv')
    strat_df = strat_df[['asset', 'best_buy_ma', 'best_sell_ma']].copy()

    # preload asset closes
    closes = {a: load_close(p)['close'] for a, p in assets.items()}

    rows = []
    for _, s in strat_df.iterrows():
        strat_name = s['asset']
        buy_ma = int(s['best_buy_ma'])
        sell_ma = int(s['best_sell_ma'])

        for asset_name, close in closes.items():
            res = run_strategy(close, buy_ma, sell_ma)
            rows.append({
                'strategy_from': strat_name,
                'buy_ma': buy_ma,
                'sell_ma': sell_ma,
                'asset_tested': asset_name,
                **res
            })

    out = pd.DataFrame(rows)
    out_file = RES / 'cross_test_19_strategies_x_19_assets.csv'
    out.to_csv(out_file, index=False)

    pivot = out.pivot(index='strategy_from', columns='asset_tested', values='total_return_pct')
    pivot_file = RES / 'cross_test_19x19_return_matrix.csv'
    pivot.to_csv(pivot_file)

    # rank by average return across all tested assets
    rank = out.groupby(['strategy_from', 'buy_ma', 'sell_ma'], as_index=False)['total_return_pct'].mean()
    rank = rank.sort_values('total_return_pct', ascending=False).rename(columns={'total_return_pct': 'avg_return_pct_across_19_assets'})
    rank_file = RES / 'cross_test_19x19_strategy_rank_by_avg_return.csv'
    rank.to_csv(rank_file, index=False)

    print('saved', out_file)
    print('saved', pivot_file)
    print('saved', rank_file)
    print('\nTop 10 strategies by avg return across all 19 assets:')
    print(rank.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
