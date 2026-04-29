# 20260429-gold

Gold + multi-asset strategy research workspace (MA variants, dual-side, realistic T+1 backtests, and dashboard tooling).

## Repository Structure
- `*.py` : strategy / backtest / data scripts
- `docs/files/*.md` : README for each Python file
- `docs/` : GitHub Pages site (filter by asset + strategy)

## Quick Start
```bash
python gridsearch_btc_dual_side_realistic_1_700_t1.py
python strategy_dashboard_30428.py
```

## File-level README Index
- `backtest_ma_9_10.py` (backtest) → [docs/files/backtest_ma_9_10.md](docs/files/backtest_ma_9_10.md)
- `backtest_ma_cross.py` (backtest) → [docs/files/backtest_ma_cross.md](docs/files/backtest_ma_cross.md)
- `cross_test_19x19_strategies_assets.py` (backtest) → [docs/files/cross_test_19x19_strategies_assets.md](docs/files/cross_test_19x19_strategies_assets.md)
- `download_daily_data.py` (data) → [docs/files/download_daily_data.md](docs/files/download_daily_data.md)
- `download_extra_assets_1990.py` (data) → [docs/files/download_extra_assets_1990.md](docs/files/download_extra_assets_1990.md)
- `fetch_alt_sources.py` (data) → [docs/files/fetch_alt_sources.md](docs/files/fetch_alt_sources.md)
- `fetch_btc_gold_long_history.py` (data) → [docs/files/fetch_btc_gold_long_history.md](docs/files/fetch_btc_gold_long_history.md)
- `fetch_earliest_multisource.py` (data) → [docs/files/fetch_earliest_multisource.md](docs/files/fetch_earliest_multisource.md)
- `fetch_missing_assets_alt_sources.py` (data) → [docs/files/fetch_missing_assets_alt_sources.md](docs/files/fetch_missing_assets_alt_sources.md)
- `fetch_missing_assets_no_tushare_stooq.py` (data) → [docs/files/fetch_missing_assets_no_tushare_stooq.md](docs/files/fetch_missing_assets_no_tushare_stooq.md)
- `gridsearch_all_assets_ma_2_700.py` (backtest) → [docs/files/gridsearch_all_assets_ma_2_700.md](docs/files/gridsearch_all_assets_ma_2_700.md)
- `gridsearch_btc_dual_side_realistic_1_700_t1.py` (backtest) → [docs/files/gridsearch_btc_dual_side_realistic_1_700_t1.md](docs/files/gridsearch_btc_dual_side_realistic_1_700_t1.md)
- `gridsearch_commodities_dual_side_ma_1_700_t1.py` (backtest) → [docs/files/gridsearch_commodities_dual_side_ma_1_700_t1.md](docs/files/gridsearch_commodities_dual_side_ma_1_700_t1.md)
- `gridsearch_commodities_dual_side_ma_2_700.py` (backtest) → [docs/files/gridsearch_commodities_dual_side_ma_2_700.md](docs/files/gridsearch_commodities_dual_side_ma_2_700.md)
- `gridsearch_gold_ma.py` (backtest) → [docs/files/gridsearch_gold_ma.md](docs/files/gridsearch_gold_ma.md)
- `gridsearch_gold_ma_1_2000_vs_1_2000.py` (backtest) → [docs/files/gridsearch_gold_ma_1_2000_vs_1_2000.md](docs/files/gridsearch_gold_ma_1_2000_vs_1_2000.md)
- `gridsearch_gold_ma_1_400_vs_1_400.py` (backtest) → [docs/files/gridsearch_gold_ma_1_400_vs_1_400.md](docs/files/gridsearch_gold_ma_1_400_vs_1_400.md)
- `gridsearch_gold_ma_fast.py` (backtest) → [docs/files/gridsearch_gold_ma_fast.md](docs/files/gridsearch_gold_ma_fast.md)
- `gridsearch_gold_ma_ultra.py` (backtest) → [docs/files/gridsearch_gold_ma_ultra.md](docs/files/gridsearch_gold_ma_ultra.md)
- `gridsearch_oil_ma_1_2000_vs_1_2000.py` (backtest) → [docs/files/gridsearch_oil_ma_1_2000_vs_1_2000.md](docs/files/gridsearch_oil_ma_1_2000_vs_1_2000.md)
- `gridsearch_short_only_all_assets_ma_2_700.py` (backtest) → [docs/files/gridsearch_short_only_all_assets_ma_2_700.md](docs/files/gridsearch_short_only_all_assets_ma_2_700.md)
- `gridsearch_single_asset_ma_2_700.py` (backtest) → [docs/files/gridsearch_single_asset_ma_2_700.md](docs/files/gridsearch_single_asset_ma_2_700.md)
- `gridsearch_top10_ma_1_700.py` (backtest) → [docs/files/gridsearch_top10_ma_1_700.md](docs/files/gridsearch_top10_ma_1_700.md)
- `hunt_btc_better_than_sym139_realistic.py` (backtest) → [docs/files/hunt_btc_better_than_sym139_realistic.md](docs/files/hunt_btc_better_than_sym139_realistic.md)
- `hunt_btc_better_v2_realistic.py` (backtest) → [docs/files/hunt_btc_better_v2_realistic.md](docs/files/hunt_btc_better_v2_realistic.md)
- `hunt_btc_better_v3_realistic.py` (backtest) → [docs/files/hunt_btc_better_v3_realistic.md](docs/files/hunt_btc_better_v3_realistic.md)
- `hunt_btc_sym_trend_fast1_20_slow30_100_trend40_200.py` (backtest) → [docs/files/hunt_btc_sym_trend_fast1_20_slow30_100_trend40_200.md](docs/files/hunt_btc_sym_trend_fast1_20_slow30_100_trend40_200.md)
- `search_btc_multi_ma_long_short_realistic_t1.py` (backtest) → [docs/files/search_btc_multi_ma_long_short_realistic_t1.md](docs/files/search_btc_multi_ma_long_short_realistic_t1.md)
- `strategy_dashboard_30428.py` (strategy) → [docs/files/strategy_dashboard_30428.md](docs/files/strategy_dashboard_30428.md)

## Notes
- Large generated artifacts are ignored by default (`backtest_results/`, big datasets).
- GitHub Pages reads compact snapshots in `docs/data/backtests.json`.
