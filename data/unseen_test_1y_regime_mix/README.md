# Unseen 1Y Test Data - Regime Mix

This folder stores the second generated 365-day OHLCV test set per CNLIB coin.

Files:
- `kapcoin-usd_unseen_test_1y_regime_mix.csv`
- `metucoin-usd_unseen_test_1y_regime_mix.csv`
- `tamcoin-usd_unseen_test_1y_regime_mix.csv`

Generation strategy:
- Reads the original four-year CNLIB train parquet files from `data/cnlib_train`.
- Samples contiguous historical market-regime blocks from the training period.
- Applies small bounded trend, volatility, and volume shifts inside historical
  quantile limits.
- Starts on the day after the last training candle, matching the first test set
  date range.

This gives models a second unseen year that still respects the original market
shape, but is less smooth than the baseline synthetic test set.

Regenerate:

```powershell
python scripts/generate_unseen_test_data.py
```
