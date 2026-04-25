# Unseen 1Y Test Data

This folder stores one generated 365-day OHLCV test set per CNLIB coin.

Files:
- `kapcoin-usd_unseen_test_1y.csv`
- `metucoin-usd_unseen_test_1y.csv`
- `tamcoin-usd_unseen_test_1y.csv`

The rows start on the day after the last CNLIB training candle. They are
deterministically generated from each coin's historical CNLIB training data, so
AI strategies can be evaluated on a separate synthetic year that was not present
in the original package data.

Columns:
- `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- `Split`: always `unseen_test_1y`
- `Source`: generation source marker

Regenerate:

```powershell
python scripts/generate_unseen_test_data.py
```
