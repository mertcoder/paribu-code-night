# Realistic Test 1Y

This folder contains the third 365-day OHLCV test set for the artificial CNLIB
coins.

Files:
- `kapcoin-usd_realistic_test_1y.csv`
- `metucoin-usd_realistic_test_1y.csv`
- `tamcoin-usd_realistic_test_1y.csv`

Generation target:
- Date range: `2027-03-16` to `2028-03-14`
- Split: `realistic_test`
- Source: `realistic_random_walk`
- Lag-1 return autocorrelation: `0.00` to `0.05`
- Daily return volatility: `2%` to `4%`
- Pairwise coin return correlation: `0.3` to `0.6`
- Mean daily return: `0.01%` to `0.05%`
- Average daily `(High - Low) / Close`: `3%` to `5%`
- Volume: random between `5B` and `30B`

Regenerate:

```powershell
python scripts/generate_realistic_test_data.py
```
