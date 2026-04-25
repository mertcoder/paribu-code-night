"""
UNSEEN TEST: Train on ALL cnlib data, test on unseen 1-year data
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from strategy import HybridStrategy

# ============================================================
# 1. Load training data (full cnlib)
# ============================================================
train_coins = ['kapcoin-usd_train', 'metucoin-usd_train', 'tamcoin-usd_train']
train_dfs = {}
for coin in train_coins:
    train_dfs[coin] = pd.read_parquet(f'C:/Python311/Lib/site-packages/cnlib/data/{coin}.parquet')

# ============================================================
# 2. Load unseen test data
# ============================================================
test_map = {
    'kapcoin-usd_train': 'data/realistic_test_1y/kapcoin-usd_realistic_test_1y.csv',
    'metucoin-usd_train': 'data/realistic_test_1y/metucoin-usd_realistic_test_1y.csv',
    'tamcoin-usd_train': 'data/realistic_test_1y/tamcoin-usd_realistic_test_1y.csv',
}
test_dfs = {}
for coin, path in test_map.items():
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    test_dfs[coin] = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

print("=== UNSEEN TEST DATA ===")
for coin in train_coins:
    tdf = test_dfs[coin]
    short = coin.split('-')[0]
    print(f"  {short}: {len(tdf)} days, {tdf['Date'].min().date()} to {tdf['Date'].max().date()}, "
          f"Price: {tdf['Close'].iloc[0]:.2f} -> {tdf['Close'].iloc[-1]:.2f} "
          f"({(tdf['Close'].iloc[-1]/tdf['Close'].iloc[0]-1)*100:+.1f}%)")

# Check autocorrelation in test data
print("\n=== TEST DATA AUTOCORRELATION ===")
for coin in train_coins:
    rets = test_dfs[coin]['Close'].pct_change().dropna()
    short = coin.split('-')[0]
    ac1 = rets.autocorr(1)
    ac2 = rets.autocorr(2)
    print(f"  {short}: lag1={ac1:+.4f}, lag2={ac2:+.4f}")

# ============================================================
# 3. Train model on ALL cnlib data
# ============================================================
print("\n=== TRAINING ===")
train_indicators = {}
for coin, df in train_dfs.items():
    c = df['Close'].values.astype(float)
    h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float)
    v = df['Volume'].values.astype(float)
    train_indicators[coin] = HybridStrategy._compute_indicators(c, h, l, v)

X_train, y_train = [], []
for coin in train_coins:
    closes = train_dfs[coin]['Close'].values
    ind = train_indicators[coin]
    for i in range(50, len(closes) - 1):
        feat = HybridStrategy._feature_vector_from_cache(ind, i)
        if feat is not None:
            X_train.append(feat)
            next_ret = (closes[i+1] - closes[i]) / closes[i]
            y_train.append(1 if next_ret > 0 else 0)

model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
model.fit(np.array(X_train), np.array(y_train))
print(f"  Model trained on {len(X_train)} samples (all cnlib train data)")

# ============================================================
# 4. Concatenate train+test data for rolling indicator computation
# ============================================================
# We need history before test period for indicator computation
combined_dfs = {}
for coin in train_coins:
    combined = pd.concat([train_dfs[coin], test_dfs[coin]], ignore_index=True)
    combined_dfs[coin] = combined

# Pre-compute indicators on combined data
combined_indicators = {}
for coin, df in combined_dfs.items():
    c = df['Close'].values.astype(float)
    h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float)
    v = df['Volume'].values.astype(float)
    combined_indicators[coin] = HybridStrategy._compute_indicators(c, h, l, v)

train_len = len(train_dfs[train_coins[0]])  # 1570
test_len = len(test_dfs[train_coins[0]])    # 365
total_len = train_len + test_len            # 1935
print(f"  Train: {train_len}, Test: {test_len}, Combined: {total_len}")

# ============================================================
# 5. Run strategy on unseen test data
# ============================================================
print("\n" + "=" * 70)
print("UNSEEN TEST RESULTS")
print("=" * 70)

for max_lev in [1, 2, 3, 5]:
    port = 3000.0
    trades = 0
    skips = 0
    liqs = 0
    wins = 0
    losses = 0
    trade_rets = []

    for i in range(train_len, total_len - 1):
        best_coin = None
        best_score = -999
        best_dir = 0
        best_lev = 1

        for coin in train_coins:
            ind = combined_indicators[coin]
            closes = combined_dfs[coin]['Close'].values
            
            if i < 5:
                continue
            past_ret = (closes[i] - closes[i-5]) / closes[i-5]
            mom_dir = 1 if past_ret > 0 else -1

            feat = HybridStrategy._feature_vector_from_cache(ind, i)
            if feat is None:
                continue
            proba = model.predict_proba([feat])[0]
            ml_dir = 1 if proba[1] > 0.5 else -1
            ml_conf = max(proba) - 0.5

            if mom_dir == ml_dir:
                score = abs(past_ret) * (1 + ml_conf * 2)
            else:
                score = -1

            if score > best_score:
                best_score = score
                best_coin = coin
                best_dir = mom_dir
                if score > 0.05:
                    best_lev = min(5, max_lev)
                elif score > 0.02:
                    best_lev = min(3, max_lev)
                else:
                    best_lev = min(2, max_lev)

        if best_coin and best_score > 0:
            closes = combined_dfs[best_coin]['Close'].values
            highs = combined_dfs[best_coin]['High'].values
            lows = combined_dfs[best_coin]['Low'].values
            entry = closes[i]
            exit_p = closes[i+1]

            # Liquidation check
            if best_dir == 1:
                liq_price = entry * (1 - 1/best_lev)
                if lows[i+1] <= liq_price:
                    liqs += 1
                    port = 0
                    break
            else:
                liq_price = entry * (1 + 1/best_lev)
                if highs[i+1] >= liq_price:
                    liqs += 1
                    port = 0
                    break

            base_ret = best_dir * (exit_p - entry) / entry
            lev_ret = base_ret * best_lev
            port *= (1 + lev_ret)
            trades += 1
            trade_rets.append(base_ret)
            if base_ret > 0:
                wins += 1
            else:
                losses += 1
        else:
            skips += 1

    final_ret = (port / 3000 - 1) * 100
    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    avg_ret = np.mean(trade_rets) * 100 if trade_rets else 0
    
    print(f"\n  Max Leverage = {max_lev}x:")
    print(f"    Starting Capital  : $       3,000.00")
    print(f"    Final Portfolio   : ${port:>20,.2f}")
    print(f"    Return            :  {final_ret:>+18,.2f}%")
    print(f"    Trades            : {trades:>8}")
    print(f"    Wins              : {wins:>8} ({wr:.1f}%)")
    print(f"    Losses            : {losses:>8}")
    print(f"    Skips             : {skips:>8}")
    print(f"    Liquidations      : {liqs:>8}")
    print(f"    Avg trade (1x)    :  {avg_ret:>+8.4f}%")

# ============================================================
# 6. Benchmarks on unseen data
# ============================================================
print("\n" + "=" * 70)
print("BENCHMARKS (unseen test data)")
print("=" * 70)

# Buy & hold
for coin in train_coins:
    tdf = test_dfs[coin]
    closes = tdf['Close'].values
    short = coin.split('-')[0]
    bh_ret = (closes[-1] / closes[0] - 1) * 100
    print(f"  {short} Buy&Hold 1x: {bh_ret:+.1f}%")

# Random signal
print()
np.random.seed(42)
for trial in range(3):
    port = 3000.0
    for i in range(train_len, total_len - 1):
        coin = train_coins[np.random.randint(3)]
        direction = np.random.choice([1, -1])
        closes = combined_dfs[coin]['Close'].values
        entry = closes[i]
        exit_p = closes[i+1]
        ret = direction * 3 * (exit_p - entry) / entry
        port *= (1 + ret)
        if port <= 0:
            port = 0
            break
    print(f"  Random 3x trial {trial+1}: ${port:>12,.0f} ({(port/3000-1)*100:>+10.1f}%)")

# Pure momentum (no ML)
print()
for lev in [1, 3, 5]:
    port = 3000.0
    for i in range(train_len, total_len - 1):
        best_coin = None
        best_mom = -999
        best_dir = 0
        for coin in train_coins:
            closes = combined_dfs[coin]['Close'].values
            if i < 5: continue
            past_ret = (closes[i] - closes[i-5]) / closes[i-5]
            if abs(past_ret) > best_mom:
                best_mom = abs(past_ret)
                best_coin = coin
                best_dir = 1 if past_ret > 0 else -1
        
        if best_coin:
            closes = combined_dfs[best_coin]['Close'].values
            highs = combined_dfs[best_coin]['High'].values
            lows = combined_dfs[best_coin]['Low'].values
            entry = closes[i]
            exit_p = closes[i+1]
            if best_dir == 1:
                if lows[i+1] <= entry * (1 - 1/lev):
                    port = 0; break
            else:
                if highs[i+1] >= entry * (1 + 1/lev):
                    port = 0; break
            ret = best_dir * lev * (exit_p - entry) / entry
            port *= (1 + ret)
            if port <= 0:
                port = 0; break
    
    print(f"  Pure Momentum(5d) {lev}x: ${port:>15,.0f} ({(port/3000-1)*100:>+12.1f}%)")
