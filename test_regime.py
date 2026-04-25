"""
Test enhanced regime-adaptive strategy on ALL 3 datasets
v2: 100-candle autocorr + negative autocorr + volatility filter
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from strategy import HybridStrategy

# Load training data
train_coins = ['kapcoin-usd_train', 'metucoin-usd_train', 'tamcoin-usd_train']
train_dfs = {}
for coin in train_coins:
    train_dfs[coin] = pd.read_parquet(f'C:/Python311/Lib/site-packages/cnlib/data/{coin}.parquet')

# Train model
indicators = {}
for coin, df in train_dfs.items():
    c = df['Close'].values.astype(float)
    h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float)
    v = df['Volume'].values.astype(float)
    indicators[coin] = HybridStrategy._compute_indicators(c, h, l, v)

X_train, y_train = [], []
for coin in train_coins:
    closes = train_dfs[coin]['Close'].values
    ind = indicators[coin]
    for i in range(50, len(closes) - 1):
        feat = HybridStrategy._feature_vector_from_cache(ind, i)
        if feat is not None:
            X_train.append(feat)
            next_ret = (closes[i+1] - closes[i]) / closes[i]
            y_train.append(1 if next_ret > 0 else 0)

model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
model.fit(np.array(X_train), np.array(y_train))
print(f"Model trained on {len(X_train)} samples\n")

# Test datasets
test_sets = {
    "Synthetic (unseen_test_1y)": {
        'kapcoin-usd_train': 'data/unseen_test_1y/kapcoin-usd_unseen_test_1y.csv',
        'metucoin-usd_train': 'data/unseen_test_1y/metucoin-usd_unseen_test_1y.csv',
        'tamcoin-usd_train': 'data/unseen_test_1y/tamcoin-usd_unseen_test_1y.csv',
    },
    "Regime Mix (bear+bull)": {
        'kapcoin-usd_train': 'data/unseen_test_1y_regime_mix/kapcoin-usd_unseen_test_1y_regime_mix.csv',
        'metucoin-usd_train': 'data/unseen_test_1y_regime_mix/metucoin-usd_unseen_test_1y_regime_mix.csv',
        'tamcoin-usd_train': 'data/unseen_test_1y_regime_mix/tamcoin-usd_unseen_test_1y_regime_mix.csv',
    },
    "Realistic (random walk)": {
        'kapcoin-usd_train': 'data/realistic_test_1y/kapcoin-usd_realistic_test_1y.csv',
        'metucoin-usd_train': 'data/realistic_test_1y/metucoin-usd_realistic_test_1y.csv',
        'tamcoin-usd_train': 'data/realistic_test_1y/tamcoin-usd_realistic_test_1y.csv',
    },
}

train_len = len(train_dfs[train_coins[0]])

def run_test(test_name, test_paths):
    test_dfs = {}
    for coin, path in test_paths.items():
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
        test_dfs[coin] = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    combined = {}
    combined_ind = {}
    for coin in train_coins:
        combined[coin] = pd.concat([train_dfs[coin], test_dfs[coin]], ignore_index=True)
        c = combined[coin]['Close'].values.astype(float)
        h = combined[coin]['High'].values.astype(float)
        l = combined[coin]['Low'].values.astype(float)
        v = combined[coin]['Volume'].values.astype(float)
        combined_ind[coin] = HybridStrategy._compute_indicators(c, h, l, v)

    total_len = len(combined[train_coins[0]])

    # Test data autocorrelation
    test_acs = []
    for coin in train_coins:
        rets = test_dfs[coin]['Close'].pct_change().dropna()
        test_acs.append(rets.autocorr(1))
    avg_test_ac = np.mean(test_acs)

    port = 3000.0
    trades = 0; wins = 0; losses = 0; skips = 0; liqs = 0
    regime_log = []

    for i in range(train_len, total_len - 1):
        # --- Regime detection (100-candle window) ---
        autocorrs = []
        volatilities = []
        for coin in train_coins:
            closes = combined[coin]['Close'].values
            if i >= 102:
                window = closes[i-101:i+1]
                rets = np.diff(window) / window[:-1]
                r1 = rets[:-1] - np.mean(rets[:-1])
                r2 = rets[1:] - np.mean(rets[1:])
                denom = np.sqrt(np.sum(r1**2) * np.sum(r2**2))
                if denom > 0:
                    autocorrs.append(np.sum(r1 * r2) / denom)
            if i >= 31:
                recent_rets = np.diff(closes[i-30:i+1]) / closes[i-30:i]
                volatilities.append(np.std(recent_rets))

        avg_ac = np.mean(autocorrs) if autocorrs else 0.0
        avg_vol = np.mean(volatilities) if volatilities else 0.02

        # Regime
        if avg_ac < -0.05:
            max_lev, alloc, min_thresh = 1, 0.1, 0.08
        elif avg_ac < 0.05:
            max_lev, alloc, min_thresh = 1, 0.15, 0.03
        elif avg_ac < 0.15:
            max_lev, alloc, min_thresh = 2, 0.4, 0.02
        elif avg_ac < 0.30:
            max_lev, alloc, min_thresh = 3, 0.7, 0.01
        else:
            max_lev, alloc, min_thresh = 5, 1.0, 0.0

        # Volatility filter
        vol_baseline = 0.025
        if avg_vol > vol_baseline * 2.5:
            vol_scale = 0.3
        elif avg_vol > vol_baseline * 1.5:
            vol_scale = 0.6
        else:
            vol_scale = 1.0

        final_alloc = alloc * vol_scale
        regime_log.append((avg_ac, max_lev, final_alloc, avg_vol))

        # Find best coin
        best_coin = None; best_score = -999; best_dir = 0; best_lev = 1
        for coin in train_coins:
            closes = combined[coin]['Close'].values
            ind = combined_ind[coin]
            if i < 5: continue
            past_ret = (closes[i] - closes[i-5]) / closes[i-5]
            mom_dir = 1 if past_ret > 0 else -1
            feat = HybridStrategy._feature_vector_from_cache(ind, i)
            if feat is None: continue
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
                if score > 0.05: best_lev = min(5, max_lev)
                elif score > 0.02: best_lev = min(3, max_lev)
                else: best_lev = min(2, max_lev)

        if best_coin and best_score > min_thresh:
            closes = combined[best_coin]['Close'].values
            highs = combined[best_coin]['High'].values
            lows = combined[best_coin]['Low'].values
            entry = closes[i]; exit_p = closes[i+1]
            # Liq check
            if best_dir == 1:
                if lows[i+1] <= entry * (1 - 1/best_lev):
                    liqs += 1; port = 0; break
            else:
                if highs[i+1] >= entry * (1 + 1/best_lev):
                    liqs += 1; port = 0; break
            base_ret = best_dir * (exit_p - entry) / entry
            lev_ret = base_ret * best_lev * final_alloc
            port *= (1 + lev_ret)
            trades += 1
            if base_ret > 0: wins += 1
            else: losses += 1
        else:
            skips += 1

    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    ret = (port / 3000 - 1) * 100
    avg_lev = np.mean([r[1] for r in regime_log])
    avg_alloc = np.mean([r[2] for r in regime_log])
    avg_ac_live = np.mean([r[0] for r in regime_log])

    print(f"  {test_name:30s} | DataAC={avg_test_ac:.3f} LiveAC={avg_ac_live:.3f} | "
          f"WR={wr:.0f}% | Tr={trades:>3} Sk={skips:>3} Lq={liqs} | "
          f"Lev={avg_lev:.1f} Alloc={avg_alloc:.2f} | "
          f"Final=${port:>15,.0f} | Ret={ret:>+15,.1f}%")

print("=" * 170)
print("ENHANCED REGIME-ADAPTIVE STRATEGY (100g AC + neg AC + vol filter)")
print("=" * 170)
for name, paths in test_sets.items():
    run_test(name, paths)
