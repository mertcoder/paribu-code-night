"""
Robustness test: different train/test splits
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

coins = ['kapcoin-usd_train', 'metucoin-usd_train', 'tamcoin-usd_train']
dfs = {}
for coin in coins:
    df = pd.read_parquet(f'C:/Python311/Lib/site-packages/cnlib/data/{coin}.parquet')
    dfs[coin] = df

n = len(dfs[coins[0]])  # 1570
from strategy import HybridStrategy

# Pre-compute indicators once
indicators = {}
for coin, df in dfs.items():
    c = df['Close'].values.astype(float)
    h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float)
    v = df['Volume'].values.astype(float)
    indicators[coin] = HybridStrategy._compute_indicators(c, h, l, v)

def run_split(train_start, train_end, test_start, test_end, label):
    # Build train set
    X_train, y_train = [], []
    for coin in coins:
        closes = dfs[coin]['Close'].values
        ind = indicators[coin]
        for i in range(max(50, train_start), train_end):
            feat = HybridStrategy._feature_vector_from_cache(ind, i)
            if feat is not None:
                next_ret = (closes[i+1] - closes[i]) / closes[i]
                X_train.append(feat)
                y_train.append(1 if next_ret > 0 else 0)
    
    model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(np.array(X_train), np.array(y_train))
    
    # Test
    port = 3000.0
    trades = 0
    skips = 0
    liqs = 0
    
    for i in range(max(50, test_start), test_end):
        best_coin = None
        best_score = -999
        best_dir = 0
        best_lev = 1
        
        for coin in coins:
            closes = dfs[coin]['Close'].values
            ind = indicators[coin]
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
                    best_lev = 5
                elif score > 0.02:
                    best_lev = 3
                else:
                    best_lev = 2
        
        if best_coin and best_score > 0:
            closes = dfs[best_coin]['Close'].values
            highs = dfs[best_coin]['High'].values
            lows = dfs[best_coin]['Low'].values
            entry = closes[i]
            exit_p = closes[i+1]
            
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
            
            ret = best_dir * best_lev * (exit_p - entry) / entry
            port *= (1 + ret)
            trades += 1
        else:
            skips += 1
    
    test_days = test_end - max(50, test_start)
    ret_pct = (port / 3000 - 1) * 100
    print(f"  {label:45s} | Train={len(X_train):>5} | TestDays={test_days:>4} | Trades={trades:>4} | Skips={skips:>3} | Liqs={liqs} | Final=${port:>20,.0f} | Return={ret_pct:>+18,.1f}%")
    return port, liqs

print("=" * 180)
print("ROBUSTNESS TEST: Different Train/Test Splits")
print("=" * 180)

# --- 1. Percentage-based splits ---
print("\n--- Percentage Splits (train first X%, test rest) ---")
for train_pct in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
    split = int(n * train_pct)
    label = f"Train 0-{split} ({train_pct:.0%}), Test {split}-{n-1} ({1-train_pct:.0%})"
    run_split(0, split, split, n - 1, label)

# --- 2. Reversed: train on LATER data, test on EARLIER ---
print("\n--- Reversed Splits (train later, test earlier) ---")
for train_pct in [0.50, 0.60, 0.70, 0.75]:
    split = int(n * (1 - train_pct))
    label = f"Train {split}-{n-1} (last {train_pct:.0%}), Test 0-{split} (first {1-train_pct:.0%})"
    run_split(split, n - 1, 0, split, label)

# --- 3. Middle-out: train on edges, test on middle ---
print("\n--- Middle-out (train edges, test middle) ---")
for edge_pct in [0.25, 0.30, 0.35]:
    edge = int(n * edge_pct)
    mid_start = edge
    mid_end = n - edge
    label = f"Train 0-{edge}+{n-edge}-{n} (edges {edge_pct:.0%}), Test {mid_start}-{mid_end} (middle)"
    
    # Build combined train
    X_train, y_train = [], []
    for coin in coins:
        closes = dfs[coin]['Close'].values
        ind = indicators[coin]
        for i in list(range(50, edge)) + list(range(n - edge, n - 1)):
            feat = HybridStrategy._feature_vector_from_cache(ind, i)
            if feat is not None:
                next_ret = (closes[i+1] - closes[i]) / closes[i]
                X_train.append(feat)
                y_train.append(1 if next_ret > 0 else 0)
    
    model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(np.array(X_train), np.array(y_train))
    
    port = 3000.0
    trades = 0
    skips = 0
    liqs = 0
    for i in range(mid_start, mid_end - 1):
        best_coin = None
        best_score = -999
        best_dir = 0
        best_lev = 1
        for coin in coins:
            closes = dfs[coin]['Close'].values
            ind = indicators[coin]
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
                if score > 0.05: best_lev = 5
                elif score > 0.02: best_lev = 3
                else: best_lev = 2
        
        if best_coin and best_score > 0:
            closes = dfs[best_coin]['Close'].values
            highs = dfs[best_coin]['High'].values
            lows = dfs[best_coin]['Low'].values
            entry = closes[i]
            exit_p = closes[i+1]
            if best_dir == 1:
                if lows[i+1] <= entry * (1 - 1/best_lev):
                    liqs += 1; port = 0; break
            else:
                if highs[i+1] >= entry * (1 + 1/best_lev):
                    liqs += 1; port = 0; break
            ret = best_dir * best_lev * (exit_p - entry) / entry
            port *= (1 + ret)
            trades += 1
        else:
            skips += 1
    
    test_days = mid_end - mid_start - 1
    ret_pct = (port / 3000 - 1) * 100
    print(f"  {label:45s} | Train={len(X_train):>5} | TestDays={test_days:>4} | Trades={trades:>4} | Skips={skips:>3} | Liqs={liqs} | Final=${port:>20,.0f} | Return={ret_pct:>+18,.1f}%")

# --- 4. K-Fold style: 5 sequential blocks ---
print("\n--- 5-Fold Sequential (each fold is test, rest is train) ---")
fold_size = (n - 50) // 5
for fold in range(5):
    test_s = 50 + fold * fold_size
    test_e = test_s + fold_size if fold < 4 else n - 1
    
    # Train on everything except this fold
    X_train, y_train = [], []
    for coin in coins:
        closes = dfs[coin]['Close'].values
        ind = indicators[coin]
        for i in range(50, n - 1):
            if test_s <= i < test_e:
                continue  # skip test fold
            feat = HybridStrategy._feature_vector_from_cache(ind, i)
            if feat is not None:
                next_ret = (closes[i+1] - closes[i]) / closes[i]
                X_train.append(feat)
                y_train.append(1 if next_ret > 0 else 0)
    
    model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(np.array(X_train), np.array(y_train))
    
    port = 3000.0
    trades = 0
    skips = 0
    liqs = 0
    for i in range(test_s, min(test_e, n - 1)):
        best_coin = None
        best_score = -999
        best_dir = 0
        best_lev = 1
        for coin in coins:
            closes = dfs[coin]['Close'].values
            ind = indicators[coin]
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
                if score > 0.05: best_lev = 5
                elif score > 0.02: best_lev = 3
                else: best_lev = 2
        
        if best_coin and best_score > 0:
            closes = dfs[best_coin]['Close'].values
            highs = dfs[best_coin]['High'].values
            lows = dfs[best_coin]['Low'].values
            entry = closes[i]
            exit_p = closes[i+1]
            if best_dir == 1:
                if lows[i+1] <= entry * (1 - 1/best_lev):
                    liqs += 1; port = 0; break
            else:
                if highs[i+1] >= entry * (1 + 1/best_lev):
                    liqs += 1; port = 0; break
            ret = best_dir * best_lev * (exit_p - entry) / entry
            port *= (1 + ret)
            trades += 1
        else:
            skips += 1
    
    test_days = min(test_e, n-1) - test_s
    ret_pct = (port / 3000 - 1) * 100
    label = f"Fold {fold+1}: Test {test_s}-{test_e} ({test_days}d)"
    print(f"  {label:45s} | Train={len(X_train):>5} | TestDays={test_days:>4} | Trades={trades:>4} | Skips={skips:>3} | Liqs={liqs} | Final=${port:>20,.0f} | Return={ret_pct:>+18,.1f}%")
