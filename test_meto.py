"""
Test different realistic-regime strategies on 10 meto-test scenarios
WITHOUT touching synthetic performance
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from strategy import HybridStrategy

train_coins = ['kapcoin-usd_train', 'metucoin-usd_train', 'tamcoin-usd_train']
train_dfs = {}
for coin in train_coins:
    train_dfs[coin] = pd.read_parquet(f'C:/Python311/Lib/site-packages/cnlib/data/{coin}.parquet')

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
            y_train.append(1 if (closes[i+1] - closes[i]) / closes[i] > 0 else 0)

model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
model.fit(np.array(X_train), np.array(y_train))
train_len = len(train_dfs[train_coins[0]])

file_map = {
    'kapcoin-usd_train': 'kapcoin-usd_realistic_test_1y.csv',
    'metucoin-usd_train': 'metucoin-usd_realistic_test_1y.csv',
    'tamcoin-usd_train': 'tamcoin-usd_realistic_test_1y.csv',
}

# Pre-load all scenarios
scenarios = {}
for sc in range(1, 11):
    folder = f'meto-test/scenario_{sc:02d}'
    combined = {}; combined_ind = {}; test_dfs = {}
    for coin, fname in file_map.items():
        df = pd.read_csv(f'{folder}/{fname}')
        df['Date'] = pd.to_datetime(df['Date'])
        test_dfs[coin] = df[['Date','Open','High','Low','Close','Volume']]
        combined[coin] = pd.concat([train_dfs[coin], test_dfs[coin]], ignore_index=True)
        c = combined[coin]['Close'].values.astype(float)
        h = combined[coin]['High'].values.astype(float)
        l = combined[coin]['Low'].values.astype(float)
        v = combined[coin]['Volume'].values.astype(float)
        combined_ind[coin] = HybridStrategy._compute_indicators(c, h, l, v)
    scenarios[sc] = (combined, combined_ind, test_dfs)

def run_strategy(strategy_name, realistic_config):
    """
    realistic_config = {
        'alloc': float,         # allocation cap
        'max_lev': int,         # max leverage
        'min_thresh': float,    # min score threshold
        'min_ml_conf': float,   # min ML confidence to trade
    }
    """
    results = []
    for sc in range(1, 11):
        combined, combined_ind, test_dfs = scenarios[sc]
        total_len = len(combined[train_coins[0]])

        port = 3000.0
        trades = 0; wins = 0; losses = 0; skips = 0; liqs = 0

        for i in range(train_len, total_len - 1):
            # Regime detection
            autocorrs = []
            volatilities = []
            for coin in train_coins:
                closes = combined[coin]['Close'].values
                if i >= 102:
                    window = closes[i-101:i+1]
                    rets = np.diff(window) / window[:-1]
                    r1 = rets[:-1] - np.mean(rets[:-1])
                    r2 = rets[1:] - np.mean(rets[1:])
                    d = np.sqrt(np.sum(r1**2) * np.sum(r2**2))
                    if d > 0: autocorrs.append(np.sum(r1*r2)/d)
                if i >= 31:
                    rr = np.diff(closes[i-30:i+1]) / closes[i-30:i]
                    volatilities.append(np.std(rr))

            ac = np.mean(autocorrs) if autocorrs else 0.0
            vol = np.mean(volatilities) if volatilities else 0.02

            # Regime rules
            if ac >= 0.30:
                max_lev, alloc, min_thresh, min_ml = 5, 1.0, 0.0, 0.0
            elif ac >= 0.15:
                max_lev, alloc, min_thresh, min_ml = 3, 0.7, 0.01, 0.0
            else:
                # REALISTIC REGIME — use custom config
                max_lev = realistic_config['max_lev']
                alloc = realistic_config['alloc']
                min_thresh = realistic_config['min_thresh']
                min_ml = realistic_config['min_ml_conf']

            # Vol filter (only for non-aggressive regimes)
            if ac < 0.30:
                vb = 0.025
                if vol > vb*2.5: alloc *= 0.3
                elif vol > vb*1.5: alloc *= 0.6

            best_coin = None; best_score = -999; best_dir = 0; best_lev = 1; best_ml_conf = 0
            for coin in train_coins:
                closes = combined[coin]['Close'].values
                ind = combined_ind[coin]
                if i < 5: continue
                pr = (closes[i] - closes[i-5]) / closes[i-5]
                md = 1 if pr > 0 else -1
                feat = HybridStrategy._feature_vector_from_cache(ind, i)
                if feat is None: continue
                proba = model.predict_proba([feat])[0]
                mld = 1 if proba[1] > 0.5 else -1
                mc = max(proba) - 0.5
                if md == mld: score = abs(pr) * (1 + mc*2)
                else: score = -1
                if score > best_score:
                    best_score = score; best_coin = coin; best_dir = md; best_ml_conf = mc
                    if score > 0.05: best_lev = min(5, max_lev)
                    elif score > 0.02: best_lev = min(3, max_lev)
                    else: best_lev = min(2, max_lev)

            if best_coin and best_score > min_thresh and best_ml_conf >= min_ml:
                closes = combined[best_coin]['Close'].values
                highs = combined[best_coin]['High'].values
                lows = combined[best_coin]['Low'].values
                entry = closes[i]; exit_p = closes[i+1]
                if best_dir == 1:
                    if lows[i+1] <= entry*(1-1/best_lev): liqs+=1; port=0; break
                else:
                    if highs[i+1] >= entry*(1+1/best_lev): liqs+=1; port=0; break
                br = best_dir * (exit_p - entry) / entry
                port *= (1 + br * best_lev * alloc)
                trades += 1
                if br > 0: wins += 1
                else: losses += 1
            else:
                skips += 1

        wr = wins/(wins+losses)*100 if (wins+losses) > 0 else 0
        results.append(port)

    avg = np.mean(results)
    wins_count = sum(1 for p in results if p >= 3000)
    worst = min(results)
    best = max(results)
    print(f"  {strategy_name:<45s} | {wins_count}/10 win | Avg=${avg:>8,.0f} | Min=${worst:>8,.0f} | Max=${best:>8,.0f}")
    return avg

print("=" * 120)
print("REALISTIC REGIME STRATEGY COMPARISON (10 meto-test scenarios)")
print("=" * 120)
print()

# A: Current strategy
run_strategy("A: Current (alloc=0.15, lev=1, thresh=0.03)", 
    {'alloc': 0.15, 'max_lev': 1, 'min_thresh': 0.03, 'min_ml_conf': 0.0})

# B: No trading at all in realistic regime
run_strategy("B: Flat (no trade when AC<0.15)", 
    {'alloc': 0.0, 'max_lev': 1, 'min_thresh': 999.0, 'min_ml_conf': 999.0})

# C: Only trade with very high ML confidence
run_strategy("C: High-conf only (ML conf>0.20, alloc=0.10)", 
    {'alloc': 0.10, 'max_lev': 1, 'min_thresh': 0.03, 'min_ml_conf': 0.20})

# D: Ultra-selective (ML conf>0.25, higher thresh)
run_strategy("D: Ultra-selective (ML conf>0.25, thresh=0.05)", 
    {'alloc': 0.08, 'max_lev': 1, 'min_thresh': 0.05, 'min_ml_conf': 0.25})

# E: Tiny bets with high threshold
run_strategy("E: Tiny bets (alloc=0.05, thresh=0.04)", 
    {'alloc': 0.05, 'max_lev': 1, 'min_thresh': 0.04, 'min_ml_conf': 0.0})

# F: Mean-reversion: flip direction in low AC
# We need to hack this differently - skip for now

# G: Only trade ML-only (ignore momentum agreement in realistic)
run_strategy("G: ML-only high-conf (conf>0.20, alloc=0.15)", 
    {'alloc': 0.15, 'max_lev': 1, 'min_thresh': 0.0, 'min_ml_conf': 0.20})

# H: Moderate approach
run_strategy("H: Moderate (alloc=0.10, thresh=0.02, conf>0.15)", 
    {'alloc': 0.10, 'max_lev': 1, 'min_thresh': 0.02, 'min_ml_conf': 0.15})

# I: Very tiny (alloc=0.03)
run_strategy("I: Micro bets (alloc=0.03, thresh=0.03)", 
    {'alloc': 0.03, 'max_lev': 1, 'min_thresh': 0.03, 'min_ml_conf': 0.0})

# J: Balanced
run_strategy("J: Balanced (alloc=0.08, lev=1, thresh=0.03, conf>0.10)", 
    {'alloc': 0.08, 'max_lev': 1, 'min_thresh': 0.03, 'min_ml_conf': 0.10})

# Also test on synthetic to make sure nothing breaks
print()
print("=" * 120)
print("SYNTHETIC SANITY CHECK (unseen_test_1y)")
print("=" * 120)

# Load synthetic test
syn_combined = {}; syn_ind = {}
syn_map = {
    'kapcoin-usd_train': 'data/unseen_test_1y/kapcoin-usd_unseen_test_1y.csv',
    'metucoin-usd_train': 'data/unseen_test_1y/metucoin-usd_unseen_test_1y.csv',
    'tamcoin-usd_train': 'data/unseen_test_1y/tamcoin-usd_unseen_test_1y.csv',
}
for coin, path in syn_map.items():
    df = pd.read_csv(path); df['Date'] = pd.to_datetime(df['Date'])
    syn_combined[coin] = pd.concat([train_dfs[coin], df[['Date','Open','High','Low','Close','Volume']]], ignore_index=True)
    c = syn_combined[coin]['Close'].values.astype(float)
    h = syn_combined[coin]['High'].values.astype(float)
    l = syn_combined[coin]['Low'].values.astype(float)
    v = syn_combined[coin]['Volume'].values.astype(float)
    syn_ind[coin] = HybridStrategy._compute_indicators(c, h, l, v)

total_len_syn = len(syn_combined[train_coins[0]])

for label, cfg in [
    ("Current", {'alloc':0.15,'max_lev':1,'min_thresh':0.03,'min_ml_conf':0.0}),
    ("Flat", {'alloc':0.0,'max_lev':1,'min_thresh':999.0,'min_ml_conf':999.0}),
    ("Best realistic", None),  # placeholder
]:
    if label == "Best realistic": continue
    port = 3000.0
    for i in range(train_len, total_len_syn - 1):
        autocorrs = []
        for coin in train_coins:
            closes = syn_combined[coin]['Close'].values
            if i >= 102:
                window = closes[i-101:i+1]
                rets = np.diff(window) / window[:-1]
                r1 = rets[:-1] - np.mean(rets[:-1])
                r2 = rets[1:] - np.mean(rets[1:])
                d = np.sqrt(np.sum(r1**2) * np.sum(r2**2))
                if d > 0: autocorrs.append(np.sum(r1*r2)/d)
        ac = np.mean(autocorrs) if autocorrs else 0.0
        
        if ac >= 0.30: max_lev, alloc, min_thresh, min_ml = 5, 1.0, 0.0, 0.0
        elif ac >= 0.15: max_lev, alloc, min_thresh, min_ml = 3, 0.7, 0.01, 0.0
        else: max_lev, alloc, min_thresh, min_ml = cfg['max_lev'], cfg['alloc'], cfg['min_thresh'], cfg['min_ml_conf']
        
        best_coin = None; best_score = -999; best_dir = 0; best_lev = 1; best_mc = 0
        for coin in train_coins:
            closes = syn_combined[coin]['Close'].values
            ind = syn_ind[coin]
            if i < 5: continue
            pr = (closes[i] - closes[i-5]) / closes[i-5]
            md = 1 if pr > 0 else -1
            feat = HybridStrategy._feature_vector_from_cache(ind, i)
            if feat is None: continue
            proba = model.predict_proba([feat])[0]
            mld = 1 if proba[1] > 0.5 else -1
            mc = max(proba) - 0.5
            if md == mld: score = abs(pr) * (1 + mc*2)
            else: score = -1
            if score > best_score:
                best_score = score; best_coin = coin; best_dir = md; best_mc = mc
                if score > 0.05: best_lev = min(5, max_lev)
                elif score > 0.02: best_lev = min(3, max_lev)
                else: best_lev = min(2, max_lev)
        
        if best_coin and best_score > min_thresh and best_mc >= min_ml:
            closes = syn_combined[best_coin]['Close'].values
            entry = closes[i]; exit_p = closes[i+1]
            br = best_dir * (exit_p - entry) / entry
            port *= (1 + br * best_lev * alloc)

    print(f"  {label:20s} Synthetic → ${port:>15,.0f} ({(port/3000-1)*100:>+.1f}%)")
