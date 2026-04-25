"""
Final comprehensive test: improved strategy v2
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

def run_test(label, test_paths, file_key_map=None):
    test_dfs = {}
    for coin, path in test_paths.items():
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'])
        test_dfs[coin] = df[['Date','Open','High','Low','Close','Volume']]
    
    combined = {}; combined_ind = {}
    for coin in train_coins:
        combined[coin] = pd.concat([train_dfs[coin], test_dfs[coin]], ignore_index=True)
        c = combined[coin]['Close'].values.astype(float)
        h = combined[coin]['High'].values.astype(float)
        l = combined[coin]['Low'].values.astype(float)
        v = combined[coin]['Volume'].values.astype(float)
        combined_ind[coin] = HybridStrategy._compute_indicators(c, h, l, v)
    
    total_len = len(combined[train_coins[0]])
    test_acs = [test_dfs[c]['Close'].pct_change().dropna().autocorr(1) for c in train_coins]
    avg_ac = np.mean(test_acs)
    
    port = 3000.0; trades = 0; wins = 0; losses = 0; skips = 0; liqs = 0
    
    for i in range(train_len, total_len - 1):
        autocorrs = []; volatilities = []
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
        
        # NEW REGIME LOGIC
        if ac < 0.10:
            max_lev, alloc, mt = 1, 0.0, 999.0
        elif ac < 0.30:
            t = (ac - 0.10) / 0.20
            alloc = 0.15 + t * 0.85
            max_lev = int(2 + t * 3)
            mt = 0.04 - t * 0.035
        else:
            max_lev, alloc, mt = 5, 1.0, 0.0
        
        # Vol filter
        if vol > 0:
            vr = vol / 0.025
            if vr > 3.0: vs = 0.2
            elif vr > 2.0: vs = 0.4
            elif vr > 1.5: vs = 0.7
            else: vs = 1.0
        else: vs = 1.0
        fa = alloc * vs
        
        best_coin = None; best_score = -999; best_dir = 0; best_lev = 1
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
                best_score = score; best_coin = coin; best_dir = md
                if score > 0.05: best_lev = min(5, max_lev)
                elif score > 0.02: best_lev = min(3, max_lev)
                else: best_lev = min(2, max_lev)
        
        if best_coin and best_score > mt and fa > 0:
            closes = combined[best_coin]['Close'].values
            highs = combined[best_coin]['High'].values
            lows = combined[best_coin]['Low'].values
            entry = closes[i]; exit_p = closes[i+1]
            if best_dir == 1:
                if lows[i+1] <= entry*(1-1/best_lev): liqs+=1; port=0; break
            else:
                if highs[i+1] >= entry*(1+1/best_lev): liqs+=1; port=0; break
            br = best_dir * (exit_p - entry) / entry
            port *= (1 + br * best_lev * fa)
            trades += 1
            if br > 0: wins += 1
            else: losses += 1
        else:
            skips += 1
    
    wr = wins/(wins+losses)*100 if (wins+losses) > 0 else 0
    ret = (port/3000-1)*100
    status = "✅" if port >= 3000 else "⚠️" if port >= 2000 else "🔴"
    print(f"  {status} {label:<40s} AC={avg_ac:>+.3f} WR={wr:>4.0f}% Tr={trades:>3} Sk={skips:>3} Lq={liqs} ${port:>15,.0f} {ret:>+13.1f}%")
    return port

print("=" * 120)
print("IMPROVED STRATEGY v2: AC<0.10=FLAT + Linear Transition + Adaptive Vol")
print("=" * 120)

# Synthetic tests
print("\n--- Synthetic Tests ---")
run_test("Synthetic (unseen_test_1y)", {
    'kapcoin-usd_train': 'data/unseen_test_1y/kapcoin-usd_unseen_test_1y.csv',
    'metucoin-usd_train': 'data/unseen_test_1y/metucoin-usd_unseen_test_1y.csv',
    'tamcoin-usd_train': 'data/unseen_test_1y/tamcoin-usd_unseen_test_1y.csv',
})
run_test("Regime Mix (bear+bull)", {
    'kapcoin-usd_train': 'data/unseen_test_1y_regime_mix/kapcoin-usd_unseen_test_1y_regime_mix.csv',
    'metucoin-usd_train': 'data/unseen_test_1y_regime_mix/metucoin-usd_unseen_test_1y_regime_mix.csv',
    'tamcoin-usd_train': 'data/unseen_test_1y_regime_mix/tamcoin-usd_unseen_test_1y_regime_mix.csv',
})

# Realistic
print("\n--- Realistic Tests ---")
run_test("Realistic (random walk)", {
    'kapcoin-usd_train': 'data/realistic_test_1y/kapcoin-usd_realistic_test_1y.csv',
    'metucoin-usd_train': 'data/realistic_test_1y/metucoin-usd_realistic_test_1y.csv',
    'tamcoin-usd_train': 'data/realistic_test_1y/tamcoin-usd_realistic_test_1y.csv',
})

# 10 Meto scenarios
print("\n--- 10 Meto-Test Scenarios ---")
file_map = {
    'kapcoin-usd_train': 'kapcoin-usd_realistic_test_1y.csv',
    'metucoin-usd_train': 'metucoin-usd_realistic_test_1y.csv',
    'tamcoin-usd_train': 'tamcoin-usd_realistic_test_1y.csv',
}
meto_results = []
for sc in range(1, 11):
    folder = f'meto-test/scenario_{sc:02d}'
    paths = {coin: f'{folder}/{fname}' for coin, fname in file_map.items()}
    p = run_test(f"Meto #{sc:02d}", paths)
    meto_results.append(p)

wins_count = sum(1 for p in meto_results if p >= 3000)
print(f"\n  METO SUMMARY: {wins_count}/10 profitable | Avg=${np.mean(meto_results):,.0f} | Min=${min(meto_results):,.0f} | Max=${max(meto_results):,.0f}")

# Real coins
print("\n--- Real Coins (BTC/ETH/SOL) ---")
run_test("Real: BTC/ETH/SOL", {
    'kapcoin-usd_train': 'gercek-coin-test/btc-usd.csv',
    'metucoin-usd_train': 'gercek-coin-test/eth-usd.csv',
    'tamcoin-usd_train': 'gercek-coin-test/sol-usd.csv',
})
