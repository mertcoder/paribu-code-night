"""
Test strategy on REAL coin data (BTC, ETH, SOL)
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

# Map real coins to train coin slots
real_map = {
    'kapcoin-usd_train': 'gercek-coin-test/btc-usd.csv',
    'metucoin-usd_train': 'gercek-coin-test/eth-usd.csv',
    'tamcoin-usd_train': 'gercek-coin-test/sol-usd.csv',
}

test_dfs = {}
for coin, path in real_map.items():
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    test_dfs[coin] = df[['Date','Open','High','Low','Close','Volume']]

print("=== REAL COIN DATA ===")
names = {'kapcoin-usd_train': 'BTC', 'metucoin-usd_train': 'ETH', 'tamcoin-usd_train': 'SOL'}
for coin in train_coins:
    tdf = test_dfs[coin]
    n = names[coin]
    ac = tdf['Close'].pct_change().dropna().autocorr(1)
    start_p = tdf['Close'].iloc[0]
    end_p = tdf['Close'].iloc[-1]
    bh = (end_p/start_p - 1)*100
    print(f"  {n}: {len(tdf)} days, {tdf['Date'].min().date()} to {tdf['Date'].max().date()}")
    print(f"       Price: ${start_p:,.0f} -> ${end_p:,.0f} ({bh:+.1f}%), AC(lag1)={ac:+.4f}")

# Combine
combined = {}; combined_ind = {}
for coin in train_coins:
    combined[coin] = pd.concat([train_dfs[coin], test_dfs[coin]], ignore_index=True)
    c = combined[coin]['Close'].values.astype(float)
    h = combined[coin]['High'].values.astype(float)
    l = combined[coin]['Low'].values.astype(float)
    v = combined[coin]['Volume'].values.astype(float)
    combined_ind[coin] = HybridStrategy._compute_indicators(c, h, l, v)

total_len = len(combined[train_coins[0]])

print(f"\n  Train: {train_len}, Test: {total_len - train_len}, Combined: {total_len}")

# Run
print(f"\n{'='*70}")
print("STRATEGY ON REAL COINS (BTC/ETH/SOL)")
print(f"{'='*70}")

port = 3000.0
trades = 0; wins = 0; losses = 0; skips = 0; liqs = 0
regime_log = []
trade_log = []

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

    if ac < -0.05: max_lev, alloc, mt = 1, 0.1, 0.08
    elif ac < 0.05: max_lev, alloc, mt = 1, 0.15, 0.03
    elif ac < 0.15: max_lev, alloc, mt = 2, 0.4, 0.02
    elif ac < 0.30: max_lev, alloc, mt = 3, 0.7, 0.01
    else: max_lev, alloc, mt = 5, 1.0, 0.0

    vb = 0.025
    if vol > vb*2.5: alloc *= 0.3
    elif vol > vb*1.5: alloc *= 0.6

    regime_log.append((ac, max_lev, alloc, vol))

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

    if best_coin and best_score > mt:
        closes = combined[best_coin]['Close'].values
        highs = combined[best_coin]['High'].values
        lows = combined[best_coin]['Low'].values
        entry = closes[i]; exit_p = closes[i+1]
        
        if best_dir == 1:
            if lows[i+1] <= entry*(1-1/best_lev):
                liqs += 1; port = 0
                print(f"  !! LIQUIDATION at day {i-train_len} {names[best_coin]} lev={best_lev}x")
                break
        else:
            if highs[i+1] >= entry*(1+1/best_lev):
                liqs += 1; port = 0
                print(f"  !! LIQUIDATION at day {i-train_len} {names[best_coin]} lev={best_lev}x")
                break
        
        br = best_dir * (exit_p - entry) / entry
        port *= (1 + br * best_lev * alloc)
        trades += 1
        if br > 0: wins += 1
        else: losses += 1
        trade_log.append((names[best_coin], best_dir, best_lev, alloc, br, port))
    else:
        skips += 1

wr = wins/(wins+losses)*100 if (wins+losses) > 0 else 0
ret = (port/3000-1)*100

print(f"\n  Starting Capital  : $       3,000.00")
print(f"  Final Portfolio   : ${port:>15,.2f}")
print(f"  Return            :  {ret:>+12.2f}%")
print(f"  Trades            : {trades:>8}")
print(f"  Wins              : {wins:>8} ({wr:.1f}%)")
print(f"  Losses            : {losses:>8}")
print(f"  Skips             : {skips:>8}")
print(f"  Liquidations      : {liqs:>8}")

# Regime stats
acs = [r[0] for r in regime_log]
levs = [r[1] for r in regime_log]
allocs = [r[2] for r in regime_log]
vols = [r[3] for r in regime_log]
print(f"\n  Avg Autocorr      :  {np.mean(acs):>+.4f}")
print(f"  Avg Leverage      :  {np.mean(levs):>6.2f}")
print(f"  Avg Allocation    :  {np.mean(allocs):>6.2f}")
print(f"  Avg Volatility    :  {np.mean(vols):>6.4f}")

# Buy & hold benchmarks
print(f"\n--- Buy & Hold Benchmarks ---")
for coin in train_coins:
    tdf = test_dfs[coin]
    bh = (tdf['Close'].iloc[-1] / tdf['Close'].iloc[0] - 1) * 100
    print(f"  {names[coin]}: {bh:+.1f}%")

# Coin selection breakdown
print(f"\n--- Trade Breakdown by Coin ---")
from collections import Counter
coin_counts = Counter(t[0] for t in trade_log)
for c, cnt in coin_counts.most_common():
    coin_trades = [t for t in trade_log if t[0] == c]
    coin_wins = sum(1 for t in coin_trades if t[4] > 0)
    print(f"  {c}: {cnt} trades, {coin_wins} wins ({coin_wins/cnt*100:.0f}%)")
