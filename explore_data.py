import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

coins = ['kapcoin-usd_train', 'metucoin-usd_train', 'tamcoin-usd_train']
dfs = {}
for coin in coins:
    df = pd.read_parquet(f'C:/Python311/Lib/site-packages/cnlib/data/{coin}.parquet')
    dfs[coin] = df

n = len(dfs[coins[0]])
train_end = int(n * 0.75)

# ============================================================
# Pre-compute ALL indicators vectorized for each coin
# ============================================================

def precompute_indicators(closes, highs, lows, volumes):
    """Compute all indicators once, return dict of arrays"""
    c = closes
    h = highs
    l = lows
    v = volumes
    N = len(c)
    
    ind = {}
    
    # Momentum returns
    for lb in [1, 2, 3, 5, 10, 20, 30]:
        mom = np.zeros(N)
        mom[lb:] = (c[lb:] - c[:-lb]) / c[:-lb]
        ind[f'mom_{lb}d'] = mom
    
    # MA ratios
    for w in [5, 10, 20, 50]:
        ma = pd.Series(c).rolling(w).mean().values
        ind[f'ma{w}_ratio'] = np.where(ma != 0, c / ma - 1, 0)
    
    # EMA ratios
    for span in [8, 13, 21]:
        ema = pd.Series(c).ewm(span=span).mean().values
        ind[f'ema{span}_ratio'] = np.where(ema != 0, c / ema - 1, 0)
    
    # Volatility
    rets = pd.Series(c).pct_change().values
    ind['vol_std'] = pd.Series(rets).rolling(20).std().values
    ind['vol_mean'] = pd.Series(rets).rolling(20).mean().values
    
    # Volume ratio
    vol_ma = pd.Series(v).rolling(20).mean().values
    ind['vol_ratio'] = np.where(vol_ma != 0, v / vol_ma - 1, 0)
    
    # Range
    high5 = pd.Series(c).rolling(5).max().values
    low5 = pd.Series(c).rolling(5).min().values
    ind['range_5d'] = np.where(c != 0, (high5 - low5) / c, 0)
    
    # RSI
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    ind['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = pd.Series(c).ewm(span=12).mean().values
    ema26 = pd.Series(c).ewm(span=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9).mean().values
    ind['macd_hist'] = np.where(c != 0, (macd_line - signal_line) / c, 0)
    ind['macd_line'] = np.where(c != 0, macd_line / c, 0)
    
    # Bollinger Bands
    bb_ma = pd.Series(c).rolling(20).mean().values
    bb_std = pd.Series(c).rolling(20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_range = bb_upper - bb_lower
    ind['bb_pctb'] = np.where(bb_range != 0, (c - bb_lower) / bb_range, 0.5)
    ind['bb_width'] = np.where(bb_ma != 0, bb_range / bb_ma, 0)
    
    # Stochastic
    high14 = pd.Series(h).rolling(14).max().values
    low14 = pd.Series(l).rolling(14).min().values
    stoch_range = high14 - low14
    ind['stoch_k'] = np.where(stoch_range != 0, 100 * (c - low14) / stoch_range, 50)
    ind['stoch_d'] = pd.Series(ind['stoch_k']).rolling(3).mean().values
    
    # CCI
    tp = (h + l + c) / 3
    tp_ma = pd.Series(tp).rolling(20).mean().values
    tp_md = pd.Series(tp).rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    ind['cci'] = np.where(tp_md != 0, (tp - tp_ma) / (0.015 * tp_md), 0)
    
    # Williams %R
    ind['williams_r'] = np.where(stoch_range != 0, -100 * (high14 - c) / stoch_range, -50)
    
    # ATR normalized
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = pd.Series(tr).rolling(14).mean().values
    atr_full = np.concatenate([[np.nan], atr])
    ind['atr_norm'] = np.where(c != 0, atr_full / c, 0)
    
    # MFI
    mf = tp * v
    pos_mf = np.where(np.diff(tp, prepend=tp[0]) > 0, mf, 0)
    neg_mf = np.where(np.diff(tp, prepend=tp[0]) < 0, mf, 0)
    pos_sum = pd.Series(pos_mf).rolling(14).sum().values
    neg_sum = pd.Series(neg_mf).rolling(14).sum().values
    ind['mfi'] = np.where(neg_sum > 0, 100 - 100 / (1 + pos_sum / neg_sum), 50)
    
    # OBV momentum
    obv = np.cumsum(np.where(np.diff(c, prepend=c[0]) > 0, v, 
                   np.where(np.diff(c, prepend=c[0]) < 0, -v, 0)))
    obv_10 = np.zeros(N)
    for i in range(10, N):
        obv_10[i] = (obv[i] - obv[i-10]) / (abs(obv[i-10]) + 1)
    ind['obv_mom'] = obv_10
    
    # ADX direction
    plus_dm = np.maximum(np.diff(h, prepend=h[0]), 0)
    minus_dm = np.maximum(-np.diff(l, prepend=l[0]), 0)
    mask_p = plus_dm > minus_dm
    plus_dm_clean = np.where(mask_p, plus_dm, 0)
    minus_dm_clean = np.where(~mask_p, minus_dm, 0)
    
    tr_full = np.maximum(h - np.roll(l, 1), np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr_full[0] = h[0] - l[0]
    atr_adx = pd.Series(tr_full).rolling(14).mean().values
    plus_di = np.where(atr_adx > 0, 100 * pd.Series(plus_dm_clean).rolling(14).mean().values / atr_adx, 0)
    minus_di = np.where(atr_adx > 0, 100 * pd.Series(minus_dm_clean).rolling(14).mean().values / atr_adx, 0)
    ind['adx_dir'] = plus_di - minus_di
    
    return ind

# Pre-compute for all coins
print("Pre-computing indicators...")
all_indicators = {}
for coin in coins:
    df = dfs[coin]
    all_indicators[coin] = precompute_indicators(
        df['Close'].values, df['High'].values, df['Low'].values, df['Volume'].values)
print("Done.\n")

# ============================================================
# Define feature sets to compare
# ============================================================

original_features = [
    'mom_1d','mom_2d','mom_3d','mom_5d','mom_10d','mom_20d','mom_30d',
    'ma5_ratio','ma10_ratio','ma20_ratio','ma50_ratio',
    'vol_std','vol_mean','vol_ratio','range_5d'
]

extended_features = original_features + [
    'rsi_14','macd_hist','macd_line','bb_pctb','bb_width',
    'stoch_k','cci','williams_r','atr_norm','mfi',
    'ema8_ratio','ema13_ratio','ema21_ratio'
]

full_features = extended_features + [
    'stoch_d','obv_mom','adx_dir'
]

def build_dataset(feature_list, start, end):
    X, y = [], []
    for coin in coins:
        ind = all_indicators[coin]
        closes = dfs[coin]['Close'].values
        for i in range(max(50, start), min(end, n - 1)):
            feat = [ind[f][i] for f in feature_list]
            if not any(np.isnan(feat)):
                X.append(feat)
                next_ret = (closes[i+1] - closes[i]) / closes[i]
                y.append(1 if next_ret > 0 else 0)
    return np.array(X), np.array(y)

# ============================================================
# Test different feature sets
# ============================================================
print("=" * 70)
print("ML ACCURACY COMPARISON: Different Feature Sets")
print("=" * 70)

for name, feat_list in [
    ("Original (15 features)", original_features),
    ("Extended (28 features)", extended_features),
    ("Full (31 features)", full_features),
]:
    X_train, y_train = build_dataset(feat_list, 0, train_end)
    X_test, y_test = build_dataset(feat_list, train_end, n)
    
    model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    
    # Calculate return
    proba = model.predict_proba(X_test)
    
    print(f"\n  {name}:")
    print(f"    Accuracy: {acc:.4f} ({(acc*100):.1f}%)")
    print(f"    Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Feature importance
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"    Top 10 features:")
    for rank in range(min(10, len(feat_list))):
        idx = sorted_idx[rank]
        bar = "█" * int(importances[idx] * 150)
        print(f"      {rank+1:2d}. {feat_list[idx]:18s} {importances[idx]:.4f} {bar}")

# ============================================================
# Walk-forward RETURN comparison with best feature set
# ============================================================
print("\n" + "=" * 70)
print("WALK-FORWARD RETURN: Original vs Extended vs Full")
print("=" * 70)

for name, feat_list in [
    ("Original (15)", original_features),
    ("Extended (28)", extended_features),
    ("Full (31)", full_features),
]:
    X_train, y_train = build_dataset(feat_list, 0, train_end)
    X_test, y_test = build_dataset(feat_list, train_end, n)
    
    model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    # Per-coin walk-forward return
    for coin in coins:
        ind = all_indicators[coin]
        closes = dfs[coin]['Close'].values
        
        rets = []
        for i in range(max(50, train_end), n - 1):
            feat = [ind[f][i] for f in feat_list]
            if not any(np.isnan(feat)):
                proba = model.predict_proba([feat])[0]
                direction = 1 if proba[1] > 0.5 else -1
                next_ret = (closes[i+1] - closes[i]) / closes[i]
                rets.append(direction * next_ret)
        
        total_ret = (1 + pd.Series(rets)).cumprod().iloc[-1] - 1
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(365) if np.std(rets) > 0 else 0
        short_name = coin.split('-')[0]
        print(f"  {name:20s} {short_name}: Return={total_ret*100:+8.1f}%, Sharpe={sharpe:.2f}")
    print()

# ============================================================
# HYBRID: Full features ML + Momentum rotation
# ============================================================
print("=" * 70)
print("HYBRID COIN ROTATION: Full-feature ML + Momentum + Dynamic Lev")
print("=" * 70)

X_train_full, y_train_full = build_dataset(full_features, 0, train_end)
model_full = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
model_full.fit(X_train_full, y_train_full)

for lev in [2, 3, 5]:
    port = 3000.0
    for i in range(max(50, train_end), n - 1):
        best_coin = None
        best_score = -999
        best_dir = 0
        
        for coin in coins:
            ind = all_indicators[coin]
            closes = dfs[coin]['Close'].values
            
            feat = [ind[f][i] for f in full_features]
            if any(np.isnan(feat)):
                continue
            
            # Momentum
            past_ret = (closes[i] - closes[i-5]) / closes[i-5]
            mom_dir = 1 if past_ret > 0 else -1
            
            # ML
            proba = model_full.predict_proba([feat])[0]
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
        
        if best_coin and best_score > 0:
            closes = dfs[best_coin]['Close'].values
            highs = dfs[best_coin]['High'].values
            lows = dfs[best_coin]['Low'].values
            entry = closes[i]
            exit_p = closes[i + 1]
            
            # Liq check
            if best_dir == 1:
                liq_price = entry * (1 - 1/lev)
                if lows[i+1] <= liq_price:
                    port = 0
                    break
            else:
                liq_price = entry * (1 + 1/lev)
                if highs[i+1] >= liq_price:
                    port = 0
                    break
            
            ret = best_dir * lev * (exit_p - entry) / entry
            port *= (1 + ret)
    
    total_ret = (port / 3000 - 1) * 100
    print(f"  Full-feature Hybrid {lev}x: Final=${port:,.0f}, Return={total_ret:+.1f}%")
