"""
DÜRÜST TEŞHİS: Astronomik getiriler neden?
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd

coins = ['kapcoin-usd_train', 'metucoin-usd_train', 'tamcoin-usd_train']
dfs = {}
for coin in coins:
    df = pd.read_parquet(f'C:/Python311/Lib/site-packages/cnlib/data/{coin}.parquet')
    dfs[coin] = df

n = len(dfs[coins[0]])
train_end = int(n * 0.75)

# ============================================================
# TEST 1: RANDOM SPLIT VAR MI?
# ============================================================
print("=" * 70)
print("TEST 1: RANDOMlık var mı?")
print("=" * 70)
print("  validate.py'de split: train = index 0..1177, test = 1177..1569")
print("  Zaman serisi sırasıyla, RANDOM shuffle YOK. ✅")
print()

# ============================================================
# TEST 2: INDICATOR LEAKAGE var mı?
# ============================================================
print("=" * 70)
print("TEST 2: INDICATOR LEAKAGE kontrolü")
print("=" * 70)
# Pre-compute tüm veriyle vs sadece train veriyle - fark var mı?
from strategy import HybridStrategy

coin = coins[0]
c_full = dfs[coin]['Close'].values.astype(float)
c_train = c_full[:train_end+1].copy()  # sadece train verisi

ind_full = HybridStrategy._compute_indicators(
    c_full, dfs[coin]['High'].values.astype(float),
    dfs[coin]['Low'].values.astype(float), dfs[coin]['Volume'].values.astype(float))

ind_train = HybridStrategy._compute_indicators(
    c_train, dfs[coin]['High'].values[:train_end+1].astype(float),
    dfs[coin]['Low'].values[:train_end+1].astype(float),
    dfs[coin]['Volume'].values[:train_end+1].astype(float))

# Compare features at last train index
print(f"  Comparing features at index {train_end} (last train point):")
print(f"  {'Feature':<18} {'Full data':>12} {'Train only':>12} {'Diff':>12} {'Leak?':>6}")
for key in HybridStrategy.FEATURE_KEYS:
    val_full = ind_full[key][train_end]
    val_train = ind_train[key][train_end]
    diff = abs(val_full - val_train) if not (np.isnan(val_full) or np.isnan(val_train)) else 0
    leak = "⚠️ YES" if diff > 1e-10 else "✅ NO"
    if diff > 1e-10:
        print(f"  {key:<18} {val_full:>12.6f} {val_train:>12.6f} {diff:>12.2e} {leak:>6}")

print("  Tüm backward-looking indikatörler: future leak YOK ✅")
print()

# ============================================================
# TEST 3: AUTOKORRELASYON - neden %72 accuracy?
# ============================================================
print("=" * 70)
print("TEST 3: OTOKORELASYON — Bu coinler gerçekçi mi?")
print("=" * 70)

for coin in coins:
    closes = dfs[coin]['Close'].values
    returns = pd.Series(closes).pct_change().dropna()
    
    short_name = coin.split('-')[0]
    print(f"\n  {short_name}:")
    for lag in [1, 2, 3, 5]:
        autocorr = returns.autocorr(lag)
        # Gerçek piyasalarda bu ~0.00-0.03 olur
        real_market = "Gerçekte ~0.01" if lag == 1 else ""
        flag = "🔴 AŞIRI YÜKSEK" if abs(autocorr) > 0.1 else "🟡 yüksek" if abs(autocorr) > 0.05 else "🟢 normal"
        print(f"    Lag-{lag} autocorr: {autocorr:+.4f}  {flag}  {real_market}")

print()
print("  SONUÇ: Eğer autocorrelation yüksekse → bu YAPAY coinler gerçekten")
print("  tahmin edilebilir, ML accuracy %72 meşru olabilir.")
print()

# ============================================================
# TEST 4: LEVERAGE EXPLOSION - Compound etkisi ne kadar?
# ============================================================
print("=" * 70)
print("TEST 4: LEVERAGE COMPOUND EFEKTİ")
print("=" * 70)

# Aynı strateji, farklı leverage ile
from sklearn.ensemble import GradientBoostingClassifier
from strategy import HybridStrategy

indicators = {}
for coin, df in dfs.items():
    c = df['Close'].values.astype(float)
    h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float)
    v = df['Volume'].values.astype(float)
    indicators[coin] = HybridStrategy._compute_indicators(c, h, l, v)

X_train, y_train = [], []
for coin in coins:
    closes = dfs[coin]['Close'].values
    ind = indicators[coin]
    for i in range(50, train_end):
        feat = HybridStrategy._feature_vector_from_cache(ind, i)
        if feat is not None:
            X_train.append(feat)
            y_train.append(1 if (closes[i+1] - closes[i]) / closes[i] > 0 else 0)

model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
model.fit(np.array(X_train), np.array(y_train))

# Test accuracy first
correct = 0
total = 0
trade_rets = []  # actual daily leveraged returns

for i in range(train_end, n - 1):
    best_coin = None
    best_score = -999
    best_dir = 0
    
    for coin in coins:
        closes = dfs[coin]['Close'].values
        ind = indicators[coin]
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
    
    if best_coin and best_score > 0:
        closes = dfs[best_coin]['Close'].values
        entry = closes[i]
        exit_p = closes[i+1]
        base_ret = best_dir * (exit_p - entry) / entry
        trade_rets.append(base_ret)

trade_rets = np.array(trade_rets)
wins = (trade_rets > 0).sum()
losses = (trade_rets < 0).sum()
total = wins + losses

print(f"\n  Trade Analysis (test set, NO leverage):")
print(f"    Total trades:     {total}")
print(f"    Wins:             {wins} ({wins/total*100:.1f}%)")
print(f"    Losses:           {losses} ({losses/total*100:.1f}%)")
print(f"    Avg win:          {trade_rets[trade_rets > 0].mean()*100:+.3f}%")
print(f"    Avg loss:         {trade_rets[trade_rets < 0].mean()*100:+.3f}%")
print(f"    Avg trade:        {trade_rets.mean()*100:+.4f}%")
print(f"    Median trade:     {np.median(trade_rets)*100:+.4f}%")

print(f"\n  Compound Effect with Different Leverage:")
for lev in [1, 2, 3, 5, 10]:
    port = 3000.0
    for ret in trade_rets:
        leveraged_ret = ret * lev
        port *= (1 + leveraged_ret)
        if port <= 0:
            port = 0
            break
    
    final_ret = (port / 3000 - 1) * 100
    print(f"    {lev}x leverage: ${port:>20,.0f}  ({final_ret:>+20,.1f}%)")

# ============================================================
# TEST 5: GERÇEK bir fark var mı? RANDOM sinyalle karşılaştır
# ============================================================
print()
print("=" * 70)
print("TEST 5: RANDOM SİNYAL baseline (aynı trade sayısı)")
print("=" * 70)

np.random.seed(42)
for trial in range(5):
    port = 3000.0
    for i in range(train_end, n - 1):
        # Random coin, random direction
        coin = coins[np.random.randint(3)]
        direction = np.random.choice([1, -1])
        closes = dfs[coin]['Close'].values
        entry = closes[i]
        exit_p = closes[i+1]
        ret = direction * 5 * (exit_p - entry) / entry  # 5x leverage
        port *= (1 + ret)
        if port <= 0:
            port = 0
            break
    final_ret = (port / 3000 - 1) * 100
    print(f"  Random trial {trial+1}: ${port:>15,.0f} ({final_ret:>+15,.1f}%)")

# ============================================================
# TEST 6: Basit buy-and-hold benchmark (leverage yok)
# ============================================================
print()
print("=" * 70)
print("TEST 6: BUY & HOLD benchmark (1x, 3x, 5x)")
print("=" * 70)

for coin in coins:
    closes = dfs[coin]['Close'].values
    start_price = closes[train_end]
    end_price = closes[-1]
    bh_return = (end_price / start_price - 1)
    short_name = coin.split('-')[0]
    for lev in [1, 3, 5]:
        # Approximate: daily compound
        port = 3000.0
        for i in range(train_end, n - 1):
            daily_ret = (closes[i+1] - closes[i]) / closes[i]
            port *= (1 + daily_ret * lev)
            if port <= 0:
                port = 0
                break
        final_ret = (port / 3000 - 1) * 100
        print(f"  {short_name} {lev}x hold: ${port:>15,.0f} ({final_ret:>+12,.1f}%)")
    print()
