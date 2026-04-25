"""
Portfolio equity curve: synthetic vs realistic vs regime-mix
+ AC vs Return graph
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from strategy import HybridStrategy

plt.rcParams['font.size'] = 11
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor'] = '#161b22'
plt.rcParams['text.color'] = '#e6edf3'
plt.rcParams['axes.labelcolor'] = '#e6edf3'
plt.rcParams['xtick.color'] = '#8b949e'
plt.rcParams['ytick.color'] = '#8b949e'

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

# All test datasets
test_sets = {
    "Synthetic": {
        'kapcoin-usd_train': 'data/unseen_test_1y/kapcoin-usd_unseen_test_1y.csv',
        'metucoin-usd_train': 'data/unseen_test_1y/metucoin-usd_unseen_test_1y.csv',
        'tamcoin-usd_train': 'data/unseen_test_1y/tamcoin-usd_unseen_test_1y.csv',
    },
    "Regime Mix": {
        'kapcoin-usd_train': 'data/unseen_test_1y_regime_mix/kapcoin-usd_unseen_test_1y_regime_mix.csv',
        'metucoin-usd_train': 'data/unseen_test_1y_regime_mix/metucoin-usd_unseen_test_1y_regime_mix.csv',
        'tamcoin-usd_train': 'data/unseen_test_1y_regime_mix/tamcoin-usd_unseen_test_1y_regime_mix.csv',
    },
    "Realistic": {
        'kapcoin-usd_train': 'data/realistic_test_1y/kapcoin-usd_realistic_test_1y.csv',
        'metucoin-usd_train': 'data/realistic_test_1y/metucoin-usd_realistic_test_1y.csv',
        'tamcoin-usd_train': 'data/realistic_test_1y/tamcoin-usd_realistic_test_1y.csv',
    },
}

# Add 10 meto-test scenarios
file_map = {
    'kapcoin-usd_train': 'kapcoin-usd_realistic_test_1y.csv',
    'metucoin-usd_train': 'metucoin-usd_realistic_test_1y.csv',
    'tamcoin-usd_train': 'tamcoin-usd_realistic_test_1y.csv',
}
for sc in range(1, 11):
    folder = f'meto-test/scenario_{sc:02d}'
    test_sets[f"Meto-{sc:02d}"] = {
        coin: f'{folder}/{fname}' for coin, fname in file_map.items()
    }

def run_and_get_curve(test_paths):
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
    
    # AC of test data
    test_acs = []
    for coin in train_coins:
        rets = test_dfs[coin]['Close'].pct_change().dropna()
        test_acs.append(rets.autocorr(1))
    avg_ac = np.mean(test_acs)

    port = 3000.0
    curve = [3000.0]
    ac_curve = []

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
        ac_curve.append(ac)

        if ac < -0.05: max_lev, alloc, mt = 1, 0.1, 0.08
        elif ac < 0.05: max_lev, alloc, mt = 1, 0.15, 0.03
        elif ac < 0.15: max_lev, alloc, mt = 2, 0.4, 0.02
        elif ac < 0.30: max_lev, alloc, mt = 3, 0.7, 0.01
        else: max_lev, alloc, mt = 5, 1.0, 0.0

        vb = 0.025
        if vol > vb*2.5: alloc *= 0.3
        elif vol > vb*1.5: alloc *= 0.6

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
                if lows[i+1] <= entry*(1-1/best_lev): port = 0
            else:
                if highs[i+1] >= entry*(1+1/best_lev): port = 0
            if port > 0:
                br = best_dir * (exit_p - entry) / entry
                port *= (1 + br * best_lev * alloc)
        curve.append(port)

    return curve, avg_ac, ac_curve

# Run all
print("Running all test sets...")
results = {}
for name, paths in test_sets.items():
    curve, ac, ac_curve = run_and_get_curve(paths)
    results[name] = (curve, ac, ac_curve)
    ret = (curve[-1]/3000-1)*100
    print(f"  {name:<15s} AC={ac:+.3f} Final=${curve[-1]:>15,.0f} Ret={ret:>+12.1f}%")

# ============================================================
# PLOT 1: Equity Curves (main 3 datasets)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top-left: Synthetic
ax = axes[0, 0]
curve = results["Synthetic"][0]
ax.semilogy(curve, color='#58a6ff', linewidth=2)
ax.axhline(3000, color='#8b949e', linestyle='--', alpha=0.5)
ax.set_title(f'Synthetic (AC={results["Synthetic"][1]:.3f})', color='#58a6ff', fontweight='bold')
ax.set_ylabel('Portfolio ($, log scale)')
ax.set_xlabel('Day')
final = curve[-1]
ax.text(0.98, 0.05, f'${final:,.0f}', transform=ax.transAxes, ha='right', fontsize=14, color='#3fb950', fontweight='bold')
ax.grid(alpha=0.1)

# Top-right: Regime Mix
ax = axes[0, 1]
curve = results["Regime Mix"][0]
ax.semilogy(curve, color='#d2a8ff', linewidth=2)
ax.axhline(3000, color='#8b949e', linestyle='--', alpha=0.5)
ax.set_title(f'Regime Mix (AC={results["Regime Mix"][1]:.3f})', color='#d2a8ff', fontweight='bold')
ax.set_ylabel('Portfolio ($, log scale)')
ax.set_xlabel('Day')
final = curve[-1]
ax.text(0.98, 0.05, f'${final:,.0f}', transform=ax.transAxes, ha='right', fontsize=14, color='#3fb950', fontweight='bold')
ax.grid(alpha=0.1)

# Bottom-left: Realistic
ax = axes[1, 0]
curve = results["Realistic"][0]
ax.plot(curve, color='#f85149', linewidth=2)
ax.axhline(3000, color='#8b949e', linestyle='--', alpha=0.5)
ax.set_title(f'Realistic (AC={results["Realistic"][1]:.3f})', color='#f85149', fontweight='bold')
ax.set_ylabel('Portfolio ($)')
ax.set_xlabel('Day')
final = curve[-1]
color = '#3fb950' if final >= 3000 else '#f85149'
ax.text(0.98, 0.05, f'${final:,.0f}', transform=ax.transAxes, ha='right', fontsize=14, color=color, fontweight='bold')
ax.grid(alpha=0.1)

# Bottom-right: AC vs Final Return
ax = axes[1, 1]
acs = []
rets = []
labels = []
for name, (curve, ac, _) in results.items():
    ret = (curve[-1]/3000-1)*100
    acs.append(ac)
    rets.append(ret)
    labels.append(name)

# Color by return
colors = ['#3fb950' if r > 0 else '#f85149' for r in rets]
sizes = [120 if 'Synthetic' in l or 'Regime' in l or 'Realistic' == l else 50 for l in labels]

ax.scatter(acs, [np.log10(max(r, 0.01)) if r > 0 else -np.log10(max(abs(r), 0.01)) for r in rets], 
           c=colors, s=sizes, alpha=0.8, edgecolors='white', linewidths=0.5)

# Label main ones
for name, ac, ret in zip(labels, acs, rets):
    if name in ['Synthetic', 'Regime Mix', 'Realistic']:
        logret = np.log10(max(ret, 0.01)) if ret > 0 else -np.log10(max(abs(ret), 0.01))
        ax.annotate(f'{name}\n{ret:+.0f}%', (ac, logret), fontsize=9, color='white',
                   ha='center', va='bottom', fontweight='bold')

ax.axhline(0, color='#8b949e', linestyle='--', alpha=0.5)
ax.axvline(0.15, color='#ffa657', linestyle=':', alpha=0.5, label='AC=0.15 threshold')
ax.axvline(0.30, color='#3fb950', linestyle=':', alpha=0.5, label='AC=0.30 threshold')
ax.set_xlabel('Autocorrelation (lag-1)')
ax.set_ylabel('Return (log10 scale)')
ax.set_title('Autocorrelation vs Strategy Return', fontweight='bold')
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.1)

plt.suptitle('Hybrid ML Strategy — Performance by Data Regime', fontsize=16, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.savefig('strategy_performance.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("\nSaved: strategy_performance.png")

# ============================================================
# PLOT 2: 10 Meto scenarios overlay
# ============================================================
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for sc in range(1, 11):
    name = f"Meto-{sc:02d}"
    curve, ac, _ = results[name]
    final = curve[-1]
    color = '#3fb950' if final >= 3000 else '#f85149'
    alpha = 0.8 if final >= 3000 else 0.4
    ax1.plot(curve, color=color, alpha=alpha, linewidth=1.5, label=f'#{sc} ${final:,.0f}')

ax1.axhline(3000, color='#ffa657', linestyle='--', linewidth=2, alpha=0.7, label='Starting $3K')
ax1.set_title('10 Meto-Test Scenarios (Realistic Data)', fontweight='bold')
ax1.set_xlabel('Day')
ax1.set_ylabel('Portfolio ($)')
ax1.legend(fontsize=7, loc='upper left', ncol=2)
ax1.grid(alpha=0.1)

# AC distribution of 10 scenarios
meto_acs = [results[f"Meto-{sc:02d}"][1] for sc in range(1, 11)]
meto_rets = [(results[f"Meto-{sc:02d}"][0][-1]/3000-1)*100 for sc in range(1, 11)]
colors2 = ['#3fb950' if r > 0 else '#f85149' for r in meto_rets]
ax2.bar(range(1, 11), meto_rets, color=colors2, alpha=0.8, edgecolor='white', linewidth=0.5)
ax2.axhline(0, color='#8b949e', linestyle='--')
ax2.set_xticks(range(1, 11))
ax2.set_xlabel('Scenario #')
ax2.set_ylabel('Return (%)')
ax2.set_title('10 Scenario Returns (Realistic Data)', fontweight='bold')
for i, (r, ac) in enumerate(zip(meto_rets, meto_acs)):
    ax2.text(i+1, r + (5 if r > 0 else -15), f'{r:+.0f}%\nAC={ac:.2f}', ha='center', fontsize=7, color='white')
ax2.grid(alpha=0.1)

plt.tight_layout()
plt.savefig('meto_scenarios.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("Saved: meto_scenarios.png")
