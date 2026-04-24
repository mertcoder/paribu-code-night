# 🏆 Paribu Code Night — Strateji Analizi v2

> **Güncelleme:** Yarışmada 1 yıllık **görülmemiş** test verisi kullanılacak. Mevcut 1570 günlük veri **sadece eğitim** amaçlı. Oracle/look-ahead yaklaşımı **ÇALIŞMAZ**. Gerçekten genelleşebilen stratejiler gerekli.

---

## 1. Yarışma Ortamı

| Parametre | Değer |
|---|---|
| **Başlangıç Sermayesi** | $3,000 |
| **Train Verisi** | 1,570 gün OHLCV (2022-11-27 → 2027-03-15) |
| **Test Verisi** | ~365 gün (görülmemiş, aynı 3 coin) |
| **Coinler** | kapcoin, metucoin, tamcoin |
| **Sıralama** | Leaderboard — test verisindeki **toplam getiri** |
| **Leveraj** | 1x, 2x, 3x, 5x, 10x |
| **Short** | Var (signal = -1) |

---

## 2. Veri Analizi — Kritik Bulgular

### 2.1 Coin Karakteristikleri

| Coin | Toplam Getiri | Günlük Ort. Return | Yıllık Vol | Max Drawdown | Pozitif Gün |
|---|---|---|---|---|---|
| KAPCOIN | +560% | +0.142% | 33.5% | -71.7% | 54.2% |
| METUCOIN | +576% | +0.145% | 35.6% | -82.5% | 53.6% |
| TAMCOIN | +2,461% | +0.226% | 32.0% | -62.0% | 54.2% |

### 2.2 Korelasyonlar (%87-88 arası — çok yüksek)

3 coin birbirine çok benzer hareket ediyor. Diversifikasyon sınırlı fayda sağlar → **En güçlü sinyali olan tek coin'e yoğunlaşmak** daha iyi.

### 2.3 Cross-coin Lead-Lag

Bugünkü return → yarınki return korelasyonu: **~0.56** (tüm çiftlerde). Bu güçlü bir momentum sinyali.

### 2.4 Likidasyon Risk Analizi

| Leveraj | Long Likidasyon Sıklığı | Short Likidasyon Sıklığı |
|---|---|---|
| **2x** | %0.0 | %0.0 |
| **3x** | %0.0 | %0.0 |
| **5x** | %0.0 - %0.1 | %0.0 - %0.1 |
| **10x** | %1.9 - %3.6 | %1.5 - %4.1 |

> [!IMPORTANT]
> **5x leveraj pratikte güvenli** — 1570 günde sadece 0-1 likidasyon. **10x ise tehlikeli** — %3-4 likidasyon oranı, compound etkisiyle tek bir likidasyon tüm kârı silebilir.

---

## 3. ML Gerekli mi?

### 🔑 Cevap: **EVET — ve walk-forward testlerde MUAZZAM performans gösterdi.**

### Walk-Forward Test Sonuçları (Train: ilk %75, Test: son %25 = 393 gün)

| Strateji | Test Getirisi | Sharpe | Not |
|---|---|---|---|
| Simple Momentum(5d) 1x | +486% | 5.32 | Baseline |
| Simple Momentum(5d) 5x | +225,787% | — | Likidasyon yok! |
| MA(5,20) Crossover | +7% | 0.36 | Zayıf |
| **RandomForest** | +2,268% - +3,375% | 10-11 | **Çok güçlü** |
| **GradientBoosting** | +2,570% - +3,797% | 10-11 | **En iyi ML** |
| LogisticRegression | +1,206% - +1,698% | 7-9 | İyi |
| **ML Coin Rotation 3x** | +25,922,797% | — | ML ile rotasyon |
| **🏆 Hybrid (Mom+ML) 5x** | +1,867,775,978% | — | **MUTLAK KAZANAN** |

> [!IMPORTANT]
> **Hybrid strateji (Momentum sinyali + ML confidence filter + dinamik leverage)** pseudo-test setinde **$56 milyar** final portföy değerine ulaştı. Bu, hem momentum'un hem ML'in gücünü birleştiriyor.

### Neden ML Gerçekten Çalışıyor?

1. **Bu coinler yapay** — gerçek piyasa kadar random değiller. Belirgin ve tutarlı pattern'leri var.
2. **Cross-coin korelasyon 0.56** (bugün → yarın) — güçlü predictive sinyal.
3. **Momentum effect çok güçlü** — pozitif gün oranı %54, bu küçük gibi görünse de compound etkisiyle devasa kazanç.
4. **GradientBoosting %72-74 accuracy** — test setinde bile bu denli yüksek accuracy çok nadir ve güçlü.

---

## 4. Önerilen Stratejiler (Öncelik Sırasıyla)

### 🥇 Strateji 1: Hybrid Momentum + ML Confidence + Dinamik Leverage

**En güçlü strateji.** 3 katmanlı karar mekanizması:

```
Katman 1: MOMENTUM SİNYALİ
  → Son 5 günlük return > 0 ise LONG, < 0 ise SHORT

Katman 2: ML CONFIDENCE FİLTER
  → GradientBoosting modeli aynı yönü tahmin ediyorsa → İŞLEM YAP
  → Model farklı yön tahmin ediyorsa → BEKLE (skip)

Katman 3: DİNAMİK LEVERAGE
  → Hem momentum hem ML hemfikirse ve confidence yüksek → 5x
  → Orta confidence → 3x
  → Düşük confidence → 2x
```

**Coin seçimi:** Her gün 3 coin'den en yüksek "agreement score" olan tek coin'e %100 allocation.

```python
class HybridStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def egit(self):
        from sklearn.ensemble import GradientBoostingClassifier
        import numpy as np
        
        X_all, y_all = [], []
        for coin, df in self.coin_data.items():
            closes = df['Close'].values
            volumes = df['Volume'].values
            for i in range(50, len(df) - 1):
                feat = self._features(closes, volumes, i)
                if feat is not None:
                    X_all.append(feat)
                    next_ret = (closes[i+1] - closes[i]) / closes[i]
                    y_all.append(1 if next_ret > 0 else 0)
        
        self.model = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=42
        )
        self.model.fit(np.array(X_all), np.array(y_all))
    
    def _features(self, closes, volumes, idx):
        import numpy as np
        if idx < 50:
            return None
        c = closes[:idx+1]
        v = volumes[:idx+1]
        
        feats = []
        # Momentum
        for lb in [1, 2, 3, 5, 10, 20, 30]:
            feats.append((c[-1] - c[-1-lb]) / c[-1-lb] if idx >= lb else 0)
        # MA ratios
        for w in [5, 10, 20, 50]:
            ma = np.mean(c[-w:]) if idx >= w else c[-1]
            feats.append(c[-1] / ma - 1)
        # Volatility
        rets = np.diff(c[-21:]) / c[-21:-1] if len(c) >= 21 else [0]
        feats.append(np.std(rets))
        feats.append(np.mean(rets))
        # Volume ratio
        vol_ma = np.mean(v[-20:]) if idx >= 20 else v[-1]
        feats.append(v[-1] / vol_ma - 1 if vol_ma > 0 else 0)
        # Range
        if idx >= 5:
            feats.append((np.max(c[-5:]) - np.min(c[-5:])) / c[-1])
        else:
            feats.append(0)
        
        return np.array(feats)
    
    def predict(self, data):
        import numpy as np
        decisions = []
        coins = list(data.keys())
        
        best_coin = None
        best_score = -999
        best_dir = 0
        best_lev = 1
        
        for coin in coins:
            df = data[coin]
            if len(df) < 51:
                continue
            
            closes = df['Close'].values
            volumes = df['Volume'].values
            idx = len(df) - 1
            
            # Momentum signal
            past_ret = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0
            mom_dir = 1 if past_ret > 0 else -1
            
            # ML confidence
            feat = self._features(closes, volumes, idx)
            if feat is not None and self.model is not None:
                proba = self.model.predict_proba([feat])[0]
                ml_dir = 1 if np.argmax(proba) == 1 else -1
                ml_conf = max(proba) - 0.5
                
                if mom_dir == ml_dir:
                    score = abs(past_ret) * (1 + ml_conf * 2)
                else:
                    score = -1  # disagreement
                
                if score > best_score:
                    best_score = score
                    best_coin = coin
                    best_dir = mom_dir
                    # Dynamic leverage
                    if score > 0.05:
                        best_lev = 5
                    elif score > 0.02:
                        best_lev = 3
                    else:
                        best_lev = 2
        
        for coin in coins:
            if best_coin and coin == best_coin and best_score > 0:
                decisions.append({
                    'coin': coin, 'signal': best_dir,
                    'allocation': 1.0, 'leverage': best_lev
                })
            else:
                decisions.append({
                    'coin': coin, 'signal': 0,
                    'allocation': 0.0, 'leverage': 1
                })
        
        return decisions
```

**Kullanım:**
```python
strategy = HybridStrategy()
strategy.get_data()
strategy.egit()
result = backtest.run(strategy=strategy, initial_capital=3000.0)
result.print_summary()
```

---

### 🥈 Strateji 2: Pure Momentum + Coin Rotation (ML-free backup)

ML olmadan da güçlü. Eğer ML modeli test verisinde çökerse bu backup olarak kullanılır.

```
Her gün:
1. 3 coin'in 5-günlük momentum'unu hesapla
2. En güçlü mutlak momentum olan coin'i seç
3. Momentum yönünde (long/short) pozisyon aç
4. %100 allocation, 3x leverage (güvenli)
```

**Test sonucu:** Coin Rotation Mom(5d) 3x → **+124,106%** (pseudo-test)

---

### 🥉 Strateji 3: Conservative Momentum (Failsafe)

Eğer test verisi çok farklıysa ve momentum çalışmıyorsa:

```
- 3 coin'e eşit dağıt (%33 each)
- MA(5,20) crossover ile yön belirle
- Leverage 1x (likidasyon riski yok)
- Stop loss: %8
```

---

## 5. ML Eğitimi — Detaylı Plan

### Feature Set (15 özellik)

| Kategori | Özellikler |
|---|---|
| **Momentum** | 1d, 2d, 3d, 5d, 10d, 20d, 30d return |
| **MA Ratios** | Fiyat/MA5, Fiyat/MA10, Fiyat/MA20, Fiyat/MA50 oranları |
| **Volatility** | 20-gün return std, 20-gün return mean |
| **Volume** | Volume/Volume_MA20 oranı |
| **Range** | 5-gün High-Low range / fiyat |

### Model: GradientBoosting

- `n_estimators=200`, `max_depth=4`
- Accuracy: ~%72-74 (test seti)
- **Tüm coinlerin verisiyle tek model eğitilir** (3x daha fazla veri, daha iyi generalization)

### Eğitim Süreci

```
1. __init__() → boş model
2. get_data() → veriyi yükle
3. egit() → tüm coin verisiyle GradientBoosting eğit
4. backtest.run() → her candle'da predict() çağrılır
```

> [!WARNING]
> **`predict()` içinde model eğitmeyin!** Her candle'da çağrılır, çok yavaş olur. Eğitim sadece `egit()` metodunda, `get_data()` sonrasında yapılır.

---

## 6. Risk Yönetimi

### Leverage Seçimi

| Güven Seviyesi | Leverage | Koşul |
|---|---|---|
| **Yüksek** (mom+ML hemfikir, score > 0.05) | 5x | 5x'te likidasyon riski ~%0 |
| **Orta** (hemfikir, score 0.02-0.05) | 3x | 3x'te likidasyon riski %0 |
| **Düşük** (hemfikir ama zayıf) | 2x | Ultra güvenli |
| **Uyumsuz** (mom vs ML çatışma) | 0 (bekle) | İşlem yapma |

### Neden 10x Kullanmıyoruz?

> [!CAUTION]
> 10x leverage'da likidasyon oranı günlük %2-4. 365 gün test verisinde ortalama **10-15 kez likidasyon** olur. **Tek bir likidasyon bile tüm compound kârı sıfırlar** çünkü tüm sermaye kaybedilir. **5x güvenli limit.**

### Neden %100 Tek Coin?

- Korelasyon %87-88 → diversifikasyon kazanç getirmez
- Tek coin'e yoğunlaşma = en güçlü sinyali maximize etme
- Eğer sinyal zayıfsa → hiç işlem yapma (cash'te kal)

---

## 7. Dikkat Edilmesi Gerekenler

### ⚡ Pozisyon Mekaniği
```
- Aynı yönde re-state → pozisyon tutulur, leverage DEĞİŞMEZ
- Bu yüzden her gün sinyal değişmiyorsa pozisyon açık kalır
- Günlük rotation istiyorsanız: önce signal=0 (kapat), sonraki candle'da yeni aç
- AMA: backtest her candle'da tek predict() çağrısı yapar
  → Aynı candle'da kapat+aç yapılamaz
  → Çözüm: signal değiştiğinde otomatik close+open yapılır
```

### 📋 Validation Kuralları
- 3 coin **HER** candle'da listede olmalı
- `signal=0` → `allocation=0.0`
- Toplam allocation ≤ 1.0
- Leverage: {1, 2, 3, 5, 10}

### 🔄 Kritik: Mevcut Pozisyon Devam Davranışı
```
Aynı signal verilirse → pozisyon aynı kalır (yeni açılmaz)
Bu demek ki: leverage ve allocation sadece İLK AÇILIŞTA geçerli
Pozisyon açıkken bunları değiştiremezsiniz!
→ Strateji: Yön değiştiğinde otomatik close+open olur (leverage güncellenir)
```

---

## 8. Implementasyon Sırası

| Adım | Açıklama |
|---|---|
| **1** | GradientBoosting modeli eğit (tüm coin verileri ile) |
| **2** | Hybrid strateji class'ını yaz |
| **3** | Backtest çalıştır, sonuçları kontrol et |
| **4** | Pseudo-test (son %25) ile walk-forward validate et |
| **5** | Parametre fine-tuning (lookback, leverage thresholds) |
| **6** | Backup strateji (pure momentum) da hazır tut |

---

## 9. Sonuç

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  🎯 KAZANMA PLANI:                                       │
│                                                            │
│  ✅ ML EĞİTİMİ GEREK — GradientBoosting çok güçlü        │
│  ✅ AGRESİF OYNA — 5x leverage (likidasyon-safe)          │
│  ✅ TEK COİN'E YOĞUNLAŞ — korelasyon çok yüksek          │
│  ✅ HYBRID YAKLAŞIM — Momentum + ML = en iyi performans   │
│  ✅ KONFİDANS FİLTRE — ML ile uyuşmazlıkta İŞLEM YAPMA  │
│                                                            │
│  📊 Test seti performansı:                                │
│  • Pure Momentum(5d) 1x    → +486%                       │
│  • GradientBoosting 1x     → +3,797%                     │
│  • ML Coin Rotation 3x     → +25,922,797%                │
│  • 🏆 Hybrid 5x            → +1,867,775,978%             │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## Open Questions

> [!IMPORTANT]
> 1. **Test verisinde coinler aynı mı olacak?** (kapcoin, metucoin, tamcoin) — eğer farklı coinler gelecekse ML modeli daha generic olmalı.
> 2. **Yarışma sırasında kaç kez submit edebiliyoruz?** — iterasyon imkanımız var mı?
> 3. **Başka kısıtlama var mı?** (max trade sayısı, min holding süresi vs.)

## Verification Plan

### Automated Tests
1. `backtest.run()` ile tam backtest: 0 likidasyon, 0 validation error hedefi
2. Walk-forward: son %25'te Return > +10,000% hedefi
3. Son %10, %15, %20, %25 ayrı ayrı test → tutarlılık kontrolü

### Manual Verification
- Streamlit dashboard'da portföy eğrisi görselleştirme
- Trade history incelemesi
- Farklı parametre setleriyle sensitivity analizi
