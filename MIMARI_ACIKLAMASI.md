# Mimari Aciklamasi

Bu projede amacimiz CNLIB formatina uygun, tek dosyada calisabilen ama piyasa
rejimine gore kendini sakinlestirebilen bir trade stratejisi kurmakti.

Ana submission dosyamiz `strategy.py` icindeki `HybridStrategy` sinifidir.
Bu sinif `BaseStrategy`'den miras alir ve her candle kapanisinda 3 coin icin
karar dondurur: long, short veya flat.

## Stratejinin Kalbi

Strateji uc ana parcadan olusur:

1. **ML modeli**
   - GradientBoostingClassifier kullanilir.
   - Momentum, hareketli ortalama, volatilite, RSI, MACD, Bollinger, volume ve
     benzeri teknik feature'larla egitilir.
   - Modelin gorevi bir sonraki fiyat yonu icin yardimci sinyal uretmektir.

2. **Momentum sinyali**
   - Son fiyat hareketinden kisa vadeli yon okunur.
   - ML modeli ile momentum ayni yonu soyluyorsa trade adayi guclenir.
   - Sinyaller uyusmuyorsa strateji daha temkinli davranir.

3. **Rejim filtresi**
   - Son 100 gunluk return autocorrelation hesaplanir.
   - Autocorrelation yuksekse piyasa momentum karakteri tasiyor kabul edilir.
   - Autocorrelation dusukse piyasa random-walk gibi gorulur ve strateji
     agresiflesmek yerine sermayeyi korur.

## Risk Yonetimi

Strateji sadece sinyal uretmeye odaklanmiyor; ne kadar risk alinacagini da
piyasa kosuluna gore ayarliyor.

- Dusuk autocorrelation: trade yok veya cok defansif davranis.
- Orta autocorrelation: kademeli allocation ve leverage.
- Yuksek autocorrelation: daha agresif allocation ve leverage.
- Yuksek volatilite: pozisyon boyutu azaltilir.
- Toplam allocation 1.0 sinirini asmaz.
- Her candle'da 3 coin icin de karar dondurulur.

Bu nedenle strateji sadece "kar kovala" mantiginda degil; uygun rejim yoksa
islem acmama secenegini de bilincli olarak kullanir.

## Veri ve Test Katmani

Orijinal CNLIB training verisi korunur. Bunun yaninda stratejinin farkli piyasa
kosullarinda nasil davrandigini gormek icin ek test setleri olusturduk:

- `data/unseen_test_1y/`: baseline synthetic test yili.
- `data/unseen_test_1y_regime_mix/`: farkli rejim bloklariyla uretilmis test.
- `data/realistic_test_1y/`: gercek kripto piyasasina daha yakin random-walk test.

Bu ek veriler resmi CNLIB datasinin yerine gecmez; stratejinin overfit olup
olmadigini anlamak ve farkli rejimlerde davranisini incelemek icindir.

## Gorsellestirme

`app.py` opsiyonel bir Streamlit dashboard'dur.

Juri stratejiyi calistirmak icin buna ihtiyac duymaz, ama isterse:

```powershell
python -m streamlit run app.py
```

komutuyla train ve test verilerini mum grafikleri uzerinde gorebilir.
UI'da farkli test setleri secilebilir ve test yillari ana grafikte ayri renkle
gosterilir.

## Calistirma

Degerlendirme icin temel akis:

```powershell
python -m pip install -r requirements.txt
python strategy.py
```

`strategy.py` modeli egitir, backtest'i calistirir ve sonucu terminale yazdirir.

## Kisa Ozet

Bu mimari, ML tahmini ile momentum sinyalini birlestirir; fakat asil guvenlik
katmani 100 gunluk autocorrelation ve volatilite filtresidir. Piyasa momentum
tasiyorsa strateji daha cesur, piyasa rastgele yuruyorsa daha korumaci davranir.

Yani sistemin temel fikri sudur:

```text
ML sinyali + momentum onayi + rejim filtresi + volatilite kontrolu = final trade karari
```

