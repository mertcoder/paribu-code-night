# Paribu Code Night - Submission

Bu repo, CNLIB formatinda calisan ana trade stratejisini ve ek analiz/test
dosyalarini icerir. Yarismada degerlendirilmesi gereken ana dosya:

```text
strategy.py
```

## Hizli Calistirma

Python 3.10+ onerilir.

```powershell
python -m pip install -r requirements.txt
python strategy.py
```

`python strategy.py` su akisi calistirir:

1. CNLIB training verisini yukler.
2. `HybridStrategy` icindeki ML modelini egitir.
3. `cnlib.backtest.run(...)` ile backtest calistirir.
4. Sonuclari terminale yazdirir.

Programatik kullanim:

```python
from cnlib import backtest
from strategy import HybridStrategy

strategy = HybridStrategy()
strategy.get_data()
strategy.egit()

result = backtest.run(strategy=strategy, initial_capital=3000.0)
result.print_summary()
```

## Strateji Ozeti

`HybridStrategy`, CNLIB `BaseStrategy` sinifindan miras alir ve her candle icin
3 coin hakkinda karar dondurur:

```python
{"coin": "...", "signal": 1 | -1 | 0, "allocation": 0.0-1.0, "leverage": 1 | 2 | 3 | 5}
```

Yuksek seviyede karar akisi:

- GradientBoosting tabanli ML modeli sonraki yonu tahmin eder.
- Momentum sinyali ile ML sinyali ayni yondeyse aday trade olusur.
- Son 100 gunluk return autocorrelation piyasa rejimi olarak kullanilir.
- Autocorrelation dusukse strateji defansif/flat kalir.
- Autocorrelation yuksekse allocation ve leverage daha agresif ayarlanir.
- Son 30 gunluk volatilite asiri yuksekse allocation azaltilir.
- Toplam aktif allocation 1.0 sinirini asmaz.

## Dosya Yapisi

```text
strategy.py                         Ana CNLIB strateji/submission dosyasi
requirements.txt                    Gerekli Python paketleri
base-doc.md                         CNLIB kullanim notlari
app.py                              Opsiyonel Streamlit veri gorsellestirme UI
data/cnlib_train/                   Lokal CNLIB train parquet snapshot'i
data/unseen_test_1y/                Synthetic test set 1
data/unseen_test_1y_regime_mix/     Synthetic test set 2
data/realistic_test_1y/             Realistic random-walk test set 3
scripts/                            Test verisi uretim scriptleri
test_*.py, validate.py, diagnose.py Opsiyonel analiz/dogrulama araclari
```

## Opsiyonel UI

Strateji degerlendirmesi icin UI gerekli degildir. Verileri gormek isteyenler
su komutla lokal dashboard acabilir:

```powershell
python -m streamlit run app.py
```

Dashboard acildiktan sonra:

```text
http://127.0.0.1:8501
```

UI icinde `Test verisi` secimiyle train verisi, baseline synthetic test,
regime-mix test ve realistic random-walk test setleri gosterilebilir.

## Test Verileri

Ek test setleri yarismadaki resmi degerlendirme yerine gecmez; stratejinin
farkli piyasa rejimlerine tepkisini lokal olarak gormek icin eklenmistir.

Realistic random-walk test seti:

- Konum: `data/realistic_test_1y/`
- Date: `2027-03-16` - `2028-03-14`
- Split: `realistic_test`
- Source: `realistic_random_walk`
- Lag-1 autocorrelation hedefi: `0.00` - `0.05`
- Gunluk volatilite hedefi: `%2` - `%4`
- Coinler arasi korelasyon hedefi: `0.3` - `0.6`

Yeniden uretmek icin:

```powershell
python scripts/generate_realistic_test_data.py
python scripts/generate_unseen_test_data.py
```

## Notlar

- Ana degerlendirme icin sadece `strategy.py` ve `requirements.txt` yeterlidir.
- `app.py`, `data/` ve `test_*.py` dosyalari analiz/gorsellestirme icindir.
- Repo mevcut mantik korunarak hazirlanmistir; bu dokuman sadece calistirma ve
  degerlendirme surecini netlestirmek icindir.
