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

## Juri Degerlendirmesi (1 Yillik Test Verisi)

### Nasil calisir?

`cnlib.backtest.run()` her gun (candle) kapanisinda `strategy.predict(data)` cagiriyor.
`data` o gune kadar gelen tum fiyat gecmisini icerir; strateji her gun 3 coin icin
sinyal + allocation + leverage donduruyor. Yani juri kendi 1-yillik verisini verdigi
anda sistem bunu **gun gun, candle candle** isliyor — ekstra kod yazmaya gerek yok.

### Tek komutla tam degerlendirme

`run_jury_eval.py` su adimlarin tamamini otomatik yapar:
veriyi hazirlar, modeli egitir, backtest calistirir, 4 CSV + ozet dosyasi uretir.

```powershell
python run_jury_eval.py --data-dir jury_data/
```

Beklenen giris (herhangi bir isimde, CSV veya Parquet):

```text
jury_data/
  kapcoin*.csv   veya   kapcoin*.parquet
  metucoin*.csv  veya   metucoin*.parquet
  tamcoin*.csv   veya   tamcoin*.parquet
```

Gerekli kolonlar: `Date, Open, High, Low, Close, Volume`

Uretilen cikti dosyalari (`eval_output/` klasorunde):

```text
summary.txt          Backtest ozeti (return, P&L, likidasyonlar)
portfolio_daily.csv  Her gunun portfoy degeri
trades.csv           Sadece islem acilan/kapanan gunler
signals_daily.csv    Her gun her coin icin sinyal + rejim bilgisi
```

Ek secenekler:

```powershell
# Farkli cikti klasoru ve sermaye
python run_jury_eval.py --data-dir jury_data/ --output-dir sonuclar/ --capital 5000

# Ilerleme ciktisini gizle
python run_jury_eval.py --data-dir jury_data/ --silent
```

### Adim adim (manuel) kullanim

Veriyi manuel hazirlamak isteyenler:

```powershell
# 1. CSV/Parquet -> cnlib-uyumlu parquet
python scripts/prepare_jury_data.py --input-dir jury_data/ --output-dir jury_parquet/

# 2. Stratejiyi calistir
python strategy.py --test-data-dir jury_parquet/
```

### Programatik kullanim

```python
from pathlib import Path
from cnlib import backtest
from strategy import HybridStrategy

strategy = HybridStrategy()
strategy.get_data()        # her zaman cnlib 4-yillik egitim verisi
strategy.egit()

# jury_parquet/ icinde 3 parquet dosyasi olmali
result = backtest.run(strategy=strategy, initial_capital=3000.0,
                      data_dir=Path("jury_parquet/"))
result.print_summary()

# Gun gun sinyal logu
import pandas as pd
signals = pd.DataFrame(strategy.decisions_log)
print(signals.head())
```

Model her zaman CNLIB paketinin icindeki 4 yillik egitim verisiyle egitilir;
backtest yalnizca `--test-data-dir` / `data_dir` altindaki yeni veri uzerinde calisir.

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
run_jury_eval.py                    Juri degerlendirme scripti (tek komutla calisir)
requirements.txt                    Gerekli Python paketleri
base-doc.md                         CNLIB kullanim notlari
app.py                              Opsiyonel Streamlit veri gorsellestirme UI
data/cnlib_train/                   Lokal CNLIB train parquet snapshot'i
data/unseen_test_1y/                Synthetic test set 1
data/unseen_test_1y_regime_mix/     Synthetic test set 2
data/realistic_test_1y/             Realistic random-walk test set 3
scripts/prepare_jury_data.py        Juri CSV/Parquet -> cnlib parquet donusturucu
scripts/generate_*.py               Test verisi uretim scriptleri
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
