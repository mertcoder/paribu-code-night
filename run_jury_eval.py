"""
Juri degerlendirme scripti — tek komutla calisir.

Kullanim:
    python run_jury_eval.py --data-dir <juri_veri_klasoru>

Ornek:
    python run_jury_eval.py --data-dir jury_data/

Beklenen giris dosyalari (CSV veya Parquet, Date/Open/High/Low/Close/Volume kolonlu):
    jury_data/kapcoin-usd_train.parquet   (veya kapcoin.csv, kapcoin-usd.csv vb.)
    jury_data/metucoin-usd_train.parquet
    jury_data/tamcoin-usd_train.parquet

Cikti dosyalari (--output-dir ile degistirilebilir, varsayilan: eval_output/):
    eval_output/summary.txt          Backtest ozeti
    eval_output/portfolio_daily.csv  Her gunun portfoy degeri
    eval_output/trades.csv           Sadece islem olan gunler
    eval_output/signals_daily.csv    Her gunun coin sinyalleri + rejim bilgisi
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from cnlib import backtest
from strategy import HybridStrategy


def _prepare_data(input_dir: Path, work_dir: Path) -> Path:
    """
    input_dir icindeki dosyalari cnlib-uyumlu parquet'e donusturur.
    Dosyalar zaten dogru isimde parquet ise dogrudan kullanir.
    """
    cnlib_names = {
        "kapcoin-usd_train",
        "metucoin-usd_train",
        "tamcoin-usd_train",
    }

    # Zaten hazir mi? (3 parquet dosyasi dogru isimle varsa)
    ready = all((input_dir / f"{n}.parquet").exists() for n in cnlib_names)
    if ready:
        return input_dir

    # Yoksa prepare scriptini calistir
    out = work_dir / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "scripts" / "prepare_jury_data.py"),
        "--input-dir", str(input_dir),
        "--output-dir", str(out),
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("HATA: Veri hazirlama basarisiz.", file=sys.stderr)
        sys.exit(1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Juri 1-yillik test verisi uzerinde stratejiyi calistir ve gunu gune raporla"
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, metavar="DIR",
        help="Juri OHLCV dosyalarinin bulundugu klasor (CSV veya Parquet)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("eval_output"), metavar="DIR",
        help="Cikti dosyalarinin yazilacagi klasor (varsayilan: eval_output/)"
    )
    parser.add_argument(
        "--capital", type=float, default=3000.0, metavar="FLOAT",
        help="Baslangic sermayesi USD (varsayilan: 3000.0)"
    )
    parser.add_argument(
        "--silent", action="store_true",
        help="Backtest ilerleme ciktisini gizle"
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"HATA: --data-dir '{args.data_dir}' bulunamadi.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Veri hazirlama ---
    print(f"[1/3] Veri hazirlaniyor: {args.data_dir}")
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = _prepare_data(args.data_dir, Path(tmp))

        # --- Model egitimi ---
        print("[2/3] Model egitiliyor (cnlib 4-yillik egitim verisi)...")
        strategy = HybridStrategy()
        strategy.get_data()   # bundled training data
        strategy.egit()

        # --- Backtest ---
        print(f"[3/3] Backtest calistiriliyor ({data_dir})...")
        result = backtest.run(
            strategy=strategy,
            initial_capital=args.capital,
            data_dir=data_dir,
            silent=args.silent,
        )

    # === Ciktilari yaz ===

    # 1) Ozet
    summary_path = args.output_dir / "summary.txt"
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result.print_summary()
    summary_text = buf.getvalue()
    print(summary_text)
    summary_path.write_text(summary_text, encoding="utf-8")

    # 2) Gunluk portfoy
    portfolio_path = args.output_dir / "portfolio_daily.csv"
    pd.DataFrame(result.portfolio_series).to_csv(portfolio_path, index=False)

    # 3) Islemler
    trades_path = args.output_dir / "trades.csv"
    if result.trade_history:
        def _coins(entries):
            if not entries:
                return ""
            first = entries[0]
            if isinstance(first, dict):
                return ", ".join(e.get("coin", str(e)) for e in entries)
            return ", ".join(str(e) for e in entries)

        rows = []
        for t in result.trade_history:
            rows.append({
                "candle_index": t["candle_index"],
                "date": t["timestamp"],
                "opened": _coins(t["opened"]),
                "closed": _coins(t["closed"]),
                "liquidated": len(t["liquidated"]),
                "portfolio_value": t["portfolio_value"],
            })
        pd.DataFrame(rows).to_csv(trades_path, index=False)
    else:
        pd.DataFrame(columns=["candle_index", "date", "opened", "closed", "liquidated", "portfolio_value"]).to_csv(trades_path, index=False)

    # 4) Gun gunu sinyaller
    signals_path = args.output_dir / "signals_daily.csv"
    if strategy.decisions_log:
        pd.DataFrame(strategy.decisions_log).to_csv(signals_path, index=False)
    else:
        pd.DataFrame(columns=["date", "coin", "signal", "allocation", "leverage", "regime_autocorr", "avg_vol"]).to_csv(signals_path, index=False)

    print(f"\nCikti dosyalari: {args.output_dir}/")
    print(f"  {summary_path.name:<25} — backtest ozeti")
    print(f"  {portfolio_path.name:<25} — {len(result.portfolio_series)} gunluk portfoy serisi")
    print(f"  {trades_path.name:<25} — {len(result.trade_history)} islem gunu")
    print(f"  {signals_path.name:<25} — {len(strategy.decisions_log)} satir sinyal logu")


if __name__ == "__main__":
    main()
