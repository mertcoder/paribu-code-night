"""
Convert jury-provided 1-year OHLCV files into cnlib-compatible parquet files.

Expected input files (CSV or Parquet) with columns:
    Date, Open, High, Low, Close, Volume

Coin mapping — rename input files to the cnlib internal names:
    kapcoin  →  kapcoin-usd_train.parquet
    metucoin →  metucoin-usd_train.parquet
    tamcoin  →  tamcoin-usd_train.parquet

Usage:
    python scripts/prepare_jury_data.py --input-dir jury_data/ --output-dir jury_parquet/

    # If your files are already named with the cnlib suffix:
    python scripts/prepare_jury_data.py --input-dir jury_data/ --output-dir jury_parquet/ --no-rename

Then run the strategy on the prepared data:
    python strategy.py --test-data-dir jury_parquet/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Volume"}

# Prefix → cnlib output filename.
# Any file whose stem *starts with* one of these prefixes is matched.
COIN_PREFIXES: list[tuple[str, str]] = [
    ("kapcoin",  "kapcoin-usd_train"),
    ("metucoin", "metucoin-usd_train"),
    ("tamcoin",  "tamcoin-usd_train"),
]

def _resolve_coin(stem: str) -> str | None:
    """Return the cnlib target name for a given file stem, or None if unrecognised."""
    s = stem.lower()
    for prefix, target in COIN_PREFIXES:
        if s == prefix or s.startswith(prefix + "-") or s.startswith(prefix + "_"):
            return target
    return None

CNLIB_COINS = {"kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"}


def _load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _validate(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        print(f"  ERROR: {source.name} is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    df = df[list(REQUIRED_COLUMNS)].copy()
    df["Date"] = df["Date"].astype(str)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["Open", "High", "Low", "Close", "Volume"]].isna().any().any():
        print(f"  WARNING: {source.name} contains NaN values after conversion.", file=sys.stderr)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare jury data for cnlib backtest")
    parser.add_argument("--input-dir", type=Path, required=True, metavar="DIR",
                        help="Directory containing jury OHLCV files (CSV or Parquet)")
    parser.add_argument("--output-dir", type=Path, required=True, metavar="DIR",
                        help="Directory where cnlib-ready parquet files will be written")
    parser.add_argument("--no-rename", action="store_true",
                        help="Skip coin alias mapping; input stems must already match cnlib names")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: --input-dir '{args.input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        p for p in args.input_dir.iterdir()
        if p.suffix.lower() in {".csv", ".parquet"} and not p.name.startswith(".")
    )
    if not candidates:
        print(f"ERROR: No CSV/Parquet files found in '{args.input_dir}'.", file=sys.stderr)
        sys.exit(1)

    written: set[str] = set()

    for src in candidates:
        stem = src.stem.lower()

        if args.no_rename:
            if stem not in CNLIB_COINS:
                print(f"  SKIP  {src.name}  (stem '{stem}' not a cnlib coin name)")
                continue
            target_stem = stem
        else:
            target_stem = _resolve_coin(stem)
            if target_stem is None:
                print(f"  SKIP  {src.name}  (unrecognised coin '{stem}')")
                continue

        print(f"  {src.name}  ->  {target_stem}.parquet")
        df = _load(src)
        df = _validate(df, src)

        out = args.output_dir / f"{target_stem}.parquet"
        df.to_parquet(out, index=False)
        written.add(target_stem)

    missing_coins = CNLIB_COINS - written
    if missing_coins:
        print(
            f"\nWARNING: The following coins were not found in input and will be missing:\n"
            f"  {missing_coins}\n"
            "The backtest requires all three coins.",
            file=sys.stderr,
        )
    else:
        print(f"\nDone. {len(written)} parquet files written to '{args.output_dir}'.")
        print(f"Run:  python strategy.py --test-data-dir {args.output_dir}")


if __name__ == "__main__":
    main()
