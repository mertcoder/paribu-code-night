from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
BASELINE_OUTPUT_DIR = ROOT_DIR / "data" / "unseen_test_1y"
REGIME_OUTPUT_DIR = ROOT_DIR / "data" / "unseen_test_1y_regime_mix"
TRAIN_DATA_DIR = ROOT_DIR / "data" / "cnlib_train"
DAYS_TO_GENERATE = 365

COIN_SEEDS = {
    "kapcoin-usd_train": 240421,
    "metucoin-usd_train": 240422,
    "tamcoin-usd_train": 240423,
}
REGIME_COIN_SEEDS = {
    "kapcoin-usd_train": 250421,
    "metucoin-usd_train": 250422,
    "tamcoin-usd_train": 250423,
}

OHLCV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
TRAIN_PARQUET_FILES = tuple(f"{coin}.parquet" for coin in COIN_SEEDS)


def has_train_parquets(data_dir: Path) -> bool:
    return all((data_dir / file_name).exists() for file_name in TRAIN_PARQUET_FILES)


def resolve_train_data_dir() -> Path:
    if has_train_parquets(TRAIN_DATA_DIR):
        return TRAIN_DATA_DIR

    package_data_dir = Path(__import__("cnlib").__file__).resolve().parent / "data"
    if has_train_parquets(package_data_dir):
        return package_data_dir

    raise FileNotFoundError(
        "CNLIB train parquet files were not found. Expected them under "
        f"{TRAIN_DATA_DIR}."
    )


def normalize_market_frame(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if "Date" not in df.columns:
        df = df.reset_index()
        if "Date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Date"})

    missing = [column for column in OHLCV_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    df = df[OHLCV_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=OHLCV_COLUMNS).sort_values("Date").reset_index(drop=True)


def clipped(values: pd.Series, low_q: float = 0.01, high_q: float = 0.99) -> pd.Series:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return pd.Series([0.0])
    return clean.clip(clean.quantile(low_q), clean.quantile(high_q))


def generate_coin_test_data(coin: str, train_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(COIN_SEEDS[coin])
    df = normalize_market_frame(train_df)

    close = df["Close"].astype(float)
    returns = clipped(close.pct_change(), 0.005, 0.995)
    gaps = clipped(df["Open"] / close.shift(1) - 1, 0.01, 0.99)
    ranges = clipped((df["High"] - df["Low"]) / close, 0.01, 0.99)
    template = df.tail(DAYS_TO_GENERATE).reset_index(drop=True).copy()
    template_returns = clipped(template["Close"].pct_change(), 0.005, 0.995).reset_index(
        drop=True
    )

    ret_values = returns.to_numpy()
    gap_values = gaps.to_numpy()
    range_values = ranges.to_numpy()
    ret_mean = float(returns.mean())
    ret_std = max(float(returns.std()), 1e-8)

    last_date = df["Date"].max()
    previous_close = float(close.iloc[-1])
    rows = []

    for index in range(DAYS_TO_GENERATE):
        template_row = template.iloc[index % len(template)]
        template_ret = float(template_returns.iloc[index % len(template_returns)])
        sampled_ret = float(rng.choice(ret_values))
        noise = float(rng.normal(ret_mean * 0.1, ret_std * 0.08))
        ret = 0.65 * template_ret + 0.25 * sampled_ret + noise
        ret = float(np.clip(ret, returns.quantile(0.005), returns.quantile(0.995)))

        sampled_gap = float(rng.choice(gap_values))
        gap = 0.7 * sampled_gap + float(rng.normal(0, max(gaps.std(), 1e-8) * 0.05))
        gap = float(np.clip(gap, gaps.quantile(0.01), gaps.quantile(0.99)))

        open_price = max(previous_close * (1 + gap), 0.0001)
        close_price = max(previous_close * (1 + ret), 0.0001)

        template_range = float(
            (template_row["High"] - template_row["Low"]) / template_row["Close"]
        )
        range_pct = 0.7 * template_range + 0.3 * float(rng.choice(range_values))
        range_pct = max(float(range_pct), 0.001)

        high_price = max(open_price, close_price) * (
            1 + range_pct * float(rng.uniform(0.25, 0.65))
        )
        low_price = min(open_price, close_price) * (
            1 - range_pct * float(rng.uniform(0.25, 0.65))
        )
        low_price = max(low_price, 0.0001)

        volume_factor = 1 + 0.35 * abs(ret) / ret_std
        volume = float(template_row["Volume"]) * volume_factor * float(
            rng.lognormal(mean=0, sigma=0.12)
        )

        rows.append(
            {
                "Date": last_date + pd.Timedelta(days=index + 1),
                "Open": round(open_price, 6),
                "High": round(max(high_price, open_price, close_price), 6),
                "Low": round(min(low_price, open_price, close_price), 6),
                "Close": round(close_price, 6),
                "Volume": round(max(volume, 1.0), 2),
                "Split": "unseen_test_1y",
                "Source": "synthetic_from_cnlib_train",
            }
        )
        previous_close = close_price

    return pd.DataFrame(rows)


def generate_regime_mix_test_data(coin: str, train_df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(REGIME_COIN_SEEDS[coin])
    df = normalize_market_frame(train_df)

    close = df["Close"].astype(float)
    returns = clipped(close.pct_change(), 0.005, 0.995)
    gaps = clipped(df["Open"] / close.shift(1) - 1, 0.01, 0.99)
    ranges = clipped((df["High"] - df["Low"]) / close, 0.01, 0.99)
    volumes = clipped(df["Volume"], 0.01, 0.99)

    ret_floor = float(returns.quantile(0.005))
    ret_ceiling = float(returns.quantile(0.995))
    gap_floor = float(gaps.quantile(0.01))
    gap_ceiling = float(gaps.quantile(0.99))
    ret_std = max(float(returns.std()), 1e-8)

    last_date = df["Date"].max()
    previous_close = float(close.iloc[-1])
    rows = []
    day = 0

    # Historical block bootstrap: sample contiguous market regimes from the
    # original four years, then nudge trend/volatility inside historical bounds.
    while day < DAYS_TO_GENERATE:
        block_length = int(rng.integers(18, 62))
        max_start = max(len(df) - block_length - 1, 1)
        start = int(rng.integers(1, max_start))
        block = df.iloc[start : start + block_length].reset_index(drop=True)

        block_returns = block["Close"].pct_change().reset_index(drop=True)
        block_returns = block_returns.clip(ret_floor, ret_ceiling)
        block_gaps = (block["Open"] / block["Close"].shift(1) - 1).reset_index(
            drop=True
        )
        block_gaps = block_gaps.clip(gap_floor, gap_ceiling)

        trend_bias = float(rng.normal(0, ret_std * 0.035))
        volatility_scale = float(rng.uniform(0.85, 1.22))
        volume_scale = float(rng.uniform(0.78, 1.28))

        for index in range(len(block)):
            if day >= DAYS_TO_GENERATE:
                break

            template_row = block.iloc[index]
            template_ret = float(block_returns.iloc[index])
            if np.isnan(template_ret):
                template_ret = float(rng.choice(returns.to_numpy()))

            centered_ret = template_ret - float(returns.mean())
            shock = float(rng.normal(0, ret_std * 0.04))
            ret = float(returns.mean()) + centered_ret * volatility_scale + trend_bias + shock
            ret = float(np.clip(ret, ret_floor, ret_ceiling))

            template_gap = float(block_gaps.iloc[index])
            if np.isnan(template_gap):
                template_gap = float(rng.choice(gaps.to_numpy()))
            gap = float(np.clip(template_gap * 0.75 + shock * 0.15, gap_floor, gap_ceiling))

            open_price = max(previous_close * (1 + gap), 0.0001)
            close_price = max(previous_close * (1 + ret), 0.0001)

            range_pct = float(
                ((template_row["High"] - template_row["Low"]) / template_row["Close"])
                * volatility_scale
            )
            range_pct = float(
                np.clip(range_pct, ranges.quantile(0.01), ranges.quantile(0.99))
            )

            high_price = max(open_price, close_price) * (
                1 + range_pct * float(rng.uniform(0.3, 0.72))
            )
            low_price = min(open_price, close_price) * (
                1 - range_pct * float(rng.uniform(0.3, 0.72))
            )
            low_price = max(low_price, 0.0001)

            volume = (
                float(template_row["Volume"])
                * volume_scale
                * (1 + 0.25 * abs(ret) / ret_std)
                * float(rng.lognormal(mean=0, sigma=0.1))
            )
            volume = float(np.clip(volume, volumes.quantile(0.01), volumes.quantile(0.995) * 1.35))

            rows.append(
                {
                    "Date": last_date + pd.Timedelta(days=day + 1),
                    "Open": round(open_price, 6),
                    "High": round(max(high_price, open_price, close_price), 6),
                    "Low": round(min(low_price, open_price, close_price), 6),
                    "Close": round(close_price, 6),
                    "Volume": round(max(volume, 1.0), 2),
                    "Split": "unseen_test_1y_regime_mix",
                    "Source": "regime_block_bootstrap_from_cnlib_train",
                }
            )
            previous_close = close_price
            day += 1

    return pd.DataFrame(rows)


def baseline_output_path_for_coin(coin: str) -> Path:
    return BASELINE_OUTPUT_DIR / f"{coin.replace('_train', '')}_unseen_test_1y.csv"


def regime_output_path_for_coin(coin: str) -> Path:
    return REGIME_OUTPUT_DIR / f"{coin.replace('_train', '')}_unseen_test_1y_regime_mix.csv"


def main() -> None:
    train_data_dir = resolve_train_data_dir()
    BASELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REGIME_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for coin in COIN_SEEDS:
        train_df = pd.read_parquet(train_data_dir / f"{coin}.parquet")

        generated = generate_coin_test_data(coin, train_df)
        path = baseline_output_path_for_coin(coin)
        generated.to_csv(path, index=False)
        print(
            f"{coin}: wrote baseline {len(generated)} rows to {path} "
            f"({generated['Date'].min().date()} - {generated['Date'].max().date()})"
        )

        regime_generated = generate_regime_mix_test_data(coin, train_df)
        regime_path = regime_output_path_for_coin(coin)
        regime_generated.to_csv(regime_path, index=False)
        print(
            f"{coin}: wrote regime-mix {len(regime_generated)} rows to {regime_path} "
            f"({regime_generated['Date'].min().date()} - {regime_generated['Date'].max().date()})"
        )


if __name__ == "__main__":
    main()
