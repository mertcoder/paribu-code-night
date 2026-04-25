from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "data" / "realistic_test_1y"
START_DATE = "2027-03-16"
END_DATE = "2028-03-14"
SPLIT = "realistic_test"
SOURCE = "realistic_random_walk"

COINS = {
    "kapcoin-usd": 390.0,
    "metucoin-usd": 5700.0,
    "tamcoin-usd": 7500.0,
}

CSV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "Split", "Source"]


@dataclass(frozen=True)
class ValidationStats:
    autocorr: dict[str, float]
    volatility: dict[str, float]
    mean_return: dict[str, float]
    range_mean: dict[str, float]
    correlation: pd.DataFrame


def make_correlated_returns(
    rng: np.random.Generator,
    n_returns: int,
    target_corr: float,
) -> np.ndarray:
    corr = np.full((3, 3), target_corr)
    np.fill_diagonal(corr, 1.0)
    innovations = rng.multivariate_normal(np.zeros(3), corr, size=n_returns)

    target_means = np.array([0.00022, 0.00031, 0.00027])
    target_stds = np.array([0.027, 0.032, 0.029])
    phi = 0.025

    returns = np.zeros_like(innovations)
    returns[0] = target_means + target_stds * innovations[0]
    for idx in range(1, n_returns):
        returns[idx] = (
            target_means
            + phi * (returns[idx - 1] - target_means)
            + target_stds * np.sqrt(1 - phi**2) * innovations[idx]
        )

    # Normalize each sample to exact crypto-like mean/std bands while preserving
    # the cross-sectional shape from the correlated shocks.
    returns = (returns - returns.mean(axis=0)) / returns.std(axis=0, ddof=1)
    returns = returns * target_stds + target_means
    return np.clip(returns, -0.12, 0.12)


def build_coin_frame(
    coin: str,
    start_price: float,
    dates: pd.DatetimeIndex,
    returns: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    closes = [start_price]
    for daily_return in returns:
        closes.append(closes[-1] * (1 + float(daily_return)))

    rows = []
    previous_close = start_price
    range_targets = rng.lognormal(mean=np.log(0.04), sigma=0.18, size=len(dates))
    range_targets = np.clip(range_targets, 0.03, 0.05)
    volumes = rng.uniform(5_000_000_000, 30_000_000_000, size=len(dates))

    for index, date in enumerate(dates):
        close_price = float(closes[index])
        gap = float(np.clip(rng.normal(0, 0.0035), -0.012, 0.012))
        open_price = previous_close * (1 + gap) if index else start_price * (1 + gap)

        body_range = abs(close_price - open_price) / close_price
        range_pct = max(float(range_targets[index]), body_range * 1.08)
        range_pct = min(range_pct, 0.085)
        target_range = range_pct * close_price
        body = abs(close_price - open_price)
        remaining_range = max(target_range - body, close_price * 0.004)
        upper_extra = remaining_range * float(rng.uniform(0.35, 0.65))
        lower_extra = remaining_range - upper_extra

        high_price = max(open_price, close_price) + upper_extra
        low_price = max(min(open_price, close_price) - lower_extra, 0.0001)

        rows.append(
            {
                "Date": date.strftime("%Y-%m-%d"),
                "Open": round(open_price, 6),
                "High": round(high_price, 6),
                "Low": round(low_price, 6),
                "Close": round(close_price, 6),
                "Volume": round(float(volumes[index]), 2),
                "Split": SPLIT,
                "Source": SOURCE,
            }
        )
        previous_close = close_price

    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def validate(frames: dict[str, pd.DataFrame]) -> ValidationStats:
    returns = {
        coin: frame["Close"].pct_change().dropna()
        for coin, frame in frames.items()
    }
    ranges = {
        coin: ((frame["High"] - frame["Low"]) / frame["Close"]).mean()
        for coin, frame in frames.items()
    }
    returns_df = pd.DataFrame(returns)

    return ValidationStats(
        autocorr={coin: float(series.autocorr(1)) for coin, series in returns.items()},
        volatility={coin: float(series.std()) for coin, series in returns.items()},
        mean_return={coin: float(series.mean()) for coin, series in returns.items()},
        range_mean={coin: float(value) for coin, value in ranges.items()},
        correlation=returns_df.corr(),
    )


def passes_constraints(stats: ValidationStats, frames: dict[str, pd.DataFrame]) -> bool:
    autocorr_ok = all(0.0 <= value <= 0.05 for value in stats.autocorr.values())
    volatility_ok = all(0.02 <= value <= 0.04 for value in stats.volatility.values())
    mean_ok = all(0.0001 <= value <= 0.0005 for value in stats.mean_return.values())
    range_ok = all(0.03 <= value <= 0.05 for value in stats.range_mean.values())

    corr_values = []
    coins = list(frames)
    for left_index, left in enumerate(coins):
        for right in coins[left_index + 1 :]:
            corr_values.append(float(stats.correlation.loc[left, right]))
    corr_ok = all(0.3 <= value <= 0.6 for value in corr_values)

    volume_ok = all(
        frame["Volume"].between(5_000_000_000, 30_000_000_000).all()
        for frame in frames.values()
    )

    return autocorr_ok and volatility_ok and mean_ok and range_ok and corr_ok and volume_ok


def generate_dataset() -> tuple[dict[str, pd.DataFrame], ValidationStats]:
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    n_returns = len(dates) - 1

    for attempt in range(1, 20_001):
        rng = np.random.default_rng(900_000 + attempt)
        target_corr = float(rng.uniform(0.38, 0.52))
        returns = make_correlated_returns(rng, n_returns, target_corr)

        frames = {}
        for index, (coin, start_price) in enumerate(COINS.items()):
            frames[coin] = build_coin_frame(
                coin,
                start_price,
                dates,
                returns[:, index],
                rng,
            )

        stats = validate(frames)
        if passes_constraints(stats, frames):
            return frames, stats

    raise RuntimeError("Could not generate realistic test data within constraints.")


def main() -> None:
    frames, stats = generate_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for coin, frame in frames.items():
        path = OUTPUT_DIR / f"{coin}_realistic_test_1y.csv"
        frame.to_csv(path, index=False)
        print(f"wrote {len(frame)} rows to {path}")

    print("lag-1 autocorr:", stats.autocorr)
    print("daily volatility:", stats.volatility)
    print("mean daily return:", stats.mean_return)
    print("average range:", stats.range_mean)
    print("correlation:")
    print(stats.correlation)


if __name__ == "__main__":
    main()
