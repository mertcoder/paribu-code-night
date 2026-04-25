from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


COIN_LABELS = {
    "kapcoin-usd_train": "KAPCOIN / USD",
    "metucoin-usd_train": "METUCOIN / USD",
    "tamcoin-usd_train": "TAMCOIN / USD",
}

ROOT_DIR = Path(__file__).resolve().parent
TRAIN_DATA_DIR = ROOT_DIR / "data" / "cnlib_train"
TRAIN_SPLIT = "cnlib_train"
TEST_DATASETS = {
    "none": {
        "label": "Sadece CNLIB train",
        "dir": None,
        "suffix": "",
        "split": TRAIN_SPLIT,
        "caption": "Ek test verisi eklenmez.",
    },
    "baseline": {
        "label": "Test 1 - Baseline 1Y",
        "dir": ROOT_DIR / "data" / "unseen_test_1y",
        "suffix": "unseen_test_1y",
        "split": "unseen_test_1y",
        "caption": "Konum: data/unseen_test_1y",
    },
    "regime_mix": {
        "label": "Test 2 - Regime mix 1Y",
        "dir": ROOT_DIR / "data" / "unseen_test_1y_regime_mix",
        "suffix": "unseen_test_1y_regime_mix",
        "split": "unseen_test_1y_regime_mix",
        "caption": "Konum: data/unseen_test_1y_regime_mix",
    },
    "realistic": {
        "label": "Test 3 - Realistic random walk 1Y",
        "dir": ROOT_DIR / "data" / "realistic_test_1y",
        "suffix": "realistic_test_1y",
        "split": "realistic_test",
        "caption": "Konum: data/realistic_test_1y",
    },
}
TEST_SPLITS = {
    config["split"] for key, config in TEST_DATASETS.items() if key != "none"
}
TEST_STYLES = {
    "unseen_test_1y": {
        "name": "Test 1 - Baseline 1Y",
        "increasing_line": "#2563eb",
        "increasing_fill": "#60a5fa",
        "decreasing_line": "#f97316",
        "decreasing_fill": "#fb923c",
        "volume_up": "#2563eb",
        "volume_down": "#f97316",
        "shade": "#fef3c7",
        "annotation": "#92400e",
    },
    "unseen_test_1y_regime_mix": {
        "name": "Test 2 - Regime mix 1Y",
        "increasing_line": "#0891b2",
        "increasing_fill": "#22d3ee",
        "decreasing_line": "#be123c",
        "decreasing_fill": "#fb7185",
        "volume_up": "#0891b2",
        "volume_down": "#be123c",
        "shade": "#cffafe",
        "annotation": "#155e75",
    },
    "realistic_test": {
        "name": "Test 3 - Realistic random walk 1Y",
        "increasing_line": "#7c3aed",
        "increasing_fill": "#a78bfa",
        "decreasing_line": "#e11d48",
        "decreasing_fill": "#fb7185",
        "volume_up": "#7c3aed",
        "volume_down": "#e11d48",
        "shade": "#ede9fe",
        "annotation": "#5b21b6",
    },
}

TIME_RANGES = {
    "1W": "Son 1 hafta",
    "1M": "Son 1 ay",
    "3M": "Son 3 ay",
    "6M": "Son 6 ay",
    "YTD": "Yil basi",
    "1Y": "Son 1 yil",
    "All": "Tum veri",
}

CANDLE_PERIODS = {
    "Gunluk": None,
    "Haftalik": "W",
    "Aylik": "ME",
    "Yillik": "YE",
}

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
METADATA_COLUMNS = ["Split", "Source"]
TRAIN_PARQUET_FILES = tuple(f"{coin}.parquet" for coin in COIN_LABELS)


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


@dataclass(frozen=True)
class MarketStats:
    last_price: float
    return_pct: float
    period_high: float
    period_low: float
    total_volume: float
    candle_count: int
    test_candle_count: int


def set_page_style() -> None:
    st.set_page_config(
        page_title="CNLIB Trade Dashboard",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f8fb;
            color: #111827;
        }

        [data-testid="stSidebar"] {
            background: #101418;
            border-right: 1px solid #232a31;
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        }

        [data-testid="stMetricLabel"] {
            color: #64748b;
        }

        [data-testid="stMetricValue"] {
            color: #111827;
            font-size: 1.55rem;
        }

        .terminal-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 16px 18px;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
            margin-bottom: 18px;
        }

        .terminal-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.2;
        }

        .terminal-subtitle {
            color: #64748b;
            font-size: 0.92rem;
            margin-top: 4px;
        }

        .market-badge {
            color: #0f766e;
            background: #ccfbf1;
            border: 1px solid #99f6e4;
            border-radius: 999px;
            font-weight: 700;
            padding: 8px 12px;
            white-space: nowrap;
        }

        .test-badge {
            color: #92400e;
            background: #fef3c7;
            border: 1px solid #fbbf24;
            border-radius: 999px;
            font-weight: 700;
            padding: 8px 12px;
            white-space: nowrap;
        }

        div[data-testid="stPlotlyChart"] {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            overflow: hidden;
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_market_frame(
    frame: pd.DataFrame,
    split: str = TRAIN_SPLIT,
    source: str = "cnlib_package",
) -> pd.DataFrame:
    df = frame.copy()

    if "Date" not in df.columns:
        df = df.reset_index()
        if "Date" not in df.columns:
            first_column = df.columns[0]
            df = df.rename(columns={first_column: "Date"})

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar: {', '.join(missing)}")

    selected_columns = REQUIRED_COLUMNS + [
        column for column in METADATA_COLUMNS if column in df.columns
    ]
    df = df[selected_columns].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLUMNS)
    if "Split" not in df.columns:
        df["Split"] = split
    else:
        df["Split"] = df["Split"].fillna(split)

    if "Source" not in df.columns:
        df["Source"] = source
    else:
        df["Source"] = df["Source"].fillna(source)

    df = df.sort_values("Date").reset_index(drop=True)
    return df


def is_test_split(split: str) -> bool:
    return split in TEST_SPLITS


def test_style_for_split(split: str) -> dict[str, str]:
    return TEST_STYLES.get(split, TEST_STYLES["unseen_test_1y"])


def volume_color_for_candle(open_price: float, close_price: float, split: str) -> str:
    if is_test_split(split):
        style = test_style_for_split(split)
        return style["volume_up"] if close_price >= open_price else style["volume_down"]
    return "#16a34a" if close_price >= open_price else "#dc2626"


def merge_split(values: pd.Series) -> str:
    for value in values:
        if is_test_split(str(value)):
            return str(value)
    return TRAIN_SPLIT


def test_data_path_for_coin(coin: str, dataset_key: str) -> Path | None:
    dataset = TEST_DATASETS[dataset_key]
    test_dir = dataset["dir"]
    if test_dir is None:
        return None
    return test_dir / f"{coin.replace('_train', '')}_{dataset['suffix']}.csv"


@st.cache_data(show_spinner=False)
def load_market_data(test_dataset_key: str) -> dict[str, pd.DataFrame]:
    train_data_dir = resolve_train_data_dir()
    market_data = {}
    test_dataset = TEST_DATASETS.get(test_dataset_key, TEST_DATASETS["none"])

    for coin in COIN_LABELS:
        train_path = train_data_dir / f"{coin}.parquet"
        if train_path.exists():
            train_df = normalize_market_frame(
                pd.read_parquet(train_path),
                split=TRAIN_SPLIT,
                source=str(train_path),
            )
            test_path = test_data_path_for_coin(coin, test_dataset_key)
            if test_path is not None and test_path.exists():
                test_df = normalize_market_frame(
                    pd.read_csv(test_path),
                    split=str(test_dataset["split"]),
                    source=str(test_path),
                )
                train_df = (
                    pd.concat([train_df, test_df], ignore_index=True)
                    .sort_values("Date")
                    .reset_index(drop=True)
                )
            market_data[coin] = train_df

    if not market_data:
        raise RuntimeError(
            "Beklenen coin parquet dosyalari okunamadi. Kontrol edilen klasor: "
            f"{train_data_dir}"
        )

    return market_data


def filter_time_range(df: pd.DataFrame, range_key: str) -> pd.DataFrame:
    if df.empty or range_key == "All":
        return df.copy()

    end_date = df["Date"].max()

    if range_key == "1W":
        start_date = end_date - pd.Timedelta(days=7)
    elif range_key == "1M":
        start_date = end_date - pd.DateOffset(months=1)
    elif range_key == "3M":
        start_date = end_date - pd.DateOffset(months=3)
    elif range_key == "6M":
        start_date = end_date - pd.DateOffset(months=6)
    elif range_key == "YTD":
        start_date = pd.Timestamp(year=end_date.year, month=1, day=1)
    elif range_key == "1Y":
        start_date = end_date - pd.DateOffset(years=1)
    else:
        start_date = df["Date"].min()

    filtered = df[df["Date"] >= start_date].copy()
    return filtered if not filtered.empty else df.tail(1).copy()


def resample_ohlcv(df: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    if rule is None or df.empty:
        return df.copy()

    aggregations = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    if "Split" in df.columns:
        aggregations["Split"] = merge_split
    if "Source" in df.columns:
        aggregations["Source"] = "last"

    aggregated = (
        df.set_index("Date")
        .resample(rule)
        .agg(aggregations)
        .dropna()
        .reset_index()
    )
    return aggregated


def calculate_stats(df: pd.DataFrame, active_test_split: str | None = None) -> MarketStats:
    first_close = float(df["Close"].iloc[0])
    last_price = float(df["Close"].iloc[-1])
    return_pct = ((last_price / first_close) - 1) * 100 if first_close else 0.0
    if active_test_split and "Split" in df.columns:
        test_candle_count = int(df["Split"].eq(active_test_split).sum())
    else:
        test_candle_count = 0

    return MarketStats(
        last_price=last_price,
        return_pct=return_pct,
        period_high=float(df["High"].max()),
        period_low=float(df["Low"].min()),
        total_volume=float(df["Volume"].sum()),
        candle_count=len(df),
        test_candle_count=test_candle_count,
    )


def format_price(value: float) -> str:
    return f"${value:,.2f}"


def format_pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def format_volume(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:,.0f}"


def make_market_chart(
    df: pd.DataFrame,
    coin_label: str,
    active_test_split: str | None = None,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )

    has_split = "Split" in df.columns
    if has_split and active_test_split:
        train_df = df[df["Split"].ne(active_test_split)].copy()
        test_df = df[df["Split"].eq(active_test_split)].copy()
    else:
        train_df = df.copy()
        test_df = df.iloc[0:0].copy()

    if not train_df.empty:
        fig.add_trace(
            go.Candlestick(
                x=train_df["Date"],
                open=train_df["Open"],
                high=train_df["High"],
                low=train_df["Low"],
                close=train_df["Close"],
                name="CNLIB train",
                increasing_line_color="#16a34a",
                increasing_fillcolor="#22c55e",
                decreasing_line_color="#dc2626",
                decreasing_fillcolor="#ef4444",
            ),
            row=1,
            col=1,
        )

    if not test_df.empty:
        for split, split_df in test_df.groupby("Split", sort=False):
            style = test_style_for_split(str(split))
            fig.add_trace(
                go.Candlestick(
                    x=split_df["Date"],
                    open=split_df["Open"],
                    high=split_df["High"],
                    low=split_df["Low"],
                    close=split_df["Close"],
                    name=style["name"],
                    increasing_line_color=style["increasing_line"],
                    increasing_fillcolor=style["increasing_fill"],
                    decreasing_line_color=style["decreasing_line"],
                    decreasing_fillcolor=style["decreasing_fill"],
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=split_df["Date"],
                    y=split_df["Close"],
                    name=f"{style['name']} close",
                    mode="lines",
                    line={"color": style["increasing_line"], "width": 3},
                    hovertemplate="%{x|%d.%m.%Y}<br>Close: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig.add_vrect(
                x0=split_df["Date"].min(),
                x1=split_df["Date"].max(),
                fillcolor=style["shade"],
                opacity=0.34,
                line_width=0,
                layer="below",
                annotation_text=style["name"],
                annotation_position="top left",
                annotation_font_color=style["annotation"],
            )
            test_start = split_df["Date"].min()
            fig.add_shape(
                type="line",
                x0=test_start,
                x1=test_start,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line={
                    "color": style["annotation"],
                    "width": 2,
                    "dash": "dash",
                },
            )
            fig.add_annotation(
                x=test_start,
                y=1,
                xref="x",
                yref="paper",
                text=f"{style['name']} baslangic",
                showarrow=False,
                yanchor="bottom",
                font={"color": style["annotation"], "size": 12},
            )

    volume_colors = [
        volume_color_for_candle(
            float(open_),
            float(close),
            str(split),
        )
        for open_, close, split in zip(
            df["Open"],
            df["Close"],
            df["Split"] if "Split" in df.columns else [TRAIN_SPLIT] * len(df),
        )
    ]
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Volume"],
            name="Hacim",
            marker_color=volume_colors,
            opacity=0.65,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title={
            "text": coin_label,
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 20, "color": "#111827"},
        },
        template="plotly_white",
        height=680,
        margin={"l": 32, "r": 24, "t": 56, "b": 28},
        hovermode="x unified",
        showlegend=not test_df.empty,
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "Arial, sans-serif", "color": "#111827"},
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#eef2f7",
        linecolor="#e5e7eb",
        tickfont={"color": "#475569"},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#eef2f7",
        linecolor="#e5e7eb",
        tickfont={"color": "#475569"},
        title_text="Fiyat",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#eef2f7",
        linecolor="#e5e7eb",
        tickfont={"color": "#475569"},
        title_text="Hacim",
        row=2,
        col=1,
    )
    return fig


def render_header(
    coin_label: str,
    date_min: pd.Timestamp,
    date_max: pd.Timestamp,
    has_test_data: bool,
    test_label: str,
) -> None:
    test_badge = (
        f'<div class="test-badge">{test_label} aktif</div>' if has_test_data else ""
    )
    st.markdown(
        f"""
        <div class="terminal-bar">
            <div>
                <div class="terminal-title">CNLIB Trade Dashboard</div>
                <div class="terminal-subtitle">
                    {date_min:%d.%m.%Y} - {date_max:%d.%m.%Y} arasi piyasa verisi
                </div>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
                {test_badge}
                <div class="market-badge">{coin_label}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(stats: MarketStats) -> None:
    col_price, col_return, col_high_low, col_volume, col_candles, col_test = st.columns(6)
    col_price.metric("Son fiyat", format_price(stats.last_price))
    col_return.metric("Donem getirisi", format_pct(stats.return_pct))
    col_high_low.metric(
        "Yuksek / Dusuk",
        f"{format_price(stats.period_high)} / {format_price(stats.period_low)}",
    )
    col_volume.metric("Toplam hacim", format_volume(stats.total_volume))
    col_candles.metric("Mum sayisi", f"{stats.candle_count:,}")
    col_test.metric("Test mumlari", f"{stats.test_candle_count:,}")


def render_test_panel(
    chart_df: pd.DataFrame,
    coin_label: str,
    active_test_split: str | None,
    selected_test_label: str,
    selected_period: str,
) -> None:
    if active_test_split is None or "Split" not in chart_df.columns:
        return

    test_df = chart_df[chart_df["Split"].eq(active_test_split)].copy()
    if test_df.empty:
        st.warning(
            "Secili zaman araligi test yilini icermiyor. Testi ana grafikte "
            "train sonuna eklenmis gormek icin Zaman araligi -> All sec."
        )
        return

    st.markdown(f"**Secili test yili detayi: {selected_test_label}**")
    st.caption(
        f"{test_df['Date'].min():%d.%m.%Y} - {test_df['Date'].max():%d.%m.%Y} "
        f"arasi {len(test_df):,} test mumu"
    )
    st.plotly_chart(
        make_market_chart(
            test_df,
            f"{coin_label} - {selected_test_label} detay / {selected_period}",
            active_test_split,
        ),
        use_container_width=True,
        key=f"test-detail-chart-{active_test_split}-{coin_label}-{selected_period}",
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "responsive": True,
        },
    )


def main() -> None:
    set_page_style()

    st.sidebar.title("Piyasa")
    selected_test_dataset = st.sidebar.selectbox(
        "Test verisi",
        options=list(TEST_DATASETS.keys()),
        index=1,
        format_func=lambda key: str(TEST_DATASETS[key]["label"]),
    )
    st.sidebar.caption(str(TEST_DATASETS[selected_test_dataset]["caption"]))

    with st.spinner("CNLIB market verisi yukleniyor..."):
        market_data = load_market_data(selected_test_dataset)

    available_coins = [coin for coin in COIN_LABELS if coin in market_data]

    selected_coin = st.sidebar.selectbox(
        "Coin",
        options=available_coins,
        format_func=lambda coin: COIN_LABELS.get(coin, coin),
    )
    selected_range = st.sidebar.radio(
        "Zaman araligi",
        options=list(TIME_RANGES.keys()),
        index=list(TIME_RANGES.keys()).index("All"),
        format_func=lambda key: f"{key} - {TIME_RANGES[key]}",
        horizontal=False,
    )
    selected_period = st.sidebar.radio(
        "Mum periyodu",
        options=list(CANDLE_PERIODS.keys()),
        horizontal=False,
    )

    source_df = market_data[selected_coin]
    filtered_df = filter_time_range(source_df, selected_range)
    chart_df = resample_ohlcv(filtered_df, CANDLE_PERIODS[selected_period])
    active_test_split = (
        str(TEST_DATASETS[selected_test_dataset]["split"])
        if selected_test_dataset != "none"
        else None
    )
    selected_test_label = str(TEST_DATASETS[selected_test_dataset]["label"])

    coin_label = COIN_LABELS.get(selected_coin, selected_coin)
    has_test_data = (
        active_test_split is not None
        and "Split" in chart_df.columns
        and chart_df["Split"].eq(active_test_split).any()
    )
    render_header(
        coin_label,
        source_df["Date"].min(),
        source_df["Date"].max(),
        has_test_data,
        selected_test_label,
    )

    stats = calculate_stats(chart_df, active_test_split)
    render_metrics(stats)

    st.plotly_chart(
        make_market_chart(
            chart_df,
            f"{coin_label} - {selected_test_label} - {selected_range} / {selected_period}",
            active_test_split,
        ),
        use_container_width=True,
        key=(
            f"market-chart-{selected_test_dataset}-"
            f"{selected_coin}-{selected_range}-{selected_period}"
        ),
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "responsive": True,
        },
    )

    render_test_panel(
        chart_df,
        coin_label,
        active_test_split,
        selected_test_label,
        selected_period,
    )

    with st.expander("Son mumlar", expanded=False):
        display_df = chart_df.tail(12).sort_values("Date", ascending=False).copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%d.%m.%Y")
        st.dataframe(display_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
