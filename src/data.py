"""Market data fetching, alignment, and validation using Yahoo Finance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import re

import pandas as pd
import streamlit as st
import yfinance as yf

BENCHMARK_TICKER = "SPY"
MAX_LOOKBACK_YEARS = 10
MIN_OVERLAP_DAYS = 30
# Allow standard Yahoo Finance symbols (letters, digits, dot, hyphen, caret, equals).
_TICKER_PATTERN = re.compile(r"^[A-Z0-9.\-^=]{1,12}$")


class DataValidationError(ValueError):
    """Raised when user ticker input or fetched market data fails validation."""


@dataclass(frozen=True)
class PriceBundle:
    """Aligned adjusted close prices for the user basket and SPY benchmark."""

    asset_prices: pd.DataFrame
    benchmark_prices: pd.Series
    requested_years: int
    actual_years: float
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    truncated: bool


def parse_tickers(raw_text: str) -> list[str]:
    """Parse, normalize, deduplicate, and validate a user-entered ticker basket."""
    if not raw_text or not raw_text.strip():
        raise DataValidationError(
            "Please enter at least 2 valid ticker symbols (for example: AAPL, MSFT, NVDA, TLT, GLD)."
        )

    tokens = [tok.strip().upper() for tok in re.split(r"[\s,;]+", raw_text.strip()) if tok.strip()]

    # Deduplicate while preserving the user's original display order.
    unique_tickers: list[str] = list(dict.fromkeys(tokens))
    malformed = [t for t in unique_tickers if not _TICKER_PATTERN.match(t)]
    if malformed:
        raise DataValidationError(
            f"Invalid ticker symbol format: {', '.join(malformed)}. "
            "Use standard Yahoo Finance symbols separated by commas."
        )

    if len(unique_tickers) < 2:
        raise DataValidationError(
            "Please enter at least 2 unique ticker symbols to run portfolio optimization."
        )

    return unique_tickers


def _extract_adjusted_close(raw: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    """Normalize yfinance download output into a flat DataFrame of adjusted closes."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=requested)

    if isinstance(raw.columns, pd.MultiIndex):
        level_0 = raw.columns.get_level_values(0)
        level_1 = raw.columns.get_level_values(1)

        # yfinance >= 0.2.50 with auto_adjust=True exposes adjusted prices under "Close".
        if "Close" in level_0:
            close_df = raw["Close"].copy()
        elif "Adj Close" in level_0:
            close_df = raw["Adj Close"].copy()
        elif "Close" in level_1:
            close_df = raw.xs("Close", axis=1, level=1).copy()
        elif "Adj Close" in level_1:
            close_df = raw.xs("Adj Close", axis=1, level=1).copy()
        else:
            raise DataValidationError(
                "Unexpected data format returned by Yahoo Finance (missing Close prices)."
            )
    else:
        # Single-ticker fallback: extract the Close column and name it after the ticker.
        col_name = "Close" if "Close" in raw.columns else "Adj Close"
        if col_name not in raw.columns:
            raise DataValidationError(
                "Unexpected data format returned by Yahoo Finance (missing Close prices)."
            )
        close_df = raw[[col_name]].rename(columns={col_name: requested[0]}).copy()

    if isinstance(close_df, pd.Series):
        close_df = close_df.to_frame(name=requested[0])

    # Ensure every requested column exists so missing/delisted tickers are caught explicitly.
    for ticker in requested:
        if ticker not in close_df.columns:
            close_df[ticker] = pd.NA

    close_df = close_df[requested].apply(pd.to_numeric, errors="coerce")

    # Normalize timezone to naive dates so lookback slicing behaves uniformly.
    idx = pd.to_datetime(close_df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    close_df.index = idx
    close_df = close_df.sort_index()
    close_df = close_df[~close_df.index.duplicated(keep="last")]
    return close_df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_history(
    tickers: tuple[str, ...],
    max_years: int = MAX_LOOKBACK_YEARS,
) -> pd.DataFrame:
    """Download and cache adjusted close prices for tickers plus SPY (1-hour RAM TTL).

    Fetches up to ``max_years`` once per unique ticker basket so both the active lookback
    window and Tab 4 sensitivity analysis reuse the same cached Yahoo Finance payload.
    """
    download_tickers = list(dict.fromkeys([*tickers, BENCHMARK_TICKER]))
    end_dt = date.today() + timedelta(days=1)
    # Add a small calendar buffer to guarantee full trading-year coverage across holidays.
    start_dt = end_dt - timedelta(days=int(max_years * 365.25) + 21)

    try:
        raw = yf.download(
            tickers=download_tickers,
            start=start_dt.isoformat(),
            end=end_dt.isoformat(),
            auto_adjust=True,
            progress=False,
            # Disable multi-threaded downloads to prevent yfinance SQLite cache locks on Windows.
            threads=False,
        )
    except Exception as exc:
        raise DataValidationError(
            f"Unable to download market data from Yahoo Finance: {exc}"
        ) from exc

    prices = _extract_adjusted_close(raw, download_tickers)
    if prices.empty or prices.dropna(how="all").empty:
        raise DataValidationError(
            "No historical price data was returned from Yahoo Finance. "
            "Check your internet connection and verify the ticker symbols."
        )

    # Report any ticker that returned no usable price history (invalid or delisted).
    invalid_tickers = [
        t for t in download_tickers if prices[t].dropna().shape[0] < 2
    ]
    if invalid_tickers:
        if BENCHMARK_TICKER in invalid_tickers and len(invalid_tickers) == 1:
            raise DataValidationError(
                "Unable to retrieve benchmark data for SPY from Yahoo Finance."
            )
        user_invalid = [t for t in invalid_tickers if t != BENCHMARK_TICKER]
        raise DataValidationError(
            f"Invalid, delisted, or unavailable ticker(s): {', '.join(user_invalid)}. "
            "Remove or correct these symbols and try again."
        )

    return prices.dropna(how="all")


def slice_price_bundle(
    history: pd.DataFrame,
    tickers: tuple[str, ...],
    lookback_years: int,
) -> PriceBundle:
    """Slice a trailing lookback window and align trading dates cleanly across assets + SPY."""
    if history.empty:
        raise DataValidationError("Historical price table is empty.")

    required_cols = list(dict.fromkeys([*tickers, BENCHMARK_TICKER]))
    missing = [col for col in required_cols if col not in history.columns]
    if missing:
        raise DataValidationError(f"Missing price columns for: {', '.join(missing)}.")

    latest_date = history.index.max()
    cutoff_date = latest_date - pd.DateOffset(years=lookback_years)

    # Slice first, then drop incomplete rows so younger tickers only constrain the active window.
    window = history.loc[history.index >= cutoff_date, required_cols]
    aligned = window.dropna(how="any")

    if len(aligned) < MIN_OVERLAP_DAYS:
        raise DataValidationError(
            f"Only {len(aligned)} overlapping trading days were found for "
            f"{', '.join(tickers)} and {BENCHMARK_TICKER}. "
            f"At least {MIN_OVERLAP_DAYS} overlapping trading days are required."
        )

    asset_prices = aligned[list(tickers)].copy()
    benchmark_prices = aligned[BENCHMARK_TICKER].copy()

    start_ts = aligned.index.min()
    end_ts = aligned.index.max()
    actual_years = max((end_ts - start_ts).days / 365.25, 0.01)
    truncated = actual_years < (lookback_years - 0.25)

    return PriceBundle(
        asset_prices=asset_prices,
        benchmark_prices=benchmark_prices,
        requested_years=lookback_years,
        actual_years=actual_years,
        start_date=start_ts,
        end_date=end_ts,
        truncated=truncated,
    )


def load_market_data(
    tickers: tuple[str, ...],
    lookback_years: int,
) -> tuple[PriceBundle, pd.DataFrame]:
    """Load cached market history and return both the active lookback bundle and full history."""
    max_years = max(MAX_LOOKBACK_YEARS, lookback_years)
    history = fetch_price_history(tickers=tickers, max_years=max_years)
    bundle = slice_price_bundle(history=history, tickers=tickers, lookback_years=lookback_years)
    return bundle, history
