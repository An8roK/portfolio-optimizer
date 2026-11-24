"""Portfolio & benchmark risk/return metrics, growth of $10,000, and underwater drawdowns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
INITIAL_CAPITAL = 10_000.0
VAR_CONFIDENCE_TAIL = 5.0  # 5th percentile corresponds to 95% Historical VaR / CVaR.


@dataclass(frozen=True)
class PerformanceMetrics:
    """Institutional performance and tail-risk metrics for a return stream."""

    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: float
    var_95: float
    cvar_95: float
    ending_value: float


def compute_portfolio_daily_returns(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Compute static daily-rebalanced portfolio simple returns from asset returns and weights."""
    aligned_weights = weights.reindex(asset_returns.columns).fillna(0.0).to_numpy(dtype=float)
    port_returns = asset_returns.to_numpy(dtype=float) @ aligned_weights
    return pd.Series(port_returns, index=asset_returns.index, name="Portfolio")


def compute_growth_and_drawdown(
    daily_returns: pd.Series,
    initial_capital: float = INITIAL_CAPITAL,
) -> tuple[pd.Series, pd.Series]:
    """Compute cumulative $10,000 growth and underwater drawdown series."""
    clean = daily_returns.dropna()
    if clean.empty:
        empty = pd.Series(dtype=float)
        return empty, empty

    growth = initial_capital * (1.0 + clean).cumprod()
    # Anchor running peak at initial_capital so any immediate day-1 drop registers as drawdown.
    running_peak = np.maximum(growth.cummax(), initial_capital)
    drawdown = growth / running_peak - 1.0
    return growth, drawdown


def compute_metrics(
    daily_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float,
    initial_capital: float = INITIAL_CAPITAL,
) -> PerformanceMetrics:
    """Calculate annualized return, volatility, Sharpe, Sortino, drawdown, Beta, VaR, and CVaR."""
    aligned = pd.concat(
        [daily_returns.rename("port"), benchmark_returns.rename("bench")],
        axis=1,
    ).dropna()

    if len(aligned) < 2:
        return PerformanceMetrics(
            annualized_return=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            beta=0.0,
            var_95=0.0,
            cvar_95=0.0,
            ending_value=initial_capital,
        )

    r = aligned["port"].to_numpy(dtype=float)
    b = aligned["bench"].to_numpy(dtype=float)
    n_days = len(r)

    growth, drawdown = compute_growth_and_drawdown(aligned["port"], initial_capital=initial_capital)
    ending_value = float(growth.iloc[-1])

    # Geometric CAGR aligns the annualized return headline with the $10,000 cumulative growth curve.
    total_multiple = max(ending_value / initial_capital, 1e-12)
    cagr = float(total_multiple ** (TRADING_DAYS / n_days) - 1.0)

    ann_vol = float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))

    # Sharpe uses arithmetic annualized excess return to stay consistent with Markowitz MVO.
    arith_ann_return = float(np.mean(r) * TRADING_DAYS)
    sharpe = float((arith_ann_return - risk_free_rate) / ann_vol) if ann_vol > 1e-12 else 0.0

    # Downside deviation measures semi-deviation below the daily risk-free hurdle rate.
    daily_rf = risk_free_rate / TRADING_DAYS
    downside_diff = np.minimum(r - daily_rf, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside_diff**2)) * np.sqrt(TRADING_DAYS))
    sortino = float((arith_ann_return - risk_free_rate) / downside_dev) if downside_dev > 1e-12 else 0.0

    max_dd = float(drawdown.min())

    bench_var = float(np.var(b, ddof=1))
    if bench_var > 1e-14:
        cov_pb = float(np.cov(r, b, ddof=1)[0, 1])
        beta = cov_pb / bench_var
    else:
        beta = 0.0

    var_95 = float(np.percentile(r, VAR_CONFIDENCE_TAIL))
    tail_losses = r[r <= var_95]
    cvar_95 = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_95

    return PerformanceMetrics(
        annualized_return=cagr,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        beta=beta,
        var_95=var_95,
        cvar_95=cvar_95,
        ending_value=ending_value,
    )


def build_metrics_comparison_table(
    portfolio_metrics: PerformanceMetrics,
    spy_metrics: PerformanceMetrics,
    portfolio_label: str = "Optimized Portfolio",
) -> pd.DataFrame:
    """Format side-by-side institutional risk and return metrics for display in Streamlit."""
    rows = [
        ("Annualized Return (CAGR)", f"{portfolio_metrics.annualized_return:.2%}", f"{spy_metrics.annualized_return:.2%}"),
        ("Annualized Volatility", f"{portfolio_metrics.annualized_volatility:.2%}", f"{spy_metrics.annualized_volatility:.2%}"),
        ("Sharpe Ratio", f"{portfolio_metrics.sharpe_ratio:.2f}", f"{spy_metrics.sharpe_ratio:.2f}"),
        ("Sortino Ratio", f"{portfolio_metrics.sortino_ratio:.2f}", f"{spy_metrics.sortino_ratio:.2f}"),
        ("Maximum Drawdown", f"{portfolio_metrics.max_drawdown:.2%}", f"{spy_metrics.max_drawdown:.2%}"),
        ("Beta vs. SPY", f"{portfolio_metrics.beta:.2f}", f"{spy_metrics.beta:.2f}"),
        ("95% Historical VaR (Daily)", f"{portfolio_metrics.var_95:.2%}", f"{spy_metrics.var_95:.2%}"),
        ("95% Historical CVaR (Daily)", f"{portfolio_metrics.cvar_95:.2%}", f"{spy_metrics.cvar_95:.2%}"),
        ("Ending Value ($10,000 Start)", f"${portfolio_metrics.ending_value:,.0f}", f"${spy_metrics.ending_value:,.0f}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", portfolio_label, "SPY Benchmark"])
