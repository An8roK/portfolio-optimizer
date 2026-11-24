"""End-to-end verification script for the Portfolio Optimization Tool.

Checks:
1. All application modules import cleanly.
2. Optimizer weights sum to 1.0 and respect [min_weight, max_weight] bounds across
   GMV, Max Sharpe, Target Return, 40-point Efficient Frontier, and 5,000 Monte Carlo portfolios.
3. All portfolio and SPY metrics are finite floats and satisfy mathematical invariants.
4. All Plotly figures build cleanly without error.
5. Live Yahoo Finance fetch succeeds on default basket and raises DataValidationError on invalid tickers.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on sys.path when invoked directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402, F401
from src.charts import (  # noqa: E402
    plot_backtest_growth,
    plot_correlation_heatmap,
    plot_efficient_frontier,
    plot_ips_weights,
    plot_sensitivity_sharpe_heatmap,
    plot_sensitivity_weight_heatmap,
    plot_underwater_drawdown,
)
from src.data import (  # noqa: E402
    DataValidationError,
    load_market_data,
    parse_tickers,
)
from src.metrics import (  # noqa: E402
    build_metrics_comparison_table,
    compute_growth_and_drawdown,
    compute_metrics,
    compute_portfolio_daily_returns,
)
from src.optimizer import (  # noqa: E402
    OptimizationError,
    compute_efficient_frontier,
    compute_equal_weight_portfolio,
    compute_return_stats,
    compute_sensitivity_grid,
    get_feasible_return_range,
    optimize_max_sharpe,
    optimize_min_variance,
    optimize_target_return,
    run_monte_carlo,
    validate_weight_bounds,
)


def _assert_valid_weights(weights: pd.Series, min_w: float, max_w: float, label: str) -> None:
    total = float(weights.sum())
    assert abs(total - 1.0) < 1e-6, f"{label}: weights sum to {total}, expected 1.0"
    assert (weights >= min_w - 1e-6).all(), f"{label}: weight below min_w ({min_w}): {weights.to_dict()}"
    assert (weights <= max_w + 1e-6).all(), f"{label}: weight above max_w ({max_w}): {weights.to_dict()}"
    assert np.isfinite(weights.to_numpy()).all(), f"{label}: non-finite weights detected"


def _assert_finite_metrics(m, label: str) -> None:
    fields = [
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "beta",
        "var_95",
        "cvar_95",
        "ending_value",
    ]
    for field in fields:
        val = getattr(m, field)
        assert isinstance(val, float) and math.isfinite(val), f"{label}.{field} is not finite: {val}"
    assert m.max_drawdown <= 1e-12, f"{label}.max_drawdown should be <= 0, got {m.max_drawdown}"
    assert m.cvar_95 <= m.var_95 + 1e-12, f"{label}: CVaR ({m.cvar_95}) should be <= VaR ({m.var_95})"
    assert m.annualized_volatility >= 0.0, f"{label}: negative volatility"


def run_synthetic_suite() -> None:
    """Verify mathematical invariants deterministically on synthetic market prices."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=1260)
    tickers = ("AAPL", "MSFT", "NVDA", "TLT", "GLD")
    daily_mu = np.array([0.0007, 0.0006, 0.0011, 0.0001, 0.0003, 0.00045])
    cov = np.full((6, 6), 0.00005)
    np.fill_diagonal(cov, [0.0003, 0.00025, 0.0007, 0.0001, 0.00012, 0.00016])
    rets = rng.multivariate_normal(daily_mu, cov, size=len(dates))
    prices = pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=[*tickers, "SPY"])

    # Feasibility validation checks
    try:
        validate_weight_bounds(5, 0.0, 0.15)
        raise AssertionError("Expected OptimizationError for infeasible max weight")
    except OptimizationError:
        pass

    stats = compute_return_stats(prices[list(tickers)])
    min_w, max_w, rf = 0.05, 0.40, 0.0425

    gmv = optimize_min_variance(stats, rf, min_w, max_w)
    tangency = optimize_max_sharpe(stats, rf, min_w, max_w)
    eq = compute_equal_weight_portfolio(stats, rf)
    min_r, max_r = get_feasible_return_range(stats, min_w, max_w)
    mid_target = 0.5 * (gmv.expected_return + max_r)
    target_port = optimize_target_return(stats, mid_target, rf, min_w, max_w)

    _assert_valid_weights(gmv.weights, min_w, max_w, "GMV")
    _assert_valid_weights(tangency.weights, min_w, max_w, "Max Sharpe")
    _assert_valid_weights(target_port.weights, min_w, max_w, "Target Return")
    assert abs(target_port.expected_return - mid_target) < 1e-4, "Target return constraint missed"
    assert tangency.sharpe_ratio >= gmv.sharpe_ratio - 1e-5, "Tangency Sharpe must be >= GMV Sharpe"
    assert gmv.volatility <= eq.volatility + 1e-5, "GMV volatility must be <= Equal Weight volatility"

    frontier = compute_efficient_frontier(stats, rf, min_w, max_w, n_points=40)
    assert len(frontier) == 40, f"Expected 40 frontier points, got {len(frontier)}"
    assert np.isfinite(frontier.to_numpy()).all(), "Non-finite values in efficient frontier"

    mc_df = run_monte_carlo(stats, rf, min_w, max_w, n_portfolios=5000)
    assert len(mc_df) == 5000, f"Expected 5,000 Monte Carlo portfolios, got {len(mc_df)}"
    assert np.isfinite(mc_df.to_numpy()).all(), "Non-finite values in Monte Carlo output"

    spy_rets = prices["SPY"].pct_change().dropna()
    port_rets = compute_portfolio_daily_returns(stats.daily_returns, tangency.weights)
    port_m = compute_metrics(port_rets, spy_rets, rf)
    spy_m = compute_metrics(spy_rets, spy_rets, rf)

    _assert_finite_metrics(port_m, "Synthetic Portfolio")
    _assert_finite_metrics(spy_m, "Synthetic SPY")
    assert abs(spy_m.beta - 1.0) < 1e-9, f"SPY beta vs SPY must be 1.0, got {spy_m.beta}"
    assert abs(port_m.sharpe_ratio - tangency.sharpe_ratio) < 1e-6, "Sharpe mismatch between optimizer and metrics"

    port_g, port_dd = compute_growth_and_drawdown(port_rets)
    spy_g, spy_dd = compute_growth_and_drawdown(spy_rets)
    table = build_metrics_comparison_table(port_m, spy_m)
    assert len(table) == 9

    sensitivity = compute_sensitivity_grid(prices, tickers, min_w, max_w, rf)
    assert not sensitivity.sharpe_matrix.empty

    # Verify every Plotly chart constructs cleanly.
    figs = [
        plot_efficient_frontier(mc_df, frontier, tangency, gmv, eq, rf, target_port),
        plot_ips_weights(tangency.weights, min_w, max_w),
        plot_correlation_heatmap(stats.corr_matrix),
        plot_backtest_growth(port_g, spy_g),
        plot_underwater_drawdown(port_dd, spy_dd),
        plot_sensitivity_sharpe_heatmap(sensitivity),
        plot_sensitivity_weight_heatmap(sensitivity, tickers[0]),
    ]
    assert all(len(f.data) > 0 for f in figs), "One or more Plotly figures have no traces"
    print("[PASS] Synthetic math, constraint, metrics, sensitivity, and Plotly chart suite")


def run_live_yfinance_smoke_test() -> None:
    """Verify live Yahoo Finance download for default basket and invalid-ticker error handling."""
    default_tickers = parse_tickers("AAPL, MSFT, NVDA, TLT, GLD")
    bundle, history = load_market_data(tuple(default_tickers), lookback_years=5)
    assert len(bundle.asset_prices) > 200, "Expected > 200 aligned trading days for default basket"
    assert bundle.asset_prices.isna().sum().sum() == 0, "Aligned asset prices contain NaNs"
    assert bundle.benchmark_prices.isna().sum() == 0, "Aligned SPY prices contain NaNs"

    stats = compute_return_stats(bundle.asset_prices)
    tangency = optimize_max_sharpe(stats, risk_free_rate=0.0425, min_weight=0.0, max_weight=0.40)
    _assert_valid_weights(tangency.weights, 0.0, 0.40, "Live Tangency")

    spy_rets = bundle.benchmark_prices.pct_change().dropna()
    port_rets = compute_portfolio_daily_returns(stats.daily_returns, tangency.weights)
    port_m = compute_metrics(port_rets, spy_rets, 0.0425)
    spy_m = compute_metrics(spy_rets, spy_rets, 0.0425)
    _assert_finite_metrics(port_m, "Live Portfolio")
    _assert_finite_metrics(spy_m, "Live SPY")

    # Verify sensitivity grid runs cleanly on live history.
    sens = compute_sensitivity_grid(history, tuple(default_tickers), 0.0, 0.40, 0.0425)
    assert not sens.sharpe_matrix.empty

    # Verify invalid ticker produces a clear DataValidationError naming the bad symbol.
    try:
        load_market_data(("AAPL", "ZZZZINVALID999"), lookback_years=2)
        raise AssertionError("Expected DataValidationError for invalid ticker")
    except DataValidationError as exc:
        assert "ZZZZINVALID999" in str(exc), f"Error did not name bad ticker: {exc}"

    print(
        f"[PASS] Live Yahoo Finance smoke test ({len(bundle.asset_prices)} trading days | "
        f"Max Sharpe={tangency.sharpe_ratio:.2f} | "
        f"Weights={ {k: round(float(v), 3) for k, v in tangency.weights.items()} })"
    )


if __name__ == "__main__":
    run_synthetic_suite()
    run_live_yfinance_smoke_test()
    print("ALL VERIFICATION CHECKS PASSED.")
