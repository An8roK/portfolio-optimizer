"""Constrained mean-variance optimization, Monte Carlo simulation, and sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.data import DataValidationError, slice_price_bundle

TRADING_DAYS = 252
DEFAULT_MC_PORTFOLIOS = 5000
FRONTIER_POINTS = 40
_BOUND_TOL = 1e-8


class OptimizationError(ValueError):
    """Raised when portfolio constraints are infeasible or optimization fails."""


@dataclass(frozen=True)
class ReturnStats:
    """Daily simple returns and annualized mean/covariance statistics."""

    tickers: list[str]
    daily_returns: pd.DataFrame
    mean_returns: pd.Series
    cov_matrix: pd.DataFrame
    corr_matrix: pd.DataFrame


@dataclass(frozen=True)
class PortfolioResult:
    """Solved portfolio allocation and annualized risk/return summary."""

    name: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float


@dataclass(frozen=True)
class SensitivityResult:
    """Grid of optimal Max Sharpe ratios and weights across lookback windows and risk-free rates."""

    lookback_labels: list[str]
    risk_free_rates: list[float]
    sharpe_matrix: pd.DataFrame
    weights_by_cell: dict[tuple[str, float], pd.Series]


def validate_weight_bounds(n_assets: int, min_weight: float, max_weight: float) -> None:
    """Ensure long-only per-asset weight bounds admit a feasible fully invested portfolio."""
    if n_assets < 2:
        raise OptimizationError("At least 2 assets are required for portfolio optimization.")
    if min_weight < 0.0 or max_weight > 1.0:
        raise OptimizationError("Weight bounds must stay within [0%, 100%] for long-only portfolios.")
    if min_weight > max_weight + 1e-12:
        raise OptimizationError(
            f"Minimum weight ({min_weight:.1%}) cannot exceed maximum weight ({max_weight:.1%})."
        )
    if n_assets * min_weight > 1.0 + 1e-9:
        raise OptimizationError(
            f"Infeasible minimum weight: {n_assets} assets × {min_weight:.1%} = "
            f"{n_assets * min_weight:.1%} > 100.0%. Lower the minimum weight or remove assets."
        )
    if n_assets * max_weight < 1.0 - 1e-9:
        raise OptimizationError(
            f"Infeasible maximum weight: {n_assets} assets × {max_weight:.1%} = "
            f"{n_assets * max_weight:.1%} < 100.0%. Raise the maximum weight or add assets."
        )


def compute_return_stats(asset_prices: pd.DataFrame) -> ReturnStats:
    """Compute daily simple returns, annualized expected returns, covariance, and correlation."""
    daily_returns = asset_prices.pct_change().dropna(how="any")
    if daily_returns.empty or len(daily_returns) < 2:
        raise OptimizationError("Insufficient return observations to compute covariance matrix.")

    mean_returns = daily_returns.mean() * TRADING_DAYS
    cov_matrix = daily_returns.cov() * TRADING_DAYS
    corr_matrix = daily_returns.corr()

    return ReturnStats(
        tickers=list(asset_prices.columns),
        daily_returns=daily_returns,
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        corr_matrix=corr_matrix,
    )


def _portfolio_stats(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
) -> tuple[float, float, float]:
    """Return annualized (expected_return, volatility, sharpe_ratio) for a weight vector."""
    exp_ret = float(weights @ mu)
    variance = max(float(weights @ cov @ weights), 0.0)
    vol = float(np.sqrt(variance))
    sharpe = float((exp_ret - risk_free_rate) / vol) if vol > 1e-12 else 0.0
    return exp_ret, vol, sharpe


def _project_bounded_simplex(vectors: np.ndarray, low: float, high: float) -> np.ndarray:
    """Project rows onto {w : sum(w) = 1, low <= w_i <= high} via exact vectorized bisection.

    Solving sum(clip(v - tau, low, high)) = 1 guarantees exact feasibility even under tight
    box constraints where naive Dirichlet rejection sampling would stall.
    """
    single = vectors.ndim == 1
    arr = np.atleast_2d(np.asarray(vectors, dtype=float))

    tau_low = np.min(arr - high, axis=1, keepdims=True) - 1.0
    tau_high = np.max(arr - low, axis=1, keepdims=True) + 1.0

    for _ in range(52):
        tau_mid = 0.5 * (tau_low + tau_high)
        clipped_sum = np.sum(np.clip(arr - tau_mid, low, high), axis=1, keepdims=True)
        tau_low = np.where(clipped_sum > 1.0, tau_mid, tau_low)
        tau_high = np.where(clipped_sum <= 1.0, tau_mid, tau_high)

    projected = np.clip(arr - 0.5 * (tau_low + tau_high), low, high)
    return projected[0] if single else projected


def _greedy_extreme_weights(
    mu: np.ndarray,
    min_weight: float,
    max_weight: float,
    maximize: bool,
) -> np.ndarray:
    """Solve max/min w^T mu over the bounded simplex analytically via greedy allocation.

    Because the objective is linear and all box constraints are symmetric, allocating surplus
    weight (1 - N * min_weight) to assets sorted by expected return yields the exact optimum.
    """
    n = len(mu)
    weights = np.full(n, min_weight, dtype=float)
    remaining = max(0.0, 1.0 - n * min_weight)
    capacity = max(0.0, max_weight - min_weight)

    order = np.argsort(-mu if maximize else mu)
    for idx in order:
        if remaining <= 1e-15:
            break
        add = min(capacity, remaining)
        weights[idx] += add
        remaining -= add

    return _project_bounded_simplex(weights, min_weight, max_weight)


def get_feasible_return_range(
    stats: ReturnStats,
    min_weight: float,
    max_weight: float,
) -> tuple[float, float]:
    """Return the exact (min_feasible_return, max_feasible_return) under weight bounds."""
    validate_weight_bounds(len(stats.tickers), min_weight, max_weight)
    mu = stats.mean_returns.to_numpy(dtype=float)
    w_min = _greedy_extreme_weights(mu, min_weight, max_weight, maximize=False)
    w_max = _greedy_extreme_weights(mu, min_weight, max_weight, maximize=True)
    return float(w_min @ mu), float(w_max @ mu)


def _sample_bounded_weights(
    n_assets: int,
    n_portfolios: int,
    min_weight: float,
    max_weight: float,
    seed: int = 42,
) -> np.ndarray:
    """Generate feasible random weights spanning both interior and near-boundary regions."""
    rng = np.random.default_rng(seed)
    if abs(max_weight - min_weight) <= 1e-12:
        return np.full((n_portfolios, n_assets), 1.0 / n_assets, dtype=float)

    # Mix Dirichlet concentrations so the cloud explores both diversified and concentrated mixes.
    alphas = (0.35, 1.0, 2.5)
    accepted_chunks: list[np.ndarray] = []
    accepted_count = 0

    for _ in range(12):
        needed = n_portfolios - accepted_count
        if needed <= 0:
            break
        batch_size = min(max(needed * 3, 2000), 60000)
        alpha = alphas[len(accepted_chunks) % len(alphas)]
        raw = rng.dirichlet(np.full(n_assets, alpha), size=batch_size)
        valid_mask = (raw >= min_weight - 1e-12).all(axis=1) & (raw <= max_weight + 1e-12).all(axis=1)
        valid = raw[valid_mask]
        if len(valid) > 0:
            take = valid[:needed]
            accepted_chunks.append(take)
            accepted_count += len(take)

    if accepted_count < n_portfolios:
        # Tight bounds fallback: project varied random draws exactly onto the bounded simplex.
        remaining = n_portfolios - accepted_count
        spread = max(max_weight - min_weight, 1e-6)
        raw_extra = rng.uniform(min_weight - 0.5 * spread, max_weight + 0.5 * spread, size=(remaining, n_assets))
        projected = _project_bounded_simplex(raw_extra, min_weight, max_weight)
        accepted_chunks.append(projected)

    weights = np.vstack(accepted_chunks)[:n_portfolios]
    # Final projection guarantees numerical sum == 1.0 and strict bound compliance.
    return _project_bounded_simplex(weights, min_weight, max_weight)


def run_monte_carlo(
    stats: ReturnStats,
    risk_free_rate: float,
    min_weight: float,
    max_weight: float,
    n_portfolios: int = DEFAULT_MC_PORTFOLIOS,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate random feasible portfolios within weight bounds and return their risk/return cloud."""
    validate_weight_bounds(len(stats.tickers), min_weight, max_weight)
    mu = stats.mean_returns.to_numpy(dtype=float)
    cov = stats.cov_matrix.to_numpy(dtype=float)

    weights = _sample_bounded_weights(
        n_assets=len(stats.tickers),
        n_portfolios=n_portfolios,
        min_weight=min_weight,
        max_weight=max_weight,
        seed=seed,
    )

    returns = weights @ mu
    variances = np.maximum(np.einsum("ij,jk,ik->i", weights, cov, weights), 0.0)
    volatilities = np.sqrt(variances)
    sharpes = np.where(volatilities > 1e-12, (returns - risk_free_rate) / volatilities, 0.0)

    return pd.DataFrame(
        {
            "return": returns,
            "volatility": volatilities,
            "sharpe": sharpes,
        }
    )


def _clean_weights(weights: np.ndarray, min_weight: float, max_weight: float) -> np.ndarray:
    """Enforce exact simplex sum and box bounds after SLSQP convergence."""
    return _project_bounded_simplex(np.asarray(weights, dtype=float), min_weight, max_weight)


def _to_result(
    name: str,
    weights: np.ndarray,
    stats: ReturnStats,
    risk_free_rate: float,
) -> PortfolioResult:
    mu = stats.mean_returns.to_numpy(dtype=float)
    cov = stats.cov_matrix.to_numpy(dtype=float)
    exp_ret, vol, sharpe = _portfolio_stats(weights, mu, cov, risk_free_rate)
    return PortfolioResult(
        name=name,
        weights=pd.Series(weights, index=stats.tickers, name=name),
        expected_return=exp_ret,
        volatility=vol,
        sharpe_ratio=sharpe,
    )


def optimize_min_variance(
    stats: ReturnStats,
    risk_free_rate: float,
    min_weight: float,
    max_weight: float,
) -> PortfolioResult:
    """Solve the constrained Global Minimum Variance (GMV) portfolio via SLSQP."""
    n = len(stats.tickers)
    validate_weight_bounds(n, min_weight, max_weight)
    cov = stats.cov_matrix.to_numpy(dtype=float)
    bounds = [(min_weight, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]

    x0 = _project_bounded_simplex(np.full(n, 1.0 / n), min_weight, max_weight)

    res = minimize(
        fun=lambda w: float(w @ cov @ w),
        x0=x0,
        jac=lambda w: 2.0 * (cov @ w),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )

    weights = _clean_weights(res.x if res.success else x0, min_weight, max_weight)
    return _to_result("Min Variance", weights, stats, risk_free_rate)


def optimize_max_sharpe(
    stats: ReturnStats,
    risk_free_rate: float,
    min_weight: float,
    max_weight: float,
) -> PortfolioResult:
    """Solve the constrained Maximum Sharpe (Tangency) portfolio using multi-start SLSQP."""
    n = len(stats.tickers)
    validate_weight_bounds(n, min_weight, max_weight)
    mu = stats.mean_returns.to_numpy(dtype=float)
    cov = stats.cov_matrix.to_numpy(dtype=float)
    bounds = [(min_weight, max_weight)] * n
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]

    def neg_sharpe(w: np.ndarray) -> float:
        vol = float(np.sqrt(max(w @ cov @ w, 1e-16)))
        return -float((w @ mu - risk_free_rate) / vol)

    # Multi-start from equal-weight, GMV, and max-return feasible points to avoid local optima.
    gmv = optimize_min_variance(stats, risk_free_rate, min_weight, max_weight)
    starts = [
        _project_bounded_simplex(np.full(n, 1.0 / n), min_weight, max_weight),
        gmv.weights.to_numpy(dtype=float),
        _greedy_extreme_weights(mu, min_weight, max_weight, maximize=True),
    ]

    best_weights = starts[0]
    best_sharpe = -np.inf

    for x0 in starts:
        res = minimize(
            fun=neg_sharpe,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 500},
        )
        candidate = _clean_weights(res.x if res.success else x0, min_weight, max_weight)
        _, _, cand_sharpe = _portfolio_stats(candidate, mu, cov, risk_free_rate)
        if cand_sharpe > best_sharpe:
            best_sharpe = cand_sharpe
            best_weights = candidate

    return _to_result("Max Sharpe", best_weights, stats, risk_free_rate)


def optimize_target_return(
    stats: ReturnStats,
    target_return: float,
    risk_free_rate: float,
    min_weight: float,
    max_weight: float,
    initial_guess: np.ndarray | None = None,
) -> PortfolioResult:
    """Solve minimum-variance portfolio achieving an exact annualized target return."""
    n = len(stats.tickers)
    validate_weight_bounds(n, min_weight, max_weight)
    min_ret, max_ret = get_feasible_return_range(stats, min_weight, max_weight)

    if target_return < min_ret - 1e-6 or target_return > max_ret + 1e-6:
        raise OptimizationError(
            f"Target return ({target_return:.2%}) is outside the feasible range "
            f"[{min_ret:.2%}, {max_ret:.2%}] under the current weight bounds."
        )

    clamped_target = float(np.clip(target_return, min_ret, max_ret))
    mu = stats.mean_returns.to_numpy(dtype=float)
    cov = stats.cov_matrix.to_numpy(dtype=float)

    # Analytical endpoints avoid SLSQP degeneracy at boundary vertices of the simplex.
    if abs(clamped_target - max_ret) <= 1e-9:
        weights = _greedy_extreme_weights(mu, min_weight, max_weight, maximize=True)
        return _to_result("Target Return", weights, stats, risk_free_rate)
    if abs(clamped_target - min_ret) <= 1e-9:
        weights = _greedy_extreme_weights(mu, min_weight, max_weight, maximize=False)
        return _to_result("Target Return", weights, stats, risk_free_rate)

    bounds = [(min_weight, max_weight)] * n
    constraints = [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        {"type": "eq", "fun": lambda w, tr=clamped_target: float(w @ mu - tr)},
    ]

    if initial_guess is not None:
        x0 = _clean_weights(initial_guess, min_weight, max_weight)
    else:
        x0 = _project_bounded_simplex(np.full(n, 1.0 / n), min_weight, max_weight)

    res = minimize(
        fun=lambda w: float(w @ cov @ w),
        x0=x0,
        jac=lambda w: 2.0 * (cov @ w),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )

    if not res.success:
        # Retry from convex combination of min-return and max-return feasible extremes.
        alpha = (clamped_target - min_ret) / max(max_ret - min_ret, 1e-12)
        w_low = _greedy_extreme_weights(mu, min_weight, max_weight, maximize=False)
        w_high = _greedy_extreme_weights(mu, min_weight, max_weight, maximize=True)
        x_alt = (1.0 - alpha) * w_low + alpha * w_high
        res = minimize(
            fun=lambda w: float(w @ cov @ w),
            x0=x_alt,
            jac=lambda w: 2.0 * (cov @ w),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 500},
        )
        weights = _clean_weights(res.x if res.success else x_alt, min_weight, max_weight)
    else:
        weights = _clean_weights(res.x, min_weight, max_weight)

    return _to_result("Target Return", weights, stats, risk_free_rate)


def compute_efficient_frontier(
    stats: ReturnStats,
    risk_free_rate: float,
    min_weight: float,
    max_weight: float,
    n_points: int = FRONTIER_POINTS,
) -> pd.DataFrame:
    """Compute a smooth n-point efficient frontier from GMV return to maximum feasible return."""
    validate_weight_bounds(len(stats.tickers), min_weight, max_weight)
    gmv = optimize_min_variance(stats, risk_free_rate, min_weight, max_weight)
    _, max_ret = get_feasible_return_range(stats, min_weight, max_weight)

    if max_ret - gmv.expected_return <= 1e-7:
        return pd.DataFrame(
            [
                {
                    "return": gmv.expected_return,
                    "volatility": gmv.volatility,
                    "sharpe": gmv.sharpe_ratio,
                }
            ]
        )

    targets = np.linspace(gmv.expected_return, max_ret, n_points)
    rows: list[dict[str, float]] = [
        {
            "return": gmv.expected_return,
            "volatility": gmv.volatility,
            "sharpe": gmv.sharpe_ratio,
        }
    ]
    prev_weights = gmv.weights.to_numpy(dtype=float)

    # Warm-start each SLSQP solve from the previous frontier solution for smooth convergence.
    for target in targets[1:]:
        port = optimize_target_return(
            stats=stats,
            target_return=float(target),
            risk_free_rate=risk_free_rate,
            min_weight=min_weight,
            max_weight=max_weight,
            initial_guess=prev_weights,
        )
        prev_weights = port.weights.to_numpy(dtype=float)
        rows.append(
            {
                "return": port.expected_return,
                "volatility": port.volatility,
                "sharpe": port.sharpe_ratio,
            }
        )

    return pd.DataFrame(rows)


def compute_equal_weight_portfolio(
    stats: ReturnStats,
    risk_free_rate: float,
) -> PortfolioResult:
    """Compute the unconstrained 1/N equal-weight reference portfolio."""
    n = len(stats.tickers)
    weights = np.full(n, 1.0 / n, dtype=float)
    return _to_result("Equal Weight", weights, stats, risk_free_rate)


def compute_sensitivity_grid(
    history: pd.DataFrame,
    tickers: tuple[str, ...],
    min_weight: float,
    max_weight: float,
    current_rf: float,
    lookback_candidates: tuple[int, ...] = (1, 2, 3, 5, 7, 10),
) -> SensitivityResult:
    """Evaluate optimal Max Sharpe ratio and weights across lookback windows and risk-free rates."""
    validate_weight_bounds(len(tickers), min_weight, max_weight)

    base_rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    rf_grid = sorted(
        {round(r, 4) for r in [*base_rates, float(current_rf)] if 0.0 <= r <= 0.25}
    )

    lookback_labels: list[str] = []
    sharpe_rows: list[list[float]] = []
    weights_by_cell: dict[tuple[str, float], pd.Series] = {}
    seen_start_dates: set[pd.Timestamp] = set()

    for years in lookback_candidates:
        try:
            bundle = slice_price_bundle(history=history, tickers=tickers, lookback_years=years)
        except DataValidationError:
            continue

        # Skip redundant longer windows when a recent IPO caps the available overlapping start date.
        if bundle.start_date in seen_start_dates:
            continue
        seen_start_dates.add(bundle.start_date)

        label = f"{years}Y" if not bundle.truncated else f"{bundle.actual_years:.1f}Y (max)"
        stats = compute_return_stats(bundle.asset_prices)
        row_sharpes: list[float] = []

        for rf in rf_grid:
            opt = optimize_max_sharpe(
                stats=stats,
                risk_free_rate=rf,
                min_weight=min_weight,
                max_weight=max_weight,
            )
            row_sharpes.append(opt.sharpe_ratio)
            weights_by_cell[(label, rf)] = opt.weights

        lookback_labels.append(label)
        sharpe_rows.append(row_sharpes)

    sharpe_matrix = pd.DataFrame(
        sharpe_rows,
        index=lookback_labels,
        columns=[f"{rf:.2%}" for rf in rf_grid],
    )

    return SensitivityResult(
        lookback_labels=lookback_labels,
        risk_free_rates=rf_grid,
        sharpe_matrix=sharpe_matrix,
        weights_by_cell=weights_by_cell,
    )
