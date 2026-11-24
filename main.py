"""Stateless Institutional Portfolio Optimization Tool built with Streamlit."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.charts import (
    plot_backtest_growth,
    plot_correlation_heatmap,
    plot_efficient_frontier,
    plot_ips_weights,
    plot_sensitivity_sharpe_heatmap,
    plot_sensitivity_weight_heatmap,
    plot_underwater_drawdown,
)
from src.data import DataValidationError, load_market_data, parse_tickers
from src.metrics import (
    INITIAL_CAPITAL,
    build_metrics_comparison_table,
    compute_growth_and_drawdown,
    compute_metrics,
    compute_portfolio_daily_returns,
)
from src.optimizer import (
    DEFAULT_MC_PORTFOLIOS,
    FRONTIER_POINTS,
    OptimizationError,
    PortfolioResult,
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

DEFAULT_TICKERS = "AAPL, MSFT, NVDA, TLT, GLD"


def _configure_page() -> None:
    st.set_page_config(
        page_title="Institutional Portfolio Optimizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
            div[data-testid="stMetric"] {
                background-color: rgba(15, 44, 89, 0.03);
                border: 1px solid rgba(15, 44, 89, 0.10);
                border-radius: 8px;
                padding: 0.75rem 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_portfolio_summary_table(portfolios: list[PortfolioResult]) -> pd.DataFrame:
    rows = [
        {
            "Portfolio": p.name,
            "Expected Return": f"{p.expected_return:.2%}",
            "Annualized Volatility": f"{p.volatility:.2%}",
            "Sharpe Ratio": f"{p.sharpe_ratio:.2f}",
            "Top Holding": f"{p.weights.idxmax()} ({p.weights.max():.1%})",
        }
        for p in portfolios
    ]
    return pd.DataFrame(rows)


def _build_asset_allocation_table(
    weights: pd.Series,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> pd.DataFrame:
    vols = cov_matrix.to_numpy().diagonal() ** 0.5
    vol_series = pd.Series(vols, index=cov_matrix.index)
    ordered = weights.sort_values(ascending=False)
    rows = [
        {
            "Ticker": ticker,
            "Optimal Weight": f"{w:.2%}",
            "$10,000 Allocation": f"${w * INITIAL_CAPITAL:,.0f}",
            "Standalone Ann. Return": f"{mean_returns[ticker]:.2%}",
            "Standalone Ann. Volatility": f"{vol_series[ticker]:.2%}",
        }
        for ticker, w in ordered.items()
    ]
    return pd.DataFrame(rows)


def main() -> None:
    _configure_page()

    st.title("Institutional Portfolio Optimization Tool")
    st.caption(
        "Stateless Markowitz mean-variance optimizer, constrained Monte Carlo simulator, "
        "SPY backtest, and assumption sensitivity suite."
    )

    # --- Sidebar IPS Controls ---
    with st.sidebar:
        st.header("Investment Policy Statement")
        raw_tickers = st.text_area(
            "Ticker Basket",
            value=DEFAULT_TICKERS,
            height=72,
            help="Enter 2 or more Yahoo Finance ticker symbols separated by commas or spaces.",
        )

        lookback_years = st.slider(
            "Lookback Window (Years)",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
        )

        risk_free_pct = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=15.0,
            value=4.25,
            step=0.25,
            format="%.2f",
        )
        risk_free_rate = risk_free_pct / 100.0

        st.subheader("Long-Only Weight Bounds")
        col_min, col_max = st.columns(2)
        with col_min:
            min_weight_pct = st.number_input(
                "Min Weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                step=1.0,
                format="%.1f",
            )
        with col_max:
            max_weight_pct = st.number_input(
                "Max Weight (%)",
                min_value=0.0,
                max_value=100.0,
                value=40.0,
                step=1.0,
                format="%.1f",
            )
        min_weight = min_weight_pct / 100.0
        max_weight = max_weight_pct / 100.0

        objective = st.selectbox(
            "Optimization Objective",
            options=["Max Sharpe", "Min Variance", "Target Return"],
            index=0,
        )
        # Placeholder keeps the dynamic Target Return slider directly under the objective selector.
        target_return_placeholder = st.container()

    # --- Input Validation & Market Data Download ---
    try:
        tickers = parse_tickers(raw_tickers)
        validate_weight_bounds(len(tickers), min_weight, max_weight)
    except (DataValidationError, OptimizationError) as exc:
        st.error(str(exc))
        st.stop()

    with st.spinner("Loading cached Yahoo Finance market history..."):
        try:
            bundle, full_history = load_market_data(tuple(tickers), lookback_years)
        except DataValidationError as exc:
            st.error(str(exc))
            st.stop()

    if bundle.truncated:
        st.info(
            f"Requested {lookback_years}Y lookback, but overlapping history across all tickers "
            f"spans {bundle.actual_years:.1f} years "
            f"({bundle.start_date.strftime('%Y-%m-%d')} to {bundle.end_date.strftime('%Y-%m-%d')})."
        )

    # --- Core Optimization & Simulation ---
    try:
        stats = compute_return_stats(bundle.asset_prices)
        gmv = optimize_min_variance(stats, risk_free_rate, min_weight, max_weight)
        tangency = optimize_max_sharpe(stats, risk_free_rate, min_weight, max_weight)
        equal_weight = compute_equal_weight_portfolio(stats, risk_free_rate)
        min_feasible_ret, max_feasible_ret = get_feasible_return_range(stats, min_weight, max_weight)

        if objective == "Max Sharpe":
            active_portfolio = tangency
        elif objective == "Min Variance":
            active_portfolio = gmv
        else:
            # Constrain target return to the efficient upper branch [GMV return, max feasible return].
            lower_ret = gmv.expected_return
            upper_ret = max(max_feasible_ret, lower_ret)
            min_pct = round(lower_ret * 100.0, 2)
            max_pct = round(upper_ret * 100.0, 2)
            with target_return_placeholder:
                if max_pct <= min_pct:
                    target_ret = lower_ret
                    st.caption(f"Single feasible return point: {target_ret:.2%}")
                else:
                    default_target = min(
                        max(tangency.expected_return, lower_ret),
                        upper_ret,
                    )
                    default_pct = min(max(round(default_target * 100.0, 2), min_pct), max_pct)
                    target_pct = st.slider(
                        "Target Annual Return (%)",
                        min_value=min_pct,
                        max_value=max_pct,
                        value=default_pct,
                        step=0.10,
                        format="%.2f%%",
                    )
                    target_ret = min(max(target_pct / 100.0, min_feasible_ret), max_feasible_ret)
            active_portfolio = optimize_target_return(
                stats=stats,
                target_return=target_ret,
                risk_free_rate=risk_free_rate,
                min_weight=min_weight,
                max_weight=max_weight,
            )

        frontier_df = compute_efficient_frontier(
            stats=stats,
            risk_free_rate=risk_free_rate,
            min_weight=min_weight,
            max_weight=max_weight,
            n_points=FRONTIER_POINTS,
        )
        mc_df = run_monte_carlo(
            stats=stats,
            risk_free_rate=risk_free_rate,
            min_weight=min_weight,
            max_weight=max_weight,
            n_portfolios=DEFAULT_MC_PORTFOLIOS,
        )
    except OptimizationError as exc:
        st.error(str(exc))
        st.stop()

    # --- Backtest & Institutional Metrics ---
    spy_daily_returns = bundle.benchmark_prices.pct_change().dropna()
    port_daily_returns = compute_portfolio_daily_returns(stats.daily_returns, active_portfolio.weights)

    port_metrics = compute_metrics(port_daily_returns, spy_daily_returns, risk_free_rate)
    spy_metrics = compute_metrics(spy_daily_returns, spy_daily_returns, risk_free_rate)

    port_growth, port_dd = compute_growth_and_drawdown(port_daily_returns)
    spy_growth, spy_dd = compute_growth_and_drawdown(spy_daily_returns)

    # --- Top Executive KPI Bar ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Expected Return (MVO)", f"{active_portfolio.expected_return:.2%}")
    k2.metric("Annualized Volatility", f"{active_portfolio.volatility:.2%}")
    k3.metric("Sharpe Ratio", f"{active_portfolio.sharpe_ratio:.2f}")
    k4.metric("Max Drawdown (Backtest)", f"{port_metrics.max_drawdown:.2%}")

    st.caption(
        f"Active Objective: **{active_portfolio.name}** | "
        f"Sample: **{bundle.start_date.strftime('%Y-%m-%d')} → {bundle.end_date.strftime('%Y-%m-%d')}** "
        f"({len(stats.daily_returns):,} trading days) | "
        f"IPS Bounds: **{min_weight:.0%} – {max_weight:.0%}** | "
        f"Risk-Free Rate: **{risk_free_rate:.2%}**"
    )

    # --- Four Main Analysis Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. Efficient Frontier & Monte Carlo",
            "2. IPS Allocation & Correlation",
            "3. Backtest vs. SPY",
            "4. Sensitivity Analysis",
        ]
    )

    with tab1:
        frontier_fig = plot_efficient_frontier(
            mc_df=mc_df,
            frontier_df=frontier_df,
            tangency=tangency,
            gmv=gmv,
            equal_weight=equal_weight,
            risk_free_rate=risk_free_rate,
            selected_portfolio=active_portfolio,
        )
        st.plotly_chart(frontier_fig, width="stretch")

        summary_portfolios = [tangency, gmv, equal_weight]
        if active_portfolio.name == "Target Return":
            summary_portfolios.insert(0, active_portfolio)
        st.subheader("Portfolio Comparison")
        st.dataframe(
            _build_portfolio_summary_table(summary_portfolios),
            width="stretch",
            hide_index=True,
        )

    with tab2:
        col_left, col_right = st.columns(2)
        with col_left:
            weights_fig = plot_ips_weights(
                weights=active_portfolio.weights,
                min_weight=min_weight,
                max_weight=max_weight,
                title=f"Optimal IPS Weights ({active_portfolio.name})",
            )
            st.plotly_chart(weights_fig, width="stretch")
        with col_right:
            corr_fig = plot_correlation_heatmap(stats.corr_matrix)
            st.plotly_chart(corr_fig, width="stretch")

        st.subheader("Asset Allocation & Standalone Risk Profile")
        st.dataframe(
            _build_asset_allocation_table(
                weights=active_portfolio.weights,
                mean_returns=stats.mean_returns,
                cov_matrix=stats.cov_matrix,
            ),
            width="stretch",
            hide_index=True,
        )

    with tab3:
        st.subheader("Institutional Performance Scorecard")
        scorecard = build_metrics_comparison_table(
            portfolio_metrics=port_metrics,
            spy_metrics=spy_metrics,
            portfolio_label=f"Portfolio ({active_portfolio.name})",
        )
        st.dataframe(scorecard, width="stretch", hide_index=True)

        growth_fig = plot_backtest_growth(
            port_growth=port_growth,
            spy_growth=spy_growth,
            portfolio_label=f"Portfolio ({active_portfolio.name})",
        )
        st.plotly_chart(growth_fig, width="stretch")

        dd_fig = plot_underwater_drawdown(
            port_dd=port_dd,
            spy_dd=spy_dd,
            portfolio_label=f"Portfolio ({active_portfolio.name})",
        )
        st.plotly_chart(dd_fig, width="stretch")

    with tab4:
        st.caption(
            "Evaluates how the optimal Maximum Sharpe portfolio and asset weights shift across "
            "lookback windows and risk-free rate assumptions under the active IPS weight bounds."
        )
        sensitivity = compute_sensitivity_grid(
            history=full_history,
            tickers=tuple(tickers),
            min_weight=min_weight,
            max_weight=max_weight,
            current_rf=risk_free_rate,
        )

        sharpe_hm = plot_sensitivity_sharpe_heatmap(sensitivity)
        st.plotly_chart(sharpe_hm, width="stretch")

        default_ticker = active_portfolio.weights.idxmax()
        selected_ticker = st.selectbox(
            "Inspect Optimal Weight Shift for Asset",
            options=tickers,
            index=tickers.index(default_ticker),
        )
        weight_hm = plot_sensitivity_weight_heatmap(sensitivity, ticker=selected_ticker)
        st.plotly_chart(weight_hm, width="stretch")


if __name__ == "__main__":
    main()
