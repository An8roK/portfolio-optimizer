"""Minimalist, institutional-grade interactive Plotly visualizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.optimizer import PortfolioResult, SensitivityResult

# Cohesive institutional color palette designed for high contrast and legibility.
NAVY = "#0F2C59"
TEAL = "#0E7490"
AMBER = "#D97706"
SLATE = "#64748B"
CRIMSON = "#B91C1C"
GRID_COLOR = "#E2E8F0"
TEXT_COLOR = "#1E293B"
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"


def _apply_base_layout(
    fig: go.Figure,
    title: str,
    height: int = 470,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
) -> go.Figure:
    """Apply consistent minimalist styling, spacing, and typography to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR), x=0.0, xanchor="left"),
        template="plotly_white",
        height=height,
        font=dict(family=FONT_FAMILY, size=12, color=TEXT_COLOR),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.98)",
        margin=dict(l=60, r=36, t=64, b=56),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=11),
        ),
        xaxis=dict(
            title=xaxis_title,
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            showline=True,
            linecolor="#CBD5E1",
        ),
        yaxis=dict(
            title=yaxis_title,
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            showline=True,
            linecolor="#CBD5E1",
        ),
    )
    return fig


def plot_efficient_frontier(
    mc_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    tangency: PortfolioResult,
    gmv: PortfolioResult,
    equal_weight: PortfolioResult,
    risk_free_rate: float,
    selected_portfolio: PortfolioResult | None = None,
) -> go.Figure:
    """Plot 5,000-point Monte Carlo cloud, Efficient Frontier, CAL, and key portfolio markers."""
    fig = go.Figure()

    # 1. Monte Carlo cloud colored by Sharpe ratio.
    fig.add_trace(
        go.Scattergl(
            x=mc_df["volatility"],
            y=mc_df["return"],
            mode="markers",
            name="Monte Carlo (5,000)",
            marker=dict(
                size=4.5,
                color=mc_df["sharpe"],
                colorscale="Viridis",
                opacity=0.52,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Sharpe", side="top"),
                    thickness=12,
                    len=0.72,
                    x=1.02,
                ),
            ),
            customdata=mc_df["sharpe"],
            hovertemplate=(
                "Volatility: %{x:.2%}<br>"
                "Expected Return: %{y:.2%}<br>"
                "Sharpe Ratio: %{customdata:.2f}<extra>Monte Carlo</extra>"
            ),
        )
    )

    # 2. Efficient Frontier curve.
    fig.add_trace(
        go.Scatter(
            x=frontier_df["volatility"],
            y=frontier_df["return"],
            mode="lines",
            name="Efficient Frontier",
            line=dict(color=NAVY, width=3.0),
            customdata=frontier_df["sharpe"],
            hovertemplate=(
                "Volatility: %{x:.2%}<br>"
                "Expected Return: %{y:.2%}<br>"
                "Sharpe Ratio: %{customdata:.2f}<extra>Efficient Frontier</extra>"
            ),
        )
    )

    # 3. Capital Allocation Line from risk-free rate through the Tangency portfolio.
    if tangency.volatility > 1e-9:
        max_vol = max(
            float(frontier_df["volatility"].max()),
            float(mc_df["volatility"].max()),
            equal_weight.volatility,
            tangency.volatility,
        )
        cal_x = np.array([0.0, tangency.volatility, max_vol * 1.06])
        cal_slope = (tangency.expected_return - risk_free_rate) / tangency.volatility
        cal_y = risk_free_rate + cal_slope * cal_x

        fig.add_trace(
            go.Scatter(
                x=cal_x,
                y=cal_y,
                mode="lines",
                name="Capital Allocation Line",
                line=dict(color=AMBER, width=1.8, dash="dash"),
                hovertemplate=(
                    "Volatility: %{x:.2%}<br>"
                    "Expected Return: %{y:.2%}<extra>CAL</extra>"
                ),
            )
        )

    # 4. Key portfolio markers.
    marker_specs = [
        (tangency, "Tangency (Max Sharpe)", "star", 16, AMBER),
        (gmv, "Min Variance (GMV)", "diamond", 12, TEAL),
        (equal_weight, "Equal Weight (1/N)", "square", 11, SLATE),
    ]
    if selected_portfolio is not None and selected_portfolio.name == "Target Return":
        marker_specs.append((selected_portfolio, "Target Return (Active)", "circle", 13, CRIMSON))

    for port, label, symbol, size, color in marker_specs:
        fig.add_trace(
            go.Scatter(
                x=[port.volatility],
                y=[port.expected_return],
                mode="markers",
                name=label,
                marker=dict(
                    symbol=symbol,
                    size=size,
                    color=color,
                    line=dict(color="white", width=1.6),
                ),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "Volatility: %{x:.2%}<br>"
                    "Expected Return: %{y:.2%}<br>"
                    f"Sharpe Ratio: {port.sharpe_ratio:.2f}<extra></extra>"
                ),
            )
        )

    _apply_base_layout(
        fig,
        title="Efficient Frontier, Monte Carlo Cloud & Capital Allocation Line",
        height=520,
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Expected Return",
    )
    fig.update_xaxes(tickformat=".1%", rangemode="tozero")
    fig.update_yaxes(tickformat=".1%")
    fig.update_layout(margin=dict(l=64, r=84, t=74, b=56))
    return fig


def plot_ips_weights(
    weights: pd.Series,
    min_weight: float,
    max_weight: float,
    title: str = "Optimal IPS Portfolio Weights",
) -> go.Figure:
    """Horizontal bar chart of optimal asset weights with IPS min/max constraint markers."""
    sorted_w = weights.sort_values(ascending=True)

    fig = go.Figure(
        go.Bar(
            x=sorted_w.values,
            y=sorted_w.index,
            orientation="h",
            marker=dict(color=NAVY, line=dict(width=0)),
            text=[f"{val:.1%}" for val in sorted_w.values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Target Weight: %{x:.2%}<extra></extra>",
        )
    )

    if max_weight < 1.0:
        fig.add_vline(
            x=max_weight,
            line_width=1.5,
            line_dash="dash",
            line_color=AMBER,
            annotation_text=f"Max {max_weight:.0%}",
            annotation_position="top right",
            annotation_font=dict(size=11, color=AMBER),
        )
    if min_weight > 0.0:
        fig.add_vline(
            x=min_weight,
            line_width=1.5,
            line_dash="dot",
            line_color=TEAL,
            annotation_text=f"Min {min_weight:.0%}",
            annotation_position="top left",
            annotation_font=dict(size=11, color=TEAL),
        )

    x_upper = max(float(sorted_w.max()) * 1.22, max_weight * 1.12, 0.15)
    _apply_base_layout(
        fig,
        title=title,
        height=420,
        xaxis_title="Allocation Weight",
        yaxis_title=None,
    )
    fig.update_xaxes(tickformat=".0%", range=[0.0, min(x_upper, 1.15)])
    fig.update_layout(showlegend=False, margin=dict(l=60, r=48, t=64, b=52))
    return fig


def plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Annotated diverging correlation heatmap across the asset basket."""
    fig = go.Figure(
        go.Heatmap(
            z=corr_matrix.to_numpy(dtype=float),
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            zmin=-1.0,
            zmax=1.0,
            zmid=0.0,
            colorscale="RdBu_r",
            text=np.round(corr_matrix.to_numpy(dtype=float), 2),
            texttemplate="%{text:.2f}",
            textfont=dict(size=12),
            colorbar=dict(title=dict(text="Corr", side="top"), thickness=12, len=0.82),
            hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>",
        )
    )

    _apply_base_layout(
        fig,
        title="Asset Return Correlation Matrix",
        height=420,
        xaxis_title=None,
        yaxis_title=None,
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    fig.update_layout(margin=dict(l=60, r=60, t=64, b=52))
    return fig


def plot_backtest_growth(
    port_growth: pd.Series,
    spy_growth: pd.Series,
    portfolio_label: str = "Optimized Portfolio",
) -> go.Figure:
    """Cumulative growth of $10,000 comparing the active portfolio against SPY."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=port_growth.index,
            y=port_growth.values,
            mode="lines",
            name=portfolio_label,
            line=dict(color=NAVY, width=2.6),
            hovertemplate="$%{y:,.0f}<extra>" + portfolio_label + "</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spy_growth.index,
            y=spy_growth.values,
            mode="lines",
            name="SPY Benchmark",
            line=dict(color=SLATE, width=2.0, dash="dot"),
            hovertemplate="$%{y:,.0f}<extra>SPY Benchmark</extra>",
        )
    )

    _apply_base_layout(
        fig,
        title="Historical Growth of $10,000 (Daily Rebalanced)",
        height=400,
        xaxis_title=None,
        yaxis_title="Portfolio Value (USD)",
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    fig.update_layout(hovermode="x unified")
    return fig


def plot_underwater_drawdown(
    port_dd: pd.Series,
    spy_dd: pd.Series,
    portfolio_label: str = "Optimized Portfolio",
) -> go.Figure:
    """Underwater drawdown area curves comparing the active portfolio against SPY."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=port_dd.index,
            y=port_dd.values,
            mode="lines",
            name=portfolio_label,
            line=dict(color=CRIMSON, width=2.0),
            fill="tozeroy",
            fillcolor="rgba(185, 28, 28, 0.14)",
            hovertemplate="%{y:.2%}<extra>" + portfolio_label + "</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=spy_dd.index,
            y=spy_dd.values,
            mode="lines",
            name="SPY Benchmark",
            line=dict(color=SLATE, width=1.8, dash="dot"),
            fill="tozeroy",
            fillcolor="rgba(100, 116, 139, 0.10)",
            hovertemplate="%{y:.2%}<extra>SPY Benchmark</extra>",
        )
    )

    _apply_base_layout(
        fig,
        title="Underwater Drawdown vs. SPY",
        height=340,
        xaxis_title=None,
        yaxis_title="Drawdown from Peak",
    )
    fig.update_yaxes(tickformat=".0%", zeroline=True, zerolinecolor="#94A3B8")
    fig.update_layout(hovermode="x unified")
    return fig


def _build_weight_hover_matrix(sensitivity: SensitivityResult) -> list[list[str]]:
    """Format full optimal weight breakdowns for each (lookback, risk-free rate) heatmap cell."""
    hover_grid: list[list[str]] = []
    for lb_label in sensitivity.lookback_labels:
        row_text: list[str] = []
        for rf in sensitivity.risk_free_rates:
            weights = sensitivity.weights_by_cell[(lb_label, rf)]
            breakdown = "<br>".join(
                f"{ticker}: {w:.1%}" for ticker, w in weights.sort_values(ascending=False).items()
            )
            row_text.append(breakdown)
        hover_grid.append(row_text)
    return hover_grid


def plot_sensitivity_sharpe_heatmap(sensitivity: SensitivityResult) -> go.Figure:
    """Heatmap of optimal Max Sharpe ratio across lookback windows and risk-free rates."""
    z_vals = sensitivity.sharpe_matrix.to_numpy(dtype=float)
    hover_weights = _build_weight_hover_matrix(sensitivity)

    fig = go.Figure(
        go.Heatmap(
            z=z_vals,
            x=list(sensitivity.sharpe_matrix.columns),
            y=list(sensitivity.sharpe_matrix.index),
            colorscale="YlGnBu",
            text=np.round(z_vals, 2),
            texttemplate="%{text:.2f}",
            textfont=dict(size=12),
            customdata=hover_weights,
            colorbar=dict(title=dict(text="Sharpe", side="top"), thickness=12, len=0.85),
            hovertemplate=(
                "<b>Lookback: %{y} | Risk-Free: %{x}</b><br>"
                "Optimal Sharpe: %{z:.2f}<br><br>"
                "<b>Optimal Weights:</b><br>%{customdata}<extra></extra>"
            ),
        )
    )

    _apply_base_layout(
        fig,
        title="Optimal Max Sharpe Ratio Sensitivity (Lookback × Risk-Free Rate)",
        height=390,
        xaxis_title="Risk-Free Rate Assumption",
        yaxis_title="Historical Lookback Window",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(margin=dict(l=84, r=64, t=64, b=56))
    return fig


def plot_sensitivity_weight_heatmap(
    sensitivity: SensitivityResult,
    ticker: str,
) -> go.Figure:
    """Heatmap of a selected asset's optimal Max-Sharpe weight across lookback and rf assumptions."""
    z_rows: list[list[float]] = []
    for lb_label in sensitivity.lookback_labels:
        z_rows.append(
            [
                float(sensitivity.weights_by_cell[(lb_label, rf)].get(ticker, 0.0))
                for rf in sensitivity.risk_free_rates
            ]
        )
    z_vals = np.array(z_rows, dtype=float)
    hover_weights = _build_weight_hover_matrix(sensitivity)

    fig = go.Figure(
        go.Heatmap(
            z=z_vals,
            x=list(sensitivity.sharpe_matrix.columns),
            y=list(sensitivity.sharpe_matrix.index),
            zmin=0.0,
            zmax=max(float(z_vals.max()), 0.05),
            colorscale="Blues",
            text=z_vals,
            texttemplate="%{text:.1%}",
            textfont=dict(size=12),
            customdata=hover_weights,
            colorbar=dict(
                title=dict(text="Weight", side="top"),
                tickformat=".0%",
                thickness=12,
                len=0.85,
            ),
            hovertemplate=(
                f"<b>{ticker} Weight — Lookback: %{{y}} | Rf: %{{x}}</b><br>"
                "Optimal Weight: %{z:.1%}<br><br>"
                "<b>Full Portfolio:</b><br>%{customdata}<extra></extra>"
            ),
        )
    )

    _apply_base_layout(
        fig,
        title=f"Optimal Weight Sensitivity for {ticker} (Lookback × Risk-Free Rate)",
        height=390,
        xaxis_title="Risk-Free Rate Assumption",
        yaxis_title="Historical Lookback Window",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(margin=dict(l=84, r=64, t=64, b=56))
    return fig
