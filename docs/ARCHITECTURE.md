# Project Architecture

## Overview
A stateless, institutional-grade Portfolio Optimization Tool built locally with Python and Streamlit. It ingests historical adjusted close prices from Yahoo Finance, performs constrained Markowitz mean-variance optimization and bounded Monte Carlo simulation, benchmarks historical risk/return performance against `SPY`, and analyzes parameter sensitivity across lookback windows and risk-free rates.

## Tech Stack & Dependencies
- **Language / Runtime:** Python 3.14 inside a local virtual environment (`.venv`)
- **Database / Storage:** None (stateless architecture; market data cached in RAM for 1 hour via `@st.cache_data(ttl=3600)`)
- **Key Libraries:** `streamlit>=1.52`, `yfinance`, `scipy`, `numpy`, `pandas`, `plotly`

## Directory Layout
- `main.py` — Streamlit UI entry point, sidebar IPS controls, feasibility guards, and four analysis tabs
- `src/data.py` — Cached Yahoo Finance download (`auto_adjust=True`), trading-date alignment, lookback window slicing, and ticker validation
- `src/optimizer.py` — Daily/annualized return statistics (252 trading days), bounded-simplex 5,000-portfolio Monte Carlo simulator, SLSQP solvers (GMV, Max Sharpe, Target Return, 40-point Efficient Frontier), and sensitivity grid computation
- `src/metrics.py` — Portfolio and `SPY` metrics (CAGR, Volatility, Sharpe, Sortino, Max Drawdown, Beta vs. `SPY`, 95% Historical VaR & CVaR), `$10,000` growth curves, and underwater drawdown series
- `src/charts.py` — Minimalist interactive Plotly visualization builders for all four tabs
- `tests/verify_app.py` — Automated verification suite for imports, optimizer constraints, finite metrics, charts, and live market data validation
- `docs/ARCHITECTURE.md` — Architecture reference and quantitative conventions

## Key Conventions & Decisions
- **252 Trading Days:** All annualized expected returns, volatilities, covariance matrices, and CAGR figures use `252` trading days per year.
- **MVO vs. Backtest Consistency:** Expected return in optimization uses arithmetic annualized mean (`daily_mean * 252`), and Sharpe/Sortino ratios use arithmetic excess return so Tab 1 and Tab 3 Sharpe ratios align identically. Headline backtest annualized return uses geometric CAGR to match the `$10,000` cumulative growth curve.
- **Exact Bounded-Simplex Feasibility:** Both SLSQP post-processing and tight-bound Monte Carlo sampling project onto `{w : sum(w) = 1, min_w <= w_i <= max_w}` via exact bisection, ensuring strict weight-sum and bound compliance across arbitrary long-only constraints.
- **Single Cached Download per Basket:** `src/data.py` downloads up to 10 years once per ticker basket and slices trailing windows in memory so lookback adjustments and Tab 4 sensitivity analysis require zero redundant network calls.
