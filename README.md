# Institutional Portfolio Optimization Engine

**Live Demo:** https://portfolio-optimizer-371024701820.us-west1.run.app/

A stateless, interactive portfolio optimization and risk analytics web application built with Python and Streamlit, containerized with Docker, and deployed on Google Cloud Run (`us-west1`). Designed to demonstrate practical mean-variance optimization, constrained portfolio construction, benchmark evaluation, and parameter sensitivity analysis.

## Key Features

- **IPS-Style Sidebar Controls:** Configure custom ticker baskets (default: `AAPL`, `MSFT`, `NVDA`, `TLT`, `GLD` benchmarked against `SPY`), lookback windows (1–10 years), risk-free rate assumptions, long-only per-asset weight bounds, and optimization objectives (`Max Sharpe`, `Min Variance`, or `Target Return`).
- **Efficient Frontier & Monte Carlo Simulation:** Runs a bounded 5,000-portfolio Monte Carlo sample alongside constrained `scipy` SLSQP optimization (`max_sharpe`, `min_variance`, `target_return`, and a 40-point efficient frontier) plus a Capital Allocation Line and key portfolio markers.
- **Allocation & Correlation Diagnostics:** Visualizes optimal portfolio weights and cross-asset correlation structure to explain diversification behavior.
- **Benchmark Backtest & Risk Metrics:** Compares cumulative growth (`$10,000` starting value) and underwater drawdowns against `SPY`, reporting annualized return, annualized volatility, Sharpe ratio, Sortino ratio, maximum drawdown, beta vs. `SPY`, and 95% historical VaR and CVaR.
- **Parameter Sensitivity Analysis:** Evaluates how optimal Sharpe ratios and weights shift across lookback horizons and risk-free rate assumptions.

## Architecture & Tech Stack

- **Runtime & UI:** Python 3.14, Streamlit (`@st.cache_data(ttl=3600)` for 1-hour in-memory caching of Yahoo Finance price series)
- **Quantitative & Visualization Stack:** `numpy`, `pandas`, `scipy`, `plotly`, `yfinance`
- **Deployment:** Docker (`python:3.14-slim`) on Google Cloud Run (`us-west1`), stateless architecture with zero database overhead and scale-to-zero (`min-instances=0`, `max-instances=2`)

## Repository Structure

```text
portfolio-optimizer/
├── main.py              # Streamlit entry point, sidebar IPS controls, and 4 analysis tabs
├── src/
│   ├── data.py          # Cached yfinance price retrieval, date alignment, and ticker validation
│   ├── optimizer.py     # Return/covariance estimation, Monte Carlo sampling, and SLSQP solvers
│   ├── metrics.py       # Risk/return metrics, $10,000 growth curve, and drawdown calculations
│   └── charts.py        # Interactive Plotly charts across all four tabs
├── tests/
│   └── verify_app.py    # Automated verification checks for modules, math constraints, and data
├── docs/
│   └── ARCHITECTURE.md  # Quantitative conventions and system design notes
├── Dockerfile           # Container build and Cloud Run startup configuration
├── .gcloudignore        # Excludes local virtual environments, tests, and docs from cloud uploads
└── requirements.txt     # Pinned/required Python dependencies