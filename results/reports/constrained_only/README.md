# Constrained-Only Portfolio Optimization Results

## Overview

This folder contains portfolio optimization results using **constrained (long-only, no short-selling)** portfolios only. These results represent economically achievable investment opportunities that can actually be implemented in practice.

## Contents

- `portfolio_optimization_results.csv` - Constrained-only portfolio results
  - Minimum-Variance Portfolio (Constrained)
  - Optimal Risky Portfolio (Tangency, Constrained)
  - Equal-Weighted Portfolio

## Why Constrained-Only?

While CAPM assumes frictionless markets with unlimited short-selling, the unconstrained (short-selling) portfolios achieve mathematically optimal but economically unachievable results:

1. **Unrealistic Volatility**: The unconstrained minimum-variance portfolio achieves near-zero volatility (~0.002%), which is not achievable in practice due to:
   - Transaction costs (bid-ask spreads, commissions)
   - Regulatory restrictions on short selling
   - Margin requirements and borrowing costs
   - Liquidity constraints
   - Market impact of large positions

2. **Investment Objective**: Our goal is to provide actionable investment recommendations. The constrained solution, while not perfectly aligned with CAPM theory, yields economically meaningful results that can be implemented.

## Comparison with Unconstrained Results

For comparison with unconstrained (short-selling) portfolios, see:
- `../portfolio_optimization_results.csv` - Contains both constrained and unconstrained results
- `../../figures/efficient_frontier.png` - Plot showing both constrained and unconstrained frontiers

## Risk-Free Rate

All calculations use **German 3-month Bund (EUR)** as the risk-free rate for all countries. This ensures consistency across the entire analysis, as all stock returns and market returns are converted to EUR.

## Related Files

- Constrained-only efficient frontier plot: `../../figures_constrained/efficient_frontier.png`
- Full results (with unconstrained): `../portfolio_optimization_results.csv`
- Full efficient frontier plot: `../../figures/efficient_frontier.png`

