# Stochastic Calculus & Monte Carlo for Options Pricing

Monte Carlo simulation for options pricing, risk measurement, and convergence analysis.

## Overview

This project implements the fundamental numerical method of quantitative finance: Monte Carlo simulation. It prices European options by simulating thousands of possible price paths for the underlying asset, computes the expected payoff under the risk-neutral measure, and compares results to the closed-form Black-Scholes solution.

### Why Monte Carlo?

Some derivatives can't be priced analytically. Path-dependent options (Asians, barriers, lookbacks) and multi-asset options require numerical methods. Monte Carlo is the most flexible and widely used approach:

1. Simulate N possible futures of the underlying asset price
2. Compute the option payoff for each path
3. Discount back to present value
4. Average across all paths → option price

By the Law of Large Numbers, as N → ∞, the Monte Carlo estimate converges to the true price. The standard error decreases as 1/√N — so quadrupling paths halves the error.

## Features

- **Geometric Brownian Motion (GBM)** — standard model for stock price dynamics
- **European call/put pricing** via Monte Carlo with confidence intervals
- **Black-Scholes analytical formula** — closed-form benchmark for comparison
- **VaR and CVaR** — risk metrics for a portfolio of the simulated asset
- **Convergence analysis** — visualize how price estimate improves with more paths
- **Visualization** — price paths, payoff distribution, convergence plots

## Project Structure

```
stochastic-calculus-monte-carlo/
├── mc/
│   ├── __init__.py
│   ├── gbm.py           # Geometric Brownian Motion path generator
│   ├── options.py       # European option Monte Carlo pricer
│   ├── black_scholes.py # Analytical Black-Scholes formula
│   └── risk.py          # VaR / CVaR computation
├── examples/
│   ├── european_option.py      # Price a European option with MC + BS comparison
│   └── convergence_analysis.py # Show how price converges as N increases
├── tests/
│   └── test_mc.py
├── requirements.txt
└── README.md
```

## The Math

### Geometric Brownian Motion

The standard model for stock prices:

```
dS = μ·S·dt + σ·S·dW
```

Where:
- S = stock price
- μ = drift (expected return)
- σ = volatility
- dW = Wiener process (Brownian motion increment)

Discretized for simulation (Euler-Maruyama):

```
S(t+Δt) = S(t) · exp((μ - σ²/2)·Δt + σ·√Δt·Z)
```

Where Z ~ N(0,1) is a standard normal random variable.

### Black-Scholes (European Call)

```
C = S₀ · N(d₁) - K·e^(-rT) · N(d₂)

d₁ = (ln(S₀/K) + (r + σ²/2)·T) / (σ·√T)
d₂ = d₁ - σ·√T
```

### Monte Carlo (European Call)

```
C = e^(-rT) · (1/N) · Σ max(Sᵢ(T) - K, 0)
```

Where Sᵢ(T) is the i-th simulated terminal price.

### Risk Metrics

**VaR (95%)**: The loss that will not be exceeded with 95% confidence.
**CVaR (95%)**: The expected loss given that the loss exceeds VaR.

## How It Works (Step by Step)

### 1. GBM Path Generator (`mc/gbm.py`)

Generates N price paths with M time steps each:

```
Input:  S₀=100, μ=0.05, σ=0.2, T=1.0, N=10000, M=252
Output: (N × M+1) matrix of simulated price paths
```

Each path starts at S₀ and evolves via the GBM discretization. Visualized, they form a fan of possible futures.

### 2. Option Pricer (`mc/options.py`)

For each simulated path:
1. Take the terminal price S(T)
2. Compute payoff: max(S(T) - K, 0) for a call
3. Discount back: payoff × e^(-rT)
4. Average all discounted payoffs → MC price
5. Compute standard error and 95% confidence interval

### 3. Black-Scholes Benchmark (`mc/black_scholes.py`)

Computes the analytical price using the closed-form formula. This serves as the "ground truth" to validate the Monte Carlo estimate.

### 4. Risk Calculator (`mc/risk.py`)

Takes the simulated terminal prices and computes:
- **VaR**: Sort returns, take the 5th percentile
- **CVaR**: Average all returns below the 5th percentile

## Example Output

```
=== European Call Option ===
Spot:        $100.00
Strike:      $105.00
Maturity:    1.0 years
Volatility:  20.0%
Risk-free:   5.0%

Monte Carlo Price:   $12.45  ± $0.08  (95% CI)
Black-Scholes Price: $12.52
Error:               0.56%

VaR (95%):          -$2.34
CVaR (95%):         -$3.87
```

## Convergence Analysis

The convergence example shows how the MC price approaches the BS price as N increases:

```
Paths:    1,000   → Price: $12.81  Error: 2.32%
Paths:   10,000   → Price: $12.45  Error: 0.56%
Paths:  100,000   → Price: $12.50  Error: 0.16%
Paths: 1,000,000  → Price: $12.52  Error: 0.04%
```

The standard error decreases proportionally to 1/√N.

## Getting Started

```bash
pip install -r requirements.txt
python examples/european_option.py
python examples/convergence_analysis.py
```

## Why This Matters for Quant Trading

Options pricing is the most fundamental quantitative finance skill. This project demonstrates:
- **Understanding of stochastic processes** — can you simulate and reason about random processes?
- **Numerical methods** — can you implement Monte Carlo correctly with proper error estimation?
- **Model validation** — can you compare numerical results to analytical benchmarks?
- **Risk measurement** — do you understand VaR, CVaR, and their limitations?
- **Code quality** — can you write clean, tested numerical code in Python?

Every quant trader interviews with at least one options pricing question. Having built this from scratch gives you the intuition to answer those questions deeply.
