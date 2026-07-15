# Stochastic Calculus & Monte Carlo

Options pricing and risk simulation using Monte Carlo methods.

## Features

- Geometric Brownian Motion (GBM) path simulation
- European call/put pricing via Monte Carlo
- Comparison to Black-Scholes analytical price
- Value at Risk (VaR) and Conditional VaR (CVaR) computation
- Visualization of simulated price paths and convergence

## Motivation

Options pricing is the bedrock of quantitative finance. Monte Carlo is the standard tool for pricing path-dependent derivatives and measuring portfolio risk.

## Getting Started

```bash
pip install -r requirements.txt
python examples/european_option.py
```

## Project Structure

```
├── mc/
│   ├── gbm.py           # GBM path generator
│   ├── options.py       # European option pricer
│   ├── black_scholes.py # Analytical BS price
│   └── risk.py          # VaR / CVaR
├── examples/
│   ├── european_option.py
│   └── convergence_analysis.py
├── tests/
├── requirements.txt
└── README.md
```

## Example Output

```
Monte Carlo Price:   $12.45
Black-Scholes Price: $12.52
Error:                0.56%
VaR (95%):           -$2.34
CVaR (95%):          -$3.87
```
