import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from mc.gbm import GeometricBrownianMotion
from mc.options import MCOptionPricer
from mc.black_scholes import BlackScholes
from mc.risk import RiskMetrics

if __name__ == "__main__":
    s0, k, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2

    gbm = GeometricBrownianMotion(mu=r, sigma=sigma, s0=s0)
    paths = gbm.simulate(T=T, N=100000, M=252, seed=42)

    pricer = MCOptionPricer(s0=s0, k=k, T=T, r=r, sigma=sigma)
    mc_price, mc_se, mc_ci = pricer.price_european_call(paths)

    bs = BlackScholes(s0=s0, k=k, T=T, r=r, sigma=sigma)
    bs_price = bs.call_price()

    risk = RiskMetrics(paths)
    risk_summary = risk.summary()

    print("=" * 50)
    print("   EUROPEAN CALL OPTION PRICING")
    print("=" * 50)
    print(f"  Spot:          ${s0:>8.2f}")
    print(f"  Strike:        ${k:>8.2f}")
    print(f"  Maturity:      {T:>8.1f} years")
    print(f"  Volatility:    {sigma:>8.1%}")
    print(f"  Risk-free:     {r:>8.1%}")
    print(f"  Paths:         {100000:>8,d}")
    print("-" * 50)
    print(f"  MC Price:      ${mc_price:>8.2f}  ± ${1.96*mc_se:.2f} (95% CI)")
    print(f"  BS Price:      ${bs_price:>8.2f}")
    print(f"  Error:         {abs(mc_price - bs_price) / bs_price:>8.2%}")
    print("-" * 50)
    print(f"  VaR (95%):     ${risk_summary['VaR']:>8.2%}")
    print(f"  CVaR (95%):    ${risk_summary['CVaR']:>8.2%}")
    print(f"  Prob Loss:     {risk_summary['prob_loss']:>8.2%}")
    print("=" * 50)
