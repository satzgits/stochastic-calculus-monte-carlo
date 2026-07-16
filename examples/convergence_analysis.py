import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from mc.gbm import GeometricBrownianMotion
from mc.options import MCOptionPricer
from mc.black_scholes import BlackScholes

if __name__ == "__main__":
    s0, k, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2
    bs = BlackScholes(s0=s0, k=k, T=T, r=r, sigma=sigma)
    bs_price = bs.call_price()

    gbm = GeometricBrownianMotion(mu=r, sigma=sigma, s0=s0)
    pricer = MCOptionPricer(s0=s0, k=k, T=T, r=r, sigma=sigma)

    N_values = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
    prices = []
    errors = []
    cis = []

    print("Convergence Study: MC Price vs Black-Scholes")
    print("-" * 60)
    print(f"  BS Price: ${bs_price:.4f}")
    print("-" * 60)

    for N in N_values:
        paths = gbm.simulate(T=T, N=N, M=252, seed=42)
        price, se, ci = pricer.price_european_call(paths)
        error = abs(price - bs_price) / bs_price * 100
        prices.append(price)
        errors.append(error)
        cis.append(ci)
        print(f"  N={N:<7d}  Price=${price:.4f}  Error={error:.2f}%  CI=[${ci[0]:.4f}, ${ci[1]:.4f}]")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(N_values, prices, "bo-", label="MC Price")
    ax1.axhline(y=bs_price, color="r", linestyle="--", label=f"BS Price = ${bs_price:.4f}")
    ax1.fill_between(N_values, [c[0] for c in cis], [c[1] for c in cis], alpha=0.2, color="blue")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Paths (N)")
    ax1.set_ylabel("Option Price ($)")
    ax1.set_title("Monte Carlo Convergence to Black-Scholes")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(N_values, errors, "ro-")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Number of Paths (N)")
    ax2.set_ylabel("Error (%)")
    ax2.set_title("Convergence Rate (Error ~ 1/√N)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("convergence_analysis.png", dpi=150)
    print("\nConvergence plot saved to convergence_analysis.png")
