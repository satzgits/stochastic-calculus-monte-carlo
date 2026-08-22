import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from mc.gbm import GeometricBrownianMotion
from mc.options import MCOptionPricer
from mc.black_scholes import BlackScholes
from mc.risk import RiskMetrics


def test_gbm_shape():
    gbm = GeometricBrownianMotion()
    paths = gbm.simulate(T=1.0, N=100, M=252, seed=42)
    assert paths.shape == (100, 253), f"Expected (100, 253), got {paths.shape}"
    assert np.allclose(paths[:, 0], 100.0), "All paths should start at S0"
    print("  ✓ GBM shape and initial value")


def test_gbm_antithetic():
    paths = GeometricBrownianMotion.simulate_antithetic(N=100, seed=42)
    assert paths.shape[0] == 100
    print("  ✓ Antithetic variates shape")


def test_gbm_estimate_params():
    mu, sigma = 0.08, 0.25
    gbm = GeometricBrownianMotion(mu=mu, sigma=sigma)
    paths = gbm.simulate(T=1.0, N=50000, M=252, seed=1)
    mu_hat, sigma_hat = GeometricBrownianMotion.estimate_params(paths)
    assert abs(mu_hat - mu) < 0.02, f"mu estimate {mu_hat:.4f} too far from {mu}"
    assert abs(sigma_hat - sigma) < 0.01, f"sigma estimate {sigma_hat:.4f} too far from {sigma}"
    print("  ✓ GBM parameter calibration recovers mu and sigma")


def test_mc_vs_bs():
    s0, k, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2
    gbm = GeometricBrownianMotion(mu=r, sigma=sigma, s0=s0)

    bs = BlackScholes(s0=s0, k=k, T=T, r=r, sigma=sigma)
    bs_price = bs.call_price()

    pricer = MCOptionPricer(s0=s0, k=k, T=T, r=r, sigma=sigma)

    for N in [10000, 50000]:
        paths = gbm.simulate(T=T, N=N, M=252, seed=42)
        mc_price, _, _ = pricer.price_european_call(paths)
        error = abs(mc_price - bs_price) / bs_price
        assert error < 0.05, f"MC error {error:.2%} exceeded 5% for N={N}"
    print("  ✓ MC converges to BS within 5%")


def test_put_call_parity():
    s0, k, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    gbm = GeometricBrownianMotion(mu=r, sigma=sigma, s0=s0)
    paths = gbm.simulate(T=T, N=50000, M=252, seed=42)

    pricer = MCOptionPricer(s0=s0, k=k, T=T, r=r, sigma=sigma)
    call, _, _ = pricer.price_european_call(paths)
    put, _, _ = pricer.price_european_put(paths)

    parity_diff = abs((call - put) - (s0 - k * np.exp(-r * T)))
    assert parity_diff < 0.50, f"Put-call parity violation: {parity_diff:.4f}"
    print("  ✓ Put-call parity holds")


def test_var_is_negative():
    s0, k, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.3
    gbm = GeometricBrownianMotion(mu=r, sigma=sigma, s0=s0)
    paths = gbm.simulate(T=T, N=10000, M=252, seed=42)
    risk = RiskMetrics(paths)
    var = risk.value_at_risk()
    assert var < 0, "VaR should be negative (downside risk)"
    cvar = risk.conditional_var()
    assert cvar <= var, "CVaR should be <= VaR"
    print("  ✓ VaR and CVaR are valid")


def test_bs_greeks():
    bs = BlackScholes(s0=100.0, k=100.0, T=1.0, r=0.05, sigma=0.2)
    assert 0 < bs.delta("call") < 1, "Call delta between 0 and 1"
    assert -1 < bs.delta("put") < 0, "Put delta between -1 and 0"
    assert bs.gamma() > 0, "Gamma should be positive"
    assert bs.vega() > 0, "Vega should be positive"
    print("  ✓ Greeks are in valid ranges")


def test_implied_vol_recovery():
    bs = BlackScholes(s0=100.0, k=105.0, T=1.0, r=0.05, sigma=0.2)
    market_price = bs.call_price()
    implied = bs.implied_vol(market_price, option_type="call")
    assert implied is not None, "Implied vol should be found for a valid price"
    assert abs(implied - 0.2) < 1e-3, f"Expected ~0.2, got {implied}"
    print(f"  ✓ Implied vol recovered: {implied:.5f}")


def test_implied_vol_out_of_range():
    bs = BlackScholes(s0=100.0, k=105.0, T=1.0, r=0.05, sigma=0.2)
    assert bs.implied_vol(1000.0) is None, "Impossible price should return None"
    print("  ✓ Out-of-range price returns None")


def test_max_drawdown_and_win_rate():
    s0, k, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.3
    gbm = GeometricBrownianMotion(mu=r, sigma=sigma, s0=s0)
    paths = gbm.simulate(T=T, N=10000, M=252, seed=42)
    risk = RiskMetrics(paths)
    assert risk.max_drawdown() <= 0, "Max drawdown should be non-positive"
    assert risk.mean_max_drawdown() <= 0, "Mean max drawdown should be non-positive"
    assert risk.mean_max_drawdown() >= risk.max_drawdown(), "Mean DD should be above worst-case DD"
    assert 0 <= risk.win_rate() <= 1, "Win rate should be a probability"
    assert "max_drawdown" in risk.summary()
    print("  \u2713 Max drawdown and win rate are valid")


if __name__ == "__main__":
    test_gbm_shape()
    test_gbm_antithetic()
    test_gbm_estimate_params()
    test_mc_vs_bs()
    test_put_call_parity()
    test_var_is_negative()
    test_bs_greeks()
    test_implied_vol_recovery()
    test_implied_vol_out_of_range()
    test_max_drawdown_and_win_rate()
    print("\nAll Monte Carlo tests passed!")
