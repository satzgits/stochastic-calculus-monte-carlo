import numpy as np
from scipy import stats as scipy_stats


class BlackScholes:
    def __init__(self, s0=100.0, k=105.0, T=1.0, r=0.05, sigma=0.2):
        self.s0 = s0
        self.k = k
        self.T = T
        self.r = r
        self.sigma = sigma

    def d1(self):
        return (np.log(self.s0 / self.k) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        d1 = self.d1()
        d2 = self.d2()
        return self.s0 * scipy_stats.norm.cdf(d1) - self.k * np.exp(-self.r * self.T) * scipy_stats.norm.cdf(d2)

    def put_price(self):
        d1 = self.d1()
        d2 = self.d2()
        return self.k * np.exp(-self.r * self.T) * scipy_stats.norm.cdf(-d2) - self.s0 * scipy_stats.norm.cdf(-d1)

    def delta(self, option_type="call"):
        if option_type == "call":
            return scipy_stats.norm.cdf(self.d1())
        return -scipy_stats.norm.cdf(-self.d1())

    def gamma(self):
        return scipy_stats.norm.pdf(self.d1()) / (self.s0 * self.sigma * np.sqrt(self.T))

    def vega(self):
        return self.s0 * scipy_stats.norm.pdf(self.d1()) * np.sqrt(self.T)

    def theta(self, option_type="call"):
        d1 = self.d1()
        d2 = self.d2()
        term1 = -self.s0 * scipy_stats.norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))
        if option_type == "call":
            term2 = -self.r * self.k * np.exp(-self.r * self.T) * scipy_stats.norm.cdf(d2)
        else:
            term2 = self.r * self.k * np.exp(-self.r * self.T) * scipy_stats.norm.cdf(-d2)
        return term1 + term2

    def rho(self, option_type="call"):
        d2 = self.d2()
        if option_type == "call":
            return self.k * self.T * np.exp(-self.r * self.T) * scipy_stats.norm.cdf(d2)
        return -self.k * self.T * np.exp(-self.r * self.T) * scipy_stats.norm.cdf(-d2)
