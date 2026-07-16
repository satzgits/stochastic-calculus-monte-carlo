import numpy as np


class MCOptionPricer:
    def __init__(self, s0=100.0, k=105.0, T=1.0, r=0.05, sigma=0.2):
        self.s0 = s0
        self.k = k
        self.T = T
        self.r = r
        self.sigma = sigma

    def price_european_call(self, paths):
        terminal = paths[:, -1]
        payoffs = np.maximum(terminal - self.k, 0)
        discounted = payoffs * np.exp(-self.r * self.T)
        price = np.mean(discounted)
        se = np.std(discounted, ddof=1) / np.sqrt(len(discounted))
        ci = (price - 1.96 * se, price + 1.96 * se)
        return price, se, ci

    def price_european_put(self, paths):
        terminal = paths[:, -1]
        payoffs = np.maximum(self.k - terminal, 0)
        discounted = payoffs * np.exp(-self.r * self.T)
        price = np.mean(discounted)
        se = np.std(discounted, ddof=1) / np.sqrt(len(discounted))
        ci = (price - 1.96 * se, price + 1.96 * se)
        return price, se, ci

    def convergence_study(self, path_generator, N_values=None):
        if N_values is None:
            N_values = [100, 500, 1000, 5000, 10000, 50000, 100000]

        results = []
        for N in N_values:
            paths = path_generator(N=N)
            price, se, ci = self.price_european_call(paths)
            results.append({
                "N": N,
                "price": price,
                "se": se,
                "ci_lower": ci[0],
                "ci_upper": ci[1]
            })

        return results
