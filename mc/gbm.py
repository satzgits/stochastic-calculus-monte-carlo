import numpy as np


class GeometricBrownianMotion:
    def __init__(self, mu=0.05, sigma=0.2, s0=100.0):
        self.mu = mu
        self.sigma = sigma
        self.s0 = s0

    def simulate(self, T=1.0, N=10000, M=252, seed=None):
        if seed is not None:
            np.random.seed(seed)

        dt = T / M
        paths = np.zeros((N, M + 1))
        paths[:, 0] = self.s0

        Z = np.random.standard_normal((N, M))
        drift = (self.mu - 0.5 * self.sigma ** 2) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z

        for t in range(1, M + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion[:, t - 1])

        return paths

    @staticmethod
    def estimate_params(paths, T=1.0, M=252):
        """Estimate (mu, sigma) of a GBM from simulated paths.

        For a geometric Brownian motion, the log returns are
            log(S_t/M_(t-1)) = (mu - sigma^2/2) * dt + sigma * sqrt(dt) * Z
        so we recover sigma from the standard deviation of log returns and mu by
        adding back the (sigma^2/2) term. This is the standard calibration used
        when fitting real market data to a GBM.
        """
        dt = T / M
        log_returns = np.diff(np.log(paths), axis=1)          # (N, M) log returns
        sigma_hat = float(np.std(log_returns, ddof=1) / np.sqrt(dt))
        mean_lr = float(np.mean(log_returns))
        mu_hat = mean_lr / dt + 0.5 * sigma_hat ** 2
        return mu_hat, sigma_hat

    @staticmethod
    def simulate_antithetic(mu=0.05, sigma=0.2, s0=100.0, T=1.0, N=10000, M=252, seed=None):
        if seed is not None:
            np.random.seed(seed)

        dt = T / M
        N_half = N // 2
        paths = np.zeros((N, M + 1))
        paths[:, 0] = s0

        Z = np.random.standard_normal((N_half, M))
        Z_all = np.vstack([Z, -Z])

        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * Z_all

        for t in range(1, M + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion[:, t - 1])

        return paths
