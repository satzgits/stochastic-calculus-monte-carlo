import numpy as np


class RiskMetrics:
    def __init__(self, paths, confidence_level=0.95):
        self.paths = paths
        self.terminal = paths[:, -1]
        self.s0 = paths[0, 0]
        self.returns = (self.terminal - self.s0) / self.s0
        self.confidence_level = confidence_level

    def value_at_risk(self):
        var_percentile = 1 - self.confidence_level
        return np.percentile(self.returns, var_percentile * 100)

    def conditional_var(self):
        var = self.value_at_risk()
        tail = self.returns[self.returns <= var]
        return np.mean(tail) if len(tail) > 0 else var

    def expected_value(self):
        return np.mean(self.terminal)

    def probability_of_loss(self):
        return np.mean(self.returns < 0)

    def summary(self):
        var = self.value_at_risk()
        cvar = self.conditional_var()
        return {
            "VaR": var,
            "CVaR": cvar,
            "expected_terminal": self.expected_value(),
            "prob_loss": self.probability_of_loss(),
            "terminal_std": np.std(self.terminal, ddof=1)
        }
