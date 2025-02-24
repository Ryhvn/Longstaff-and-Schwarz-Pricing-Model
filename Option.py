import numpy as np
import scipy.stats as stats

# ---------------- Option Class ----------------
class Option:
    def __init__(self, K, T, opt_type="put"):
        self.K = K  # Strike
        self.T = T  # Maturité
        self.opt_type = opt_type  # "call" ou "put"

    def payoff(self, S):
        return np.maximum(S - self.K, 0) if self.opt_type == "call" else np.maximum(self.K - S, 0)

    def black_scholes_price(self, market):
        S0, K, T, r, sigma = market.S0, self.K, self.T, market.r, market.sigma
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if self.opt_type == "call":
            price = S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:  # Put
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1)

        return price