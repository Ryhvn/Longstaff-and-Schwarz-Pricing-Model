import numpy as np

# ---------------- Option Class ----------------
class Option:
    def __init__(self, K, maturity, opt_type="put"):
        self.K = K  # Strike
        self.T = maturity  # Maturité
        self.opt_type = opt_type.lower()  # "call" ou "put"

    def payoff(self, S):
        return np.maximum(S - self.K, 0) if self.opt_type.lower() == "call" else np.maximum(self.K - S, 0)