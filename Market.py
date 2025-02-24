# ---------------- Market Class ----------------
class Market:
    def __init__(self, S0, r, sigma, dividend):
        self.S0 = S0  # Prix initial du sous-jacent
        self.r = r  # Taux d'intérêt sans risque
        self.sigma = sigma  # Volatilité
        self.dividend = dividend  # Dividende