from Market import Market
from Option import Option, Call, Put
import numpy as np
import scipy.stats as stats

# ---------------- Black-Scholes Pricer ----------------
class BlackScholesPricer:
    def __init__(self, market: Market, option: Option, t_div, dt, T):
        self.market = market  # Informations sur le marché (spot, taux, etc.)
        self.option = option  # Option (call ou put)
        self.T = T
        self.S0 = self._adjust_initial_price(t_div, dt)
        self.q = self._compute_dividend_yield()
        self.d1, self.d2 = None, None

    def _adjust_initial_price(self, t_div, dt):
        """ Ajuste le prix initial en fonction des dividendes. """
        if self.market.div_type == "discrete":
            return self.market.S0 - self.market.dividend * np.exp(-self.market.r * t_div * dt)
        return self.market.S0

    def _compute_dividend_yield(self):
        """ Calcule le taux de dividende en fonction du type de dividende. """
        return self.market.dividend if self.market.div_type == "continuous" else 0

    def _compute_d1_d2(self):
        """ Calcule d1 et d2 pour la formule de Black-Scholes. """
        sigma_sqrt_T = self.market.sigma * np.sqrt(self.T)
        d1 = (np.log(self.S0 / self.option.K) + (
                    self.market.r - self.q + 0.5 * self.market.sigma ** 2) * self.T) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        return d1, d2

    def price(self):
        """ Calcul du prix de l'option via Black-Scholes. """
        #if self.option.exercise.lower() == "european":
        self.d1, self.d2 = self._compute_d1_d2()
        if isinstance(self.option, Call):
            return self.S0 * np.exp(-self.q * self.T) * stats.norm.cdf(self.d1) - self.option.K * np.exp(
                -self.market.r * self.T) * stats.norm.cdf(self.d2)

        elif isinstance(self.option, Put):
            return self.option.K * np.exp(-self.market.r * self.T) * stats.norm.cdf(
                -self.d2) - self.S0 * np.exp(-self.q * self.T) * stats.norm.cdf(-self.d1)

        else:
            raise ValueError("Option type not supported.")

        #else:
            #raise NotImplementedError("Non-European options not yet supported in this pricer.")