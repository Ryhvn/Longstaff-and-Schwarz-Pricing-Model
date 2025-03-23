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
        self.d1, self.d2 = None, None  # Initialisation des valeurs de d1 et d2

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
        if self.d1 is None and self.d2 is None:
            sigma_sqrt_T = self.market.sigma * np.sqrt(self.T)
            self.d1 = (np.log(self.S0 / self.option.K) +
                  (self.market.r - self.q + 0.5 * self.market.sigma ** 2) * self.T) / sigma_sqrt_T
            self.d2 = self.d1 - sigma_sqrt_T

    def price(self):
        """ Calcul du prix de l'option via Black-Scholes. """
        if self.option.exercise.lower() == "european":
            self._compute_d1_d2()
            if isinstance(self.option, Call):
                return self.S0 * np.exp(-self.q * self.T) * stats.norm.cdf(self.d1) - \
                    self.option.K * np.exp(-self.market.r * self.T) * stats.norm.cdf(self.d2)

            elif isinstance(self.option, Put):
                return self.option.K * np.exp(-self.market.r * self.T) * stats.norm.cdf(-self.d2) - \
                    self.S0 * np.exp(-self.q * self.T) * stats.norm.cdf(-self.d1)

            else:
                return "NA"

        else:
            return "NA"

    # ---------------- Greek Calculations ----------------

    def delta(self):
        """ Calcul du Delta (sensibilité au spot). """
        if self.option.exercise.lower() != "european":
            return "NA"
        self._compute_d1_d2()

        if isinstance(self.option, Call):
            return np.exp(-self.q * self.T) * stats.norm.cdf(self.d1)
        elif isinstance(self.option, Put):
            return np.exp(-self.q * self.T) * (stats.norm.cdf(self.d1) - 1)

    def gamma(self):
        """ Calcul du Gamma (convexité par rapport au spot). """
        if self.option.exercise.lower() != "european":
            return "NA"
        self._compute_d1_d2()

        return np.exp(-self.q * self.T) * stats.norm.pdf(self.d1) / (self.S0 * self.market.sigma * np.sqrt(self.T))

    def vega(self):
        """ Calcul du Vega (sensibilité à la volatilité). """
        if self.option.exercise.lower() != "european":
            return "NA"
        self._compute_d1_d2()

        return self.S0 * np.exp(-self.q * self.T) * stats.norm.pdf(self.d1) * np.sqrt(self.T) / 100

    def theta(self):
        """ Calcul du Theta (décroissance temporelle). """
        if self.option.exercise.lower() != "european":
            return "NA"
        self._compute_d1_d2()

        first_term = - (self.S0 * self.market.sigma * np.exp(-self.q * self.T) * stats.norm.pdf(self.d1)) / (
                    2 * np.sqrt(self.T))
        if isinstance(self.option, Call):
            second_term = self.q * self.S0 * np.exp(-self.q * self.T) * stats.norm.cdf(self.d1)
            third_term = - self.market.r * self.option.K * np.exp(-self.market.r * self.T) * stats.norm.cdf(self.d2)
            return (first_term + second_term + third_term)/self.market.days_convention
        elif isinstance(self.option, Put):
            second_term = self.q * self.S0 * np.exp(-self.q * self.T) * stats.norm.cdf(-self.d1)
            third_term = self.market.r * self.option.K * np.exp(-self.market.r * self.T) * stats.norm.cdf(-self.d2)
            return (first_term - second_term + third_term)/self.market.days_convention

    def rho(self):
        """ Calcul du Rho (sensibilité aux taux d'intérêt). """
        if self.option.exercise.lower() != "european":
            return "NA"
        self._compute_d1_d2()

        if isinstance(self.option, Call):
            return self.option.K * self.T * np.exp(-self.market.r * self.T) * stats.norm.cdf(self.d2) / 100
        elif isinstance(self.option, Put):
            return -self.option.K * self.T * np.exp(-self.market.r * self.T) * stats.norm.cdf(-self.d2) / 100

    def all_greeks(self):
        return [self.delta(),self.gamma(),self.vega(),self.theta(),self.rho()]