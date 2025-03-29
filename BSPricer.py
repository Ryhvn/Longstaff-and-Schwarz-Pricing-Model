from Market import Market
from Option import Option, Call, Put
import numpy as np
import scipy.stats as stats
from typing import Union, List

# ---------------- Black-Scholes Pricer ----------------
class BlackScholesPricer:
    def __init__(self, market: Market, option: Option, t_div: float, dt: float, T: float) -> None:
        self.market: Market = market
        self.option: Option = option
        self.T: float = T
        self.S0: float = self._adjust_initial_price(t_div, dt)
        self.q: float = self._compute_dividend_yield()
        self.d1: Union[float, None] = None
        self.d2: Union[float, None] = None

    def european_exercise_check(self) -> bool:
        if self.option.exercise.lower() == "european":
            return True
        if isinstance(self.option, Call) and (not self.market.dividend) and self.market.r > 0:
            return True
        if isinstance(self.option, Put) and (not self.market.dividend) and self.market.r < 0:
            return True
        return False

    def _adjust_initial_price(self, t_div: float, dt: float) -> float: # ajuste le prix initial en fonction du type de dividende
        if self.market.div_type == "discrete" and self.market.div_date is not None:
            return self.market.S0 - self.market.dividend * np.exp(-self.market.r * t_div * dt)
        return self.market.S0

    def _compute_dividend_yield(self) -> float:
        return self.market.dividend if self.market.div_type == "continuous" else 0.0

    def _compute_d1_d2(self) -> None:
        if self.d1 is None and self.d2 is None:
            sigma_sqrt_T = self.market.sigma * np.sqrt(self.T)
            self.d1 = (np.log(self.S0 / self.option.K) +
                       (self.market.r - self.q + 0.5 * self.market.sigma ** 2) * self.T) / sigma_sqrt_T
            self.d2 = self.d1 - sigma_sqrt_T

    def price(self) -> Union[float, str]: #formule de Black Scholes
        if self.european_exercise_check():
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
        
    def delta(self) -> Union[float, str]: #calcul du delta Black Scholes
        if not self.european_exercise_check():
            return "NA"
        self._compute_d1_d2()
        if isinstance(self.option, Call):
            return np.exp(-self.q * self.T) * stats.norm.cdf(self.d1)
        elif isinstance(self.option, Put):
            return np.exp(-self.q * self.T) * (stats.norm.cdf(self.d1) - 1)

    def gamma(self) -> Union[float, str]: #calcul du gamma Black Scholes
        if not self.european_exercise_check():
            return "NA"
        self._compute_d1_d2()
        return np.exp(-self.q * self.T) * stats.norm.pdf(self.d1) / (self.S0 * self.market.sigma * np.sqrt(self.T))

    def vega(self) -> Union[float, str]: #calcul du vega Black Scholes
        if not self.european_exercise_check():
            return "NA"
        self._compute_d1_d2()
        return self.S0 * np.exp(-self.q * self.T) * stats.norm.pdf(self.d1) * np.sqrt(self.T) / 100

    def theta(self) -> Union[float, str]: #calcul du theta Black Scholes
        if not self.european_exercise_check():
            return "NA"
        self._compute_d1_d2()
        first_term = -(
    self.S0 * self.market.sigma * np.exp(-self.q * self.T) * stats.norm.pdf(self.d1)) / (2 * np.sqrt(self.T))
        if isinstance(self.option, Call):
            second_term = self.q * self.S0 * np.exp(-self.q * self.T) * stats.norm.cdf(self.d1)
            third_term = - self.market.r * self.option.K * np.exp(-self.market.r * self.T) * stats.norm.cdf(self.d2)
            return (first_term + second_term + third_term) / self.market.days_convention
        elif isinstance(self.option, Put):
            second_term = self.q * self.S0 * np.exp(-self.q * self.T) * stats.norm.cdf(-self.d1)
            third_term = self.market.r * self.option.K * np.exp(-self.market.r * self.T) * stats.norm.cdf(-self.d2)
            return (first_term - second_term + third_term) / self.market.days_convention

    def rho(self) -> Union[float, str]: #calcul du rho Black Scholes
        if not self.european_exercise_check():
            return "NA"
        self._compute_d1_d2()
        if isinstance(self.option, Call):
            return self.option.K * self.T * np.exp(-self.market.r * self.T) * stats.norm.cdf(self.d2) / 100
        elif isinstance(self.option, Put):
            return -self.option.K * self.T * np.exp(-self.market.r * self.T) * stats.norm.cdf(-self.d2) / 100

    def all_greeks(self) -> List[Union[float, str]]: #retourne une liste de greeks
        return [self.delta(), self.gamma(), self.vega(), self.theta(), self.rho()]
