from abc import ABC
from Market import Market
from Option import Option
from BSPricer import BlackScholesPricer
import numpy as np

# ---------------- Model Abstract Class ----------------
class Engine(ABC):
    def __init__(self, market: Market, option: Option, pricing_date, n_paths, n_steps):
        self.market = market
        self._option = option
        self.pricing_date = pricing_date
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.T = self._calculate_T()
        self.dt = self.T / n_steps
        self.df = np.exp(-self.market.r * self.dt)
        # Conversion de div_date en indice temporel si dividende discret
        self._calculate_t_div()
        self.bsm = BlackScholesPricer(self.market,self.option,self.t_div, self.dt, self.T)

    def _calculate_T(self):
        """
        Méthode pour calculer le temps jusqu'à l'expiration de l'option (T).
        Retourne la durée en années entre la date de pricing et la date de maturité.
        """
        return (self.option.T - self.pricing_date).days / self.market.days_convention

    def _calculate_t_div(self):
        """
        Calcule l'indice temporel pour le dividende si le dividende est discret.
        """
        if self.market.div_type == "discrete" and self.market.div_date is not None:
            T_div = (self.market.div_date - self.pricing_date).days / self.market.days_convention  # Conversion en année
            self.t_div = int(T_div / self.dt)  # Conversion en indice de pas de temps

    @property
    def option(self):
        return self._option

    @option.setter
    def option(self, new_option):
        T = self.option.T
        self._option = new_option
        self.bsm.option = new_option
        if new_option.T != T: # évites le recalcul des périodes si inchangé
            self.T = self._calculate_T()
            self.dt = self.T / self.n_steps
            self._calculate_t_div()
            self.bsm = BlackScholesPricer(self.market, self.option, self.t_div, self.dt, self.T)