from PricingModel import Engine
from Regression import Regression
from BrownianMotion import BrownianMotion
from Market import Market
import numpy as np
from scipy.stats import norm
from Option import Option
from datetime import date
from typing import Optional, Tuple, Union
# ---------------- PathGenerator Class ----------------
import numpy as np
from typing import Optional, Union

class PathGenerator:
    def __init__(
        self,market : Market, brownian : BrownianMotion,n_paths: int, n_steps: int, dt: float, t_div: Optional[int],
        compute_antithetic: bool = False
    ) -> None:
        self.market = market
        self.brownian = brownian
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = dt
        self.paths_scalar: Optional[np.ndarray] = None
        self.paths_vectorized: Optional[np.ndarray] = None
        self.t_div = t_div
        self.compute_antithetic = compute_antithetic
    # Génération des chemins de prix
    def get_factors(self, dW: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        growth_adjustment: float = (
            self.market.r
            - (self.market.dividend if self.market.div_type == "continuous" else 0)
            - 0.5 * self.market.sigma ** 2
        ) * self.dt

        if isinstance(dW, float):
            return float(np.exp(growth_adjustment + self.market.sigma * dW))

        elif isinstance(dW, np.ndarray):
            increments = growth_adjustment + self.market.sigma * dW
            np.cumsum(increments, axis=1, out=increments)
            return np.exp(increments)
    # Ajuste le prix pour les dividendes
    def _adjust_price_for_dividend(self, price: float, t: int) -> float:
        if (
            self.market.div_type == "discrete"
            and self.t_div is not None
            and t == self.t_div + 1
        ):
            if 0 < self.t_div <= self.n_steps:
                price -= self.market.dividend
        return price
    # Applique les dividendes sur les chemins
    def _apply_dividends(self, paths: np.ndarray) -> None:
        if self.market.div_type == "discrete" and self.t_div is not None:
            div_reduction_factor = 1 - (self.market.dividend / paths[:, self.t_div])
            paths[:, self.t_div + 1:] *= div_reduction_factor[:, np.newaxis]
    # Génération des chemins en scalaire
    def generate_paths_scalar(self) -> np.ndarray:
        if self.paths_scalar is None:
            paths = np.empty((self.n_paths, self.n_steps + 1))

            for i in range(self.n_paths):
                price: float = self.market.S0
                paths[i, 0] = price

                for t in range(1, self.n_steps + 1):
                    dW: float = self.brownian.scalar_motion()
                    price = self._adjust_price_for_dividend(price, t)
                    price *= self.get_factors(dW)
                    paths[i, t] = price

            self.paths_scalar = paths

        return self.paths_scalar
    
    # Génération des chemins en vectorisé
    def generate_paths_vectorized(self) -> np.ndarray:
        if self.paths_vectorized is None:
            paths = np.empty((self.n_paths, self.n_steps + 1))
            paths[:, 0] = self.market.S0

            if self.compute_antithetic and self.n_paths % 2 == 0:
                dW = self.brownian.vectorized_motion(self.n_paths // 2, self.n_steps)
                dW = np.concatenate((dW, -dW), axis=0)
            else:
                dW = self.brownian.vectorized_motion(self.n_paths, self.n_steps)

            increments = self.get_factors(dW)
            paths[:, 1:] = self.market.S0 * increments
            self._apply_dividends(paths)
            self.paths_vectorized = paths

        return self.paths_vectorized

# ---------------- Classe MCEngine ----------------

class MonteCarloEngine(Engine):
    def __init__(
        self, market : Market,option : Option,
        pricing_date :date, n_paths: int, n_steps: int, seed: Optional[int] = None,
        ex_frontier: str = "Quadratic",
        compute_antithetic: bool = False
    ) -> None:
        super().__init__(market, option, pricing_date, n_steps)
        self.n_paths: int = n_paths
        self.reg_type: str = ex_frontier
        self.eu_payoffs: Optional[np.ndarray] = None
        self.am_payoffs: Optional[np.ndarray] = None
        self.american_price_by_time: Optional[list[float]] = None
        self.PathGenerator = PathGenerator(
            self.market,
            BrownianMotion(self.dt, seed),
            self.n_paths,
            self.n_steps,
            self.dt,
            self.t_div,
            compute_antithetic
        )
    #Calcul le prix d'une option américaine 
    def _price_american_lsm(self, paths: np.ndarray, analysis: bool = False) -> float:
        global price_by_time
        CF = self.option.payoff(paths[:, -1])
        if analysis:
            price_by_time = []
            price_by_time.append(CF.mean() * np.exp(-self.market.r * self.T))
        for t in range(self.n_steps - 2, -1, -1):
            CF *= self.df
            immediate = self.option.payoff(paths[:, t])
            in_money = immediate > 0
            if np.any(in_money):
                cont_val = Regression.fit(self.reg_type, paths[in_money, t], CF[in_money])
                exercise = immediate[in_money] >= cont_val
                CF[in_money] = np.where(exercise, immediate[in_money], CF[in_money])

                if analysis:
                    price_by_time.append(CF.mean() * np.exp(-self.market.r * (t + 1) * self.dt))
        self.am_payoffs = CF
        if analysis:
            self.american_price_by_time = price_by_time # utilisé pour l'analyse
        return float(CF.mean())

    def _discounted_payoffs_by_method(self, type: str) -> np.ndarray: # retourne les payoffs actualisés selon la méthode choisie
        if type == "MC":
            if self.eu_payoffs is None:
                self.eu_payoffs = self._discounted_eu_payoffs(
                    self.PathGenerator.generate_paths_vectorized()
                )
            return self.eu_payoffs
        else:
            if self.am_payoffs is None:
                self._price_american_lsm(self.PathGenerator.generate_paths_vectorized())
            return self.am_payoffs

    def _discounted_eu_payoffs(self, paths: np.ndarray) -> np.ndarray:
        payoffs = self.option.payoff(paths[:, -1])
        return np.exp(-self.market.r * self.T) * payoffs

    def get_variance(self, type: str = "MC") -> float: # retourne la variance des payoffs actualisés selon la méthode choisie
        discounted_payoffs = self._discounted_payoffs_by_method(type)

        if self.PathGenerator.compute_antithetic:
            discounted_payoffs = (
                discounted_payoffs[: self.n_paths // 2]
                + discounted_payoffs[self.n_paths // 2 :]
            ) / 2

        return float(discounted_payoffs.var())
    # retourne la liste des prix américains par pas de temps
    def get_american_price_path(self) -> Optional[list[float]]: 
        self._price_american_lsm(self.PathGenerator.generate_paths_vectorized(), analysis=True)
        return self.american_price_by_time
    # calcule le prix européen
    def _european_price(self, paths: np.ndarray) -> float: 
        payoffs = self._discounted_eu_payoffs(paths)
        return float(np.mean(payoffs))
    # calcule l'écart-type des payoffs actualisés selon la méthode choisie
    def get_std(self, type: str = "MC") -> float: 
        std = np.sqrt(self.get_variance(type) / self.n_paths)
        return float(std)

    #retourne l'intervalle de confiance du prix selon la méthode choisie
    def price_confidence_interval(self, alpha: float = 0.05, type: str = "MC") -> Tuple[float, float]: 
        discounted_payoffs = self._discounted_payoffs_by_method(type)
        mean_price = np.mean(discounted_payoffs)
        std_dev = self.get_std(type)
        z = norm.ppf(1 - alpha / 2)
        CI_half_width = z * std_dev
        CI_lower = mean_price - CI_half_width
        CI_upper = mean_price + CI_half_width
        return (float(CI_upper), float(CI_lower))
    #retourne le prix selon la méthode choisie
    def price(self, type: str = "MC") -> float:
        if type == "Longstaff":
            return self.american_price_vectorized()
        else:
            return self.european_price_vectorized()
        
    #----------------- Pricing Methods -----------------
    def european_price_scalar(self) -> float:
        return self._european_price(self.PathGenerator.generate_paths_scalar())

    def european_price_vectorized(self) -> float:
        return self._european_price(self.PathGenerator.generate_paths_vectorized())

    def american_price_scalar(self) -> float:
        return self._price_american_lsm(self.PathGenerator.generate_paths_scalar())

    def american_price_vectorized(self) -> float:
        return self._price_american_lsm(self.PathGenerator.generate_paths_vectorized())