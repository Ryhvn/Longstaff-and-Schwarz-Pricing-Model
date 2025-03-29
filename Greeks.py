from MCPricer import MonteCarloEngine
from TreePricer import TreeModel
from datetime import timedelta, date
from typing import Optional, Any, Union
import copy
import numpy as np

class GreeksCalculator:
    def __init__(
        self,
        model: Union[MonteCarloEngine, TreeModel],
        epsilon: float = 1e-2,
        type: str = "MC"
    ) -> None:
        """
        Initialise le calculateur de Greeks avec un modèle MC ou Tree.

        :param model: Instance du modèle.
        :param epsilon: Pas de variation pour les dérivées numériques.
        :param type: Méthode de pricing ("Longstaff" ou "MC").
        """
        self.epsilon: float = epsilon
        self._type: str = type
        self._original_model: Union[MonteCarloEngine, TreeModel] = copy.copy(model)
        self._cached_prices: dict[str, float] = {}
        self._alpha: Optional[float] = getattr(self._original_model, "alpha", None)

    def _recreate_model(self, **kwargs: Any) -> Union[MonteCarloEngine, TreeModel]:
        new_params: dict[str, Any] = {
            "market": copy.deepcopy(self._original_model.market),
            "option": copy.deepcopy(self._original_model.option),
            "pricing_date": self._original_model.pricing_date,
            "n_steps": self._original_model.n_steps,
        }

        if isinstance(self._original_model, MonteCarloEngine):
            new_params["n_paths"] = self._original_model.n_paths
            new_params["seed"] = self._original_model.PathGenerator.brownian.seed

        new_params.update(kwargs)
        return type(self._original_model)(**new_params)
    #méthode générique
    def _get_price(
        self,
        model: Union[MonteCarloEngine, TreeModel],
        key: str,
        up: bool = False,
        down: bool = False
    ) -> float:
        if key not in self._cached_prices:
            if isinstance(model, TreeModel):
                self._cached_prices[key] = model.price(up=up, down=down)
            else:
                self._cached_prices[key] = model.price(self._type)
        return self._cached_prices[key]
    #calcul du delta
    def delta(self) -> float:
        S0: float = self._original_model.market.S0
        key_up, key_down = "S0_up", "S0_down"

        if isinstance(self._original_model, TreeModel):
            price_up = self._get_price(self._original_model, key_up, up=True)
            price_down = self._get_price(self._original_model, key_down, down=True)
            return (price_up - price_down) / (S0 * self._alpha - S0 / self._alpha)

        mc_up = self._recreate_model(market=self._original_model.market.copy(S0=S0 * (1 + self.epsilon)))
        price_up = self._get_price(mc_up, key_up, up=True)

        mc_down = self._recreate_model(market=self._original_model.market.copy(S0=S0 * (1 - self.epsilon)))
        price_down = self._get_price(mc_down, key_down, down=True)

        return (price_up - price_down) / (2 * S0 * self.epsilon)
    #calcul du gamma
    def gamma(self) -> float:
        S0: float = self._original_model.market.S0
        key_up, key_down, key_mid = "S0_up", "S0_down", "S0"

        if isinstance(self._original_model, TreeModel):
            model_up, model_down = self._original_model, self._original_model
        else:
            model_up = self._recreate_model(market=self._original_model.market.copy(S0=S0 * (1 + self.epsilon)))
            model_down = self._recreate_model(market=self._original_model.market.copy(S0=S0 * (1 - self.epsilon)))

        price = self._get_price(self._original_model, key_mid)
        price_up = self._get_price(model_up, key_up, up=True)
        price_down = self._get_price(model_down, key_down, down=True)

        if isinstance(self._original_model, TreeModel):
            d_up = (price_up - price) / (self._alpha * S0 - S0)
            d_down = (price - price_down) / (S0 - S0 / self._alpha)
            return (d_up - d_down) / ((self._alpha * S0 - S0 / self._alpha) / 2)

        return (price_up - 2 * price + price_down) / (S0 ** 2 * self.epsilon ** 2)
    #calcul du vega
    def vega(self) -> float:
        sigma: float = self._original_model.market.sigma

        model_up = self._recreate_model(market=self._original_model.market.copy(sigma=sigma + self.epsilon))
        model_down = self._recreate_model(market=self._original_model.market.copy(sigma=sigma - self.epsilon))

        price_up = self._get_price(model_up, "sigma_up")
        price_down = self._get_price(model_down, "sigma_down")

        return (price_up - price_down) / (2 * self.epsilon) / 100
    #calcul du theta
    def theta(self) -> float:
        pricing_date: date = self._original_model.pricing_date + timedelta(days=1)

        model_new = self._recreate_model(pricing_date=pricing_date)
        price_new = self._get_price(model_new, "t_future")
        price_old = self._get_price(self._original_model, "t_now")

        return price_new - price_old
    #calcul du rho
    def rho(self) -> float:
        r: float = self._original_model.market.r

        model_up = self._recreate_model(market=self._original_model.market.copy(r=r + self.epsilon))
        model_down = self._recreate_model(market=self._original_model.market.copy(r=r - self.epsilon))

        price_up = self._get_price(model_up, "r_up")
        price_down = self._get_price(model_down, "r_down")

        return (price_up - price_down) / (2 * self.epsilon) / 100
    #calcul de tous les greeks
    def all_greeks(self) -> list[float]:
        return [self.delta(), self.gamma(), self.vega(), self.theta(), self.rho()]
