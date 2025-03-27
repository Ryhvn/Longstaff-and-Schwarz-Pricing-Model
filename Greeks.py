from MCPricer import MonteCarloEngine
from TreePricer import TreeModel
from datetime import timedelta
import copy

class GreeksCalculator:
    def __init__(self, model, epsilon=1e-2, type="MC"):
        """
        Initialise le calculateur de Greeks avec un modèle MC.

        :param model: Instance du modèle.
        :param epsilon: Pas de variation pour les dérivées numériques.
        :param type: Méthode de pricing ("Longstaff / MC").
        """
        self.epsilon = epsilon
        self._type = type
        self._original_model = copy.copy(model)  # Sauvegarde de l'état initial
        self._cached_prices = {}  # Stocke les prix calculés pour éviter les répétitions
        self._alpha = self._original_model.alpha if hasattr(self._original_model, "alpha") else None

    def _recreate_model(self, **kwargs):
        new_params = {
            "market": copy.deepcopy(self._original_model.market),
            "option": copy.deepcopy(self._original_model.option),
            "pricing_date": self._original_model.pricing_date,
            "n_steps": self._original_model.n_steps,
        }

        if issubclass(type(self._original_model), MonteCarloEngine):
            new_params["n_paths"] = self._original_model.n_paths
            new_params["seed"] = self._original_model.PathGenerator.brownian.seed

        new_params.update(kwargs)
        return type(self._original_model)(**new_params)

    def _get_price(self, model, key, up=False, down=False):
        if key not in self._cached_prices:
            if isinstance(model, TreeModel):
                self._cached_prices[key] = model.price(up=up, down=down)
            else:
                self._cached_prices[key] = model.price(self._type)
        return self._cached_prices[key]

    def delta(self):
        """Calcul optimisé du Delta : dPrix/dS0 selon le modèle"""
        S0 = self._original_model.market.S0
        key_up, key_down = "S0_up", "S0_down"

        if isinstance(self._original_model, TreeModel):
            price_up  = self._get_price(self._original_model,key_up,up=True)
            price_down = self._get_price(self._original_model,key_down,down=True)
            return (price_up - price_down) / (S0 * self._alpha - S0 / self._alpha)

        mc_up = self._recreate_model(market=self._original_model.market.copy(S0=S0 * (1 + self.epsilon)))
        price_up = self._get_price(mc_up, key_up, up=True)

        mc_down = self._recreate_model(market=self._original_model.market.copy(S0=S0 * (1 - self.epsilon)))
        price_down = self._get_price(mc_down, key_down,down=True)

        return (price_up - price_down) / (2 * S0 * self.epsilon)


    def gamma(self):
        """Calcul optimisé du Gamma : d²Prix/dS0²"""
        S0 = self._original_model.market.S0
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

    def vega(self):
        """Calcul du Vega : dPrix/dSigma"""
        sigma = self._original_model.market.sigma

        model_up = self._recreate_model(market=self._original_model.market.copy(sigma=sigma + self.epsilon))
        model_down = self._recreate_model(market=self._original_model.market.copy(sigma=sigma - self.epsilon))

        price_up = self._get_price(model_up,"sigma_up", up=True)
        price_down = self._get_price(model_down,"sigma_down", down=True)

        if isinstance(self._original_model, TreeModel):
            return (price_up - price_down) / 2 / 100

        return (price_up - price_down) / (2 * self.epsilon) / 100

    def theta(self):
        """Calcul du Theta : dPrix/dt"""
        pricing_date = self._original_model.pricing_date + timedelta(days=1)

        model_new = self._recreate_model(pricing_date=pricing_date)
        price_new = self._get_price(model_new, "t_future")
        price_old = self._get_price(self._original_model,"t_now")

        return price_new - price_old

    def rho(self):
        """Calcul du Rho : dPrix/dr"""
        r = self._original_model.market.r

        model_up = self._recreate_model(market=self._original_model.market.copy(r=r + self.epsilon))
        model_down = self._recreate_model(market=self._original_model.market.copy(r=r - self.epsilon))

        price_up = self._get_price(model_up,"r_up")
        price_down = self._get_price(model_down,"r_down")

        return (price_up - price_down) / (2 * self.epsilon) / 100

    def all_greeks(self):
        return [self.delta(),self.gamma(),self.vega(),self.theta(),self.rho()]
