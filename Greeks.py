from MCPricer import MonteCarloEngine
from datetime import timedelta


class GreeksCalculator:
    def __init__(self, mc_model: MonteCarloEngine, epsilon=1e-4, type="MC"):
        """
        Initialise le calculateur de Greeks avec un modèle MC.

        :param mc_model: Instance du modèle de Monte Carlo.
        :param epsilon: Pas de variation pour les dérivées numériques.
        :param type: Méthode de pricing ("Longstaff / MC").
        """
        self.mc_model = mc_model
        self.epsilon = epsilon
        self.type = type
        self._cached_prices = {}  # Stocke les prix calculés pour éviter les répétitions

    def _recreate_model(self, **kwargs):
        """Crée un nouveau modèle MC avec des paramètres modifiés."""
        new_params = {
            "market": self.mc_model.market,
            "option": self.mc_model.option,
            "pricing_date": self.mc_model.pricing_date,
            "n_paths": self.mc_model.n_paths,
            "n_steps": self.mc_model.n_steps,
            "seed": self.mc_model.PathGenerator.brownian.seed
        }
        new_params.update(kwargs)

        return MonteCarloEngine(**new_params)

    def _get_price(self, mc_model):
        """Retourne le prix selon la méthode choisie."""
        if self.type == "Longstaff":
            return mc_model.american_price_vectorized()
        return mc_model.european_price_vectorized()

    def _get_cached_price(self, key, mc_model):
        """Récupère le prix depuis le cache ou le calcule si nécessaire."""
        if key not in self._cached_prices:
            self._cached_prices[key] = self._get_price(mc_model)
        return self._cached_prices[key]

    def delta(self):
        """Calcul optimisé du Delta : dPrix/dS0"""
        S0 = self.mc_model.market.S0

        mc_up = self._recreate_model(market=self.mc_model.market.copy(S0=S0 * (1 + self.epsilon)))
        mc_down = self._recreate_model(market=self.mc_model.market.copy(S0=S0 * (1 - self.epsilon)))

        price_up = self._get_cached_price("S0_up", mc_up)
        price_down = self._get_cached_price("S0_down", mc_down)

        return (price_up - price_down) / (2 * S0 * self.epsilon)

    def gamma(self):
        """Calcul optimisé du Gamma : d²Prix/dS0²"""
        S0 = self.mc_model.market.S0

        mc_up = self._recreate_model(market=self.mc_model.market.copy(S0=S0 * (1 + self.epsilon)))
        mc_down = self._recreate_model(market=self.mc_model.market.copy(S0=S0 * (1 - self.epsilon)))

        price = self._get_cached_price("S0", self.mc_model)
        price_up = self._get_cached_price("S0_up", mc_up)
        price_down = self._get_cached_price("S0_down", mc_down)

        return (price_up - 2 * price + price_down) / (S0 ** 2 * self.epsilon ** 2)

    def vega(self):
        """Calcul du Vega : dPrix/dSigma"""
        sigma = self.mc_model.market.sigma

        mc_up = self._recreate_model(market=self.mc_model.market.copy(sigma=sigma + self.epsilon))
        mc_down = self._recreate_model(market=self.mc_model.market.copy(sigma=sigma - self.epsilon))

        price_up = self._get_cached_price("sigma_up", mc_up)
        price_down = self._get_cached_price("sigma_down", mc_down)

        return (price_up - price_down) / (2 * self.epsilon) / 100

    def theta(self):
        """Calcul du Theta : dPrix/dt"""
        pricing_date = self.mc_model.pricing_date + timedelta(days=1)

        mc_new = self._recreate_model(pricing_date=pricing_date)
        price_new = self._get_cached_price("t_future", mc_new)
        price_old = self._get_cached_price("t_now", self.mc_model)

        return (price_new - price_old) / (-1)

    def rho(self):
        """Calcul du Rho : dPrix/dr"""
        r = self.mc_model.market.r

        mc_up = self._recreate_model(market=self.mc_model.market.copy(r=r + self.epsilon))
        mc_down = self._recreate_model(market=self.mc_model.market.copy(r=r - self.epsilon))

        price_up = self._get_cached_price("r_up", mc_up)
        price_down = self._get_cached_price("r_down", mc_down)

        return (price_up - price_down) / (2 * self.epsilon)

    def all_greeks(self):
        return [self.delta(),self.gamma(),self.vega(),self.theta(),self.rho()]
