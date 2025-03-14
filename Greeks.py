from MCPricer import MCModel
import numpy as np

class GreeksCalculator:
    def __init__(self, mc_model: MCModel, epsilon=1e-4):
        """
        Initialise le calculateur de Greeks avec un modèle MC.

        :param mc_model: Instance du modèle de Monte Carlo.
        :param epsilon: Pas de variation pour les dérivées numériques.
        """
        self.mc_model = mc_model
        self.epsilon = epsilon  # Petite variation pour les dérivées

    def _recreate_model(self, **kwargs):
        """Crée un nouveau modèle MC avec des paramètres modifiés."""
        new_params = {
            "market": self.mc_model.market,
            "option": self.mc_model.option,
            "pricing_date": self.mc_model.pricing_date,
            "n_paths": self.mc_model.n_paths,
            "n_steps": self.mc_model.n_steps,
        }
        new_params.update(kwargs)  # Modifier les paramètres fournis

        return MCModel(**new_params)  # Nouvelle instance de modèle MC

    def delta(self):
        """Approximation numérique du Delta : dPrix/dS0"""
        S0 = self.mc_model.market.S0

        # Nouveau modèle avec S0 augmenté
        mc_up = self._recreate_model(market=self.mc_model.market.copy(S0=S0 * (1 + self.epsilon)))
        price_up = mc_up.european_price_vectorized()

        # Nouveau modèle avec S0 diminué
        mc_down = self._recreate_model(market=self.mc_model.market.copy(S0=S0 * (1 - self.epsilon)))
        price_down = mc_down.european_price_vectorized()

        return (price_up - price_down) / (2 * S0 * self.epsilon)

    def gamma(self):
        """Approximation numérique du Gamma : d²Prix/dS0²"""
        S0 = self.mc_model.market.S0

        mc_up = self._recreate_model(market=self.mc_model.market.copy(S0=S0 * (1 + self.epsilon)))
        price_up = mc_up.european_price_vectorized()

        mc_down = self._recreate_model(market=self.mc_model.market.copy(S0=S0 * (1 - self.epsilon)))
        price_down = mc_down.european_price_vectorized()

        mc_normal = self._recreate_model()
        price = mc_normal.european_price_vectorized()

        return (price_up - 2 * price + price_down) / (S0 ** 2 * self.epsilon ** 2)

    def vega(self):
        """Approximation numérique du Vega : dPrix/dSigma"""
        sigma = self.mc_model.market.sigma

        mc_up = self._recreate_model(market=self.mc_model.market.copy(sigma=sigma + self.epsilon))
        price_up = mc_up.european_price_vectorized()

        mc_down = self._recreate_model(market=self.mc_model.market.copy(sigma=sigma - self.epsilon))
        price_down = mc_down.european_price_vectorized()

        return (price_up - price_down) / (2 * self.epsilon)

    def theta(self):
        """Approximation numérique du Theta : dPrix/dt"""
        pricing_date = self.mc_model.pricing_date + np.timedelta64(1, 'D')
        mc_new = self._recreate_model(pricing_date=pricing_date)
        price_new = mc_new.european_price_vectorized()

        price_old = self.mc_model.european_price_vectorized()

        return (price_new - price_old) / (-1)  # Theta est généralement négatif

    def rho(self):
        """Approximation numérique du Rho : dPrix/dr"""
        r = self.mc_model.market.r

        mc_up = self._recreate_model(market=self.mc_model.market.copy(r=r + self.epsilon))
        price_up = mc_up.european_price_vectorized()

        mc_down = self._recreate_model(market=self.mc_model.market.copy(r=r - self.epsilon))
        price_down = mc_down.european_price_vectorized()

        return (price_up - price_down) / (2 * self.epsilon)
