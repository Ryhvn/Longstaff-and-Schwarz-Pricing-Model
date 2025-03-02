from abc import ABC, abstractmethod
from Market import Market
from Option import Option
from Regression import Regression
from BrownianMotion import BrownianMotion
import numpy as np
import scipy.stats as stats

# ---------------- Model Abstract Class ----------------
class Model(ABC):
    def __init__(self, market: Market, option: Option, pricingDate, n_paths, n_steps):
        self.market = market
        self.option = option
        self.T = (option.T - pricingDate).days / 365.0
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = self.T / n_steps

    def black_scholes_price(self):
        """ Calcule le prix de l'option avec le modèle de Black-Scholes """
        S0, K, T, r, sigma = self.market.S0, self.option.K, self.T, self.market.r, self.market.sigma
        if T <= 0:
            return max(0, S0 - K) if self.option.opt_type == "call" else max(0, K - S0)

        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if self.option.opt_type.lower() == "call":
            price = S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:  # Put
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1)

        return price

    @abstractmethod
    def european_price_scalar(self):
        pass

    @abstractmethod
    def european_price_vectorized(self):
        pass

    @abstractmethod
    def american_price_scalar(self):
        pass

    @abstractmethod
    def american_price_vectorized(self):
        pass

# ---------------- Classe ModelMC ----------------
class ModelMC(Model):
    def __init__(self, market, option, pricingDate, n_paths, n_steps, seed=None):
        super().__init__(market, option, pricingDate, n_paths, n_steps)
        self.brownian = BrownianMotion(self.dt,seed)
        self.paths_scalar = None
        self.paths_vectorized = None

        #self.paths_scalar = np.array([
            #[1.00, 1.09, 1.08, 1.34],
            #[1.00, 1.16, 1.26, 1.54],
            #[1.00, 1.22, 1.07, 1.03],
            #[1.00, 0.93, 0.97, 0.92],
            #[1.00, 1.11, 1.56, 1.52],
            #[1.00, 0.76, 0.77, 0.90],
            #[1.00, 0.92, 0.84, 1.01],
            #[1.00, 0.88, 1.22, 1.34]
        #])

    def generate_paths_scalar(self):
        if self.paths_scalar is None:
            paths = np.empty((self.n_paths, self.n_steps + 1))  # Allocation mémoire optimisée
            paths[:, 0] = self.market.S0  # Initialisation des prix initiaux
            drift = (self.market.r - self.market.dividend - 0.5 * self.market.sigma ** 2) * self.dt

            for i in range(self.n_paths):
                price = self.market.S0  # Stocker temporairement pour éviter les accès fréquents à paths
                for t in range(1, self.n_steps + 1):
                    dW = self.brownian.scalar_motion()  # Générer à l'avance et normaliser
                    price *= np.exp(drift + self.market.sigma * dW)
                    paths[i, t] = price  # Mise à jour du chemin en une seule écriture

            self.paths_scalar = paths  # Stockage des chemins générés
        return self.paths_scalar

    def generate_paths_vectorized(self):
        if self.paths_vectorized is None:
            dW = self.brownian.vectorized_motion(self.n_paths, self.n_steps)  # Génération vectorisée
            increments = (self.market.r - self.market.dividend - 0.5 * self.market.sigma ** 2) * self.dt + self.market.sigma * dW

            paths = np.empty((self.n_paths, self.n_steps + 1))  # Allocation directe
            paths[:, 0] = self.market.S0  # Initialisation

            np.cumsum(increments, axis=1, out=increments)  # Évite une copie mémoire
            paths[:, 1:] = self.market.S0 * np.exp(increments)  # Application de l'exponentielle

            self.paths_vectorized = paths  # Stocke le résultat
        return self.paths_vectorized

    def _american_price_lsm(self,paths):

        # Initialisation avec seulement les cash flows finaux
        current_values = self.option.payoff(paths[:, -1])
        exercise_times = np.full(self.n_paths, self.n_steps)

        # On itère en arrière (sauf pour le temps final déjà traité)
        for t in range(self.n_steps - 1, 0, -1):
            # Identifier les chemins in-the-money à l'instant t
            payoff_t = self.option.payoff(paths[:, t])
            in_the_money = payoff_t > 0

            if np.any(in_the_money):
                # Récupérer les valeurs actualisées des cash flows futurs pour les chemins ITM
                future_values = current_values.copy()
                time_to_cf = exercise_times - t
                discount_factors = np.exp(-self.market.r * self.dt * time_to_cf)
                continuation_values = future_values * discount_factors

                # Régression uniquement sur les chemins in-the-money
                itm_indices = np.where(in_the_money)[0]
                continuation = Regression.fit(paths[itm_indices, t], continuation_values[itm_indices])

                # Décision d'exercice
                exercise_decision = payoff_t[itm_indices] > continuation

                # Mise à jour des valeurs et des temps d'exercice
                for i, idx in enumerate(itm_indices):
                    if exercise_decision[i]:
                        current_values[idx] = payoff_t[idx]
                        exercise_times[idx] = t

        # Actualisation finale jusqu'à t=0
        discount_to_zero = np.exp(-self.market.r * self.dt * exercise_times)
        return np.mean(current_values * discount_to_zero)

    def _european_price(self,paths):
        payoffs = self.option.payoff(paths[:, -1])
        return np.exp(-self.market.r * self.T) * np.mean(payoffs)

    def european_price_scalar(self):
        return self._european_price(self.generate_paths_scalar())

    def european_price_vectorized(self):
        self.generate_paths_vectorized()
        return self._european_price(self.generate_paths_vectorized())

    def american_price_scalar(self):
        return self._american_price_lsm(self.generate_paths_scalar())

    def american_price_vectorized(self):
        return self._american_price_lsm(self.generate_paths_vectorized())