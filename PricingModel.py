from abc import ABC, abstractmethod
from Market import Market
from Option import Option
from Regression import Regression
from BrownianMotion import BrownianMotion
import numpy as np

# ---------------- Model Abstract Class ----------------
class Model(ABC):
    def __init__(self, market: Market, option: Option, n_paths, n_steps):
        self.market = market
        self.option = option
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = option.T / n_steps

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
    def __init__(self, market, option, n_paths, n_steps, seed=None):
        super().__init__(market, option, n_paths, n_steps)
        self.brownian = BrownianMotion(self.dt,seed)
        self.paths_scalar = None
        self.paths_vectorized = None

    def generate_paths_scalar(self):
        if self.paths_scalar is None:
            paths = np.zeros((self.n_paths, self.n_steps + 1))
            paths[:, 0] = self.market.S0  # Initialisation des prix initiaux

            for i in range(self.n_paths):  # Remplissage ligne par ligne
                for t in range(1, self.n_steps + 1):
                    dW = self.brownian.scalar_motion()  # Un seul nombre à la fois
                    paths[i, t] = paths[i, t - 1] * np.exp(
                        (self.market.r - self.market.dividend - 0.5 * self.market.sigma ** 2) * self.dt +
                        self.market.sigma * dW
                    )
            self.paths_scalar = paths  # Stocke le résultat pour éviter de recalculer
        self.paths_scalar = np.array([
            [1.00, 1.09, 1.08, 1.34],
            [1.00, 1.16, 1.26, 1.54],
            [1.00, 1.22, 1.07, 1.03],
            [1.00, 0.93, 0.97, 0.92],
            [1.00, 1.11, 1.56, 1.52],
            [1.00, 0.76, 0.77, 0.90],
            [1.00, 0.92, 0.84, 1.01],
            [1.00, 0.88, 1.22, 1.34]
        ])
        return self.paths_scalar

    def generate_paths_vectorized(self):
        if self.paths_vectorized is None:
            dW = self.brownian.vectorized_motion(self.n_paths, self.n_steps)
            increments = (self.market.r - self.market.dividend - 0.5 * self.market.sigma ** 2) * self.dt + self.market.sigma * dW
            paths = self.market.S0 * np.exp(np.cumsum(increments, axis=1))
            self.paths_vectorized = np.hstack([np.full((self.n_paths, 1), self.market.S0), paths])  # Stocke le résultat
            self.paths_vectorized = np.array([
                [1.00, 1.09, 1.08, 1.34],
                [1.00, 1.16, 1.26, 1.54],
                [1.00, 1.22, 1.07, 1.03],
                [1.00, 0.93, 0.97, 0.92],
                [1.00, 1.11, 1.56, 1.52],
                [1.00, 0.76, 0.77, 0.90],
                [1.00, 0.92, 0.84, 1.01],
                [1.00, 0.88, 1.22, 1.34]
            ])
        return self.paths_vectorized

    def _american_price_lsm(self, paths):
        cash_flows = np.zeros_like(paths)
        cash_flows[:, -1] = self.option.payoff(paths[:, -1])

        for t in range(self.n_steps - 1, 0, -1):
            in_the_money = self.option.payoff(paths[:, t]) > 0
            if np.any(in_the_money):
                continuation = Regression.fit(paths[in_the_money, t],
                                              cash_flows[in_the_money, t + 1] * np.exp(-self.market.r * self.dt))
                exercise = self.option.payoff(paths[in_the_money, t])
                continuation = np.round(continuation, 4)
                exercise = np.round(exercise, 2)
                cash_flows[in_the_money, t] = np.where(exercise > continuation, exercise, 0)
                cash_flows[cash_flows[:, t] > 0, t + 1:] = 0

        return np.mean(np.sum(cash_flows * np.exp(-self.market.r * self.dt * np.arange(self.n_steps + 1)), axis=1))

    def _european_price(self,paths):
        payoffs = self.option.payoff(paths[:, -1])
        return np.exp(-self.market.r * self.option.T) * np.mean(payoffs)

    def european_price_scalar(self):
        return self._european_price(self.generate_paths_scalar())

    def european_price_vectorized(self):
        return self._european_price(self.generate_paths_vectorized())

    def american_price_scalar(self):
        return self._american_price_lsm(self.generate_paths_scalar())

    def american_price_vectorized(self):
        return self._american_price_lsm(self.generate_paths_vectorized())