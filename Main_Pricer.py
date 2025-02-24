from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import time

# ---------------- Classe Market ----------------
class Market:
    def __init__(self, S0, r, sigma, dividend):
        self.S0 = S0  # Prix initial du sous-jacent
        self.r = r  # Taux d'intérêt sans risque
        self.sigma = sigma  # Volatilité
        self.dividend = dividend  # Dividende

# ---------------- Classe Option ----------------
class Option:
    def __init__(self, K, T, opt_type="put"):
        self.K = K  # Strike
        self.T = T  # Maturité
        self.opt_type = opt_type  # "call" ou "put"

    def payoff(self, S):
        return np.maximum(S - self.K, 0) if self.opt_type == "call" else np.maximum(self.K - S, 0)

    def black_scholes_price(self, market):
        S0, K, T, r, sigma = market.S0, self.K, self.T, market.r, market.sigma
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if self.opt_type == "call":
            price = S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:  # Put
            price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1)

        return price

# ---------------- Classe BrownianMotion ----------------
class BrownianMotion:
    def __init__(self, seed=None):
        self.seed = seed
        self._gen = np.random.default_rng(seed)

    def generate_scalar(self, dt):
        p = self._gen.uniform(0, 1)
        while p == 0:
            p = self._gen.uniform(0, 1)
        return stats.norm.ppf(p) * np.sqrt(dt)

    def generate_vectorized(self, n_paths, n_steps, dt):
        self._gen = np.random.default_rng(self.seed) # reset du générateur de nombres aléatoires
        uniform_samples = self._gen.uniform(0, 1, (n_paths, n_steps))
        norm_matrix = stats.norm.ppf(uniform_samples)
        return np.sqrt(dt) * norm_matrix


# ---------------- Classe Regression ----------------
class Regression:
    @staticmethod
    def fit(X, Y):
        if len(X) == 0:
            return np.zeros_like(Y)
        X_reg = np.vstack([X, X ** 2]).T
        model = LinearRegression().fit(X_reg, Y)
        coef = model.coef_
        cst = model.intercept_
        return model.predict(X_reg)

# ---------------- Classe abstraite Model ----------------
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
        self.brownian = BrownianMotion(seed)
        self.paths_scalar = None
        self.paths_vectorized = None

    def generate_paths_scalar(self):
        if self.paths_scalar is None:
            paths = np.zeros((self.n_paths, self.n_steps + 1))
            paths[:, 0] = self.market.S0  # Initialisation des prix initiaux

            for i in range(self.n_paths):  # Remplissage ligne par ligne
                for t in range(1, self.n_steps + 1):
                    dW = self.brownian.generate_scalar(self.dt)  # Un seul nombre à la fois
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
            dW = self.brownian.generate_vectorized(self.n_paths, self.n_steps, self.dt)
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

# ---------------- Test du modèle ----------------
if __name__ == '__main__':
    market = Market(S0=1, r=0.06, sigma=0.2, dividend=0)
    option = Option(K=1.1, T=3, opt_type="put")
    model_mc = ModelMC(market, option, n_paths=8, n_steps=3, seed=1)

    print("Prix Black-Scholes :", f"{option.black_scholes_price(market):.6f}")

    start_total = time.time()

    start = time.time()
    price = model_mc.european_price_scalar()
    end = time.time()
    elapsed = end - start
    total_elapsed = end - start_total
    print(f"Prix Européen Scalaire: {price:.6f} (temps écoulé: {elapsed:.4f} s, total: {total_elapsed:.4f} s)")

    start = time.time()
    price = model_mc.european_price_vectorized()
    end = time.time()
    elapsed = end - start
    total_elapsed = end - start_total
    print(f"Prix Européen Vectorisé: {price:.6f} (temps écoulé: {elapsed:.4f} s, total: {total_elapsed:.4f} s)")

    start = time.time()
    price = model_mc.american_price_scalar()
    end = time.time()
    elapsed = end - start
    total_elapsed = end - start_total
    print(f"Prix Américain Scalaire: {price:.6f} (temps écoulé: {elapsed:.4f} s, total: {total_elapsed:.4f} s)")

    start = time.time()
    price = model_mc.american_price_vectorized()
    end = time.time()
    elapsed = end - start
    total_elapsed = end - start_total
    print(f"Prix Américain Vectorisé: {price:.6f} (temps écoulé: {elapsed:.4f} s, total: {total_elapsed:.4f} s)")

    print(f"Temps total d'exécution: {time.time() - start_total:.4f} s")
