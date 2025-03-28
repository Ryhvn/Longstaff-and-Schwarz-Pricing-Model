from PricingModel import Engine
from Regression import Regression
from BrownianMotion import BrownianMotion
import numpy as np
from scipy.stats import norm

#import matplotlib.pyplot as plt

# Scatter plot des valeurs de continuation estimées vs. les cash flows actualisés
#plt.figure(figsize=(8, 6))
#plt.scatter(cont_val, CF[in_money], alpha=0.5, label="CF vs Cont. Value")
#plt.plot([min(cont_val), max(cont_val)], [min(cont_val), max(cont_val)], color='red', linestyle='--',
#        label="y = x (Référence)")
#plt.xlabel("Valeur de continuation estimée")
#plt.ylabel("Cash Flow actualisé")
#plt.title(f"Comparaison Cont. Value vs CF - Step {t}")
#plt.legend()
#plt.grid()
#plt.show()


# ---------------- PathGenerator Class ----------------
class PathGenerator:
    def __init__(self, market, brownian, n_paths, n_steps, dt, t_div, compute_antithetic=False):
        self.market = market
        self.brownian = brownian
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = dt
        self.paths_scalar = None
        self.paths_vectorized = None
        self.t_div = t_div
        self.compute_antithetic = compute_antithetic

    def get_factors(self, dW: float) -> float:
        """
        Calcul de l'incrément selon la formule : exp((r - q - 0.5 * sigma^2) * dt + sigma * dW)
        Si dW est un scalaire, effectue le calcul pour une seule trajectoire.
        Si dW est une matrice, effectue le calcul pour toutes les trajectoires.
        """
        growth_adjustment = (self.market.r - (self.market.dividend if self.market.div_type == "continuous" else 0)
                             - 0.5 * self.market.sigma ** 2) * self.dt

        # Si dW est un scalaire (pour la méthode scalaire)
        if isinstance(dW, float):
            return np.exp(growth_adjustment + self.market.sigma * dW)

        # Si dW est une matrice (pour la méthode vectorielle)
        elif isinstance(dW, np.ndarray):
            increments = growth_adjustment + self.market.sigma * dW
            # Effectuer le cumul des incréments si dW est un tableau
            np.cumsum(increments, axis=1, out=increments)
            return np.exp(increments)

    def _adjust_price_for_dividend(self, price: float, t: int) -> float:
        """
        Ajuste le prix de l'actif en fonction des dividendes, si applicable, pour la méthode scalaire.
        """
        if self.market.div_type == "discrete" and self.t_div is not None and t == self.t_div + 1:
            if 0 < self.t_div <= self.n_steps:  # Vérifier que t_div est bien dans l'intervalle
                price -= self.market.dividend
        return price

    def _apply_dividends(self, paths: np.ndarray):
        """
        Applique les dividendes discrets dans le cas de la méthode vectorielle.
        """
        if self.market.div_type == "discrete" and self.t_div is not None:
            div_reduction_factor = 1 - (self.market.dividend / paths[:, self.t_div])
            paths[:, self.t_div + 1:] *= div_reduction_factor[:, np.newaxis]

    def generate_paths_scalar(self):
        """
        Génère des trajectoires avec une méthode scalaire (avec boucles).
        """
        if self.paths_scalar is None:
            # Initialisation des chemins
            paths = np.empty((self.n_paths, self.n_steps + 1))

            for i in range(self.n_paths):
                price = self.market.S0
                paths[i, 0] = price

                # Générer les trajectoires pour chaque chemin
                for t in range(1, self.n_steps + 1):
                    dW = self.brownian.scalar_motion()  # Génère un seul dW pour chaque pas

                    # Appliquer l'ajustement des dividendes
                    price = self._adjust_price_for_dividend(price, t)

                    # Calculer l'incrément et ajuster le prix
                    price *= self.get_factors(dW)
                    paths[i, t] = price

            self.paths_scalar = paths

        return self.paths_scalar

    def generate_paths_vectorized(self):
        """
        Génère des trajectoires avec une méthode vectorielle (sans boucles).
        """
        if self.paths_vectorized is None:
            # Initialisation des chemins
            paths = np.empty((self.n_paths, self.n_steps + 1))
            paths[:, 0] = self.market.S0  # Initialisation à S0

            if self.compute_antithetic and self.n_paths % 2 == 0:
                dW = self.brownian.vectorized_motion(self.n_paths//2, self.n_steps)
                dW = np.concatenate((dW, -dW), axis=0)
                #if self.n_paths % 2 == 1:
                    #dW_extra = self.brownian.vectorized_motion(1, self.n_steps)
                    #dW = np.concatenate((dW, dW_extra), axis=0)
            else  :
                dW = self.brownian.vectorized_motion(self.n_paths, self.n_steps)

            # Calcul des incréments pour tous les chemins
            increments = self.get_factors(dW)

            # Diffusions
            paths[:, 1:] = self.market.S0 * increments

            # Application des dividendes discrets éventuels
            self._apply_dividends(paths)

            self.paths_vectorized = paths

        return self.paths_vectorized

# ---------------- Classe MCModel ----------------
class MonteCarloEngine(Engine):
    def __init__(self, market, option, pricing_date, n_paths, n_steps, seed=None, ex_frontier="Quadratic",compute_antithetic=False):
        super().__init__(market, option, pricing_date, n_steps)
        self.n_paths = n_paths
        self.reg_type = ex_frontier
        self.eu_payoffs = None
        self.am_payoffs = None
        self.american_price_by_time = None
        self.PathGenerator = PathGenerator(
            self.market, BrownianMotion(self.dt, seed), self.n_paths, self.n_steps, self.dt, self.t_div,compute_antithetic)

    def _price_american_lsm(self, paths, analysis=False):
        global price_by_time

        CF = self.option.payoff(paths[:,-1]) # Valeur du payoff à l'échéance

        if analysis:
            price_by_time = []
            price_by_time.append(CF.mean() * np.exp(-self.market.r * self.T))

        for t in range(self.n_steps - 2, -1, -1):

            CF *= self.df # Actualisation en un seul calcul
            immediate = self.option.payoff(paths[:, t])
            in_money = immediate > 0  # Mask des options ITM

            if np.any(in_money):  # Vérifie si au moins une option est ITM
                cont_val = Regression.fit(self.reg_type, paths[in_money, t], CF[in_money])
                # print(f"Step {t}, Mean CF (avant régression) :", np.mean(CF[in_money]))
                # print(f"Step {t}, Mean Immediate Payoff :", np.mean(immediate[in_money]))
                # print(f"Step {t}, Mean Cont. Value :", np.mean(cont_val))
                # epsilon = 0.5  # Petit seuil de tolérance pour ajuster le bruit de la prédiction
                # exercise = immediate[in_money] >= (cont_val + epsilon)
                exercise = immediate[in_money] >= cont_val
                # exercised_before_T = np.sum(exercise)  # Nombre d'exercices avant T
                # print("Nombre d'exercices anticipés (devrait être 0) :", exercised_before_T)
                CF[in_money] = np.where(exercise, immediate[in_money], CF[in_money])

                if analysis:
                    price_by_time.append(CF.mean() * np.exp(-self.market.r * (t + 1) * self.dt))

        #CF *= self.df pas nécessaire d'actualiser encore un coup ? en t=0 je rentre dans la boucle et j'actualise déjà la matrice de CF t=1
        self.am_payoffs = CF

        if analysis:
            self.american_price_by_time = price_by_time

        return CF.mean()

    def _discounted_payoffs_by_method(self, type):
        """Calcule les payoffs associés à la méthode de pricing"""
        if type == "MC":
            if self.eu_payoffs is None:
                self.eu_payoffs = self._discounted_eu_payoffs(
                    self.PathGenerator.generate_paths_vectorized())
            return self.eu_payoffs.copy()
        else:
            if self.am_payoffs is None:
                self._price_american_lsm(self.PathGenerator.generate_paths_vectorized())
            return self.am_payoffs.copy()

    def _discounted_eu_payoffs(self, paths):
        """Calcule les payoffs actualisés pour un pricing européen."""
        payoffs = self.option.payoff(paths[:, -1])  # Payoff à maturité
        return np.exp(-self.market.r * self.T) * payoffs # Actualisation
      
    def get_variance(self, type="MC"):
        """Calcule la variance des payoffs actualisés pour la méthode de prix associée"""
        discounted_payoffs = self._discounted_payoffs_by_method(type)

        if self.PathGenerator.compute_antithetic:
            discounted_payoffs = (discounted_payoffs[:self.n_paths//2] + discounted_payoffs[self.n_paths//2:]) / 2
        return discounted_payoffs.var()
    
    def get_american_price_path(self):
        #if self.american_price_by_time is None:
        self._price_american_lsm(self.PathGenerator.generate_paths_vectorized(),analysis=True)
        return self.american_price_by_time
    
    def _european_price(self, paths):
        """Calcule le prix européen moyen."""
        payoffs=self._discounted_eu_payoffs(paths)
        return np.mean(payoffs)

    def get_std(self, type="MC"):
        std = np.sqrt(self.get_variance(type).copy() / self.n_paths) 
        return std

    def price_confidence_interval(self, alpha=0.05, type="MC"):
        """Calcule le prix et son intervalle de confiance Monte Carlo."""
        # Attention, la méthode conditionnelle ci-dessous fonctionne seulement si
        # aucun paramètre de calcul américain (ex reg_type n'est modifié) autrement on rappelera toujours le payoff mémoire sans recalcul
        discounted_payoffs = self._discounted_payoffs_by_method(type)
           
         # Récupère les payoffs actualisés
        mean_price = np.mean(discounted_payoffs)  # Prix moyen estimé

        std_dev =self.get_std()  
        # Quantile de la loi normale pour l'intervalle de confiance (avec numpy)
        z = norm.ppf(1 - alpha / 2)  # Approximation sans scipy

        # Calcul de la marge d'erreur
        CI_half_width = z * std_dev

        CI_lower = mean_price - CI_half_width
        CI_upper = mean_price + CI_half_width

        return (CI_upper, CI_lower)
    
    def price(self, type="MC"):
        """ Retourne le prix associé au type d'option enregistré"""
        if type == "Longstaff":
            return self.american_price_vectorized()
        else:
            return self.european_price_vectorized()

    def european_price_scalar(self):
        return self._european_price(self.PathGenerator.generate_paths_scalar())

    def european_price_vectorized(self):
        return self._european_price(self.PathGenerator.generate_paths_vectorized())

    def american_price_scalar(self):
        return self._price_american_lsm(self.PathGenerator.generate_paths_scalar())

    def american_price_vectorized(self):
        return self._price_american_lsm(self.PathGenerator.generate_paths_vectorized())