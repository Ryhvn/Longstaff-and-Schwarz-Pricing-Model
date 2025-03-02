import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------- Regression Class  ----------------
class Regression1:
    @staticmethod
    def fit(X, Y):
        if len(X) == 0:
            return np.zeros_like(Y)
        X_reg = np.vstack([X, X ** 2]).T # Continuation value with polynomial regression
        model = LinearRegression().fit(X_reg, Y)
        return model.predict(X_reg)

class Regression:
    @staticmethod
    def fit(X, Y):
        if len(X) == 0:
            return np.zeros_like(Y)

        # Construction de la matrice X_reg avec une colonne de 1 pour l'intercept
        X_reg = np.vstack([np.ones_like(X), X, X ** 2]).T

        # Régression par moindres carrés (résout AX = B)
        coeffs, _, _, _ = np.linalg.lstsq(X_reg, Y, rcond=None)  # rcond=None pour éviter les warnings

        # Affichage des coefficients
        #print(f"Intercept: {coeffs[0]:.6f}, Coefficients: {coeffs[1]:.6f}, {coeffs[2]:.6f}")

        # Calcul des valeurs prédites (valeur de continuation)
        continuation = X_reg @ coeffs  # Produit matriciel entre X_reg et les coefficients

        return continuation