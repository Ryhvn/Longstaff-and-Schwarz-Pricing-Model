import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------- Regression Class  ----------------
class Regression:
    @staticmethod
    def fit(X, Y):
        if len(X) == 0:
            return np.zeros_like(Y)
        X_reg = np.vstack([X, X ** 2]).T # Continuation value with polynomial regression
        model = LinearRegression().fit(X_reg, Y)
        return model.predict(X_reg)