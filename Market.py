from typing import Optional, Any
from datetime import date

# ---------------- Market Class ----------------
class Market:
    def __init__(
        self,
        S0: float,
        r: float,
        sigma: float,
        dividend: float,
        div_type: str = "continuous",
        div_date: Optional[date] = None,
        days_convention: int = 365
    ) -> None:
        self.S0: float = S0  # Prix initial du sous-jacent
        self.r: float = r  # Taux d'intérêt sans risque
        self.sigma: float = sigma  # Volatilité
        self.dividend: float = dividend  # Dividende (yield / discret)
        self.div_date: Optional[date] = div_date
        self.div_type: str = div_type.lower()
        self.days_convention: int = days_convention

        if self.div_type not in ["continuous", "discrete"]:
            raise ValueError("div_type must be 'continuous' or 'discrete'")

    def copy(self, **kwargs: Any) -> 'Market':
        return Market(
        S0=kwargs.get("S0", self.S0),
        r=kwargs.get("r", self.r),
        sigma=kwargs.get("sigma", self.sigma),
        dividend=kwargs.get("dividend", self.dividend),
        div_type=kwargs.get("div_type", self.div_type),
        div_date=kwargs.get("div_date", self.div_date),
        days_convention=kwargs.get("days_convention", self.days_convention)
    )