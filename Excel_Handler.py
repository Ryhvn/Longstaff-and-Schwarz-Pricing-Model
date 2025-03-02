import xlwings as xw
from abc import ABC, abstractmethod
from Market import Market
from Option import Option
from PricingModel import ModelMC

# 🔹 Classe Abstraite SheetHandler
class SheetHandler(ABC):
    def __init__(self, file_path, sheet_name):
        self.file_path = file_path
        self.wb = xw.Book(file_path)
        self.sheet = self.wb.sheets[sheet_name]
        self.param_prefix = ""  # Préfixe pour les noms de cellule

    def get_value(self, cell_range):
        """Récupère une valeur unique depuis une cellule Excel."""
        return self.sheet.range(cell_range).value

    @abstractmethod
    def get_options(self):
        """Méthode abstraite pour récupérer les options (différente selon la feuille)."""
        pass

    def get_market(self):
        """Instancie la classe Market à partir des valeurs Excel."""
        return Market(
            S0=self.get_value(f"{self.param_prefix}Spot"),
            r=self.get_value(f"{self.param_prefix}Rate"),
            sigma=self.get_value(f"{self.param_prefix}Vol"),
            dividend=self.get_value(f"{self.param_prefix}Div")
        )

    def get_model(self, option):
        """Instancie UN SEUL modèle Monte Carlo et l'utilise pour tous les strikes."""
        return ModelMC(
            self.get_market(),
            option,
            pricingDate=self.get_value(f"{self.param_prefix}PrDate"),
            n_paths=int(self.get_value(f"{self.param_prefix}Paths")),
            n_steps=int(self.get_value(f"{self.param_prefix}Steps")),
            seed=int(self.get_value(f"{self.param_prefix}Seed"))
        )

    @abstractmethod
    def write_results(self, *args):
        """Méthode abstraite pour écrire les résultats dans Excel."""
        pass


# 🔹 Classe pour la feuille Pricing Models
class PricingModels(SheetHandler):
    def __init__(self, file_path):
        super().__init__(file_path, "Pricing Models")
        self.param_prefix = ""  # Pas de préfixe sur cette feuille

    def get_options(self):
        """Ici, pas besoin d’option dynamique, on peut en créer une seule."""
        return Option(
            K=self.get_value("Strike"),  # Une seule cellule pour le strike
            maturity=self.get_value("Maturity"),
            opt_type=self.get_value("OptType")
        )

    def write_results(self, prices, times):
        """Écrit les résultats de la première feuille."""
        self.sheet.range("ModelPrices").value = prices
        self.sheet.range("Times").value = times

    def get_model(self, option):
        """Instancie un modèle Monte Carlo pour la feuille Pricing Models."""
        return ModelMC(
            self.get_market(),
            self.get_options(),
            pricingDate=self.sheet.range("PrDate").value,
            n_paths=int(self.sheet.range("Paths").value),
            n_steps=int(self.sheet.range("Steps").value),
            seed=int(self.sheet.range("Seed").value)
        )

# 🔹 Classe pour la feuille Convergence LS vs BS
class ConvLSvsBS(SheetHandler):
    def __init__(self, file_path):
        super().__init__(file_path, "Conv LS vs BS")
        self.param_prefix = "Conv_LS_"

    def get_options(self):
        """Lit la liste des strikes et crée les options dynamiquement."""
        strikes = self.sheet.range("Strike_range").value
        return [Option(K=strike,
                       maturity=self.get_value(f"{self.param_prefix}Maturity"),
                       opt_type=self.get_value(f"{self.param_prefix}OptType"))
                for strike in strikes]

    def get_steps_list(self):
        """Lit la liste des steps à partir de la cellule 'Conv_LS_S1'."""
        steps = []
        cell = self.sheet.range("Conv_LS_S1")  # Première cellule des steps
        while cell.value is not None:
            steps.append(int(cell.value))
            cell = cell.offset(1, 0)  # Descend d'une ligne
        return steps

    def get_model(self, option, **kwargs):
        """Instancie un modèle Monte Carlo avec des paramètres ajustables."""
        return ModelMC(
            self.get_market(),
            option,
            pricingDate=kwargs.get("pricing_date", self.get_value(f"{self.param_prefix}PrDate")),
            n_paths=int(kwargs.get("n_paths", self.get_value(f"{self.param_prefix}Paths"))),
            n_steps=int(kwargs.get("n_steps", self.get_value(f"{self.param_prefix}Steps"))),
            seed=int(kwargs.get("seed", self.get_value(f"{self.param_prefix}Seed")))
        )

    def write_results(self, bs_prices, ls_matrix):
        """Écrit les résultats des prix BS et LS dans la feuille Convergence LS."""
        self.sheet.range("BS_range").value = bs_prices  # Range contenant les prix BS
        self.sheet.range("Conv_LS_P1").value = ls_matrix  # Première cellule des résultats LS
