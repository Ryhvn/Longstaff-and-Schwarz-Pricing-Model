import xlwings as xw
from abc import ABC, abstractmethod
from Market import Market
from Option import Call, Put
from MCPricer import MonteCarloEngine
from TreePricer import TreeModel

# Classe Abstraite SheetHandler
class SheetHandler(ABC):
    def __init__(self, file_path, sheet_name):
        self.file_path = file_path
        self.wb = xw.Book(file_path)
        self.sheet = self.wb.sheets[sheet_name]
        self.param_prefix = ""  # Préfixe pour les noms de cellule

    def get_value(self, cell_range):
        """Récupère une valeur unique depuis une cellule Excel."""
        return self.sheet.range(cell_range).value

    def get_market(self):
        """Instancie la classe Market à partir des valeurs Excel."""
        div_type = self.get_value(f"{self.param_prefix}DivType")

        if div_type == "Continuous Dividend:":
            dividend = self.get_value(f"{self.param_prefix}Div")
            div_date = None
        elif div_type == "Discrete Dividend:":
            dividend = self.get_value(f"{self.param_prefix}Div")
            div_date = self.get_value(f"{self.param_prefix}DivDate")
        else:
            raise ValueError("DivType must be named 'Continuous Dividend:' or 'Discrete Dividend:'")
        if dividend is None:
            dividend = 0

        return Market(
            S0=self.get_value(f"{self.param_prefix}Spot"),
            r=self.get_value(f"{self.param_prefix}Rate"),
            sigma=self.get_value(f"{self.param_prefix}Vol"),
            dividend=dividend,
            div_type="continuous" if div_type == "Continuous Dividend:" else "discrete",
            div_date=div_date
        )

    def get_options(self, **kwargs):
        """ Crée dynamiquement une option en fonction de 'OptType' (Call ou Put). """
        opt_type = self.get_value(f"{self.param_prefix}OptType").lower()

        # Vérification du type d'option et création de l'objet correspondant
        if opt_type not in ["call", "put"]:
            raise ValueError(f"Option type '{opt_type}' non supporté, veuillez spécifier 'call' ou 'put'.")

        return {
            "call": Call,
            "put": Put
        }[opt_type](
            K=kwargs.get("strike", self.get_value(f"{self.param_prefix}Strike")),
            maturity=kwargs.get("maturity", self.get_value(f"{self.param_prefix}Maturity")),
            exercise=kwargs.get("exercise", self.get_value(f"{self.param_prefix}Exercise"))
        )

    def get_mcmodel(self, **kwargs):
        """Instancie un modèle Monte Carlo avec des paramètres ajustables."""
        return MonteCarloEngine(
            self.get_market(),
            self.get_options(),
            pricing_date=kwargs.get("pricing_date", self.get_value(f"{self.param_prefix}PrDate")),
            n_paths=int(kwargs.get("n_paths", self.get_value(f"{self.param_prefix}Paths"))),
            n_steps=int(kwargs.get("n_steps", self.get_value(f"{self.param_prefix}Steps"))),
            seed=int(kwargs.get("seed", self.get_value(f"{self.param_prefix}Seed"))),
            ex_frontier=kwargs.get("ex_frontier",self.get_value(f"{self.param_prefix}Ex_frontier")),
            compute_antithetic = kwargs.get("compute_antithetic", self.get_value(f"{self.param_prefix}Compute_Antithetic")).lower() == "true"
        )

    @abstractmethod
    def write_results(self, *args):
        """Méthode abstraite pour écrire les résultats dans Excel."""
        pass

# Classe pour la feuille EU Pricing
class EUPricing(SheetHandler):
    def __init__(self, file_path):
        super().__init__(file_path, "EU Pricing")
        self.param_prefix = ""  # Pas de préfixe sur cette feuille

    def write_results(self, prices, ci, times):
        """Écrit les résultats de la première feuille."""
        self.sheet.range("ModelPrices").value = prices
        self.sheet.range("Times").value = times
        self.sheet.range("UPB_MC").value = ci[0]
        self.sheet.range("LPB_MC").value = ci[1]

    def write_paths_results(self, prices, bs_price):
        """Écrit les résultats Monte Carlo dans la feuille EU Pricing pour différents nombres de chemins."""
        self.sheet.range("MCPrices").options(transpose=True).value = prices
        self.sheet.range("BS_Price").value = bs_price

    def get_treemodel(self):
        """Instancie un modèle d'arbre trinomial pour la feuille EU Pricing."""
        return TreeModel(
            self.get_market(),
            self.get_options(),
            pricing_date=self.sheet.range("PrDate").value,
            n_paths=int(self.sheet.range("Paths").value),
            n_steps=int(self.sheet.range("Steps").value),
        )

    def get_paths_list(self):
        """Lit la liste des paths à partir de la cellule 'Paths_Main'."""
        steps = []
        cell = self.sheet.range("Paths_Main")  # Première cellule des paths
        while cell.value is not None:
            steps.append(int(cell.value))
            cell = cell.offset(1, 0)  # Descend d'une ligne
        return steps

# Classe pour la feuille Convergence LS vs BS
class ConvLSvsBS(SheetHandler):
    def __init__(self, file_path):
        super().__init__(file_path, "Conv LS vs BS")
        self.param_prefix = "Conv_LS_"

    def get_options_list(self):
        """Lit la liste des strikes et crée les options dynamiquement."""
        strikes = self.sheet.range("Strike_range").value
        # Utilisation d'une fonction pour créer l'option pour chaque strike
        return [self.get_options(strike=strike) for strike in strikes]

    def get_paths_list(self):
        """Lit la liste des paths à partir de la cellule 'Conv_LS_S1'."""
        steps = []
        cell = self.sheet.range("Conv_LS_PList")  # Première cellule des paths
        while cell.value is not None:
            steps.append(int(cell.value))
            cell = cell.offset(1, 0)  # Descend d'une ligne
        return steps

    def write_results(self, bs_prices, ls_matrix):
        """Écrit les résultats des prix BS et LS dans la feuille Convergence LS."""
        self.sheet.range("BS_range").value = bs_prices  # Range contenant les prix BS
        self.sheet.range("Conv_LS_P1").value = ls_matrix  # Première cellule des résultats LS


class RegressComp(SheetHandler):
    def __init__(self, file_path):
        super().__init__(file_path, "Regres. Comp")
        self.param_prefix = "RC_"
        self.reg_list = ["Linear", "Quadratic", "Cubic", "Quartic", "Quintic", "Sextic"]

    def get_steps_list(self):
        """Lit la liste des paths à partir de la cellule 'Conv_LS_S1'."""
        steps = []
        cell = self.sheet.range("RC_StepsList")  # Première cellule des steps
        while cell.value is not None:
            steps.append(int(cell.value))
            cell = cell.offset(1, 0)  # Descend d'une ligne
        return steps

    def write_results(self, ls_matrix):
        """Écrit les résultats Monte Carlo dans la feuille EU Pricing pour différents nombres de chemins."""
        self.sheet.range("RCPrices").value = ls_matrix
