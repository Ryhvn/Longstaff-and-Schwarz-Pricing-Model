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
        """Instancie un Engine Monte Carlo avec des paramètres ajustables."""
        return MonteCarloEngine(
            self.get_market(),
            self.get_options(),
            pricing_date=kwargs.get("pricing_date", self.get_value(f"{self.param_prefix}PrDate")),
            n_paths=int(kwargs.get("n_paths", self.get_value(f"{self.param_prefix}Paths"))),
            n_steps=int(kwargs.get("n_steps", self.get_value(f"{self.param_prefix}Steps"))),
            seed=int(kwargs.get("seed", self.get_value(f"{self.param_prefix}Seed"))),
            ex_frontier=kwargs.get("ex_frontier",self.get_value(f"{self.param_prefix}Ex_frontier")),
            compute_antithetic=kwargs.get("compute_antithetic", self.get_value(f"{self.param_prefix}Antithetic") == "True")
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

    def clear_step_list(self):
        """Efface la liste des steps de pricing à partir de la cellule RCTimes, sans descendre trop bas."""
        cell = self.sheet.range("RCTimes")  # Première cellule des steps

        # Vérifier si la cellule contient des valeurs avant de tenter l'effacement
        if cell.value is not None:
            last_filled_cell = cell.end("down")  # Trouve la dernière cellule non vide
            self.sheet.range(cell, last_filled_cell).value = None  # Efface tout le range

    def write_results(self, ls_matrix):
        """Écrit les résultats LS dans la feuille Regres. Comp pour différents types de regressions."""
        self.sheet.range("RCPrices").value = ls_matrix

    def write_price_by_time_results(self,times, price_matrix):
        """Écrit les résultats de prix LS à chaque pas de temps dans la feuille Regres. Comp."""
        self.sheet.range("RCTimes").options(transpose=True).value = times
        self.sheet.range("RCPriceByT").value = price_matrix

class VarianceComp(SheetHandler):
    def __init__(self, file_path):
        super().__init__(file_path, "Variance Comp")
        self.param_prefix = "VC_"

    def get_paths_list(self):
        steps = []
        cell = self.sheet.range("VC_PathsList")  # Première cellule des paths
        while cell.value is not None:
            steps.append(int(cell.value))
            cell = cell.offset(1, 0)  # Descend d'une ligne
        return steps

    def write_results(self, var_matrix):
        self.sheet.range("VCPrices").value = var_matrix
    
# Classe pour la feuille Pricing Menu
class PricingMenu(SheetHandler):
    def __init__(self, file_path):
        super().__init__(file_path, "Pricing Menu")
        self.param_prefix = "Menu_"

    def write_results(self, prices, times, bs_greeks, model_greeks,CI,std ):
        """Écrit les résultats de la première feuille."""
        self.sheet.range(f"{self.param_prefix}ModelPrices").value = prices
        self.sheet.range(f"{self.param_prefix}Times").value = times
        self.sheet.range(f"{self.param_prefix}BSGreeks").options(transpose=True).value = bs_greeks
        self.sheet.range(f"{self.param_prefix}ModelGreeks").options(transpose=True).value = model_greeks
        self.sheet.range(f"{self.param_prefix}UPB_MC").value = CI[0]
        self.sheet.range(f"{self.param_prefix}LPB_MC").value = CI[1]
        self.sheet.range(f"{self.param_prefix}Std_MC").value = std

    def get_treemodel(self):
        """Instancie un modèle d'arbre trinomial pour la feuille PricingMenu."""
        return TreeModel(
            self.get_market(),
            self.get_options(),
            pricing_date=self.sheet.range(f"{self.param_prefix}PrDate").value,
            n_steps=int(self.sheet.range(f"{self.param_prefix}Steps").value),
        )

    def get_selected_model(self):
        """ Instancie le modèle spécifique au paramétrage utilisateur de l'option"""
        if self.get_value(f"{self.param_prefix}Model") == "MC":
            return self.get_mcmodel(), "MC"
        elif self.get_value(f"{self.param_prefix}Model") == "Longstaff":
            return self.get_mcmodel(), "Longstaff"
        elif self.get_value(f"{self.param_prefix}Model") == "Trinomial":
            return self.get_treemodel(), "Trinomial"
        else:
            raise ValueError("Le modèle sélectionné ne figure pas parmi les choix MC, Longstaff ou Trinomial.")

class ConfidenceInterval(SheetHandler):
    def __init__(self, file_path):
        super().__init__(file_path, "Conf. Interval")
        self.param_prefix = "CI_"
        self.reg_list = ["Linear", "Quadratic", "Cubic", "Quartic", "Quintic", "Sextic"]

    def get_seeds_list(self):
        steps = []
        cell = self.sheet.range("CI_SeedList")  # Première cellule des seeds
        while cell.value is not None:
            steps.append(int(cell.value))
            cell = cell.offset(1, 0)  # Descend d'une ligne
        return steps

    def write_results(self, seeds_matrix, reg_matrix):
        self.sheet.range("CI_Seed_values").value = seeds_matrix
        self.sheet.range("CI_Reg_values").value = reg_matrix