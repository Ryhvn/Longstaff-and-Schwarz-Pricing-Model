from Excel_Handler import RegressComp
import numpy as np

# Programme principal
def main():
    global model

    # Initialisation de l'objet Excel
    excel = RegressComp("Models_comparisons.xlsm")

    # Récupération des steps et des types de régressions
    steps_list = excel.get_steps_list()
    regression_types = excel.reg_list

    # Matrice des prix Longstaff-Schwartz (lignes: steps, colonnes: types de régressions)
    ls_prices = np.zeros((len(steps_list), len(regression_types)))

    for i, steps in enumerate(steps_list):
        for j, reg_type in enumerate(regression_types):
            model = excel.get_mcmodel(n_steps=steps, ex_frontier=reg_type)  # Mise à jour des paramètres
            ls_prices[i, j] = model.american_price_vectorized()

    # Écriture des résultats
    excel.write_results(ls_prices)

if __name__ == "__main__":
    main()
