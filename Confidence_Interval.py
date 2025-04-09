from Excel_Handler import ConfidenceInterval
import numpy as np

def main():
    global model

    # Initialisation de l'objet Excel
    excel = ConfidenceInterval("Models_comparisons.xlsm")

    # ------------- Intervalles de confiance en fonction de la seed -----------------
    # Récupération des seeds
    seeds_list = excel.get_seeds_list()

    # Matrice des prix et bornes Longstaff-Schwartz (lignes: seeds, colonnes: Prix, UB, LB)
    ls_seed_bounds = np.zeros((len(seeds_list), 3))

    for i, seed in enumerate(seeds_list):
        model = excel.get_mcmodel(seed=seed)  # Mise à jour des paramètres
        price = model.price("Longstaff")
        lb, ub = model.price_confidence_interval(type="Longstaff")  # Déballer le tuple
        ls_seed_bounds[i] = (price,ub,lb)

    # ------------- Intervalles de confiance en fonction du type de regression -----------------
    # Récupération des seeds et des types de régressions
    regression_types = excel.reg_list
    # Matrice des prix et bornes Longstaff-Schwartz (lignes: regressions, colonnes: Prix, UB, LB)
    ls_reg_bounds = np.zeros((len(regression_types) + 1, 3))
    model = excel.get_mcmodel()
    ls_reg_bounds[0] = np.full(3, model.bsm.price()) # valeur black-scholes sans intervalle

    for i, reg_type in enumerate(regression_types):
        model.reg_type = reg_type
        price = model.price("Longstaff")
        lb, ub = model.price_confidence_interval(type="Longstaff")  # Déballer le tuple
        ls_reg_bounds[i+1] = (price, ub, lb)

    # Écriture des résultats
    excel.write_results(ls_seed_bounds, ls_reg_bounds)

if __name__ == "__main__":
    main()
