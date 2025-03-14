from Excel_Handler import ConvLSvsBS
import numpy as np

import sys
print(sys.path)

# Programme principal
def main():
    excel = ConvLSvsBS("Models_comparisons.xlsm")

    # Récupération des options et des steps
    options = excel.get_options_list()
    paths_list = excel.get_paths_list()

    # Calcul des prix Black-Scholes
    #bs_prices = np.array([
        #model.bsm.price() if setattr(model, "option", option) is None else model.bsm.price()
        #for option in options
    #])
    bs_prices = []
    model = excel.get_mcmodel()
    for option in options:
        model.option = option
        bs_prices.append(model.bsm.price())

    # Matrice de convergence LS (pré-allouée pour éviter les ralentissements)
    ls_matrix = np.zeros((len(paths_list), len(options)))

    for i, paths in enumerate(paths_list):
        model = excel.get_mcmodel(n_paths=paths)

        for j, option in enumerate(options):
            model.option = option
            ls_matrix[i, j] = model.american_price_vectorized()

    # Écriture des résultats
    excel.write_results(bs_prices, ls_matrix)

if __name__ == "__main__":
    main()
