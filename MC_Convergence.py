from Excel_Handler import EUPricing
import numpy as np

# Programme principal
def main():
    global model

    excel = EUPricing("Models_comparisons.xlsm")

    # Récupération des steps
    paths_list = excel.get_paths_list()

    # Matrice des prix Monte Carlo européen
    mc_prices = np.zeros(len(paths_list))

    for i, paths in enumerate(paths_list):
        model = excel.get_mcmodel(n_paths=paths)  # Mise à jour du nombre de pas
        mc_prices[i] = model.european_price_vectorized()

    # Écriture des résultats
    excel.write_paths_results(mc_prices,model.bsm.price())

if __name__ == "__main__":
    main()