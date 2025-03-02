from Excel_Handler import ConvLSvsBS
import numpy as np

# Programme principal
def main():
    excel = ConvLSvsBS("Models_comparisons.xlsm")

    # Récupération des options et des steps
    options = excel.get_options()
    steps_list = excel.get_steps_list()

    # Création unique du modèle Monte Carlo (avec la première option pour init)
    model = excel.get_model(options[0])

    # Calcul des prix Black-Scholes
    bs_prices = np.array([
        model.black_scholes_price() if setattr(model, "option", option) is None else model.black_scholes_price()
        for option in options
    ])

    # Matrice de convergence LS (pré-allouée pour éviter les ralentissements)
    ls_matrix = np.zeros((len(steps_list), len(options)))

    for i, steps in enumerate(steps_list):
        model = excel.get_model(options[0],n_steps=steps)

        for j, option in enumerate(options):
            model.option = option  # Mise à jour de l'option
            ls_matrix[i, j] = model.american_price_vectorized()

    # Écriture des résultats
    excel.write_results(bs_prices, ls_matrix)

if __name__ == "__main__":
    main()
