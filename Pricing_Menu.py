import time
from Excel_Handler import PricingMenu
from Greeks import GreeksCalculator
from MCPricer import MonteCarloEngine


def main():
    excel = PricingMenu("Models_comparisons.xlsm")

    # Instanciation du modèle via ExcelHandler
    model, type = excel.get_selected_model()

    # Calcul des prix & grecques
    CI = ["", ""]
    std = ""
    prices = []
    times = []

    start = time.time()
    prices.append(model.bsm.price())
    times.append(time.time() - start)
    start = time.time()
    if isinstance(model, MonteCarloEngine):
        prices.append(model.price(type))
        CI = model.price_confidence_interval(type=type)
        std = model.get_std(type=type)
    else:
        prices.append(model.price())
    times.append(time.time() - start)

    bs_greeks = model.bsm.all_greeks()

    model_greeks = GreeksCalculator(model,type=type).all_greeks()

    # Écriture des résultats dans Excel
    excel.write_results(prices, times, bs_greeks, model_greeks,CI,std)

if __name__ == "__main__":
    main()
