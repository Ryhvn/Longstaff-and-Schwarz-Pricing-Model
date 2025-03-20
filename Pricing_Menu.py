import time
from Excel_Handler import PricingMenu
from Greeks import GreeksCalculator

def main():
    excel = PricingMenu("Models_comparisons.xlsm")

    # Instanciation du modèle via ExcelHandler
    model, type = excel.get_selected_model()

    # Calcul des prix & grecques
    prices = []
    times = []
    bs_greeks = model.bsm.all_greeks()
    model_greeks = GreeksCalculator(model,type=type).all_greeks()

    start = time.time()
    prices.append(model.bsm.price())
    times.append(time.time() - start)

    start = time.time()
    prices.append(model.price(type))
    times.append(time.time() - start)

    # Écriture des résultats dans Excel
    excel.write_results(prices, times, bs_greeks, model_greeks)

if __name__ == "__main__":
    main()
