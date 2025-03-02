import time
from Excel_Handler import PricingModels

def main():
    excel = PricingModels("Models_comparisons.xlsm")

    # Instanciation du modèle via ExcelHandler
    model_mc = excel.get_model()

    # Calcul des prix
    start_total = time.time()

    prices = []
    times = []

    start = time.time()
    prices.append(model_mc.black_scholes_price())
    times.append(time.time() - start)

    start = time.time()
    prices.append(model_mc.european_price_scalar())
    times.append(time.time() - start)

    start = time.time()
    prices.append(model_mc.european_price_vectorized())
    times.append(time.time() - start)

    start = time.time()
    prices.append(model_mc.american_price_scalar())
    times.append(time.time() - start)

    start = time.time()
    prices.append(model_mc.american_price_vectorized())
    times.append(time.time() - start)

    print(f"Temps total d'exécution: {time.time() - start_total:.4f} s")

    # Écriture des résultats dans Excel
    excel.write_results(prices, times)

if __name__ == "__main__":
    main()
