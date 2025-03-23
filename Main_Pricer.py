import time
from Excel_Handler import EUPricing

def main():
    excel = EUPricing("Models_comparisons.xlsm")

    # Instanciation du modèle via ExcelHandler
    model_mc = excel.get_mcmodel()
    model_tree = excel.get_treemodel()

    # Calcul des prix
    start_total = time.time()

    prices = []
    times = []

    start = time.time()
    prices.append(model_mc.bsm.price())
    times.append(time.time() - start)

    prices.append(0)
    start = time.time()
    prices.append(model_mc.european_price_scalar())
    times.append(time.time() - start)

    start = time.time()
    prices.append(model_mc.european_price_vectorized())
    CI = model_mc.price_confidence_interval()
    times.append(time.time() - start)

    prices.append(0)
    start = time.time()
    prices.append(model_mc.american_price_scalar())
    times.append(time.time() - start)

    start = time.time()
    prices.append(model_mc.american_price_vectorized())
    times.append(time.time() - start)

    start = time.time()
    prices.append(model_tree.price())
    times.append(time.time() - start)

    print(f"Temps total d'exécution: {time.time() - start_total:.4f} s")

    # Écriture des résultats dans Excel
    excel.write_results(prices, CI, times)

if __name__ == "__main__":
    main()
