import time
from Excel_Handler import EUPricing

def main() -> None:
    excel = EUPricing("Models_comparisons.xlsm")
    model_mc = excel.get_mcmodel()
    model_tree = excel.get_treemodel()
    prices = [model_mc.bsm.price(), 0]
    times = []
    def timeit(func: callable) -> float:
        start = time.time()
        result = func()
        times.append(time.time() - start)
        return result
    prices.append(timeit(model_mc.european_price_scalar))
    start = time.time()
    price = model_mc.european_price_vectorized()
    CI = model_mc.price_confidence_interval() #intervalle de confiance
    prices.append(price)
    times.append(time.time() - start)
    prices.append(0)
    prices.append(timeit(model_mc.american_price_scalar))
    prices.append(timeit(model_mc.american_price_vectorized))
    prices.append(timeit(model_tree.price))
    print(f"Temps total d'exécution: {sum(times):.4f} s")
    excel.write_results(prices, CI, times)

if __name__ == "__main__":
    main()
