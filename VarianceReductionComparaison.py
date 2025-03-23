from Excel_Handler import VarianceComp
import numpy as np
import plotly.graph_objs as go

def main():
    excel = VarianceComp("Models_comparisons.xlsm")

    # Récupération des steps
    paths_list = excel.get_paths_list()
    n = len(paths_list)
    results = np.zeros((n, 7))

    # Listes pour stocker les variances et le nombre de paths traités
    for i, paths in enumerate(paths_list):
        engine = excel.get_mcmodel(n_paths=paths, compute_antithetic=False)
        engine_antithetic = excel.get_mcmodel(n_paths=paths, compute_antithetic=True)
        variance = engine.get_variance(type="Longstaff")
        variance_antithetic = engine_antithetic.get_variance(type="Longstaff")
        (upper_bound,lower_bound ) = engine.price_confidence_interval()
        (upper_bound_antithetic,lower_bound_antithetic ) = engine_antithetic.price_confidence_interval()
        diff = variance - variance_antithetic
        results[i, 0] = variance
        results[i, 1] = variance_antithetic
        results[i, 2] = diff
        results[i, 3] = upper_bound
        results[i, 4] = lower_bound
        results[i, 5] = upper_bound_antithetic
        results[i, 6] = lower_bound_antithetic
        print(f"Paths: {paths} traités.")
    stop=""
    excel.write_results(results)

if __name__ == '__main__':
    main()