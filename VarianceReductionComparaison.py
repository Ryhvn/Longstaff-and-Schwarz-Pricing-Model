from Excel_Handler import EUPricing
import numpy as np
import plotly.graph_objs as go

#sanity check pour la reduction de variance
def main():
    excel = EUPricing("Models_comparisons.xlsm")

    paths_list = excel.get_paths_list()
    mc_variance = []
    mc_variance_antithetic = []
    processed_paths = []

    for paths in paths_list:
        if paths > 100 and paths <= 30000 and paths%2==0: 
            engine = excel.get_mcmodel(n_paths=paths, compute_antithetic="False")
            engine_antithetic = excel.get_mcmodel(n_paths=paths, compute_antithetic="True")
            # Calcul de la variance sur les payoffs actualisés
            variance=engine.get_variance()
            variance_antithetic=engine_antithetic.get_variance()
            mc_variance.append(variance)
            mc_variance_antithetic.append(variance_antithetic)
            processed_paths.append(paths)
            print(f"Paths: {paths} traités.")

    payoff_difference = np.array(mc_variance) - np.array(mc_variance_antithetic)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=processed_paths, 
        y=payoff_difference,
        mode='lines+markers',
        name='Différence'
    ))

    fig.update_layout(
        title="Différence de variance des payoffs actualisés (Standard - Antithetic)",
        xaxis_title="Nombre de paths",
        yaxis_title="Différence de variance",
        hovermode="x unified"
    )

    fig.show()

if __name__ == '__main__':
    main()
