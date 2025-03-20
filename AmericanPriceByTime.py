from Excel_Handler import RegressComp
import numpy as np
import plotly.graph_objs as go
from Regression import Regression

def main():
    excel = RegressComp("Models_comparisons.xlsm")
    regression_types = excel.reg_list
    model_mc = excel.get_mcmodel()
    result=np.zeros((model_mc.n_steps,len(regression_types)))

    for i in range(len(regression_types)):
        reg_type = regression_types[i]
        model = excel.get_mcmodel()
        model.reg_type = reg_type
        prices_by_time=model.get_american_price_path()
        result[:,i]=prices_by_time
    times=np.arange(model_mc.n_steps)
    excel.write_results(times,result)

def get_prices_by_time(model_mc):
    paths = model_mc.PathGenerator.generate_paths_vectorized()
    option = model_mc.option
    n_steps = model_mc.n_steps

    payoff = option.payoff(paths)
    CF = payoff[:, -1].copy()  # Valeur du payoff à l'échéance
    discount_factor = np.exp(-model_mc.market.r * model_mc.dt)
    price_by_time = []
    time=[]
    price_by_time.append(CF.mean()* np.exp(-model_mc.market.r * n_steps * model_mc.dt))
    for t in range(n_steps - 2, -1, -1):
        CF *= discount_factor  

        immediate = payoff[:, t]
        in_money = immediate > 0  

        if np.any(in_money):  
            cont_val = Regression.fit(model_mc.reg_type, paths[in_money, t], CF[in_money])

            exercise = immediate[in_money] >= cont_val
            CF[in_money] = np.where(exercise, immediate[in_money], CF[in_money])

        price_by_time.append(CF.mean()* np.exp(-model_mc.market.r * (t+1) * model_mc.dt))
        time.append(t)
    return price_by_time
    
if __name__ == "__main__":
    main()




