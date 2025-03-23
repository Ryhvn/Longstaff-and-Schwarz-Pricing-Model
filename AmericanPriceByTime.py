from Excel_Handler import RegressComp
import numpy as np

def main():
    excel = RegressComp("Models_comparisons.xlsm")
    excel.clear_step_list()
    regression_types = excel.reg_list
    model = excel.get_mcmodel()
    result=np.zeros((model.n_steps,len(regression_types)))

    for i in range(len(regression_types)):
        model.reg_type = regression_types[i]
        result[:,i]=model.get_american_price_path()

    result = result[::-1, :] #Inversement des résultats pour l'affichage
    times= np.arange(model.n_steps)
    excel.write_price_by_time_results(times,result)
    
if __name__ == "__main__":
    main()




