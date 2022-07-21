from GenerateData import DynamicModel
from FunctionalLibrary import FunctionalLibrary
from HyperOpt import HyperOpt
from Optimizer import Optimizer_casadi

from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

def run_gridsearch(features : list[np.ndarray], target : list[np.ndarray], features_clean : list[np.ndarray], target_clean : list[np.ndarray], 
                    time_span : np.ndarray, parameters : dict, add_constraints : bool = False, 
                    filename : str = "saved_data\Gridsearch_results.html", title : str = "Conc vs time", 
                    ensemble_iterations : int = 1, max_workers : Optional[int] = None, seed : int = 12345):
    
    if add_constraints == "mass_balance":
        include_column = [] # "mass_balance" : [56.108, 28.05, 56.106, 56.108]
        constraints_dict = {"mass_balance" : [], "consumption" : [], "formation" : [], 
                            "stoichiometry" : np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -0.5, -1]).reshape(4, -1)}
        # mass balance : equality constraint; formation/consumption : inequality constraint
    elif add_constraints == "stoichiometry":
        include_column = [[0, 2], [0, 3], [0, 1]]
        constraints_dict = {"mass_balance" : [], "consumption" : [], "formation" : [], 
                                "stoichiometry" : np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1)}
    else:
        include_column = None
        constraints_dict = {}
    
    # model.plot(features[-1], t_span, legend=["A", "B", "C", "D"], title = title)
    opt = HyperOpt(features, target, features_clean, target_clean, time_span, parameters, Optimizer_casadi(solver_dict = {"ipopt.print_level" : 0, 
                    "print_time": 0, "ipopt.sb" : "yes"}), 
                include_column = include_column, constraints_dict = constraints_dict, ensemble_iterations = ensemble_iterations, seed = seed)

    opt.gridsearch(max_workers = max_workers)
    opt.plot(filename, title)
    return opt.df_result

def run_all(noise_level : float, parameters : dict, iterate = True, ensemble_iterations : int = 1, 
            max_workers : Optional[int] = None, seed : int = 12345):

    t_span = np.arange(0, 5, 0.01)
    model = DynamicModel("kinetic_kosir", t_span, n_expt = 15)
    features_clean = model.integrate()
    target_clean = model.approx_derivative
    features = model.add_noise(0, noise_level)
    target = model.approx_derivative

    if iterate:
        print(f"Running simulation for {noise_level} noise and without constraints")
        df_result = run_gridsearch(features, target, features_clean, target_clean, t_span, parameters, add_constraints = False, 
                    filename = f"saved_data\Gridsearch_no_con_{noise_level}.html", title = f"Without constraints {noise_level} noise", 
                    ensemble_iterations = ensemble_iterations, max_workers = max_workers, seed = seed)

        adict["MSE_test_pred"].append(df_result.loc[0, "MSE_test_pred"])
        adict["AIC"].append(df_result.loc[0, "AIC"])
        adict["MSE_test_sim"].append(df_result.loc[0, "MSE_test_sim"])
        adict["complexity"].append(df_result.loc[0, "complexity"])


        print(f"Running simulation for {noise_level} noise and with constraints")
        df_result = run_gridsearch(features, target, features_clean, target_clean, t_span, parameters, add_constraints = "mass_balance", 
                    filename = f"saved_data\Gridsearch_con_{noise_level}.html", title = f"With constraints {noise_level} noise", 
                    ensemble_iterations = ensemble_iterations, max_workers = max_workers, seed = seed) 

        adict["MSE_test_pred"].append(df_result.loc[0, "MSE_test_pred"])
        adict["AIC"].append(df_result.loc[0, "AIC"])
        adict["MSE_test_sim"].append(df_result.loc[0, "MSE_test_sim"])
        adict["complexity"].append(df_result.loc[0, "complexity"])

        print(f"Running simulation for {noise_level} noise and with stoichiometry")
        df_result = run_gridsearch(features, target, features_clean, target_clean, t_span, parameters, add_constraints = "stoichiometry", 
                    filename = f"saved_data\Gridsearch_stoichiometry_{noise_level}.html", title = f"With stoichiometry {noise_level} noise", 
                    ensemble_iterations = ensemble_iterations, max_workers = max_workers, seed = seed) 

        adict["MSE_test_pred"].append(df_result.loc[0, "MSE_test_pred"])
        adict["AIC"].append(df_result.loc[0, "AIC"])
        adict["MSE_test_sim"].append(df_result.loc[0, "MSE_test_sim"])
        adict["complexity"].append(df_result.loc[0, "complexity"])

if __name__ == "__main__": 

    # with hyperparameter optmization
    params = {"optimizer__threshold": [0.01, 0.1], 
        "optimizer__alpha": [0, 0.01, 0.1, 1], 
        "feature_library__include_bias" : [False],
        "feature_library__degree": [1, 2, 3]}

    ensemble_params = {"optimizer__threshold": [0.01, 0.1, 1, 10], 
        "optimizer__alpha": [0], 
        "feature_library__include_bias" : [False],
        "feature_library__degree": [1, 2, 3]}

    adict = defaultdict(list)
    noise_level = [0.0, ]#0.1, 0.2]
    for noise in noise_level:
        run_all(noise, params, iterate=True, ensemble_iterations=1, max_workers = 3, seed = 10) 

    # plotting results
    for key in adict.keys():
        plt.plot(noise_level, adict[key][0::3], "--o", label = "without") 
        plt.plot(noise_level, adict[key][1::3], "--*", label = "mass")
        plt.plot(noise_level, adict[key][2::3], "--+", label = "stoichiometry")
        plt.xlabel("noise level")
        plt.ylabel(key)
        plt.legend()
        plt.show() 


    """ # without hyperparameter optimization
    t_span = np.arange(0, 5, 0.01)
    model = DynamicModel("kinetic_kosir", t_span, n_expt = 15)
    features = model.integrate() # list of features
    target_clean = model.approx_derivative # list of target value

    features = model.add_noise(0, 0.2)
    target = model.approx_derivative
    print(f"Features value", features[-1][-1])
    print(f"Target value", target[-1][-1])

    # model.plot(features[-1], t_span, legend = ["A", "B", "C", "D"])
    include_column = include_column = [[0, 2], [0, 3], [0, 1]]
    constraints_dict= {"mass_balance" : [], "formation" : [], "consumption" : [], 
                                        "stoichiometry" : np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1)}

    model = Optimizer_casadi(FunctionalLibrary(1) , alpha = 0, threshold = 0.1, solver_dict={"ipopt.print_level" : 0, "print_time":0})
    model.fit(features[:12], target[:12], include_column = [], 
                constraints_dict= {})
    model.print()
    print(model.complexity) """