from GenerateData import DynamicModel
from FunctionalLibrary import FunctionalLibrary
from HyperOpt import HyperOpt
from Optimizer import Optimizer_casadi

from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt



def run_gridsearch(n_expt : int, delta_t : float, noise_level : float, parameters : dict, 
                ensemble_iterations : int = 1, max_workers : Optional[int] = None, seed : int = 12345):
    
    time_span = np.arange(0, 5, delta_t)
    model = DynamicModel("kinetic_kosir", time_span, n_expt = n_expt)
    features_clean = model.integrate()
    target_clean = model.approx_derivative
    features = model.add_noise(0, noise_level)
    target = model.approx_derivative

    for status in ["without constraints", "with constraints", "with stoichiometry"]:

        if status == "with constriants" : # mass balance constraints 
            include_column = [] # "mass_balance" : [56.108, 28.05, 56.106, 56.108]
            constraints_dict = {"consumption" : [], "formation" : [], 
                                "stoichiometry" : np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -0.5, -1]).reshape(4, -1)}
        elif status == "with stoichiometry" : 
            include_column = [[0, 2], [0, 3], [0, 1]]
            constraints_dict = {"consumption" : [], "formation" : [], 
                                "stoichiometry" : np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1)}
        else:
            include_column = None
            constraints_dict = {}

        print(f"Running simulation for {noise_level} noise and " + status)

        opt = HyperOpt(features, target, features_clean, target_clean, time_span, parameters, Optimizer_casadi(solver_dict = {"ipopt.print_level" : 0, 
                    "print_time": 0, "ipopt.sb" : "yes"}), 
                include_column = include_column, constraints_dict = constraints_dict, ensemble_iterations = ensemble_iterations, seed = seed)

        opt.gridsearch(max_workers = max_workers)
        opt.plot(filename = f"saved_data\Gridsearch_{status}_noise{noise_level}_ensemble{ensemble_iterations}.html", title = f"{status} and {noise_level} noise")
        df_result = opt.df_result

        adict["MSE_test_pred"].append(df_result.loc[0, "MSE_test_pred"])
        adict["AIC"].append(df_result.loc[0, "AIC"])
        adict["MSE_test_sim"].append(df_result.loc[0, "MSE_test_sim"])
        adict["complexity"].append(df_result.loc[0, "complexity"])


if __name__ == "__main__": 

    # with hyperparameter optmization
    params = {"optimizer__threshold": [0.01], # 0.01, 0.1
        "optimizer__alpha": [0], # 0, 0.01, 0.1, 1 
        "feature_library__include_bias" : [False],
        "feature_library__degree": [1, 2]} # 1, 2, 3

    ensemble_params = {"optimizer__threshold": [0.01, 0.1, 1, 10], 
        "optimizer__alpha": [0], 
        "feature_library__include_bias" : [False],
        "feature_library__degree": [1, 2, 3]}

    adict = defaultdict(list)
    noise_level = [0.0, 0.1, 0.2]
    for noise in noise_level:
        run_gridsearch(noise_level = noise, parameters = params, ensemble_iterations=1, max_workers = 3, seed = 10) 

    # plotting results
    for key in adict.keys():
        plt.plot(noise_level, adict[key][0::3], "--o", label = "without") 
        plt.plot(noise_level, adict[key][1::3], "--*", label = "mass")
        plt.plot(noise_level, adict[key][2::3], "--+", label = "stoichiometry")
        plt.xlabel("noise level")
        plt.ylabel(key)
        plt.legend()
        plt.show() 

