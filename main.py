from GenerateData import DynamicModel
from FunctionalLibrary import FunctionalLibrary
from HyperOpt import HyperOpt
from Optimizer import Optimizer_casadi

from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt



def run_gridsearch(n_expt : int, delta_t : float, noise_level : float, parameters : dict, perform_grid_search : bool = True, 
                ensemble_iterations : int = 1, max_workers : Optional[int] = None, seed : int = 12345, name : str = "ensemble",
                result_dict : dict = {}):
    
    solver_dict = {"ipopt.print_level" : 0, "print_time": 0, "ipopt.sb" : "yes"}
    # generate clean testing data to be used later for calculating errors
    time_span_clean = np.arange(0, 5, 0.01)
    model = DynamicModel("kinetic_kosir", time_span_clean, n_expt = 15)
    features_clean = model.integrate()
    target_clean = model.approx_derivative

    # generate training data with varying experiments and sampling time
    time_span = np.arange(0, 5, delta_t)
    model = DynamicModel("kinetic_kosir", time_span, n_expt = n_expt)
    features = model.integrate()
    features = model.add_noise(0, noise_level)
    target = model.approx_derivative

    for status in ["without constraints", "with constraints", "with stoichiometry"]:

        if status == "with constraints" : # mass balance constraints 
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

        if perform_grid_search:
            # does grid_serch over parameters 
            print(f"Running simulation for {noise_level} noise, {n_expt} experiments, {delta_t} sampling time, and " + status)
            opt = HyperOpt(features, target, features_clean, target_clean, time_span, time_span_clean, parameters, Optimizer_casadi(solver_dict = solver_dict), 
                    include_column = include_column, constraints_dict = constraints_dict, ensemble_iterations = ensemble_iterations, seed = seed)

            opt.gridsearch(max_workers = max_workers)
            opt.plot(filename = f"saved_data\Gridsearch_{name}_{status}_noise{noise_level}_eniter{ensemble_iterations}_expt{n_expt}_delta_{delta_t}.html", 
                        title = f"{status} and {noise_level} noise")
            df_result = opt.df_result

            result_dict["MSE_test_pred"].append(df_result.loc[0, "MSE_test_pred"])
            result_dict["AIC"].append(df_result.loc[0, "AIC"])
            result_dict["MSE_test_sim"].append(df_result.loc[0, "MSE_test_sim"])
            result_dict["complexity"].append(df_result.loc[0, "complexity"])

        else:
            # does normal model fitting
            opt = Optimizer_casadi(solver_dict = solver_dict)
            opt.set_params(**parameters) # set parameters
            opt.fit(features, target, include_column = include_column, constraints_dict = constraints_dict, ensemble_iterations = ensemble_iterations, 
                    seed = seed, max_workers = max_workers)
            opt.print()

def plot_adict(x : list, adict : dict, x_label : str):
    # plotting results
    for key in adict.keys():
        plt.plot(x, adict[key][0::3], "--o", label = "without") 
        plt.plot(x, adict[key][1::3], "--*", label = "mass")
        plt.plot(x, adict[key][2::3], "--+", label = "stoichiometry")
        plt.xlabel(x_label)
        plt.ylabel(key)
        plt.legend()
        plt.savefig(f"{x_label}_{key}")
        plt.close()

if __name__ == "__main__": 

    # with hyperparameter optmization
    params = {"optimizer__threshold": [0.01, 0.1, 1], 
        "optimizer__alpha": [0, 0.01, 0.1, 1], 
        "feature_library__include_bias" : [False],
        "feature_library__degree": [1, 2, 3]}

    ensemble_params = {"optimizer__threshold": [2, 1.25, 1.6], 
        "optimizer__alpha": [0], 
        "feature_library__include_bias" : [False],
        "feature_library__degree": [1, 2, 3]}

    # Perfrom simulations
    ensemble_study = True # if True performs bootstrapping to eliminate parameters else normal sindy
    noise_study = False # if True performs hyperparameter optimization for varying noise levels
    experiments_study = True # if True performs hyperparameter optimization for varying initial conditions
    sampling_study = False # if True performs hyperparameter optimization for varying sampling frequencies

    ########################################################################################################################
    if noise_study :

        adict_noise = defaultdict(list)
        noise_level = [0.0, 0.1, 0.2]
        for noise in noise_level:
            run_gridsearch(15, 0.01, noise_level = noise, parameters = ensemble_params if ensemble_study else params, 
                            ensemble_iterations = 1000 if ensemble_study else 1, max_workers = 3, seed = 10, 
                            name = "ensemble", result_dict = adict_noise)

        plot_adict(noise_level, adict_noise, x_label = "noise")

    ########################################################################################################################
    if experiments_study :

        experiments_params = {"optimizer__threshold": [2], 
                    "optimizer__alpha": [0], 
                    "feature_library__include_bias" : [False],
                    "feature_library__degree": [1]}
    
        print("------"*100)
        print("Starting experiment study")

        experiments = [2, 4, 6]
        adict_experiments = defaultdict(list)
        for expt in experiments:
            run_gridsearch(expt, 0.01, noise_level= 0, parameters = ensemble_params if ensemble_study else params, 
                            ensemble_iterations = 2 if ensemble_study else 1, max_workers = 2, seed = 10, 
                            name = "experiments", result_dict = adict_experiments)
        plot_adict(experiments, adict_experiments, x_label = "experiments")
    
    ########################################################################################################################
    if sampling_study :
    
        print("------"*100)
        print("Starting sampling study")

        sampling = [0.01, 0.05, 0.1]
        adict_sampling = defaultdict(list)
        for sample in sampling:
            run_gridsearch(15, sample, noise_level= 0, parameters = ensemble_params if ensemble_study else params, 
                            ensemble_iterations = 1000 if ensemble_study else 1, max_workers = 2, seed = 10, 
                            name = "sampling", result_dict = adict_sampling)

        plot_adict(sampling, adict_sampling, x_label = "sampling")