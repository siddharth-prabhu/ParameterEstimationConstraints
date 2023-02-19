# type: ignore

from collections import defaultdict
from typing import Optional, List
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import repeat
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from GenerateData import DynamicModel
from FunctionalLibrary import FunctionalLibrary
from HyperOpt import HyperOpt
from Optimizer import Optimizer_casadi
from energy import EnergySindy


# only runs hyperparameter optimization for different training conditions

parser = argparse.ArgumentParser("ParameterEstimationSINDy")
parser.add_argument("--ensemble_study", choices = [1, 2], type = int, default = 1, 
                    help = "If 1 performs bootstrapping and 2 performs covariance elimination")
parser.add_argument("--noise_study", choices = [0, 1], type = int, default = 0, 
                    help = "If True performs hyperparameter optimization for varying noise levels") 
parser.add_argument("--experiments_study", choices = [0, 1], type = int, default = 0, 
                    help = "If True performs hyperparameter optimization for varying initial conditions") 
parser.add_argument("--sampling_study", choices = [0, 1], type = int, default = 0, 
                    help = "If True performs hyperparameter optimization for varying sampling frequencies") 
parser.add_argument("--max_workers", default = 0, type = int)
parser.add_argument("--kind", choices = ["NLS", "LS"], type = str, default = "LS", 
                    help = "Nonlinear Least Square or Least Square")
parser.add_argument("--degree", type = int, default = 1, help = "The polynomial degree of terms in functional library")
parser.add_argument("--ensemble_iter", type = int, default = 1000, help = "The number of emsemble iterations")
args = parser.parse_args()


def run_gridsearch(n_expt : int, delta_t : float, noise_level : float, parameters : List[dict], kind : str = "LS", ensemble_iterations : int = 1, 
                    variance_elimination : bool = False, max_workers : Optional[int] = None, seed : int = 12345, path : Optional[str] = None):
    
    """
    parameters : List[ensemble_params, sindy_params]
    """

    plugin_dict = {"ipopt.print_level" : 0, "print_time": 0, "ipopt.sb" : "yes", "ipopt.max_iter" : 1000}
    
    # generate clean testing data to be used later for calculating errors
    time_span_clean = np.arange(0, 5, 0.01)
    arguments_temperature = [(360, 8.314), (370, 8.314), (380, 8.314), (390, 8.314), (373, 8.314), (385, 8.314)]
    assert n_expt <= len(arguments_temperature), "Please sepcify more temperature values"
    
    arguments_clean = [(373, 8.314)] if kind == "LS" else arguments_temperature
    model = DynamicModel("kinetic_kosir", time_span_clean, n_expt = 6, arguments = arguments_clean)
    features_clean = model.integrate()
    target_clean = model.actual_derivative # use actual derivatives for testing 
    arguments = model.arguments

    # generate training data with varying experiments and sampling time
    time_span = np.arange(0, 5, delta_t)
    arguments = [(373, 8.314)] if kind == "LS" else arguments_temperature[:n_expt]
    model = DynamicModel("kinetic_kosir", time_span, n_expt = n_expt, arguments = arguments, seed = seed)
    features = model.integrate()
    features = model.add_noise(0, noise_level)
    target = model.approx_derivative
    arguments = model.arguments

    mse_pred, aic, mse_sim, comp = [], [], [], []
    for status in ["without constraints", "with constraints", "with stoichiometry", "sindy"]:

        # for NLS no need to perform unconstrained and mass balance formulation
        if kind == "NLS" and status in ["without constraints", "with constraints"]:
            continue

        if status == "with constraints" : # mass balance constraints 
            include_column = [] # "mass_balance" : [56.108, 28.05, 56.106, 56.108]
            constraints_dict = {"consumption" : [], "formation" : [], 
                                "stoichiometry" : np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -0.5, -1]).reshape(4, -1)}
        elif status == "with stoichiometry" : # chemistry constraints
            include_column = [[0, 2], [0, 3], [0, 1]]
            constraints_dict = {"consumption" : [], "formation" : [], 
                                "stoichiometry" : np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1)}
        elif status == "sindy":
            # compare derivative free method with sindy for NLS
            if kind == "NLS":
                include_column = [[0, 2], [0, 3], [0, 1]]
                constraints_dict = {"consumption" : [], "formation" : [], 
                                    "stoichiometry" : np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1)}
            else:
                include_column = None
                constraints_dict = {}

        else : # unconstrained formulation
            include_column = None
            constraints_dict = {}
        
        # does grid_serch over parameters 
        print(f"Running simulation for {noise_level} noise, {n_expt} experiments, {delta_t} sampling time, and " + status)
        opt = HyperOpt(features, target, features_clean, target_clean, 
                        time_span, time_span_clean, parameters = parameters[1], # parameters[status == "sindy"], 
                        model = Optimizer_casadi(plugin_dict = plugin_dict) if kind == "LS" else EnergySindy(plugin_dict = plugin_dict),
                        arguments = arguments if kind == "NLS" else None, arguments_clean =  arguments_clean if kind == "NLS" else None, 
                        meta = {"include_column" : include_column, "constraints_dict" : constraints_dict, "ensemble_iterations" : ensemble_iterations, 
                        "variance_elimination" : False, # no need to perform bootstrapping 
                        "seed" : seed, "derivative_free" : False if status == "sindy" else True})

        opt.gridsearch(max_workers = max_workers)

        opt.plot(filename = f'Gridsearch_{parameters[0]["feature_library__degree"][0]}_{status}_noise{noise_level}_eniter{ensemble_iterations}_expt{n_expt}_delta_{delta_t}.html', 
                    path = path, title = f"{status} and {noise_level} noise")
        df_result = opt.df_result

        # if no models were discovered then return all zero entries else return the first entry
        mse_pred.append(df_result.get("MSE_Prediction", [0])[0])
        aic.append(df_result.get("AIC", [0])[0])
        mse_sim.append(df_result.get("MSE_Integration", [0])[0])
        comp.append(df_result.get("complexity", [0])[0])

    return mse_pred, aic, mse_sim, comp 

def plot_adict(x : list, adict : dict, x_label : str, path : Optional[str] = None, title : Optional[str] = None):
    # plotting results
    if not path:
        path = os.getcwd()
    
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    
    width = (x[1] - x[0])/5 # the width of bar plots

    kind = adict.pop("kind", "LS")
    
    with plt.style.context(["science", "notebook", "vibrant"]):
        for key, value in adict.items():
            
            value = np.array(value)

            if kind == "LS":
                plt.bar(x - 0.5*width, value[0::4], label = "unconstrained", width = width, align = "center")
                plt.bar(x + 0.5*width, value[1::4], label = "mass balance", width = width, align = "center")
                plt.bar(x + 1.5*width, value[2::4], label = "chemistry", width = width, align = "center")
                plt.bar(x - 1.5*width, value[3::4], label = "sindy", width = width, align = "center")
            else:
                plt.bar(x + 0.5*width, value[0::2], label = "chemistry", width = width, align = "center")
                plt.bar(x - 0.5*width, value[1::2], label = "sindy", width = width, align = "center")

            if key in ["MSE_Integration", "MSE_Prediction"]:
                plt.yscale("log")
            
            if title:
                plt.title(title)

            plt.xlabel(x_label)
            plt.ylabel(key)
            plt.xticks(x)
            plt.legend()
            plt.savefig(os.path.join(path, f'{x_label}_{key}_{"_".join(title.split())}'))
            plt.close()

if __name__ == "__main__": 

    # Perfrom simulations
    ensemble_study = args.ensemble_study # if True performs bootstrapping to eliminate parameters else normal sindy
    noise_study = args.noise_study # if True performs hyperparameter optimization for varying noise levels
    experiments_study = args.experiments_study # if True performs hyperparameter optimization for varying initial conditions
    sampling_study = args.sampling_study # if True performs hyperparameter optimization for varying sampling frequencies
    kind = args.kind # either Least Square or energy sindy 
    variance_elimination = True if ensemble_study >= 1 else False 
    max_workers = None if args.max_workers <= 0 else args.max_workers 
    degree = args.degree
    ensemble_iterations = args.ensemble_iter

    if variance_elimination:
        elimination = "covariance" if ensemble_study == 2 else "boostrapping"
    else:
        elimination = "normal"

    path = os.path.join(os.getcwd(), "LS" if kind == "LS" else "NLS", elimination)

    # with hyperparameter optmization
    sindy_params = {"optimizer__threshold": [0.1, 0.5, 1], 
        "optimizer__alpha": [0, 0.01, 0.1], 
        "feature_library__include_bias" : [False],
        "feature_library__degree": [degree]
        }

    ensemble_params = {"optimizer__threshold": [2, 1.25, 1.6], # 95%, 80%, 90%
        "optimizer__alpha": [0], 
        "feature_library__include_bias" : [False],
        "feature_library__degree": [degree]}

    ########################################################################################################################
    if noise_study :

        print("------"*100)
        print("Starting experiment study")
        adict_noise = defaultdict(list)
        noise_level = [0.0, 0.1, 0.2]
        path_noise = os.path.join(path, "noise")

        with ProcessPoolExecutor(max_workers = max_workers) as executor:   
            result = executor.map(partial(run_gridsearch, parameters = [ensemble_params, sindy_params], 
                            ensemble_iterations = ensemble_iterations, variance_elimination = variance_elimination, 
                            max_workers = max_workers, seed = 20, 
                            path = path_noise, kind = kind), repeat(6), repeat(0.01), noise_level)

            for alist in result:
                for key, value in zip(["MSE_Prediction", "AIC", "MSE_Integration", "complexity"], alist):
                    adict_noise[key].extend(value)

        adict_noise["kind"] = kind
        plot_adict(noise_level, adict_noise, x_label = "noise", path = path_noise, title = f"Polynomial degree {degree}")

    ########################################################################################################################
    if experiments_study :
        # choose dt == 0.05 to see significant difference
    
        print("------"*100)
        print("Starting experiment study")
        experiments = [2, 4, 6]
        adict_experiments = defaultdict(list)
        path_experiments = os.path.join(path, "experiments")

        with ProcessPoolExecutor(max_workers = max_workers) as executor:   
            result = executor.map(partial(run_gridsearch, delta_t = 0.05, noise_level = 0, parameters = [ensemble_params, sindy_params], 
                            ensemble_iterations = ensemble_iterations, variance_elimination = variance_elimination, 
                            max_workers = max_workers, seed = 20, 
                            path = path_experiments, kind = kind), experiments)

            for alist in result:
                for key, value in zip(["MSE_Prediction", "AIC", "MSE_Integration", "complexity"], alist):
                    adict_experiments[key].extend(value)
        
        adict_experiments["kind"] = kind
        plot_adict(experiments, adict_experiments, x_label = "experiments", path = path_experiments, title = f"Polynomial degree {degree}")
    
    ########################################################################################################################
    if sampling_study :
    
        print("------"*100)
        print("Starting sampling study")
        sampling = [0.01, 0.05, 0.1]
        adict_sampling = defaultdict(list)
        path_sampling = os.path.join(path, "sampling")

        with ProcessPoolExecutor(max_workers = max_workers) as executor:   
            result = executor.map(partial(run_gridsearch, noise_level = 0, parameters = [ensemble_params, sindy_params], 
                            ensemble_iterations = ensemble_iterations, variance_elimination = variance_elimination, 
                            max_workers = max_workers, seed = 20, 
                            path = path_sampling, kind = kind), repeat(6), sampling)

            for alist in result:
                for key, value in zip(["MSE_Prediction", "AIC", "MSE_Integration", "complexity"], alist):
                    adict_sampling[key].extend(value)

        adict_sampling["kind"] = kind
        plot_adict(sampling, adict_sampling, x_label = "sampling", path = path_sampling, title = f"Polynomial degree {degree}")

    ########################################################################################################################