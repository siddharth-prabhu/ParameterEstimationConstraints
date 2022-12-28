from GenerateData import DynamicModel
from FunctionalLibrary import FunctionalLibrary
from HyperOpt import HyperOpt
from Optimizer import Optimizer_casadi
from energy import EnergySindy

from collections import defaultdict
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import repeat

import numpy as np
import matplotlib.pyplot as plt
from utils import coefficient_difference_plot

import argparse


parser = argparse.ArgumentParser("ParameterEstimationSINDy")
parser.add_argument("--ensemble_study", choices = [0, 1], type = int, default = 0, help = "If True performs bootstrapping to eliminate parameters else normal sindy")
parser.add_argument("--noise_study", choices = [0, 1], type = int, default = 0, help = "If True performs hyperparameter optimization for varying noise levels") 
parser.add_argument("--experiments_study", choices = [0, 1], type = int, default = 0, help = "If True performs hyperparameter optimization for varying initial conditions") 
parser.add_argument("--sampling_study", choices = [0, 1], type = int, default = 0, help = "If True performs hyperparameter optimization for varying sampling frequencies") 
parser.add_argument("--max_workers", default = 0, type = int)
parser.add_argument("--kind", choices = ["energy", "normal"], type = str, default = "normal", help = "Energy model calculates the activation energy")
args = parser.parse_args()


def run_gridsearch(n_expt : int, delta_t : float, noise_level : float, parameters : dict, kind : str = "normal", ensemble_iterations : int = 1, 
                    max_workers : Optional[int] = None, seed : int = 12345, name : str = "ensemble"):
    
    plugin_dict = {"ipopt.print_level" : 0, "print_time": 0, "ipopt.sb" : "yes"}
    # generate clean testing data to be used later for calculating errors
    time_span_clean = np.arange(0, 5, 0.01)
    arguments = [(373, 8.314)] if kind == "normal" else None
    model = DynamicModel("kinetic_kosir", time_span_clean, n_expt = 15, arguments = arguments)
    features_clean = model.integrate()
    target_clean = model.approx_derivative
    arguments = model.arguments

    # generate training data with varying experiments and sampling time
    time_span = np.arange(0, 5, delta_t)
    arguments = [(373, 8.314)] if kind == "normal" else None
    model = DynamicModel("kinetic_kosir", time_span, n_expt = n_expt, arguments = arguments)
    features = model.integrate()
    features = model.add_noise(0, noise_level)
    target = model.approx_derivative
    arguments = model.arguments

    mse_pred, aic, mse_sim, comp = [], [], [], []
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

        # does grid_serch over parameters 
        print(f"Running simulation for {noise_level} noise, {n_expt} experiments, {delta_t} sampling time, and " + status)
        opt = HyperOpt(features, target, features_clean, target_clean, time_span, time_span_clean, parameters = parameters, 
                        model = Optimizer_casadi(plugin_dict = plugin_dict) if kind == "normal" else EnergySindy(plugin_dict = plugin_dict), 
                        include_column = include_column, constraints_dict = constraints_dict, ensemble_iterations = ensemble_iterations, seed = seed, 
                        arguments = arguments if kind == "energy" else None)

        opt.gridsearch(max_workers = max_workers)
        opt.plot(filename = f"saved_data\Gridsearch_{name}_{status}_noise{noise_level}_eniter{ensemble_iterations}_expt{n_expt}_delta_{delta_t}.html", 
                    title = f"{status} and {noise_level} noise")
        df_result = opt.df_result

        mse_pred.append(df_result.loc[0, "MSE_Prediction"])
        aic.append(df_result.loc[0, "AIC"])
        mse_sim.append(df_result.loc[0, "MSE_Integration"])
        comp.append(df_result.loc[0, "complexity"])

    return mse_pred, aic, mse_sim, comp 

def plot_adict(x : list, adict : dict, x_label : str):
    # plotting results
    with plt.style.context(["science", "notebook", "vibrant"]):
        for key in adict.keys():
            plt.plot(x, adict[key][0::3], "--o", linewidth = 2, markersize = 8, label = "naive") 
            plt.plot(x, adict[key][1::3], "--*", linewidth = 2, markersize = 8, label = "mass balance")
            plt.plot(x, adict[key][2::3], "--+", linewidth = 2, markersize = 8, label = "chemistry")
            plt.xlabel(x_label)
            plt.ylabel(key)
            plt.legend()
            plt.savefig(f"{x_label}_{key}")
            plt.close()

def run_without_gridsearch(noise_level : list[float], ensemble_iterations : int = 1, 
                    max_workers : Optional[int] = None, seed : int = 12345):
    
    plugin_dict = {"ipopt.print_level" : 0, "print_time": 0, "ipopt.sb" : "yes"}
    # generate clean testing data to be used later for calculating errors
    time_span_clean = np.arange(0, 5, 0.01)
    model = DynamicModel("kinetic_kosir", time_span_clean, n_expt = 15)

    # generate training data with varying experiments and sampling time
    time_span = np.arange(0, 5, 0.01)
    model = DynamicModel("kinetic_kosir", time_span, n_expt = 15)
    alist = []

    for noise in noise_level:
        
        features = model.integrate()
        features = model.add_noise(0, noise)
        target = model.approx_derivative

        opti = Optimizer_casadi(FunctionalLibrary(2) , alpha = 0.0, threshold = 0.1, plugin_dict={"ipopt.print_level" : 0, "print_time":0, "ipopt.sb" : "yes"}, 
                            max_iter = 20)
        # stoichiometry = np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1) # chemistry constraints
        # include_column = [[0, 2], [0, 3], [0, 1]]
        # stoichiometry =  np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -0.5, -1]).reshape(4, -1) # mass balance constraints
        stoichiometry = np.eye(4)

        opti.fit(features, target, include_column = [], 
                constraints_dict= {"formation" : [], "consumption" : [], 
                                    "stoichiometry" : stoichiometry}, ensemble_iterations = ensemble_iterations, seed = seed, max_workers = max_workers)

        coefficient_difference_plot(model.coefficients(), sigma = opti.adict["coefficients_dict"])
    

if __name__ == "__main__": 

    # with hyperparameter optmization
    params = {"optimizer__threshold": [0.01], # 0.1, 1], 
        "optimizer__alpha": [0], # 0.01, 0.1, 1], 
        "feature_library__include_bias" : [False],
        "feature_library__degree": [1] #, 2, 3]
        }

    ensemble_params = {"optimizer__threshold": [2, 1.25, 1.6], 
        "optimizer__alpha": [0], 
        "feature_library__include_bias" : [False],
        "feature_library__degree": [1, 2, 3]}

    # Perfrom simulations
    ensemble_study = args.ensemble_study # if True performs bootstrapping to eliminate parameters else normal sindy
    noise_study = args.noise_study # if True performs hyperparameter optimization for varying noise levels
    experiments_study = args.experiments_study # if True performs hyperparameter optimization for varying initial conditions
    sampling_study = args.sampling_study # if True performs hyperparameter optimization for varying sampling frequencies
    kind = args.kind # either normal sindy or energy sindy 
    max_workers = None if args.max_workers <= 0 else args.max_workers 

    ########################################################################################################################
    if noise_study :

        adict_noise = defaultdict(list)
        noise_level = [0.0, 0.1, 0.2]

        with ProcessPoolExecutor(max_workers = max_workers) as executor:   
            result = executor.map(partial(run_gridsearch, parameters = ensemble_params if ensemble_study else params, 
                            ensemble_iterations = 1000 if ensemble_study else 1, max_workers = max_workers, seed = 10, 
                            name = "sampling", kind = kind), repeat(15), repeat(0.01), noise_level)

            for alist in result:
                for key, value in zip(["MSE_Prediction", "AIC", "MSE_Integration", "complexity"], alist):
                    adict_noise[key].extend(value)

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
        
        with ProcessPoolExecutor(max_workers = max_workers) as executor:   
            result = executor.map(partial(run_gridsearch, delta_t = 0.05, noise_level = 0, parameters = ensemble_params if ensemble_study else params, 
                            ensemble_iterations = 1000 if ensemble_study else 1, max_workers = max_workers, seed = 10, 
                            name = "sampling", kind = kind), experiments)

            for alist in result:
                for key, value in zip(["MSE_Prediction", "AIC", "MSE_Integration", "complexity"], alist):
                    adict_experiments[key].extend(value)
        
        plot_adict(experiments, adict_experiments, x_label = "experiments")
    
    ########################################################################################################################
    if sampling_study :
    
        print("------"*100)
        print("Starting sampling study")
        adict_sampling = defaultdict(list)
        sampling = [0.01, 0.05, 0.1]

        with ProcessPoolExecutor(max_workers = max_workers) as executor:   
            result = executor.map(partial(run_gridsearch, noise_level = 0, parameters = ensemble_params if ensemble_study else params, 
                            ensemble_iterations = 1000 if ensemble_study else 1, max_workers = max_workers, seed = 10, 
                            name = "sampling", kind = kind), repeat(15), sampling)

            for alist in result:
                for key, value in zip(["MSE_Prediction", "AIC", "MSE_Integration", "complexity"], alist):
                    adict_sampling[key].extend(value)

        plot_adict(sampling, adict_sampling, x_label = "sampling")

    ########################################################################################################################
    """ # adding noise breaks down the method (situational runs)
    time_span_clean = np.arange(0, 5, 0.01)
    model = DynamicModel("kinetic_kosir", time_span_clean, n_expt = 15)

    # generate training data with varying experiments and sampling time
    time_span = np.arange(0, 5, 0.01)
    model = DynamicModel("kinetic_kosir", time_span, n_expt = 15)
        
    features = model.integrate()
    features = model.add_noise(0, 0)
    target = model.approx_derivative

    opti = Optimizer_casadi(FunctionalLibrary(2) , alpha = 0.0, threshold = 0.1, plugin_dict={"ipopt.print_level" : 0, "print_time":0, "ipopt.sb" : "yes"}, 
                        max_iter = 20)
    # stoichiometry = np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1) # chemistry constraints
    # include_column = [[0, 2], [0, 3], [0, 1]]
    # stoichiometry =  np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -0.5, -1]).reshape(4, -1) # mass balance constraints
    stoichiometry = np.eye(4)

    opti.fit(features, target, include_column = [], 
            constraints_dict= {"formation" : [], "consumption" : [], 
                                "stoichiometry" : stoichiometry}, ensemble_iterations = 1, seed = 10, max_workers = max_workers)

    coefficient_difference_plot(model.coefficients(), sigma = opti.adict["coefficients_dict"])

    features = model.integrate()
    features = model.add_noise(0, 0.1)
    target = model.approx_derivative

    opti = Optimizer_casadi(FunctionalLibrary(2) , alpha = 0.0, threshold = 0.1, plugin_dict={"ipopt.print_level" : 0, "print_time":0, "ipopt.sb" : "yes"}, 
                        max_iter = 20)
    # stoichiometry = np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1) # chemistry constraints
    # include_column = [[0, 2], [0, 3], [0, 1]]
    # stoichiometry =  np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -0.5, -1]).reshape(4, -1) # mass balance constraints
    stoichiometry = np.eye(4)

    opti.fit(features, target, include_column = [], 
            constraints_dict= {"formation" : [], "consumption" : [], 
                                "stoichiometry" : stoichiometry}, ensemble_iterations = 1, seed = 10, max_workers = max_workers)

    coefficient_difference_plot(model.coefficients(), sigma = opti.adict["coefficients_dict"]) """