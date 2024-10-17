# type: ignore

import os
from collections import defaultdict
from typing import Optional, List
from concurrent.futures import ProcessPoolExecutor
import argparse

import numpy as np
import matplotlib.pyplot as plt
import sympy as smp

from generate_data import DynamicModel
from hyper_opt import HyperOpt
from optimizer import Optimizer_casadi
from energy import EnergySindy
from adiabatic import AdiabaticSindy
from utils import coefficients_plot


# only runs hyperparameter optimization for different training conditions

parser = argparse.ArgumentParser("ParameterEstimationSINDy")
parser.add_argument("--noise_study", choices = [0, 1], type = int, default = 0, 
                    help = "If True performs hyperparameter optimization for varying noise levels") 
parser.add_argument("--experiments_study", choices = [0, 1], type = int, default = 0, 
                    help = "If True performs hyperparameter optimization for varying initial conditions")   
parser.add_argument("--sampling_study", choices = [0, 1], type = int, default = 0, 
                    help = "If True performs hyperparameter optimization for varying sampling frequency")             
parser.add_argument("--max_workers", default = 1, type = int)
parser.add_argument("--problem", choices = ["NLS", "LS", "AD"], type = str, default = "LS", help = "The type of problem to solve")
parser.add_argument("--system", choices = ["kosir", "menten", "carb"], type = str, default = "kosir", help = "The type of reaction network to run")
parser.add_argument("--degree", type = int, default = 1, help = "The polynomial degree of terms in functional library")
pargs = parser.parse_args()


def run_gridsearch(n_expt : int, delta_t : float, noise_level : float, parameters : dict, kind : str = "LS", 
                    max_workers : Optional[int] = None, seed : int = 12345, path : Optional[str] = None):
    

    plugin_dict = {"ipopt.print_level" : 0, "print_time": 0, "ipopt.sb" : "yes", "ipopt.max_iter" : 3000}
    solver_dict = {"solver" : "ipopt", "tol" : 1e-4}
    
    # generate clean testing data to be used later for calculating errors
    time_span_clean = np.arange(0, 5, 0.01)
    arguments_temperature = [(365, 8.314), (370, 8.314), (380, 8.314), (390, 8.314), (373, 8.314), (385, 8.314)]
    assert n_expt <= len(arguments_temperature), "Please sepcify more temperature values"
    
    arguments_clean = [(373, 8.314)] if kind == "LS" else arguments_temperature
    model = DynamicModel("kinetic_kosir", time_span_clean, n_expt = 6, arguments = arguments_clean)
    features_clean = model.integrate()
    target_clean = model.actual_derivative # use actual derivatives for testing 
    arguments_clean = model.arguments

    # generate training data with varying experiments and sampling time
    time_span = np.arange(0, 5, delta_t)
    arguments = [(373, 8.314)] if kind == "LS" else arguments_temperature[:n_expt]
    model = DynamicModel("kinetic_kosir", time_span, n_expt = n_expt, arguments = arguments, seed = seed)
    features = model.integrate()
    features = model.add_noise(0, noise_level)
    target = model.approx_derivative
    arguments = model.arguments

    actual_coefficients = [model.coefficients(pre_stoichiometry = True if kind == "NLS" else False)]
    mse_pred, aic, mse_sim, comp, coefficients, coefficients_pre_stoichiometriy = [], [], [], [], [], []
    for status in ["without constraints", "with constraints", "with stoichiometry", "sindy"]:

        # for NLS no need to perform unconstrained and mass balance formulation
        if kind == "NLS" and status in ["without constraints", "with constraints"]:
            continue

        if status == "with constraints" : # mass balance constraints 
            include_column = [] # "mass_balance" : [56.108, 28.05, 56.106, 56.108]
            constraints_dict = {"stoichiometry" : np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -0.5, -1]).reshape(4, -1)}
        elif status == "with stoichiometry" : # chemistry constraints
            include_column = [[0, 1], [0, 2], [0, 3]] 
            constraints_dict = {"stoichiometry" : np.array([-1, -1, -1, 2, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(4, -1)}
        elif status == "sindy":
            # compare derivative free method with sindy for NLS
            if kind == "NLS":
                include_column = [[0, 1], [0, 2], [0, 3]] 
                constraints_dict = {"stoichiometry" : np.array([-1, -1, -1, 2, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(4, -1)}
            else:
                include_column = None
                constraints_dict = {"stoichiometry" : np.eye(4)}

        else : # unconstrained formulation
            include_column = None
            constraints_dict = {"stoichiometry" : np.eye(4)}
        
        # does grid_serch over parameters 
        _path = os.path.join(path, status)
        opt = HyperOpt(features, target, features_clean, target_clean, 
                        time_span, time_span_clean, parameters = parameters, 
                        model = Optimizer_casadi(plugin_dict = plugin_dict, solver_dict = solver_dict) if kind == "LS" else EnergySindy(plugin_dict = plugin_dict, solver_dict = solver_dict),
                        arguments = arguments if kind == "NLS" else None, arguments_clean =  arguments_clean if kind == "NLS" else None, 
                        meta = {"include_column" : include_column, "constraints_dict" : constraints_dict,
                        "seed" : seed, "derivative_free" : False if status == "sindy" else True}, 
                        _dir = _path)

        opt.gridsearch(max_workers = max_workers)
        opt.plot(filename = 'Gridsearch.html', path = _path, title = f"{status} and {noise_level} noise")
        df_result = opt.df_result

        # if no models were discovered then return all zero entries else return the first entry
        mse_pred.append(df_result.get("MSE_Prediction", [0])[0])
        aic.append(df_result.get("AIC", [0])[0])
        mse_sim.append(df_result.get("MSE_Integration", [0])[0])
        comp.append(df_result.get("complexity", [0])[0])

        _dummy_dict = [{} for _ in range(constraints_dict["stoichiometry"].shape[-1])]
        coefficients.append(df_result.get("coefficients", [_dummy_dict])[0])
        coefficients_pre_stoichiometriy.append(df_result.get("coefficients_pre_stoichiometry", [_dummy_dict])[0])

    return mse_pred, aic, mse_sim, comp, coefficients, coefficients_pre_stoichiometriy, actual_coefficients


def run_adiabatic(n_expt : int, delta_t : float, noise_level : float, parameters : dict, kind : str = "LS", 
                    max_workers : Optional[int] = None, seed : int = 12345, path : Optional[str] = None):

    # lower the tolerance for noise
    plugin_dict = {"ipopt.print_level" : 0, "print_time": 0, "ipopt.sb" : "yes", "ipopt.max_iter" : 3000, "ipopt.tol" : 1e-5}

    if noise_level > 0 : plugin_dict["ipopt.tol"] = 1e-5 # set lower tolerance for noise

    # generate clean testing data to be used later for calculating errors
    time_span_clean = np.arange(0, 5, 0.01)
    model = DynamicModel("kinetic_kosir_temperature", time_span_clean, n_expt = 6)
    features_clean = model.integrate()
    target_clean = [tar[:, :-1] for tar in model.actual_derivative] # use actual derivatives for testing 
    arguments_clean = [np.column_stack((feat[:, -1], np.tile(8.314, (len(feat), 1)))) for feat in features_clean]

    # generate training data with varying experiments and sampling time
    time_span = np.arange(0, 5, delta_t)
    model = DynamicModel("kinetic_kosir_temperature", time_span, n_expt = n_expt)
    features = model.integrate()
    features = model.add_noise(0, noise_level)
    target = [tar[:, :-1] for tar in model.approx_derivative]
    arguments = [np.column_stack((feat[:, -1], np.tile(8.314, (len(feat), 1)))) for feat in features]
    
    include_column = [[0, 1], [0, 2], [0, 3]] 
    constraints_dict = {"stoichiometry" : np.array([-1, -1, -1, 2, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(4, -1)}

    actual_coefficients = [model.coefficients(pre_stoichiometry = True)]
    mse_pred, aic, mse_sim, comp, coefficients, coefficients_pre_stoichiometry = [], [], [], [], [], []
    for status in ["sindy", "df-sindy"]:
            
        # does grid_serch over parameters 
        _path = os.path.join(path, status)
        print(f"Running simulation for {noise_level} noise, {n_expt} experiments, {delta_t} sampling time, and " + status, flush = True)
        opt = HyperOpt([feat[:, :-1] for feat in features] if status == "sindy" else features, 
                        target if status == "sindy" else [feat[:, :-1] for feat in features], 
                        [feat[:, :-1] for feat in features_clean] if status == "sindy" else features_clean, 
                        target_clean if status == "sindy" else [feat[:, :-1] for feat in features_clean], 
                        time_span, time_span_clean, 
                        parameters = parameters, 
                        model = EnergySindy(plugin_dict = plugin_dict) if status == "sindy" else AdiabaticSindy(plugin_dict = plugin_dict),
                        arguments = arguments, 
                        arguments_clean = arguments_clean, 
                        meta = {"include_column" : include_column, "constraints_dict" : constraints_dict, 
                        "seed" : seed, "derivative_free" : False},
                        _dir = _path
                        )

        opt.gridsearch(max_workers = max_workers)
        opt.plot(filename = 'Gridsearch.html', path = _path, title = f"{status}")
        df_result = opt.df_result

        # if no models were discovered then return all zero entries else return the first entry
        mse_pred.append(df_result.get("MSE_Prediction", [0])[0])
        aic.append(df_result.get("AIC", [0])[0])
        mse_sim.append(df_result.get("MSE_Integration", [0])[0])
        comp.append(df_result.get("complexity", [0])[0])

        _dummy_dict = [{} for _ in range(constraints_dict["stoichiometry"].shape[-1])]
        coefficients.append(df_result.get("coefficients", [_dummy_dict])[0])
        coefficients_pre_stoichiometry.append(df_result.get("coefficients_pre_stoichiometry", [_dummy_dict])[0])

    return mse_pred, aic, mse_sim, comp, coefficients, coefficients_pre_stoichiometry, actual_coefficients


def plot_adict(x : list, adict : dict, x_label : str, path : Optional[str] = None, title : Optional[str] = None) -> None :
    # plotting results
    if path is None :
        path = os.getcwd()
    
    if not isinstance(x, np.ndarray):
        if isinstance(x[0], str):
            xtick_label, x = x, np.arange(0, len(x))
        else:
            xtick_label, x = x, np.array(x)
    
    width = (x[1] - x[0])/5 # the width of bar plots
    kind = adict.pop("kind", False)
    assert kind, "kind is not provided while plotting"
    
    with plt.style.context(["science", "notebook", "vibrant"]):
        for key, value in adict.items():
            value = np.array(value)

            if kind == "LS":
                plt.bar(x - 0.5*width, value[0::4], label = "unconstrained", width = width, align = "center")
                plt.bar(x + 0.5*width, value[1::4], label = "mass balance", width = width, align = "center")
                plt.bar(x + 1.5*width, value[2::4], label = "chemistry", width = width, align = "center")
                plt.bar(x - 1.5*width, value[3::4], label = "sindy", width = width, align = "center")
            elif kind == "AD":
                plt.bar(x - 0.5*width, value[0::2], label = "derivative", width = width, align = "center")
                plt.bar(x + 0.5*width, value[1::2], label = "integral", width = width, align = "center")
            else:
                plt.bar(x + 0.5*width, value[0::2], label = "chemistry", width = width, align = "center")
                plt.bar(x - 0.5*width, value[1::2], label = "sindy", width = width, align = "center")

            if key in ["MSE", "MSE_Prediction"]:
                plt.yscale("log")
            
            if title:
                plt.title(title)

            plt.xlabel(x_label)
            plt.ylabel(key)
            plt.xticks(x, labels = xtick_label)
            plt.legend()
            plt.savefig(os.path.join(path, f'{key}'))
            plt.close()


def plot_coeff(level : list, adict : dict, path : Optional[str] = None, title : Optional[str] = None) -> None :

    if path is None :
        path = os.path.join(os.getcwd(), "coefficients")

    kind = adict.pop("kind")
    actual_coefficients = adict["actual_coefficients"][0][0]
    discovered_coefficients = adict["coefficients"] if kind == "LS" else adict["coefficients_pre_stoichiometry"]

    remove_exp = lambda expr : expr.split("*exp")[0]

    def update_keys(adict):
        bdict = {}
        for key, value in adict.items():
            new_key = remove_exp(str(key))
            bdict[smp.sympify(new_key)] = value

        return bdict

    for i, val in enumerate(level):

        _discovered_coefficients = [[update_keys(_dict) for _dict in _alist] for _alist in discovered_coefficients[i]]

        coefficients_plot(
            actual_coefficients, 
            _discovered_coefficients, 
            expt_names = ["unconstrained", "mass balance", "chemistry", "sindy"] if kind == "LS" else (["derivative", "integral"] if kind == "AD" else ["chemistry", "sindy"]), 
            path = path + str(val) + ".png", 
            title = title
        )


if __name__ == "__main__": 

    # Perform simulations
    noise_study = pargs.noise_study # if True performs hyperparameter optimization for varying noise levels
    experiments_study = pargs.experiments_study # if True performs hyperparameter optimization for varying initial conditions
    sampling_study = pargs.sampling_study # if True performs hyperparameter optimization for varying sampling frequencies
    problem = pargs.problem # the type of problem to solve. Either LS or NLS or Adiabatic 
    max_workers = None if pargs.max_workers <= 0 else pargs.max_workers 
    degree = pargs.degree 

    path = os.path.join("log", pargs.system, problem)

    # with hyperparameter optmization
    sindy_params = {"optimizer__threshold": [0.1, 0.5, 1],
        "optimizer__alpha": [0, 0.01, 0.1], # 0, 0.01, 0.1
        "feature_library__include_bias" : [False],
        "feature_library__degree": [degree]
        }

    ########################################################################################################################
    if noise_study :

        print("------"*100)
        print("Starting experiment study")
        adict_noise = defaultdict(list)
        coeff_noise = defaultdict(list)
        noise_level = [0.0, 0.1, 0.2]
        path_noise = [os.path.join(path, "noise", f"degree{degree}", str(i)) for i in noise_level]

        afunc = run_adiabatic if problem == "AD" else run_gridsearch
        def afunc_partial(_noise, _path) :
            return afunc(n_expt = 6, delta_t = 0.01, noise_level = _noise, parameters = sindy_params,  
                            max_workers = max_workers, seed = 20, kind = problem, path = _path)

        with ProcessPoolExecutor(max_workers = max_workers) as executor:   
            result = executor.map(afunc_partial, noise_level, path_noise)

            for alist in result:
                for key, value in zip(["MSE_Prediction", "AIC", "MSE", "complexity", "coefficients", "coefficients_pre_stoichiometry", "actual_coefficients"], alist):
                    if key in ["coefficients", "coefficients_pre_stoichiometry", "actual_coefficients"]:
                        coeff_noise[key].append(value)
                    else:
                        adict_noise[key].extend(value)

        adict_noise["kind"] = problem
        coeff_noise["kind"] = problem
        plot_adict(noise_level, adict_noise, x_label = "noise", path = os.path.join(path, "noise", f"degree{degree}"), title = f"Polynomial degree {degree}")
        plot_coeff(noise_level, coeff_noise, path = os.path.join(path, "noise", f"degree{degree}", f"coefficients"))

    ########################################################################################################################
    if experiments_study :
        # choose dt == 0.05 to see significant difference
    
        print("------"*100)
        print("Starting experiment study")
        experiments = [2, 4, 6]
        adict_experiments = defaultdict(list)
        coeff_experiments = defaultdict(list)
        path_experiments = [os.path.join(path, "experiments", f"degree{degree}", str(i)) for i in experiments]

        afunc = run_adiabatic if problem == "AD" else run_gridsearch
        def afunc_partial(_expt, _path) :
            return afunc(n_expt = _expt, delta_t = 0.05, noise_level = 0, parameters = sindy_params, max_workers = max_workers, seed = 20, 
                                kind = problem, path = _path)

        with ProcessPoolExecutor(max_workers = max_workers) as executor:   
            result = executor.map(afunc_partial, experiments, path_experiments)

            for alist in result:
                for key, value in zip(["MSE_Prediction", "AIC", "MSE", "complexity", "coefficients", "coefficients_pre_stoichiometry", "actual_coefficients"], alist):
                    if key in ["coefficients", "coefficients_pre_stoichiometry", "actual_coefficients"]:
                        coeff_experiments[key].append(value)
                    else:
                        adict_experiments[key].extend(value)
        
        adict_experiments["kind"] = problem
        coeff_experiments["kind"] = problem
        plot_adict(experiments, adict_experiments, x_label = "experiments", path = os.path.join(path, "experiments", f"degree{degree}"), title = f"Polynomial degree {degree}")
        plot_coeff(experiments, coeff_experiments, path = os.path.join(path, "experiments", f"degree{degree}", "coefficients"))

    ########################################################################################################################
    if sampling_study :
    
        print("------"*100)
        print("Starting sampling study")
        sampling = [0.01, 0.05, 0.1]
        adict_sampling = defaultdict(list)
        coeff_sampling = defaultdict(list)
        path_sampling = [os.path.join(path, "sampling", f"degree{degree}", str(i)) for i in sampling]

        afunc = run_adiabatic if problem == "AD" else run_gridsearch
        def afunc_partial(_samp, _path) :
            return afunc(n_expt = 6, delta_t = _samp, noise_level = 0, parameters = sindy_params, max_workers = max_workers, seed = 20, 
                        kind = problem, path = _path)

        with ProcessPoolExecutor(max_workers = max_workers) as executor:   
            result = executor.map(afunc_partial, sampling, path_sampling)

            for alist in result:
                for key, value in zip(["MSE_Prediction", "AIC", "MSE", "complexity", "coefficients", "coefficients_pre_stoichiometry", "actual_coefficients"], alist):
                    if key in ["coefficients", "coefficients_pre_stoichiometry", "actual_coefficients"]:
                        coeff_sampling[key].append(value)
                    else:
                        adict_sampling[key].extend(value)

        adict_sampling["kind"] = problem
        coeff_sampling["kind"] = problem
        plot_adict(sampling, adict_sampling, x_label = "sampling", path = os.path.join(path, "sampling", f"degree{degree}"), title = f"Polynomial degree {degree}")
        plot_coeff(sampling, coeff_sampling, path = os.path.join(path, "sampling", f"degree{degree}", f"coefficients"))

    ########################################################################################################################
    