# type: ignore

import os
from collections import defaultdict
from typing import Optional, List
import itertools as it
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
from data import reaction_data
from utils import coefficients_plot


# only runs hyperparameter optimization for different training conditions

parser = argparse.ArgumentParser("ParameterEstimationSINDy")
parser.add_argument("--noise_study", type = str, default = "", help = "Comma separated varying noise levels") 
parser.add_argument("--experiments_study", type = str, default = "", help = "Comma separated varying initial conditions")   
parser.add_argument("--sampling_study", type = str, default = "", help = "Comma separated varying sampling frequency")
parser.add_argument("--mechanism_study", type = int, default = 0, help = "Whether to perform mechanism simulations or not")
parser.add_argument("--stiffness_study", type = int, default = 0, help = "Whether to perform stiffness simulations or not")
parser.add_argument("--stlsq_threshold", type = str, default = "0.01", help = "Comma separated values of thersholding parameter")
parser.add_argument("--stlsq_alpha", type = str, default = "0", help = "Comma separated values of regularization parameter")            
parser.add_argument("--degree", type = str, default = "1", help = "Comma separated varying polynomial degree of terms in functional library")
parser.add_argument("--problem", choices = ["NLS", "LS", "AD"], type = str, default = "LS", help = "The type of problem to solve")
parser.add_argument("--system", choices = ["kosir", "menten", "carb"], type = str, default = "kosir", help = "The type of reaction network to run")
parser.add_argument("--nexpt", type = int, default = 6, help = "The number of independant experiments")
parser.add_argument("--dt", type = float, default = 0.01, help = "The sampling time of independant experiments")
parser.add_argument("--max_workers", default = 1, type = int)
parser.add_argument("--seed", default = 20, type = int)
pargs = parser.parse_args()


def run_gridsearch(system : str, n_expt : int, delta_t : float, noise_level : float, parameters : dict, kind : str = "LS", 
                    max_workers : Optional[int] = None, seed : int = 12345, path : Optional[str] = None):
    

    plugin_dict = {"ipopt.print_level" : 0, "print_time": 0, "ipopt.sb" : "yes", "ipopt.max_iter" : 3000}
    solver_dict = {"solver" : "ipopt", "tol" : 1e-4}
    
    # generate clean testing data to be used later for calculating errors
    time_span_clean = np.arange(0, 10, 0.01)
    arguments_temperature = reaction_data[f"kinetic_{system}"].arguments
    while max(2, n_expt) > len(arguments_temperature) : 
        arguments_temperature.extend(arguments_temperature[:n_expt - len(arguments_temperature)])
    
    arguments_clean = arguments_temperature[:1] if kind == "LS" else arguments_temperature[:2]
    model = DynamicModel(f"kinetic_{system}", time_span_clean, n_expt = 2, arguments = arguments_clean)
    features_clean = model.integrate()
    target_clean = model.actual_derivative # use actual derivatives for testing 
    arguments_clean = model.arguments

    # generate training data with varying experiments and sampling time
    time_span = np.arange(0, 10, delta_t)
    arguments = arguments_temperature[:1] if kind == "LS" else arguments_temperature[:n_expt]
    model = DynamicModel(f"kinetic_{system}", time_span, n_expt = n_expt, arguments = arguments, seed = seed)
    features = model.integrate()
    features = model.add_noise(0, noise_level)
    target = model.approx_derivative
    arguments = model.arguments

    actual_coefficients = [model.coefficients(pre_stoichiometry = True if kind == "NLS" else False)]
    mse_pred, aic, mse_sim, comp, coefficients, coefficients_pre_stoichiometriy = [], [], [], [], [], []
    for status in ["unconstraint", "massbalance", "chemistry", "sindy"]:

        # for NLS no need to perform unconstrained and mass balance formulation
        if kind == "NLS" and status in ["unconstraint", "massbalance"]:
            continue

        if status == "massbalance" : # mass balance constraints 
            include_column = [] # 
            constraints_dict = {"stoichiometry" : reaction_data[f"kinetic_{system}"].stoichiometry_mass_balance}
        elif status == "chemistry" : # chemistry constraints
            include_column = reaction_data[f"kinetic_{system}"].include_column
            constraints_dict = {"stoichiometry" : reaction_data[f"kinetic_{system}"].stoichiometry}
        elif status == "sindy":
            # compare derivative free method with sindy for NLS
            if kind == "NLS":
                include_column = reaction_data[f"kinetic_{system}"].include_column
                constraints_dict = {"stoichiometry" : reaction_data[f"kinetic_{system}"].stoichiometry}
            else:
                include_column = None
                constraints_dict = {"stoichiometry" : reaction_data[f"kinetic_{system}"].stoichiometry_unconstrained}

        else : # unconstrained formulation
            include_column = None
            constraints_dict = {"stoichiometry" : reaction_data[f"kinetic_{system}"].stoichiometry_unconstrained}
        
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


def run_adiabatic(system : str, n_expt : int, delta_t : float, noise_level : float, parameters : dict, kind : str = "LS", 
                    max_workers : Optional[int] = None, seed : int = 12345, path : Optional[str] = None):

    # lower the tolerance for noise
    plugin_dict = {"ipopt.print_level" : 0, "print_time": 0, "ipopt.sb" : "yes", "ipopt.max_iter" : 3000, "ipopt.tol" : 1e-5}

    if noise_level > 0 : plugin_dict["ipopt.tol"] = 1e-5 # set lower tolerance for noise

    # generate clean testing data to be used later for calculating errors
    time_span_clean = np.arange(0, 10, 0.01)
    model = DynamicModel("kinetic_kosir_temperature", time_span_clean, n_expt = 2)
    features_clean = model.integrate()
    target_clean = [tar[:, :-1] for tar in model.actual_derivative] # use actual derivatives for testing 
    arguments_clean = [np.column_stack((feat[:, -1], np.tile(8.314, (len(feat), 1)))) for feat in features_clean]

    # generate training data with varying experiments and sampling time
    time_span = np.arange(0, 10, delta_t)
    model = DynamicModel("kinetic_kosir_temperature", time_span, n_expt = n_expt)
    features = model.integrate()
    features = model.add_noise(0, noise_level)
    target = [tar[:, :-1] for tar in model.approx_derivative]
    arguments = [np.column_stack((feat[:, -1], np.tile(8.314, (len(feat), 1)))) for feat in features]
    
    include_column = reaction_data["kinetic_kosir"].include_column 
    constraints_dict = {"stoichiometry" : reaction_data["kinetic_kosir"].stoichiometry}

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
                        "seed" : seed, "derivative_free" : False if status == "sindy" else True},
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
    
    width = (x[1] - x[0])/5 if len(x) > 1 else 0.2 # the width of bar plots
    kind = adict.pop("kind", False)
    assert kind, "kind is not provided while plotting"
    
    with plt.style.context(["science", "notebook", "vibrant"]):
        for key, value in adict.items():
            value = np.array(value)

            if kind == "LS":
                plt.bar(x - 1.5*width, value[3::4], label = "sindy", width = width, align = "center")
                plt.bar(x - 0.5*width, value[0::4], label = "unconstrained", width = width, align = "center")
                plt.bar(x + 0.5*width, value[1::4], label = "mass balance", width = width, align = "center")
                plt.bar(x + 1.5*width, value[2::4], label = "chemistry", width = width, align = "center")
                
            elif kind == "AD":
                plt.bar(x - 0.5*width, value[0::2], label = "derivative", width = width, align = "center")
                plt.bar(x + 0.5*width, value[1::2], label = "integral", width = width, align = "center")
            else:
                plt.bar(x - 0.5*width, value[1::2], label = "sindy", width = width, align = "center")
                plt.bar(x + 0.5*width, value[0::2], label = "chemistry", width = width, align = "center")
                

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

    map_afunc_string = lambda _astring, afunc : list(map(afunc, _astring.split(",")))

    problem = pargs.problem # the type of problem to solve. Either LS or NLS or Adiabatic 
    max_workers = None if pargs.max_workers <= 0 else pargs.max_workers 
    
    path = os.path.join("log", pargs.system, problem)
    sindy_params = {"optimizer__threshold": map_afunc_string(pargs.stlsq_threshold, float),
        "optimizer__alpha": map_afunc_string(pargs.stlsq_alpha, float), # 0, 0.01, 0.1
        "feature_library__include_bias" : [False],
        "feature_library__degree": map_afunc_string(pargs.degree, int)
        }

    ########################################################################################################################
    if len(pargs.noise_study) > 0 :

        print("------"*100)
        print("Starting experiment study")
        adict_noise = defaultdict(list)
        coeff_noise = defaultdict(list)
        noise_level = map_afunc_string(pargs.noise_study, float)
        path_noise = [os.path.join(path, "noise", str(i)) for i in noise_level]

        afunc = run_adiabatic if problem == "AD" else run_gridsearch
        def afunc_partial(_noise, _path) :
            return afunc(system = pargs.system, n_expt = pargs.nexpt, delta_t = pargs.dt, noise_level = _noise, parameters = sindy_params,  
                            max_workers = max_workers, seed = pargs.seed, kind = problem, path = _path)

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
        plot_adict(noise_level, adict_noise, x_label = "noise", path = os.path.join(path, "noise"))
        plot_coeff(noise_level, coeff_noise, path = os.path.join(path, "noise", "coefficients"))

    ########################################################################################################################
    if len(pargs.experiments_study) > 0 :
    
        print("------"*100)
        print("Starting experiment study")
        experiments = map_afunc_string(pargs.experiments_study, int)
        adict_experiments = defaultdict(list)
        coeff_experiments = defaultdict(list)
        path_experiments = [os.path.join(path, "experiments", str(i)) for i in experiments]

        afunc = run_adiabatic if problem == "AD" else run_gridsearch
        def afunc_partial(_expt, _path) :
            return afunc(system = pargs.system, n_expt = _expt, delta_t = pargs.dt, noise_level = 0, parameters = sindy_params, max_workers = max_workers, seed = pargs.seed, 
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
        plot_adict(experiments, adict_experiments, x_label = "experiments", path = os.path.join(path, "experiments"))
        plot_coeff(experiments, coeff_experiments, path = os.path.join(path, "experiments", "coefficients"))

    ########################################################################################################################
    if len(pargs.sampling_study) > 0 :
    
        print("------"*100)
        print("Starting sampling study")
        sampling = map_afunc_string(pargs.sampling_study, float)
        adict_sampling = defaultdict(list)
        coeff_sampling = defaultdict(list)
        path_sampling = [os.path.join(path, "sampling", str(i)) for i in sampling]

        afunc = run_adiabatic if problem == "AD" else run_gridsearch
        def afunc_partial(_samp, _path) :
            return afunc(system = pargs.system, n_expt = pargs.nexpt, delta_t = _samp, noise_level = 0, parameters = sindy_params, max_workers = max_workers, 
                        seed = pargs.seed, kind = problem, path = _path)

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
        plot_adict(sampling, adict_sampling, x_label = "sampling", path = os.path.join(path, "sampling"))
        plot_coeff(sampling, coeff_sampling, path = os.path.join(path, "sampling", f"coefficients"))

    ########################################################################################################################
    # Additional mechanisms in kinetic_kosir and LS problem
    if pargs.mechanism_study > 0 : 

        n_expt = 6
        plugin_dict = {"ipopt.print_level" : 0, "print_time": 0, "ipopt.sb" : "yes", "ipopt.max_iter" : 3000}
        solver_dict = {"solver" : "ipopt", "tol" : 1e-5}
        
        # generate clean testing data to be used later for calculating errors
        time_span_clean = np.arange(0, 10, 0.01)
        arguments_clean = [(373, 8.314)]
        model = DynamicModel("kinetic_kosir", time_span_clean, n_expt = n_expt, arguments = arguments_clean)
        features_clean = model.integrate()
        target_clean = model.actual_derivative # use actual derivatives for testing 
        arguments_clean = model.arguments

        # generate training data with varying experiments and sampling time
        time_span = np.arange(0, 10, 0.01)
        arguments = [(373, 8.314)] 
        model = DynamicModel("kinetic_kosir", time_span, n_expt = n_expt, arguments = arguments, seed = pargs.seed)
        features = model.integrate()
        target = model.approx_derivative
        arguments = model.arguments

        actual_coefficients = model.coefficients(pre_stoichiometry = True)

        # Chemistry information corresponds to the following assumed reaction network
        # species = [A, B, C, D]
            # reactions = [
            #       A     -- k0> 2B;
            #       A <k2 -- k1> C;
            #       A <k4 -- k3> D;
            #       C     -- k4> D ]
        include_column = [[0, 1], [0, 2], [0, 3], [2, 3]]
        constraints_dict = {"stoichiometry" : np.array([-1, -1, -1, 0, 2, 0, 0, 0, 0, 1, 0, -1, 0, 0, 1, 1]).reshape(4, -1)}

        # does grid_serch over parameters 
        _path = os.path.join("log", "kosir", "Mechanism")
        opt = HyperOpt(features, target, features_clean, target_clean, 
                        time_span, time_span_clean, parameters = sindy_params, 
                        model = Optimizer_casadi(plugin_dict = plugin_dict, solver_dict = solver_dict),
                        arguments = None, arguments_clean =  None, 
                        meta = {"include_column" : include_column, "constraints_dict" : constraints_dict,
                        "seed" : pargs.seed, "derivative_free" : True}, 
                        _dir = _path)

        opt.gridsearch(max_workers = max_workers)
        opt.plot(filename = 'Gridsearch.html', path = _path, title = "")
        _discovered_coefficients = opt.df_result["coefficients_pre_stoichiometry"][0]

        # plot coefficients pre_stoichiometry 
        actual_coefficients.append({_key : 0. for _key in _discovered_coefficients[-1].keys()}) # append dummy coefficients for the spurious reaction
        coefficients_plot(actual_coefficients, [_discovered_coefficients], expt_names = ["chemistry"], path = os.path.join(_path, "coefficients.png"))

    ########################################################################################################################
    # Stiffness simulation of Menten reaction network
    if pargs.stiffness_study > 0 : 

        stiff_params = {"optimizer__threshold" : [0.01, 0.05, 0.1],
            "optimizer__alpha" : [1e-4, 1e-3, 1e-2, 1e-1, 1, 10], # 0, 0.01, 0.1
            "feature_library__include_bias" : [False],
            "feature_library__degree" : [1, 2]
            }
        
        print("------"*100)
        print("Starting stiffness study")
        
        _path = os.path.join("log", "menten", "Stiffness")

        for _ind, _params in enumerate([
            (0.1, 0.2, 0.3), (1, 2, 0.3), (10, 20, 0.3), (100, 200, 0.3), (100, 200, 3), (1000, 2000, 0.3)
            ]) :
            
            adict_stiff = defaultdict(list)
            coeff_stiff = defaultdict(list)
            adict_stiff["kind"] = "LS"
            coeff_stiff["kind"] = "LS"

            # This prevents multiprocessing (Workaround is to pass reaction_data as an argument to the function)
            reaction_data["kinetic_menten"] = reaction_data["kinetic_menten"]._replace(arguments = [_params])

            alist = run_gridsearch(system = "menten", n_expt = 10, delta_t = 0.01, noise_level = 0, parameters = stiff_params, 
                max_workers = max_workers, seed = pargs.seed, kind = "LS", path = os.path.join(_path, f"{_ind}"))

            for key, value in zip(["MSE_Prediction", "AIC", "MSE", "complexity", "coefficients", "coefficients_pre_stoichiometry", "actual_coefficients"], alist) :
                
                coeff_stiff[key].append(value) if key in ["coefficients", "coefficients_pre_stoichiometry", "actual_coefficients"] else adict_stiff[key].extend(value) 

            plot_adict([str(_ind)], adict_stiff, x_label = "", path = os.path.join(_path, f"{_ind}"))
            plot_coeff([str(_ind)], coeff_stiff, path = os.path.join(_path, f"{_ind}", "coefficients"))
            