import os
import itertools
from dataclasses import dataclass, field
from collections import defaultdict
from functools import reduce
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, List, Callable, Tuple

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 20)
from sklearn.metrics import mean_squared_error, r2_score
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource
from bokeh.layouts import column, row

from GenerateData import DynamicModel
from Optimizer import Optimizer_casadi
from energy import EnergySindy
from adiabatic import AdiabaticSindy
from FunctionalLibrary import FunctionalLibrary

@dataclass()
class HyperOpt():

    X : List[np.ndarray] 
    y : List[np.ndarray]
    X_clean : List[np.ndarray]
    y_clean : List[np.ndarray]
    time : np.ndarray
    time_clean : np.ndarray 
    model : Callable  
    arguments : List[np.ndarray] = field(default = None)
    arguments_clean : List[np.ndarray] = field(default = None)

    parameters : dict = field(default_factory = dict)
    meta : dict = field(default_factory = dict)

    def __post_init__(self):
        assert len(self.X) > 1 and len(self.X_clean) > 1, "Need atleast 2 experiments for hyperparameter optimization"
        assert len(self.X) == len(self.y) and len(self.X_clean) == len(self.y_clean), "Features and targets should have the same length"
        if self.arguments:
            assert len(self.arguments) == len(self.X), f"Arguments (length = {len(self.arguments)}) should be equal to the number of initial conditions (length = {len(self.X)})"
            assert len(self.arguments_clean) == len(self.X_clean), f"Clean arguments (length = {len(self.arguments_clean)}) should be equal to the number of initial conditions (length = {len(self.X_clean)})"  

    @staticmethod
    def train_test_split(arrays : Tuple[List[np.ndarray]], train_split_percent : int = 80) -> Tuple[List[np.array]]:
        
        dataset_size = len(arrays[0])
        assert all(len(_) == dataset_size for _ in arrays), "All features should have the same length"
        
        # each type (noisy and clean) of data can have different data points 
        sample = dataset_size*train_split_percent//100
        
        return (*[arr[:sample] for arr in arrays], *[arr[sample :] for arr in arrays])

    def gridsearch(self, train_split_percent : int = 80, max_workers : Optional[int] = None, display_results : bool = True):

        self.X_train, self.y_train, self.X_test, self.y_test = self.train_test_split((self.X, self.y), train_split_percent)
        self.X_clean_train, self.y_clean_train, self.X_clean_test, self.y_clean_test = self.train_test_split((self.X_clean, self.y_clean), train_split_percent)
        
        if self.arguments and self.arguments_clean:
            self.arguments_train, self.arguments_test = self.train_test_split((self.arguments, ), train_split_percent)
            self.arguments_clean_train, self.arguments_clean_test = self.train_test_split((self.arguments_clean, ), train_split_percent)
        elif not (self.arguments and self.arguments_clean):
            self.arguments_train, self.arguments_test = None, None
            self.arguments_clean_train, self.arguments_clean_test = None, None
        else:
            raise ValueError("Both arguments and arguments_clean should either be specified or not")
            
        self.t_train, self.t_clean = self.time, self.time_clean

        result_dict = defaultdict(list)
        parameter_key, parameter_value = zip(*self.parameters.items()) # separate the key value pairs
        combinations = itertools.product(*parameter_value)

        # using multiple cores to run gridsearch
        with ProcessPoolExecutor(max_workers = max_workers) as executor:
            
            _gridsearch_results = executor.map(self._gridsearch_optimization, combinations, itertools.repeat(parameter_key))
            individual_results_keyword = ["complexity", "MSE_Prediction", "MSE_train_pred", "r2_test_pred", "r2_train_pred", "MSE_Integration", "MSE_train_sim", 
                    "r2_test_sim", "r2_train_sim", "AIC", "iterations", "coefficients", "coefficients_pre_stoichiometry"]
            
            for individual_results in _gridsearch_results:    
                if individual_results : # exceptions in optimizer returns None
                    param_dict = individual_results[0]
                    for key, value in param_dict.items():
                        result_dict[key].append(value)
                    
                    for key, value in zip(individual_results_keyword, individual_results[1:]):
                        result_dict[key].append(value)
                else:
                    continue

        # sort value and remove duplicates
        self.df_result = pd.DataFrame(result_dict)
        self.df_result.dropna(inplace=True)

        # there are no models that were discovered. Either the solver failed or the thresholding eliminated all the parameters
        if not self.df_result.empty:
            self.df_result.sort_values(by = ["AIC"], ascending = True, inplace = True, ignore_index = True)
            
        # self.df_result.drop_duplicates(["r2_test_pred"], keep = "first", inplace = True, ignore_index = True)

        if display_results:
            print(self.df_result.head(10))

    # function to be looped for multiprocessing
    def _gridsearch_optimization(self, combination, parameter_key):
        
        param_dict = dict(zip(parameter_key, combination)) # combine the key value part and fit the model
        self.model.set_params(**param_dict)
        print("--"*100)
        print("Running for parameter combination", param_dict)

        try:
            self.model.fit(self.X_train, self.y_train, time_span = self.t_train, arguments = self.arguments_train, **self.meta)
        
        except Exception as error:
            print(error)
            print("Failed for the parameter combination", param_dict)
            print("--"*100)
        else :
            print(f"Model for parameter combination {param_dict}", self.model.print(), sep = "\n")
            complexity = self.model.complexity
            MSE_test_pred = self.model.score(self.X_clean_test, self.y_clean_test, self.t_clean, metric = mean_squared_error, model_args = self.arguments_clean_test)
            MSE_train_pred = self.model.score(self.X_clean_train, self.y_clean_train, self.t_clean, metric = mean_squared_error, model_args = self.arguments_clean_train)

            r2_test_pred = self.model.score(self.X_clean_test, self.y_clean_test, self.t_clean, metric = r2_score, model_args = self.arguments_clean_test)
            r2_train_pred = self.model.score(self.X_clean_train, self.y_clean_train, self.t_clean, metric = r2_score, model_args = self.arguments_clean_train)

            try :
                # add integration results
                _, (MSE_test_sim, r2_test_sim) = self.model.simulate(self.X_clean_test, self.t_clean, model_args = self.arguments_clean_test, calculate_score = True, metric = [mean_squared_error, r2_score])
                _, (MSE_train_sim, r2_train_sim) = self.model.simulate(self.X_clean_train, self.t_clean, model_args = self.arguments_clean_train, calculate_score = True, metric = [mean_squared_error, r2_score])
            
            except Exception as error:
                print(f"Failed calculating integration errors because of {error}")
                MSE_test_sim = np.nan
                MSE_train_sim = np.nan
                r2_test_sim = np.nan
                r2_train_sim = np.nan
                AIC = np.nan

            else:                  
                AIC = 2*np.log(MSE_test_sim) + complexity # AIC should be based on testing data and not clean data
                
            return [param_dict, complexity, MSE_test_pred, MSE_train_pred, r2_test_pred, r2_train_pred, MSE_test_sim, MSE_train_sim, 
                    r2_test_sim, r2_train_sim, AIC, self.model.adict["iterations"], self.model.adict["coefficients_dict"], self.model.adict["coefficients_pre_stoichiometry_dict"]]

    @staticmethod
    def _bokeh_plot(fig : figure, x_label : str, y_label : str, title : str, height : int = 400, width : int = 700):

        fig.xaxis.axis_label = x_label
        fig.xaxis.axis_label_text_font_style = "bold"
        fig.yaxis.axis_label = y_label
        fig.yaxis.axis_label_text_font_style = "bold"
        fig.title.text = title
        fig.title.text_color = "blue"
        fig.title.align = "center"
        fig.title.text_font_size = "18px"
        fig.plot_height = height
        fig.plot_width = width
        fig.outline_line_color = "black"
        fig.margin = (5, 5, 5, 5) #top, right, bottom, left

    # bokeh plotting
    def plot(self, filename : str = "Gridsearch_results.html", path : Optional[str] = None, title : str = "Concentration vs time"):

        # do not run if dataframe is empty
        if self.df_result.empty:
            return

        # capture r2 values between 0 and 1
        updated_results = self.df_result[["complexity", "MSE_Prediction", "MSE_train_pred", "r2_test_pred", "r2_train_pred", "MSE_Integration", "MSE_train_sim", 
                    "r2_test_sim", "r2_train_sim", "AIC", "iterations"]]
        updated_results = updated_results.loc[(updated_results["r2_test_pred"] >= 0) & (updated_results["r2_test_pred"] <= 1) & 
                                               (updated_results["r2_test_sim"] >= 0) & (updated_results["r2_test_sim"] <= 1 )]
        
        source = ColumnDataSource(updated_results)
        tooltips = [("Index", "$index"), ("complexity", "@complexity"), ("MSE", "@MSE_Prediction"), ("r2", "@r2_test_pred"), 
                    ("threshold", "@optimizer__threshold"), ("alpha", "@optimizer__alpha"), ("degree", "@feature_library__degree")]

        fig_mse = figure(tooltips = tooltips)
        fig_mse.scatter(x = "complexity", y = "MSE_Prediction", size = 8, source = source)
        self._bokeh_plot(fig_mse, "Complexity", "MSE Prediction", title)
        
        fig_r2 = figure(tooltips = tooltips)
        fig_r2.scatter(x = "complexity", y = "r2_test_pred", size = 8, source = source)
        self._bokeh_plot(fig_r2, "Complexity", "R squared Prediction", title)
        
        fig_mse_sim = figure(tooltips = tooltips)
        fig_mse_sim.scatter(x = "complexity", y = "MSE_Integration", size = 8, source = source)
        self._bokeh_plot(fig_mse_sim, "Complexity", "MSE Integration", title)
        
        fig_r2_sim = figure(tooltips = tooltips)
        fig_r2_sim.scatter(x = "complexity", y = "r2_test_sim", size = 8, source = source)
        self._bokeh_plot(fig_r2_sim, "Complexity", "R squared Simulation", title)
        
        fig_aic = figure(tooltips = tooltips)
        fig_aic.scatter(x = "complexity", y = "AIC", size = 8, source = source)
        self._bokeh_plot(fig_aic, "Complexity", "AIC", title)

        grid = column(row(fig_mse, fig_r2), row(fig_mse_sim, fig_r2_sim), fig_aic)

        path = os.getcwd() if not path else os.path.join(os.getcwd(), path)
        if not os.path.exists(path):
            os.makedirs(path)

        output_file(os.path.join(path, filename))
        save(grid)


if __name__ == "__main__":

    params = {"optimizer__threshold": [0.01], 
        "optimizer__alpha": [0.01], 
        "feature_library__include_bias" : [False],
        "feature_library__degree": [1]}

    """
    t_span = np.arange(0, 5, 0.01)
    model_actual = DynamicModel("kinetic_kosir", t_span, n_expt = 2, arguments = [(373, 8.314)])
    features = model_actual.integrate()
    target = model_actual.approx_derivative
    # model_actual.plot(features[-1], t_span, "Time", "Concentration", ["A", "B", "C", "D"])

    opt = HyperOpt(features, target, features, target, t_span, t_span, model = Optimizer_casadi(plugin_dict = {"ipopt.print_level" : 0, "print_time":0, "ipopt.sb" : "yes"}), 
                    parameters = params)
    opt.gridsearch(max_workers = 2)

    
    ## testing for temperature varying data
    t_span = np.arange(0, 5, 0.01)
    model = DynamicModel("kinetic_kosir", t_span, n_expt = 20)
    features = model.integrate() # list of features
    target = model.approx_derivative # list of target value
    features = model.add_noise(0, 0.0)
    target = model.approx_derivative
    arguments = model.arguments

    opt = HyperOpt(features, target, features, target, t_span, t_span, model = EnergySindy(FunctionalLibrary(1) , alpha = 0.1, threshold = 0.5, solver_dict={"max_iter" : 5000}, 
                            plugin_dict = {"ipopt.print_level" : 5, "print_time":5, "ipopt.sb" : "yes", "ipopt.max_iter" : 10000, "ipopt.tol" : 1e-6}), 
                    parameters = params, arguments = arguments, arguments_clean = arguments)

    opt.gridsearch(max_workers = 2)
    
    ## testing for adiabatic conditions

    time_span = np.arange(0, 5, 0.1)
    n_expt = 2
    model = DynamicModel("kinetic_kosir_temperature", time_span, n_expt = n_expt)
    features = model.integrate()
    derivative = model.actual_derivative
    target = [tar[:, :-1] for tar in features]
    arguments = [np.column_stack((features[i][:, -1], np.ones(len(features[i]))*8.314)) for i in range(n_expt)]
    plugin_dict = {"ipopt.print_level" : 5, "print_time":5, "ipopt.sb" : "yes", "ipopt.max_iter" : 3000, "ipopt.tol" : 1e-6}
    
    stoichiometry = np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0], dtype = int).reshape(4, -1) # chemistry constraints
    include_column = [[0, 2], [0, 3], [0, 1]] # chemistry constraints

    print("--"*20)
    opt = HyperOpt(features, target, features, target, time_span, time_span, AdiabaticSindy(plugin_dict = plugin_dict), 
                    arguments, arguments, parameters = params,  
                    meta = {"include_column" : include_column, "constraints_dict" : {"stoichiometry" : stoichiometry}, "shooting_horizon" : 5})
    opt.gridsearch(max_workers = 1)
    """

    ## testing for menten system

    t_span = np.arange(0, 20, 0.01)
    model_actual = DynamicModel("kinetic_menten", t_span, n_expt = 2, arguments = [(0.1, 0.2, 0.3)])
    features = model_actual.integrate()
    target = model_actual.approx_derivative
    
    opt = HyperOpt(features, target, features, target, t_span, t_span, model = Optimizer_casadi(plugin_dict = {"ipopt.print_level" : 0, "print_time":0, "ipopt.sb" : "yes"}), 
                    parameters = params)
    opt.gridsearch(max_workers = 2)