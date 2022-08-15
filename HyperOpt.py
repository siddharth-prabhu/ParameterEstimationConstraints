import numpy as np

import pandas as pd
pd.set_option("display.max_columns", 20)
from sklearn.metrics import mean_squared_error, r2_score

import itertools
from dataclasses import dataclass, field
from collections import defaultdict
from functools import reduce
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource
from bokeh.layouts import column, row

from GenerateData import DynamicModel
from Optimizer import Optimizer_casadi

@dataclass()
class HyperOpt():

    X : list[np.ndarray] 
    y : list[np.ndarray]
    X_clean : list[np.ndarray]
    y_clean : list[np.ndarray]
    time : np.ndarray
    time_clean : np.ndarray 

    parameters : dict = field(default_factory = dict)
    model : Optimizer_casadi = field(default_factory = Optimizer_casadi) 
    include_column : list[list] = field(default = None)
    constraints_dict : dict = field(default_factory = dict)
    ensemble_iterations : int = field(default = 1)
    seed : int = field(default = 12345)

    def __post_init__(self):
        assert len(self.X) > 1 and len(self.X_clean) > 1, "Need atleast 2 experiments for hyperparameter optimization"

    @staticmethod
    def train_test_split(X : list[np.ndarray], y : list[np.ndarray], X_clean : list[np.ndarray], y_clean : list[np.ndarray], 
                        t : np.ndarray, t_clean : np.ndarray, train_split_percent : int = 80):
        
        t_len, t_clean_len = len(t), len(t_clean)
        assert sum(map(lambda x : len(x[0]) == len(x[1]) == t_len, zip(X, y))) == len(X) == len(y), "Features and targets should match with time"
        assert sum(map(lambda x : len(x[0]) == len(x[1]) == t_clean_len, zip(X_clean, y_clean))) == len(X_clean) == len(y_clean), "Clean features and clean targets should match with clean time"
        
        # each type (noisy and clean) of data can have different data points 
        sample = len(y)*train_split_percent//100
        sample_clean = len(y_clean)*train_split_percent//100

        return X[:sample], y[:sample], X_clean[:sample_clean], y_clean[:sample_clean], t, X[sample:], y[sample:], X_clean[sample_clean:], y_clean[sample_clean:], t_clean

    def gridsearch(self, train_split_percent : int = 80, max_workers : Optional[int] = None, display_results : bool = True):

        (self.X_train, self.y_train, self.X_clean_train, self.y_clean_train, self.t_train, self.X_test, self.y_test, 
        self.X_clean_test, self.y_clean_test, self.t_clean) = self.train_test_split(self.X, self.y, self.X_clean, self.y_clean, self.time, self.time_clean, train_split_percent)
        
        result_dict = defaultdict(list)
        parameter_key, parameter_value = zip(*self.parameters.items()) # separate the key value pairs
        combinations = itertools.product(*parameter_value)

        # using multiple cores to run gridsearch
        with ProcessPoolExecutor(max_workers = max_workers) as executor:
            
            _gridsearch_results = executor.map(self._gridsearch_optimization, combinations, itertools.repeat(parameter_key), itertools.repeat(max_workers))
            individual_results_keyword = ["complexity", "MSE_Prediction", "MSE_train_pred", "r2_test_pred", "r2_train_pred", "MSE_Integration", "MSE_train_sim", 
                    "r2_test_sim", "r2_train_sim", "AIC", "iterations"]
            
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
        self.df_result.sort_values(by = ["AIC"], ascending = True, inplace = True, ignore_index = True)
        # self.df_result.drop_duplicates(["r2_test_pred"], keep = "first", inplace = True, ignore_index = True)

        if display_results:
            print(self.df_result.head(10))

    # function to be looped for multiprocessing
    def _gridsearch_optimization(self, combination, parameter_key, max_workers):
        
        param_dict = dict(zip(parameter_key, combination)) # combine the key value part and fit the model
        self.model.set_params(**param_dict)
        print("--"*100)
        print("Running for parameter combination", param_dict)

        try:
            self.model.fit(self.X_train, self.y_train, include_column = self.include_column, constraints_dict = self.constraints_dict, 
                        ensemble_iterations = self.ensemble_iterations, max_workers = max_workers, seed = self.seed)  

        except Exception as error:
            print(error)
            print("Failed for the parameter combination", param_dict)
            print("--"*100)
        else :
            print(f"Model for parameter combination {param_dict}", self.model.print(), sep = "\n")
            complexity = self.model.complexity
            MSE_test_pred = self.model.score(self.X_clean_test, self.y_clean_test, metric = mean_squared_error)
            MSE_train_pred = self.model.score(self.X_clean_train, self.y_clean_train, metric = mean_squared_error)

            r2_test_pred = self.model.score(self.X_clean_test, self.y_clean_test, metric = r2_score)
            r2_train_pred = self.model.score(self.X_clean_train, self.y_clean_train, metric = r2_score)

            # add integration results
            try :
                _integration_test = self.model.simulate(self.X_clean_test, self.t_clean)
                _integration_train = self.model.simulate(self.X_clean_train, self.t_clean)
            
            except:
                MSE_test_sim = np.nan
                MSE_train_sim = np.nan
                r2_test_sim = np.nan
                r2_train_sim = np.nan
                AIC = np.nan

            else:
                MSE_test_sim = self.model.score(_integration_test, self.X_clean_test, metric=mean_squared_error, predict = False)
                MSE_train_sim = self.model.score(_integration_train, self.X_clean_train, metric = mean_squared_error, predict = False)

                r2_test_sim = self.model.score(_integration_test, self.X_clean_test, metric = r2_score, predict = False)
                r2_train_sim = self.model.score(_integration_train, self.X_clean_train, metric = r2_score, predict = False)                   

                AIC = 2*np.log((MSE_test_sim + MSE_test_pred)/2) + complexity
            
            return [param_dict, complexity, MSE_test_pred, MSE_train_pred, r2_test_pred, r2_train_pred, MSE_test_sim, MSE_train_sim, 
                    r2_test_sim, r2_train_sim, AIC, self.model.adict["iterations"]]

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
    def plot(self, filename : str = "saved_data\Gridsearch_results.html", title : str = "Concentration vs time"):
        # capture r2 values between 0 and 1
        updated_results = self.df_result
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
        output_file(filename)
        save(grid)


if __name__ == "__main__":

    t_span = np.arange(0, 5, 0.01)
    model_actual = DynamicModel("kinetic_kosir", t_span, [], 2)
    features = model_actual.integrate(())
    target = model_actual.approx_derivative
    # model_actual.plot(features[-1], t_span, "Time", "Concentration", ["A", "B", "C", "D"])

    params = {"optimizer__threshold": [0.5], 
        "optimizer__alpha": [0], 
        "feature_library__include_bias" : [False],
        "feature_library__degree": [1]}

    opt = HyperOpt(features, target, features, target, t_span, t_span, params, Optimizer_casadi(solver_dict = {"ipopt.print_level" : 0, "print_time":0, "ipopt.sb" : "yes"}))
    opt.gridsearch(max_workers = 2)
    # opt.plot()


