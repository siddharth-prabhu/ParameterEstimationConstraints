import numpy as np

import pandas as pd
pd.set_option("display.max_columns", 20)
import pysindy as ps
from sklearn.metrics import mean_squared_error, r2_score

import itertools
from dataclasses import dataclass, field
from collections import defaultdict
from functools import reduce

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
    t : np.ndarray

    parameters : dict = field(default_factory = dict)
    model : Optimizer_casadi = field(default_factory = Optimizer_casadi()) 
    include_column : list[list] = field(default = None)
    constraints_dict : dict = field(default_factory = dict)
    ensemble_iterations : int = field(default = 1)

    @staticmethod
    def train_test_split(X : list[np.ndarray], y : list[np.ndarray], X_clean : list[np.ndarray], y_clean : list[np.ndarray], 
                        t : np.ndarray, train_percent : int = 80):
        
        assert np.shape(y) == np.shape(y_clean), "Targets and targets_clean should have same dimensions"
        assert len(y) == len(y_clean) == len(X), "Features, target and target_clean should have the same length"
        sample = len(y)*train_percent//100
    
        return X[:sample], y[:sample], X_clean[:sample], y_clean[:sample], t, X[sample:], y[sample:], X_clean[sample:], y_clean[sample:], t

    def gridsearch(self, display_results : bool = True):

        (self.X_train, self.y_train, self.X_clean_train, self.y_clean_train, self.t_train, self.X_test, self.y_test, 
        self.X_clean_test, self.y_clean_test, self.t_test) = self.train_test_split(self.X, self.y, self.X_clean, self.y_clean, self.t)
        
        result_dict = defaultdict(list)
        parameter_key, parameter_value = zip(*self.parameters.items()) # separate the key value pairs
        combinations = itertools.product(*parameter_value)

        for combination in combinations: # use combinations of values
            param_dict = dict(zip(parameter_key, combination)) # combine the key value pari and fit the model
            self.model.set_params(**param_dict)
            print("--"*100)
            print("Running for parameter combination", param_dict)
            try:
                # self.model.fit(self.X_train, x_dot = self.y_train, quiet = True, multiple_trajectories = True) # for sindy
                self.model.fit(self.X_train, self.y_train, include_column = self.include_column,
                            constraints_dict = self.constraints_dict, ensemble_iterations = self.ensemble_iterations)  
                            # {"mass_balance" : [56.108, 28.05, 56.106, 56.108], "consumption" : [], "formation" : []}) # for casadi

            except Exception as error:
                print(error)
                print("Failed for the parameter combination", param_dict)
                print("--"*100)
                continue
            else:
                result_dict["complexity"].append(self.model.complexity)

                # calculate error
                for key in param_dict:
                    result_dict[key].append(param_dict[key])

                result_dict["MSE_test_pred"].append(self.model.score(self.X_clean_test, self.y_clean_test, metric = mean_squared_error))
                result_dict["MSE_train_pred"].append(self.model.score(self.X_clean_train, self.y_clean_train, metric = mean_squared_error))

                result_dict["r2_test_pred"].append(self.model.score(self.X_clean_test, self.y_clean_test, metric = r2_score))
                result_dict["r2_train_pred"].append(self.model.score(self.X_clean_train, self.y_clean_train, metric = r2_score))

                # add integration results
                try :
                    _integration_test = self.model.simulate(self.X_clean_test, self.t_test)
                    _integration_train = self.model.simulate(self.X_clean_train, self.t_train)
                except:
                    result_dict["MSE_test_sim"].append(np.nan)
                    result_dict["MSE_train_sim"].append(np.nan)

                    result_dict["r2_test_sim"].append(np.nan)
                    result_dict["r2_train_sim"].append(np.nan)

                    result_dict["AIC"].append(np.nan)
                else:
                    result_dict["MSE_test_sim"].append(self.model.score(_integration_test, self.X_clean_test, metric=mean_squared_error, predict = False))
                    result_dict["MSE_train_sim"].append(self.model.score(_integration_train, self.X_clean_train, metric = mean_squared_error, predict = False))

                    result_dict["r2_test_sim"].append(self.model.score(_integration_test, self.X_clean_test, metric = r2_score, predict = False))
                    result_dict["r2_train_sim"].append(self.model.score(_integration_train, self.X_clean_train, metric = r2_score, predict = False))                    

                    result_dict["AIC"].append(np.log((result_dict["MSE_test_sim"][-1] + result_dict["MSE_test_pred"][-1])/2) + result_dict["complexity"][-1])

        # sort value and remove duplicates
        self.df_result = pd.DataFrame(result_dict)
        self.df_result.dropna(inplace=True)
        self.df_result.sort_values(by = ["AIC"], ascending = True, inplace = True, ignore_index = True)
        # self.df_result.drop_duplicates(["r2_test_pred"], keep = "first", inplace = True, ignore_index = True)

        if display_results:
            print(self.df_result.head(10))


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
        tooltips = [("Index", "$index"), ("complexity", "@complexity"), ("MSE", "@MSE_test_pred"), ("r2", "@r2_test_pred"), 
                    ("threshold", "@optimizer__threshold"), ("alpha", "@optimizer__alpha"), ("degree", "@feature_library__degree")]

        fig_mse = figure(tooltips = tooltips)
        fig_mse.scatter(x = "complexity", y = "MSE_test_pred", size = 8, source = source)
        self._bokeh_plot(fig_mse, "Complexity", "MSE Prediction", title)
        
        fig_r2 = figure(tooltips = tooltips)
        fig_r2.scatter(x = "complexity", y = "r2_test_pred", size = 8, source = source)
        self._bokeh_plot(fig_r2, "Complexity", "R squared Prediction", title)
        
        fig_mse_sim = figure(tooltips = tooltips)
        fig_mse_sim.scatter(x = "complexity", y = "MSE_test_sim", size = 8, source = source)
        self._bokeh_plot(fig_mse_sim, "Complexity", "MSE Simulation", title)
        
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

    params = {"optimizer__threshold": [0.01, 0.1], 
        "optimizer__alpha": [0, 0.01], 
        "feature_library__include_bias" : [False, True],
        "feature_library__degree": [1, 2]}

    opt = HyperOpt(features, target, features, target, t_span, params, Optimizer_casadi(solver_dict = {"ipopt.print_level" : 0, "print_time":0}), 
                    include_column = [[0, 1], [0, 2], [0, 3]], constraints_dict = {"mass_balance" : [], "consumption" : [], "formation" : [3], 
                                "stoichiometry" : np.array([-1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 2]).reshape(4, -1)})
    opt.gridsearch()
    opt.plot()


