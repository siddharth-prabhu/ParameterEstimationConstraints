import numpy as np
np.random.seed(10)
import pandas as pd
import pysindy as ps
from sklearn.metrics import mean_squared_error, r2_score

import itertools
from dataclasses import dataclass, field
from collections import defaultdict
from functools import reduce

from bokeh.plotting import show, figure, output_file

from GenerateData import DynamicModel
from Optimizer import Optimizer_casadi

@dataclass()
class HyperOpt():

    X : list[np.ndarray] 
    y : list[np.ndarray]
    t : np.ndarray
    parameters : dict = field(default_factory = dict)
    model : Optimizer_casadi() = field(default = Optimizer_casadi()) 

    @staticmethod
    def train_test_split(X, y, t, train_percent : int = 80):
        assert np.shape(X) == np.shape(y), "Features and targets should have same dimensions"
        sample = len(y)*train_percent//100
    
        return X[:sample], y[:sample], t[:sample], X[sample:], y[sample:], t[sample:]

    def gridsearch(self, integrate_models : bool = False, display_results : bool = True):

        self.X_train, self.y_train, self.t_train, self.X_test, self.y_test, self.t_test = self.train_test_split(self.X, self.y, self.t)
        result_dict = defaultdict(list)
        parameter_key, parameter_value = zip(*self.parameters.items()) # separate the key value pairs
        combinations = itertools.product(*parameter_value)

        for combination in combinations: # use combinations of values
            param_dict = dict(zip(parameter_key, combination)) # combine the key value pari and fit the model
            self.model.set_params(**param_dict)

            try:
                # self.model.fit(self.X_train, x_dot = self.y_train, quiet = True, multiple_trajectories = True) # for sindy
                self.model.fit(self.X_train, self.y_train) # for casadi

            except Exception as e:
                print(e)
                print("Failed for the parameter combination", param_dict)
                continue
            else:
                # models with none coefficients are not considered 
                if 0 in map(lambda x : sum(abs(x)), self.model.coefficients):
                    continue

                # calculate error
                for key in param_dict:
                    result_dict[key].append(param_dict[key])

                result_dict["MSE_test_pred"].append(self.model.score(self.X_test, x_dot = self.y_test, metric = mean_squared_error, 
                                                    multiple_trajectories = True))
                result_dict["MSE_train_pred"].append(self.model.score(self.X_train, x_dot = self.y_train, metric = mean_squared_error, 
                                                    multiple_trajectories = True))

                result_dict["r2_test_pred"].append(self.model.score(self.X_test, x_dot = self.y_test, metric = r2_score, 
                                                    multiple_trajectories = True))
                result_dict["r2_train_pred"].append(self.model.score(self.X_train, x_dot = self.y_train, metric = r2_score, 
                                                    multiple_trajectories = True))

                # is not compatible with multiple trajectories
                if integrate_models:
                    try:
                        y_pred_test_sim = self.model.simulate(self.X_test[0], self.t_test, integrator_kws = {"atol" : 1e-4, "rtol" : 1e-3, "method" : "RK23"})
                        y_pred_train_sim = self.model.simulate(self.X_train[0], self.t_train, integrator_kws = {"atol" : 1e-4, "rtol" : 1e-3, "method" : "RK23"})
                        print(y_pred_test_sim.shape)
                        assert len(y_pred_train_sim) == len(self.y_train)
                        assert len(y_pred_test_sim) == len(self.y_test) 
                    except:
                        result_dict["MSE_test_sim"].append(np.inf)
                        result_dict["MSE_train_sim"].append(np.inf)

                        result_dict["r2_test_sim"].append(np.inf)
                        result_dict["r2_train_sim"].append(np.inf)
                    else:
                        result_dict["MSE_test_sim"].append(mean_squared_error(self.X_test, y_pred_test_sim))
                        result_dict["MSE_train_sim"].append(mean_squared_error(self.X_train, y_pred_train_sim))

                        result_dict["r2_test_sim"].append(r2_score(self.X_test, y_pred_test_sim))
                        result_dict["r2_train_sim"].append(r2_score(self.X_train, y_pred_train_sim))

                result_dict["complexity"].append(self.model.complexity)

        # sort value and remove duplicates
        self.df_result = pd.DataFrame(result_dict)
        self.df_result.sort_values(by = ["r2_test_pred", "complexity"], ascending = False, inplace = True, ignore_index = True)
        self.df_result.drop_duplicates(["r2_test_pred"], keep = "first", inplace = True, ignore_index = True)

        if display_results:
            print(self.df_result.head())

        return self.df_result

    # bokeh plotting
    def plot(self, filename : str = "Gridsearch_results.html"):
        pass


if __name__ == "__main__":

    t_span = np.arange(0, 5, 0.01)
    model_actual = DynamicModel("kinetic_kosir", t_span, [], 2)
    features = model_actual.integrate(())
    target = model_actual.approx_derivative
    model_actual.plot(features[-1], t_span, "Time", "Concentration", ["A", "B", "C", "D"])

    params = {"optimizer__threshold": [0.01, 0.1], 
        "optimizer__alpha": [0, 0.01], 
        "feature_library": [ps.PolynomialLibrary(include_bias=False)], "feature_library__include_bias" : [False],
        "feature_library__degree": [1, 2]}

    opt = HyperOpt(features, target, t_span, params, Optimizer_casadi(solver_dict = {"ipopt.print_level" : 0, "print_time":0}))
    opt.gridsearch()


