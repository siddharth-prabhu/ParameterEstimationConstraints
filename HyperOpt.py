import numpy as np
import pandas as pd
import pysindy as ps
from sklearn.metrics import mean_squared_error, r2_score

import itertools
from dataclasses import dataclass, field
from collections import defaultdict

from GenerateData import DynamicModel

@dataclass()
class HyperOpt():

    X : np.ndarray 
    y : np.ndarray
    t : np.ndarray
    parameters : field(default_factory = dict)
    model : ps.SINDy()

    @staticmethod
    def train_test_split(X, y, t, train_percent : int = 80):
        assert len(X) == len(y), "Features and target values are not of same leangth"
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
                self.model.fit(self.X_train, x_dot = self.y_train, quiet = True)
            except :
                print("Failed for the parameter combination", param_dict)
                continue
            else:
                # models with none coefficients are not considered 
                if not np.sum(abs(self.model.coefficients()), axis = 1).all():
                    continue

                # calculate error
                for key in param_dict:
                    result_dict[key].append(param_dict[key])

                y_pred_test = self.model.predict(self.X_test)

                result_dict["MSE_test_pred"].append(mean_squared_error(self.y_test, y_pred_test))
                result_dict["MSE_train_pred"].append(mean_squared_error(self.y_train, self.model.predict(self.X_train)))

                result_dict["r2_test_pred"].append(r2_score(self.y_test, y_pred_test))
                result_dict["r2_train_pred"].append(r2_score(self.y_train, self.model.predict(self.X_train)))

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
        self.df_result.drop_duplicates(["r2_test_pred"], keep = "first", inplace = True)
        self.df_result.sort_values(by = ["r2_test_pred"], ascending = False, inplace = True, ignore_index = True)

        if display_results:
            print(self.df_result.head())

        return self.df_result

    # bokeh plotting
    def plot(self):
        pass


if __name__ == "__main__":

    x_init = np.array([1, 2, 3, 4])
    t_span = np.arange(0, 5, 0.1)
    model_actual = DynamicModel("kinetic_kosir", x_init, t_span)
    features = model_actual.integrate(())
    target = model_actual.approx_derivative

    params = {"optimizer__threshold": [0.001, 0.01], 
        "optimizer__alpha": [0, 0.01], 
        "feature_library": [ps.PolynomialLibrary(include_bias=False)], "feature_library__include_bias" : [False],
        "feature_library__degree": [1, 2]}

    opt = HyperOpt(features, target, t_span, params, ps.SINDy())
    opt.gridsearch()
    # opt.plot()