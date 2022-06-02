import numpy as np
import pysindy as ps
from sklearn.metrics import mean_squared_error, r2_score

import itertools
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass()
class HyperOpt():

    X : np.ndarray 
    y : np.ndarray
    parameters : field(default_factory = dict)
    model : ps.SINDy()

    @staticmethod
    def train_test_split(X, y, train_percent : int = 80):
        assert len(X) == len(y), "Features and target values are not of same leangth"
        sample = len(y)*train_percent//100
    
        return X[:sample], y[:sample], X[sample:], y[sample:]

    def gridsearch(self):

        self.X_train, self.y_train, self.X_test, self.y_test = train_test_split(self.X, self.y)
        result_dict = defaultdict(list)
        key, value = zip(*self.parameters.items()) # separate the key value pairs
        for combination in itertools.product(*value): # use combinations of values
            
            param_dict = zip(key, combination) # combine the key value pari and fit the model
            self.model.set_params(**param_dict)

            try:
                self.model.fit(self.X_train, x_dot = self.y_train, quiet = True)
            except :
                print("Failed for the parameter combination", param_dict)
                continue
            else:
                # models with none coefficients are not considered 
                if not np.sum(abs(model.coefficients()), axis = 1).all():
                    continue

                # calculate error
                for key in param_dict:
                    result_dict[key].append(param_dict[key])

                y_pred_test = self.model.predict(self.X_test)
                y_pred_test_sim = self.forward_simulate(self.model, self.X_test[0], len(self.X_test))
                y_pred_train_sim = self.forward_simulate(self.model, self.X_train[0], len(self.X_train))

                result_dict["MSE_test"].append(mean_squared_error(self.y_test, y_pred_test))
                result_dict["MSE_train"].append(mean_squared_error(self.y_train, self.model.predict(self.X_train)))

                result_dict["r2_test"].append(r2_score(self.y_test, y_pred_test))
                result_dict["r2_train"].append(r2_score(self.y_train, self.model.predict(self.X_train)))

                result_dict["MSE_test_sim"].append(mean_squared_error(self.y_test, y_prec_test_sim))
                result_dict["MSE_train_sim"].append(mean_squared_error(self.y_train, y_pred_train_sim))
 
                result_dict["complexity"].append(self.model.complexity)

        return result_dict

    # bokeh plotting
    def plot():
        pass

    @staticmethod
    def forward_simulate(model : ps.SINDy, x_initial:np.ndarray, n_steps:int):
        # forward simulates a discrete model. Can only be done with full model
        for step in range(n_steps):
            if step == 0:
                xi = model.predict(x_initial).flatten()
                x_constant = x_initial[len(xi):].flatten()
                x = np.concatenate((xi, x_constant)) 
            else:
                xi = model.predict(x[-1], u_constant).flatten()
                x = np.row_stack((x, np.concatenate((xi, x_constant))))

        return x[:, :2] # remove the constant terms

    # errors
    
