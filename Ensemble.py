from dataclasses import dataclass, field

import numpy as np
import sympy as smp
from collections import defaultdict, namedtuple
import matplotlib.pyplot as plt

from Optimizer import Optimizer_casadi
from utils import ensemble_plot


@dataclass
class ensemble :

    include_column : list
    constraints_dict : dict

    casadi_model : Optimizer_casadi
    seed : int = field(default = 10)

    @staticmethod
    def _dataloader(X : np.array, y : np.array, seed : int) -> tuple:
        
        rng = np.random.default_rng(seed)
        dataset_size = X.shape[0]
        indices = np.arange(dataset_size)
        
        while True:
            permutations = rng.choice(indices, dataset_size, replace = True)
            yield (X[permutations], y[permutations])


    def fit(self, X : list[np.array], y : list[np.array], iterations : int):
        
        X, y = np.vstack(X), np.vstack(y)
        self.coefficients_list = [defaultdict(list) for _ in range(len(X[0]))]
        data_iter = ensemble._dataloader(X, y, seed = 10)
        for i in range(iterations):
            X_bootstrap, y_bootstrap = next(data_iter)

            try :
                # ensemble iterations was added after writing this script
                self.casadi_model.fit([X_bootstrap], [y_bootstrap], self.include_column, self.constraints_dict, ensemble_iterations = 1, seed = self.seed)
            except Exception as error:
                print(f"failed for iteration {i} with error {error}")
                continue
            else:
                for (equation_dict, coefficients_dict) in zip(self.casadi_model.adict["coefficients_dict"], self.coefficients_list):
                    np.vectorize(lambda key: coefficients_dict[key].append(equation_dict[key]), cache=True)(list(equation_dict.keys()))
        
        self._calculate_statistics(iterations)

    # calculates the mean, standard deviation and inclusion probability of coefficients
    def _calculate_statistics(self, iterations : int):
        distribution = namedtuple("distribution", ("mean", "deviation"))
        inclusion = namedtuple("probability", "inclusion")

        self.inclusion = [defaultdict(distribution) for _ in range(len(self.coefficients_list))]
        self.distribution = [defaultdict(inclusion) for _ in range(len(self.coefficients_list))]

        for (coefficients_dict, distribution_dict, inclusion_dict) in zip(self.coefficients_list, self.distribution, 
                                                                        self.inclusion):
            
            coefficients_dict_keys = list(coefficients_dict.keys())
            np.vectorize(lambda key: inclusion_dict.update({key : inclusion(len(coefficients_dict[key])/iterations)}))(coefficients_dict_keys)
            
            np.vectorize(lambda key : (coefficients_dict[key].extend([0]*(iterations - len(coefficients_dict[key]))),
                                    coefficients_dict[key].extend([0]*(iterations - len(coefficients_dict[key])))), cache = True)(coefficients_dict_keys)
            
            parameters = np.vectorize(lambda key : (np.mean(np.array(coefficients_dict[key], dtype=float)), 
                                                    np.std(np.array(coefficients_dict[key], dtype=float))))(coefficients_dict_keys)
            
            np.vectorize(lambda key, mean, deviation : distribution_dict.update({key : distribution(mean, deviation)}), 
                                        cache = True)(coefficients_dict_keys, parameters[0], parameters[1])



if __name__ == "__main__":

    from GenerateData import DynamicModel
    from FunctionalLibrary import FunctionalLibrary

    model = DynamicModel("kinetic_kosir", np.arange(0, 5, 0.01), n_expt = 15)
    features = model.integrate() # list of features
    target = model.approx_derivative # list of target value

    features = model.add_noise(0, 0)
    target = model.approx_derivative

    opti = Optimizer_casadi(FunctionalLibrary(1) , alpha = 0.0, threshold = 0.1, solver_dict={"ipopt.print_level" : 0, "print_time":0})
    include_column = [[0, 2], [0, 3], [0, 1]]
    stoichiometry = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -0.5, -1]).reshape(4, -1) # mass balance
    stoichiometry = np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1) # chemistry constraint
    constraints_dict= {"mass_balance" : [], "formation" : [], "consumption" : [], 
                                    "stoichiometry" : stoichiometry}
    
    opti_ensemble = ensemble(include_column = [], constraints_dict = constraints_dict, casadi_model = opti)
    opti_ensemble.fit(features, target, iterations = 2)
    alist = opti_ensemble.coefficients_list
    ensemble_plot(opti_ensemble.coefficients_list, opti_ensemble.distribution, opti_ensemble.inclusion)