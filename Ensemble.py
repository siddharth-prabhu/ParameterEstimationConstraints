from dataclasses import dataclass, field

import numpy as np
import sympy as smp
from collections import defaultdict, namedtuple

from Optimizer import Optimizer_casadi


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

        for i in range(iterations):
            X_bootstrap, y_bootstrap = next(ensemble._dataloader(X, y, seed = 10))

            try :
                self.casadi_model.fit([X_bootstrap], [y_bootstrap], self.include_column, self.constraints_dict)
            except Exception as error:
                print(f"failed for iteration {i} with error {error}")
                continue
            else:
                for (equation_symbols, coefficients, equation_dict) in zip(self.casadi_model.adict["library_labels"], 
                                                                self.casadi_model.adict["coefficients_value"] , self.coefficients_list):
                       
                    np.vectorize(lambda key, value: equation_dict[key].append(value), cache=True)(equation_symbols, coefficients)

        self._calculate_statistics()

    # calculates the mean, standard deviation and inclusion probability of coefficients
    def _calculate_statistics(self):
        distribution = namedtuple("distribution", ("mean", "deviation"))
        inclusion = namedtuple("probability", "inclusion")

        self.inclusion = [defaultdict(inclusion)]*len(self.coefficients_list)
        self.distribution = [defaultdict(distribution)]*len(self.coefficients_list)

        for (coefficients_dict, distribution_dict, inclustion_dict) in zip(self.coefficients_list, self.distribution, self.inclusion):
            
            parameters = np.vectorize(lambda x : (np.mean(coefficients_dict[x]), np.std(coefficients_dict[x])))(list(coefficients_dict.keys()))
            np.vectorize(lambda key, mean, deviation : distribution_dict.update({key : distribution(mean, deviation)}), cache = True)(list(coefficients_dict.keys()), parameters[0], parameters[1])
            np.vectorize(lambda key: inclustion_dict.update({key : inclusion(np.count_nonzero(coefficients_dict[key])/len(coefficients_dict[key]))}))(list(coefficients_dict.keys()))


    def plot(self):
        
        pass


if __name__ == "__main__":

    from GenerateData import DynamicModel
    from FunctionalLibrary import FunctionalLibrary

    model = DynamicModel("kinetic_kosir", np.arange(0, 5, 0.01), n_expt = 15)
    features = model.integrate() # list of features
    target = model.approx_derivative # list of target value

    opti = Optimizer_casadi(FunctionalLibrary(2) , alpha = 0.01, threshold = 0.1, solver_dict={"ipopt.print_level" : 0, "print_time":0})
    
    opti_ensemble = ensemble([[], [], [], []], {}, opti)
    opti_ensemble.fit(features, target, 4)
    alist = opti_ensemble.coefficients_list
    print(alist[0])
    print("length of equations", len(alist))
    print("distribution", opti_ensemble.distribution)
    print("inclusion probability", opti_ensemble.inclusion)
