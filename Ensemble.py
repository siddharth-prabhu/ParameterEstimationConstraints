from dataclasses import dataclass, field

import numpy as np
import sympy as smp
from collections import defaultdict

from Optimizer import Optimizer_casadi

# made changes in the ensemble branch
@dataclass
class ensemble :

    include_column : list
    constraints_dict : dict

    model : Optimizer_casadi
    seed : int = field(default = 10)

    @staticmethod
    def dataloader(self, X : np.array, y : np.array, seed : int) -> tuple(np.ndarray, np.ndarray):
        
        rng = np.random.default_rng(seed)
        dataset_size = X.shape[0]
        indices = np.arange(dataset_size)
        
        while True:
            permutations = rng.choice(indices, dataset_size, replace = True)
            yield (X[permutations], y[permutations])


    def fit(self, X : np.array, y : np.array, iterations : int):
        
        
        # define default dict to store value
        for i in range(iterations):
            X_bootstrap, y_bootstrap = next(dataloader(X, y, seed = 10))

            try :
                casadi_model = model.fit([X_bootstrap], [y_bootstrap], self.include_column, self.constraints_dict)
                print(casadi_model.input_symbols)
            except Exception as error:
                print(f"failed for iteration {i} with error {error}")
                continue
            else:
                pass

            

    def plot(self):
        pass