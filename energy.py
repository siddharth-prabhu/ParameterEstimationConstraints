import casadi as cd
import numpy as np
import sympy as smp

from Optimizer import Optimizer_casadi
from FunctionalLibrary import FunctionalLibrary

from typing import List, Optional
from functools import reduce

class Energy_sindy(Optimizer_casadi):

    def __init__(self, 
                    library : FunctionalLibrary,
                    input_features : List[str],
                    alpha : float = 0.0,
                    num_points : float = 0.5,
                    threshold : float = 0.01, # inverse of z_critical for boostrapping
                    max_iter : int = 20,
                    solver_dict : dict = {}):
        
        self.library = library
        self.input_features = input_features
        self.alpha = alpha
        self.num_points = num_points
        self.threshold = threshold
        self.max_iter = max_iter
        self.solver_dict = solver_dict

        super().__init__(self.library, 
                        self.input_features,
                        self.alpha,
                        self.num_points,
                        self.threshold,
                        self.max_iter,
                        self.solver_dict)

    def _generate_library(self, data : np.ndarray, include_column : List[np.ndarray]):

        # given data creates list of matix of all possible combinations of terms 
        # returns a list of number of columns of each matrix

        # define input features if not given
        if not len(self.input_features):
            self.input_features = [f"x{i}" for i in range(self._n_states)]

        # define symbols that can be converted to equations later
        self.input_symbols = smp.symbols(reduce(lambda accum, value : accum + value + ", ", self.input_features, ""))

        # done using for loop instead of list comprehension becoz each fit_transform and get_features
        # share the same instance of the class
        self.adict["library"] = []
        self.adict["library_labels"] = []
        for i in range(self._functional_library):
            # self.adict["library"] is now a list of list of np.ndarrays instead of list of np.ndarrays
            self.adict["library"].append([self.library.fit_transform(each_expt_data, include_column[i]) for each_expt_data in data])
            self.adict["library_labels"].append(self.library.get_features(self.input_features))
        
        self.adict["library_dimension"] = [xi.shape for xi in self.adict["library"]]

    def _create_decision_variables(self):
        super()._create_decision_variables()
        self.adict["coefficients_energy"] = [self.opti.variable(dimension[-1], 1) for dimension in self.adict["library_dimension"]]

    def _update_cost(self, target: np.ndarray, temperature : np.ndarray):

        # initialize the cost to zero
        self.adict["cost"] = 0
        
        # stoichiometric coefficient times the reactions 
        # need 2 for loops because of limitation of casadi
        self.adict["reactions"] = []
        for _functional_library in range(self._functional_library):
            alist = []
            for j in range(len(self.adict["library"][_functional_library])):
                reaction_constant = self.adict["coefficients"][_functional_library]*cd.exp(-self.adict["coefficients_energy"][_functional_library]/8.314/temperature[j])
                alist.append(cd.mtimes(self.adict["library"][_functional_library][j], reaction_constant))
            
            self.adict["reactions"].append(cd.vertcat(*alist)) # revert it back to the same shape as the original case

        for i in range(self._n_states):
            asum = 0
            for j in range(self._functional_library): 
                asum += self.adict["stoichiometry"][i, j]*self.adict["reactions"][j]

            self.adict["cost"] += cd.sumsqr(target[:, i] - asum)

        # normalize the cost by dividing by the number of data points
        self.adict["cost"] /= self.adict["library_dimension"][0][0] # first array with first dimension

        # add regularization to the cost function
        for j in range(self._functional_library):
            self.adict["cost"] += self.alpha*cd.sumsqr(self.adict["coefficients"][j])

    def _stlsq_solve_optimization(self, library: List, target: np.ndarray, constraints_dict: dict, permutations: List, seed: int) -> List:
        pass

    def fit():
        pass

class foo():

    def __init__(self, name):
        self.name = name

    def bfunc(self):
        return self.afunc()

    def afunc(self):
        return "help"

class boo(foo):

    def __init__(self, name):
        super().__init__(name)

    def cfunc(self):
        self.last = "another"
        return self.name, self.last, super().bfunc()

    def afunc(self):
        return "boo"

demo = boo("sindy")
print(demo.cfunc())