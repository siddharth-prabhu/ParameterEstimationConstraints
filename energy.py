import casadi as cd
import numpy as np
import sympy as smp
from sklearn.metrics import mean_squared_error

from Optimizer import Optimizer_casadi
from FunctionalLibrary import FunctionalLibrary

from typing import List, Optional, Any, Callable
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

    def _create_decision_variables(self):
        super()._create_decision_variables()
        self.adict["coefficients_energy"] = [self.opti.variable(dimension[-1], 1) for dimension in self.adict["library_dimension"]]

    def _generate_library(self, data : np.ndarray, include_column : list[np.ndarray]):
        
        super()._generate_library(data, include_column)
        # create new symbol for temperature
        self.input_symbols = (*self.input_symbols, smp.symbols("T"))

    def _update_cost(self, target: np.ndarray, temperature : np.ndarray):

        # initialize the cost to zero
        self.adict["cost"] = 0
        
        # stoichiometric coefficient times the reactions 
        # need 2 for loops because of limitation of casadi 
        # the reaction rate will be arhenius equation
        reaction_rate = [A*cd.exp(-B/8.314/temperature) for A, B in zip(self.adict["coefficients"], self.adict["coefficients_energy"])]
        self.adict["reactions"] = [cd.mtimes(self.adict["library"][j], reaction_rate[j]) for j in range(self._functional_library)]

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

        # normalize the cost by dividing by the number of data points
        self.adict["cost"] /= self.adict["library_dimension"][0][0] # first array with first dimension

        # add regularization to the cost function
        for j in range(self._functional_library):
            self.adict["cost"] += self.alpha*cd.sumsqr(self.adict["coefficients"][j])

    def _stlsq_solve_optimization(self, library: List, target: np.ndarray, 
                                    constraints_dict: dict, permutations: List, seed: int) -> List:
        # create problem from scratch since casadi cannot run the same problem once optimized
        # steps should follow a sequence 
        # dont replace if there is only one ensemble iteration. Dataset rows are constant for all reactions 
        self._create_decision_variables()  
        self.adict["library"] = [value[permutations]*self.adict["mask"][ind] for ind, value in enumerate(library)]
        self._update_cost(target[permutations], self.adict["temperature"])
        if constraints_dict:
            self._add_constraints(constraints_dict, seed)
        _solution = self._minimize(self.solver_dict) # no need to save for every iteration

        # list[np.ndarray]. additional layer of np.array and flatten to account for singular value, which casadi outputs as float
        # use the reaction rate at 300K for thresholding
        return [np.array([_solution.value(coeff)]).flatten()*np.exp(-np.array([_solution.value(coeff_energy)]).flatten()/8.314/300)
                    for coeff, coeff_energy in zip(self.adict["coefficients"], self.adict["coefficients_energy"])]


    def fit(self, features : list[np.ndarray], target : list[np.ndarray], temperature : np.ndarray, include_column : Optional[list[np.ndarray]] = None, 
            constraints_dict : dict = {} , ensemble_iterations : int = 1, max_workers : Optional[int] = None, seed : int = 12345) -> None:

        assert temperature.shape[0] == len(features) and temperature.ndim == 1, "Temperature values should be equal to the number of experiments and should be 1 dimensional"
        # match the temp with the number of data points (each array in features can have varying data points)
        self.adict["temperature"] = np.vstack([np.repeat(temp, len(feat)) for temp, feat in zip(temperature, features)]).flatten()
        super().fit(features, target, include_column, constraints_dict, ensemble_iterations, max_workers, seed)

    @property
    def coefficients(self):
        return self.adict["coefficients_value"], self.adict["coefficients_energy"]


    def _create_sympy_expressions(self, stoichiometry_row : np.ndarray) -> str:

        coefficients_value : List[np.ndarray] = self.adict["coefficients_value"]
        coefficients_energy : List[np.ndarray] = self.adict["coefficients_energy"]
        library_labels : List[List[str]] = self.adict["library_labels"]
        expr = 0
        
        for j in range(len(library_labels)):
            zero_filter = filter(lambda x : x[0], zip(coefficients_value[j], coefficients_energy[j], library_labels[j]))
            # modify expr to include arhenius equation
            expr += stoichiometry_row[j]*smp.sympify(reduce(lambda accum, value : 
                    accum + value[0] + "*e**(-" + value[1] + "/8.314/T)* " + value[-1].replace(" ", "*") + " + ",   
                    map(lambda x : ("{:.2f}".format(x[0]), "{:.2f}".format(x[1]), x[-1]), zero_filter), "+").rstrip(" +")) 
        # replaced whitespaces with multiplication element wise library labels
        # simpify already handles xor operation
        return expr

    def predict(self, X : list[np.ndarray], T : float) -> list:
        return super().predict(X, T)

    def score(self, X : list[np.ndarray], y : list[np.ndarray], T : float, metric : Callable = mean_squared_error, predict : bool = True) -> float:
        return super().score(X, y, metric, predict, T)

    # integrate the model
    def simulate(self, X : list[np.ndarray], time_span : np.ndarray, T : float, **integrator_kwargs) -> list[np.ndarray]:
        return super().simulate(X, time_span, T, **integrator_kwargs)
        
        
class foo():

    def afunc(self):
        self.bfunc()
        print("original class")
        return 10

    @staticmethod
    def bfunc():
        print("original static method")
        return 2

class boo(foo):

    def dumb(self):
        return self.afunc()

    @staticmethod
    def bfunc():
        print("new static method")
        return 3

alist = boo()
print(alist.dumb())
    