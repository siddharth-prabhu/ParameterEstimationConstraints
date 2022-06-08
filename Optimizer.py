from dataclasses import dataclass, field
from Base import Base
from typing import Optional, Callable
from functools import reduce

from FunctionalLibrary import FunctionalLibrary
import numpy as np
import casadi as cd
import sympy as smp
from sklearn.metrics import mean_squared_error


@dataclass(frozen = False)
class Optimizer_casadi(Base):

    library : FunctionalLibrary = field(default = FunctionalLibrary())
    input_features : list[str] = field(default_factory=list)
    alpha : float = field(default = 0.0)
    threshold : float = field(default = 0.01)
    max_iter : int = field(default = 10)

    def __post_init__(self):
        self._fit_flag = False
        self.adict = {}

    def set_params(self, **kwargs):
        # sets the values of various parameter for gridsearchcv

        if kwargs.get("optimizer__alpha", False):
            setattr(self, "alpha", kwargs["optimizer__alpha"])
        
        if kwargs.get("optimizer__threshold", False):
            setattr(self, "threshold", kwargs["optimizer__threshold"])
        
        if kwargs.get("optimizer__max_iter", False):
            setattr(self, "max_iter", kwargs["optimizer__max_iter"])
        
        self.library.set_params(**kwargs)


    def _generate_library(self, data : np.ndarray, include_column = list[np.ndarray]):
        # given data creates list of matix of all possible combinations of terms 
        # returns a list of number of columns of each matrix

        # define input features if not given
        if not len(self.input_features):
            self.input_features = [f"x{i}" for i in range(self._n_states)]
            print(self.input_features)

        # define symbols that can be converted to equations later
        self.input_symbols = smp.symbols(reduce(lambda accum, value : accum + value + ", ", self.input_features, ""))

        # done using for loop instead of list comprehension becoz each fit_transform and get_features
        # share the same instance of the class
        self.adict["library"] = []
        self.adict["library_labels"] = []
        for i in range(self._n_states):
            self.adict["library"].append(self.library.fit_transform(data, include_column[i]))
            self.adict["library_labels"].append(self.library.get_features(self.input_features))
        
        self.adict["library_dimension"] = [xi.shape for xi in self.adict["library"]]

    def _create_decision_variables(self):
        # initializes the number of variables that will be used in casadi optimization 

        self.opti = cd.Opti() # create casadi instance
        self.adict["coefficients"] = [self.opti.variable(dimension[-1], 1) for dimension in self.adict["library_dimension"]]
    
    # created separately becoz it needs to be run only once
    def _create_mask(self):
        self.adict["mask"] = [np.ones(dimension[-1]) for dimension in self.adict["library_dimension"]]

    def _update_cost(self, target : np.ndarray):

        self.adict["cost"] = 0
        for i in range(self._n_states):
            self.adict["cost"] += (cd.sumsqr(target[:, i] - cd.mtimes(self.adict["library"][i], self.adict["coefficients"][i]))/
                            (sum(map(lambda x : x[0], self.adict["library_dimension"]))) + 
                            self.alpha*cd.sumsqr(self.adict["coefficients"][i]))

    
    def _minimize(self, solver_dict : dict):

        self.opti.minimize(self.adict["cost"])
        self.opti.solver("ipopt", solver_dict)
        solution = self.opti.solve()
        # assert solution.success, "The solution did not converge"
        return solution

    def fit(self, features : list[np.ndarray], target : list[np.ndarray], include_column : Optional[list[np.ndarray]] = None) -> None:

        self._fit_flag = True
        self._n_states = np.shape(features)[-1]

        if include_column:
            assert len(include_column) == self._n_states, "List should match the dimensions of features"
            include_column = [list(range(self._n_states)) if len(alist) == 0 else alist for alist in include_column] 
        else:
            include_column = [list(range(self._n_states)) for _ in range(self._n_states)]

        features, target = np.vstack(features), np.vstack(target)
        self._generate_library(features, include_column)
        self._create_mask()
        coefficients_prev = [np.ones(dimension[-1]) for dimension in self.adict["library_dimension"]]
        self.adict["iterations"] = 0
 
        for i in range(self.max_iter):
            # create problem from scratch since casadi cannot run the same problem once optimized
            self._create_decision_variables()  
            self.adict["library"] = [value*self.adict["mask"][i] for i, value in enumerate(self.adict["library"])]   
            self._update_cost(target)
            self.adict["solution"] = self._minimize({"ipopt.print_level" : 5}) # no need to save for every iteration

            coefficients = [self.adict["solution"].value(coeff) for coeff in self.adict["coefficients"]] # list[np.ndarray]
            
            coefficients_next = [np.abs(coeff) >= self.threshold for coeff in coefficients] # list of boolean arrays

            if np.array([np.allclose(coeff_prev, coeff_next) for coeff_prev, coeff_next in zip(coefficients_prev, coefficients_next)]).all():
                print("Solution converged")
                break

            if not sum([np.sum(coeff) for coeff in coefficients_next]):
                raise RuntimeError("Thresholding parameter eliminated all the coefficients")
                break
            
            coefficients_prev = coefficients_next # boolean array

            # update mask of small terms to zero
            self.adict["mask"] = [mask*coefficients_next[i] for i, mask in enumerate(self.adict["mask"])]
            self.adict["iterations"] += 1

        self.adict["coefficients_value"] = coefficients

    def _create_equations(self) -> None:
        # stores the equations in adict that can later be used
        self.adict["equations"] = []
        self.adict["equations_lambdify"] = []
        for i in range(self._n_states):
            zero_filter = filter(lambda x : x[0], zip(self.adict["coefficients_value"][i], self.adict["library_labels"][i]))
            expr = (reduce(lambda accum, value : accum + value[0] + " * " + value[1] + " + ", 
                    map(lambda x : ("{:.2f}".format(x[0]), x[1]), zero_filter), "").rstrip(" +"))
        
            self.adict["equations"].append(f"{self.input_features[i]}' = " + expr)
            self.adict["equations_lambdify"].append(smp.lambdify(self.input_symbols, expr))


    def _casadi_model(self, x, t):

        if not self.adict.get("equations_lambdify", False):
            self._create_equations()

        return np.array([eqn(*x) for eqn in self.adict["equations_lambdify"]])
    

    def predict(self, X : list[np.ndarray]) -> list:
        assert self._fit_flag, "Fit the model before running predict"
        afunc = np.vectorize(self._casadi_model, signature = "(m),()->(m)")
        
        return [afunc(xi, 0) for xi in X]
        

    def score(self, X : list[np.ndarray], x_dot : [np.ndarray], metric : Callable = mean_squared_error, **kwargs) -> float:
        assert self._fit_flag, "Fit the model before running score"

        y_pred = self.predict(X)
        return metric(np.vstack(y_pred), np.vstack(x_dot))

    @property
    def complexity(self):
        if not self.adict.get("equations", False):
            self._create_equations()
        
        return sum(eqn.count("+") + 1 for eqn in self.adict["equations"])

    @property
    def coefficients(self):
        return self.adict["coefficients_value"]


    def print(self) -> None:
        assert self._fit_flag, "Fit the model before printing models"
        if not self.adict.get("equations", False):
            self._create_equations()

        for eqn in self.adict["equations"]:
            print(eqn)


if __name__ == "__main__":

    from GenerateData import DynamicModel

    model = DynamicModel("kinetic_kosir", np.arange(0, 5, 0.01), n_expt = 15)
    features = model.integrate() # list of features
    target = model.approx_derivative # list of target value

    opti = Optimizer_casadi(alpha = 0.0, threshold = 0.1)
    opti.fit(features, target, [[], [0, 1], [], []])
    opti.print()

    print("mean squared error :", opti.score(features, target))
    print(opti.complexity)
    print(opti.coefficients)
