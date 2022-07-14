from dataclasses import dataclass, field
from Base import Base
from typing import Optional, Callable
from functools import reduce
from collections import defaultdict
from tqdm import tqdm

from FunctionalLibrary import FunctionalLibrary
import numpy as np
import casadi as cd
import sympy as smp
import pickle
from sklearn.metrics import mean_squared_error
from scipy.integrate import odeint


@dataclass(frozen = False)
class Optimizer_casadi(Base):

    library : FunctionalLibrary = field(default = FunctionalLibrary())
    input_features : list[str] = field(default_factory=list)
    alpha : float = field(default = 0.0)
    alpha_mass : float = field(default = 1)
    num_points : float = field(default = 0.5) # 50 % of total data
    threshold : float = field(default = 0.01)
    max_iter : int = field(default = 20)
    solver_dict : dict = field(default_factory = dict)

    _flag_fit : bool = field(default = False, init = False)
    adict : dict = field(default_factory = dict, init = False)

    def __post_init__(self):
        pass

    def set_params(self, **kwargs):
        # sets the values of various parameter for gridsearchcv

        if "optimizer__alpha" in kwargs:
            setattr(self, "alpha", kwargs["optimizer__alpha"])
        
        if "optimizer__threshold" in kwargs:
            setattr(self, "threshold", kwargs["optimizer__threshold"])
        
        if "optimizer__max_iter" in kwargs:
            setattr(self, "max_iter", kwargs["optimizer__max_iter"])
        
        if "optimizer__alpha_mass" in kwargs:
            setattr(self, "alpha_mass", kwargs["optimizer__alpha_mass"])

        if "optimize__num_points" in kwargs:
            setattr(self, "num_points", kwargs["optimize__num_points"])

        self.library.set_params(**kwargs)


    def _generate_library(self, data : np.ndarray, include_column = list[np.ndarray]):
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
        
        # initialize the cost to zero
        self.adict["cost"] = 0
        
        # stoichiometric coefficient times the reactions 
        # need 2 for loops because of limitation of casadi 
        self.adict["reactions"] = [cd.mtimes(self.adict["library"][j], self.adict["coefficients"][j]) for j in range(self._functional_library)]

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

    def _add_constraints(self, constraints_dict, seed : int = 12345):
        
        data_points = self.adict["library_dimension"][0][0]
        rng = np.random.default_rng(seed)
        chosen_rows = rng.choice(range(data_points), int(self.num_points*data_points), replace = False)
        
        # adding mass balance constraints 
        state_mass = constraints_dict.get("mass_balance", None)
        if state_mass and len(state_mass) != self._n_states:
            raise ValueError("Masses are not equal to the states")
        
        if state_mass :
            asum = 0
            for i in range(self._n_states):
                asum += state_mass[i]*cd.mtimes(self.adict["library"][i][chosen_rows], self.adict["coefficients"][i])

            self.adict["cost"] += self.alpha_mass*cd.sum1(asum)**2/len(chosen_rows)

        # adding formation constraints
        state_formation = constraints_dict["formation"]
        if state_formation:
            for state in state_formation :
                asum = 0
                for j in range(self._functional_library):
                    asum += self.adict["stoichiometry"][state, j]*self.adict["reactions"][j]

                self.opti.subject_to(asum >= 0)

        # adding consumption constraints
        state_consumption = constraints_dict["consumption"]
        if state_consumption :
            for state in state_consumption:
                asum = 0
                for j in range(self._functional_library):
                    asum += self.adict["stoichiometry"][state, j]*self.adict["reactions"][j]
                
                self.opti.subject_to(asum <= 0)


    def _minimize(self, solver_dict : dict):

        self.opti.minimize(self.adict["cost"])
        self.opti.solver("ipopt", solver_dict, {"max_iter" : 30})
        solution = self.opti.solve()
        # assert solution.success, "The solution did not converge" add assertion 
        return solution

    def _stlsq(self, target : np.ndarray, constraints_dict : dict, ensemble_iterations : int, seed : int = 12345) -> list[np.ndarray]:
        
        rng = np.random.default_rng(seed)
        self._create_mask()
        coefficients_prev = [np.ones(dimension[-1]) for dimension in self.adict["library_dimension"]]
        self.adict["iterations"] = 0
        library = self.adict["library"]

        for _ in tqdm(range(self.max_iter)):
            
            self.adict["coefficients_ensemble"] = defaultdict(list)

            for _ in range(ensemble_iterations):
                # create problem from scratch since casadi cannot run the same problem once optimized
                # steps should follow a sequence 
                # dont replace if there is only one ensemble iteration. Dataset rows are constant for all reactions 
                permutations = rng.choice(range(self.adict["library_dimension"][0][0]), self.adict["library_dimension"][0][0], replace = (ensemble_iterations > 1))
                self._create_decision_variables()  
                self.adict["library"] = [value[permutations]*self.adict["mask"][ind] for ind, value in enumerate(library)]
                self._update_cost(target[permutations])
                if constraints_dict:
                    self._add_constraints(constraints_dict, seed)
                self.adict["solution"] = self._minimize(self.solver_dict) # no need to save for every iteration

                # list[np.ndarray]. additional layer of np.array and flatten to account for singular value, which casadi outputs as float
                for key, coeff in enumerate(self.adict["coefficients"]):
                    self.adict["coefficients_ensemble"][key].append(np.array([self.adict["solution"].value(coeff)]).flatten())
            
            # calculating mean and standard deviation 
            _mean, _deviation = [], []
            for key in self.adict["coefficients_ensemble"].keys():
                stack = np.vstack(self.adict["coefficients_ensemble"][key])
                _mean.append(np.mean(stack, axis = 0))
                _deviation.append(np.std(stack, axis = 0))
            
            # list of boolean arrays
            if ensemble_iterations > 1:
                coefficients_next = [np.abs(deviation/(mean + 1e-10)) < self.threshold for mean, deviation in zip(_mean, _deviation)]
            else:
                coefficients_next = [np.abs(self.adict["coefficients_ensemble"][key]) > self.threshold for key in self.adict["coefficients_ensemble"].keys()]

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
        
        return _mean, _deviation

    def fit(self, features : list[np.ndarray], target : list[np.ndarray], include_column : Optional[list[np.ndarray]] = None, 
            constraints_dict : dict = {} , ensemble_iterations : int = 1, seed : int = 12345) -> None:

        # constraints_dict should be of the form {"mass_balance" : [], "consumption" : [], "formation" : [], 
        #                                           "stoichiometry" : np.ndarray}
        self._flag_fit = True
        self._n_states = np.shape(features)[-1]

        if "stoichiometry" in constraints_dict and isinstance(constraints_dict["stoichiometry"], np.ndarray):
            rows, cols = constraints_dict["stoichiometry"].shape
            assert rows == self._n_states, "The rows should match the number of states"
            self._functional_library = cols
            self.adict["stoichiometry"] = constraints_dict["stoichiometry"]
        else:
            self._functional_library = self._n_states
            self.adict["stoichiometry"] = np.eye(self._n_states) 

        if include_column:
            assert len(include_column) == self._functional_library, "length of columns should match with the number of functional libraries"
            include_column = [list(range(self._n_states)) if len(alist) == 0 else alist for alist in include_column] 
        else:
            include_column = [list(range(self._n_states)) for _ in range(self._functional_library)]

        features, target = np.vstack(features), np.vstack(target)
        self._generate_library(features, include_column)

        self.adict["coefficients_value"], self.adict["coefficients_deviation"] = self._stlsq(target, constraints_dict, ensemble_iterations, seed)
        self._create_equations()

    # need to consider when stoichiometric in present
    def _create_equations(self) -> None:
        # stores the equations in adict to be used later
        self.adict["equations"] = []
        self.adict["equations_lambdify"] = []
        self.adict["coefficients_dict"] = []
        
        for i in range(self._n_states):
            expr = 0
            for j in range(self._functional_library):
                zero_filter = filter(lambda x : x[0], zip(self.adict["coefficients_value"][j], self.adict["library_labels"][j]))
                expr += self.adict["stoichiometry"][i, j]*smp.sympify(reduce(lambda accum, value : 
                        accum + value[0] + " * " + value[1].replace(" ", "*") + " + ",   
                        map(lambda x : ("{:.2f}".format(x[0]), x[1]), zero_filter), "+").rstrip(" +")) 
            # replaced whitespaces with multiplication element wise library labels
            # simpify already handles xor operation

            self.adict["equations"].append(str(expr))
            self.adict["coefficients_dict"].append(expr.as_coefficients_dict())
            self.adict["equations_lambdify"].append(smp.lambdify(self.input_symbols, expr))


    def _casadi_model(self, x : np.ndarray, t : np.ndarray):

        return np.array([eqn(*x) for eqn in self.adict["equations_lambdify"]])
    

    def predict(self, X : list[np.ndarray]) -> list:
        assert self._flag_fit, "Fit the model before running predict"
        afunc = np.vectorize(self._casadi_model, signature = "(m),()->(m)")

        return [afunc(xi, 0) for xi in X]
        

    def score(self, X : list[np.ndarray], y : list[np.ndarray], metric : Callable = mean_squared_error, predict : bool = True) -> float:
        assert self._flag_fit, "Fit the model before running score"
        y_pred = self.predict(X) if predict else X

        return metric(np.vstack(y_pred), np.vstack(y))

    # integrate the model
    def simulate(self, X : list[np.ndarray], time_span : np.ndarray, **integrator_kwargs) -> list[np.ndarray]:
        assert self._flag_fit, "Fit the model before running score"
        x_init = [xi[0].flatten() for xi in X]
        result = []
        for xi in x_init:
            assert len(xi) == self._n_states, "Initial conditions should be of right dimensions"

            try:
                _integration_solution = odeint(self._casadi_model, xi, time_span, **integrator_kwargs)
            except Exception as error:
                print(error)
                raise ValueError(f"Integration failed with error {error}") 
            else:
                if np.isnan(_integration_solution).any() or np.isinf(_integration_solution).any():
                    raise ValueError("Integration failed becoz nan or inf detected")
                result.append(_integration_solution)

        return result

    @property
    def complexity(self):
        return sum(eqn.count("+") + eqn.lstrip("-").count("-") + 1 for eqn in self.adict["equations"])

    @property
    def coefficients(self):
        return self.adict["coefficients_value"]

    # saves the equations using pickle
    def save_model(self, pathname : str = "saved_data\casadi_model") -> None:

        if not self.adict.get("equations", False):
            self._create_equations()

        with open(pathname, "wb") as file:
            pickle.dump(self.adict["equations"], file)


    def print(self) -> None:
        assert self._flag_fit, "Fit the model before printing models"
        if not self.adict.get("equations", False):
            self._create_equations()

        for i, eqn in enumerate(self.adict["equations"]):
            print(f"{self.input_features[i]}' = " + eqn)


if __name__ == "__main__":

    from GenerateData import DynamicModel

    model = DynamicModel("kinetic_kosir", np.arange(0, 5, 0.01), n_expt = 15)
    features = model.integrate() # list of features
    target = model.approx_derivative # list of target value

    opti = Optimizer_casadi(FunctionalLibrary(2) , alpha = 0.0, threshold = 0.1, solver_dict={"ipopt.print_level" : 0, "print_time":0})
    stoichiometry = np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1)
    include_column = [[0, 2], [0, 3], [0, 1]]
    # stoichiometry = None
    # opti.fit(features, target, include_column = [], 
    #         constraints_dict = {})
    opti.fit(features, target, include_column = [], 
                constraints_dict= {"mass_balance" : [], "formation" : [], "consumption" : [], 
                                    "stoichiometry" : 0}, ensemble_iterations = 1000)
    opti.print()
    print("--"*20)
    print("mean squared error :", opti.score(features, target))
    print(opti.complexity)
    print("Total number of iterations", opti.adict["iterations"])
    print("--"*20)
    print("coefficients value", opti.adict["coefficients_value"])
    print("--"*20)
    print("coefficients standard deviation", opti.adict["coefficients_deviation"])