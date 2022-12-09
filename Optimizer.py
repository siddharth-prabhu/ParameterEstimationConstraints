from dataclasses import dataclass, field
from Base import Base
from typing import Optional, Callable, List
from functools import reduce
from collections import defaultdict, namedtuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from FunctionalLibrary import FunctionalLibrary
from utils import ensemble_plot

import numpy as np
import casadi as cd
import sympy as smp
import pickle
from sklearn.metrics import mean_squared_error
from scipy.integrate import odeint
import matplotlib.pyplot as plt


@dataclass(frozen = False)
class Optimizer_casadi(Base):

    library : FunctionalLibrary = field(default = FunctionalLibrary())
    input_features : list[str] = field(default_factory=list)
    alpha : float = field(default = 0.0)
    num_points : float = field(default = 0.5)
    threshold : float = field(default = 0.01) # inverse of z_critical for boostrapping
    max_iter : int = field(default = 20)
    solver_dict : dict = field(default_factory = dict)

    _flag_fit : bool = field(default = False, init = False)
    adict : dict = field(default_factory = dict, init = False)

    def __post_init__(self):
        assert self.alpha >= 0 and self.threshold >= 0, "Regularization and thresholding parameter should be greater than equal to zero"
        assert self.max_iter >= 1, "Max iteration should be greater than zero"
        assert 0 <= self.num_points <= 1, "percent points to be considered as constraints should be in [0, 1]"

    def set_params(self, **kwargs):
        # sets the values of various parameter for gridsearchcv

        if "optimizer__alpha" in kwargs:
            setattr(self, "alpha", kwargs["optimizer__alpha"])
        
        if "optimizer__threshold" in kwargs:
            setattr(self, "threshold", kwargs["optimizer__threshold"])
        
        if "optimizer__max_iter" in kwargs:
            setattr(self, "max_iter", kwargs["optimizer__max_iter"])

        if "optimize__num_points" in kwargs:
            setattr(self, "num_points", kwargs["optimize__num_points"])

        self.library.set_params(**kwargs)


    def _generate_library(self, data : np.ndarray, include_column : list[np.ndarray]):
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

        # adding formation constraints
        state_formation = constraints_dict.get("formation", None) if constraints_dict else None
        if state_formation:
            for state in state_formation :
                asum = 0
                for j in range(self._functional_library):
                    asum += self.adict["stoichiometry"][state, j]*self.adict["reactions"][j][chosen_rows]

                self.opti.subject_to(asum >= 0)

        # adding consumption constraints
        state_consumption = constraints_dict.get("consumption", None) if constraints_dict else None
        if state_consumption :
            for state in state_consumption:
                asum = 0
                for j in range(self._functional_library):
                    asum += self.adict["stoichiometry"][state, j]*self.adict["reactions"][j][chosen_rows]
                
                self.opti.subject_to(asum <= 0)


    def _minimize(self, solver_dict : dict):

        self.opti.minimize(self.adict["cost"])
        self.opti.solver("ipopt", solver_dict, {"max_iter" : 30})
        solution = self.opti.solve()
        # assert solution.success, "The solution did not converge" add assertion 
        return solution

    # function for multiprocessing
    def _stlsq_solve_optimization(self, library : list, target : np.ndarray, constraints_dict : dict, permutations : list, seed : int) -> list:
        # create problem from scratch since casadi cannot run the same problem once optimized
        # steps should follow a sequence 
        # dont replace if there is only one ensemble iteration. Dataset rows are constant for all reactions 
        self._create_decision_variables()  
        self.adict["library"] = [value[permutations]*self.adict["mask"][ind] for ind, value in enumerate(library)]
        self._update_cost(target[permutations])
        if constraints_dict:
            self._add_constraints(constraints_dict, seed)
        _solution = self._minimize(self.solver_dict) # no need to save for every iteration

        # list[np.ndarray]. additional layer of np.array and flatten to account for singular value, which casadi outputs as float
        return [np.array([_solution.value(coeff)]).flatten() for coeff in self.adict["coefficients"]]


    def _stlsq(self, target : np.ndarray, constraints_dict : dict, ensemble_iterations : int, max_workers : Optional[int] = None,
                seed : int = 12345) -> list[np.ndarray]:
        
        rng = np.random.default_rng(seed)
        self._create_mask()
        coefficients_prev = [np.ones(dimension[-1]) for dimension in self.adict["library_dimension"]]
        # data structures compatible with multiprocessing : list, dict 
        self.adict["iterations"] = 0
        self.adict["iterations_ensemble"] = ensemble_iterations
        self.adict["coefficients_iterations"] : list[dict] = []
        library = self.adict["library"] # use the original library terms

        for _ in tqdm(range(self.max_iter)):
            
            self.adict["coefficients_casadi_ensemble"] = defaultdict(list)
            permutations = [rng.choice(range(self.adict["library_dimension"][0][0]), self.adict["library_dimension"][0][0], replace = (ensemble_iterations > 1))
                                for _ in range(self.adict["iterations_ensemble"])] 
            
            if max_workers == 0: # Do not use multiprocessing.
                _coefficients_ensemble = [self._stlsq_solve_optimization(library, target, constraints_dict, permute, seed) for permute in permutations]

                for key in range(self._functional_library):
                    self.adict["coefficients_casadi_ensemble"][key].extend(alist[key] for alist in _coefficients_ensemble)
                    
            else: # if none use all cores
                with ProcessPoolExecutor(max_workers = max_workers) as executor:           
                    _coefficients_ensemble = list(executor.map(self._stlsq_solve_optimization, repeat(library), repeat(target), repeat(constraints_dict), 
                                        permutations, repeat(seed)))

                    for key in range(self._functional_library):
                        self.adict["coefficients_casadi_ensemble"][key].extend(alist[key] for alist in _coefficients_ensemble)

            # calculating mean and standard deviation 
            _mean, _deviation = [], []
            for key in self.adict["coefficients_casadi_ensemble"].keys():
                stack = np.vstack(self.adict["coefficients_casadi_ensemble"][key])
                _mean.append(np.mean(stack, axis = 0))
                _deviation.append(np.std(stack, axis = 0))
                self.adict["coefficients_casadi_ensemble"][key] = stack
            
            # list of boolean arrays
            if ensemble_iterations > 1:
                coefficients_next = [np.abs(mean/(deviation + 1e-15)) > self.threshold for mean, deviation in zip(_mean, _deviation)]
            else:
                coefficients_next = [np.abs(self.adict["coefficients_casadi_ensemble"][key]) > self.threshold for key in self.adict["coefficients_casadi_ensemble"].keys()]

            if np.array([np.allclose(coeff_prev, coeff_next) for coeff_prev, coeff_next in zip(coefficients_prev, coefficients_next)]).all():
                print("Solution converged")
                break

            if not sum([np.sum(coeff) for coeff in coefficients_next]):
                raise RuntimeError("Thresholding parameter eliminated all the coefficients")
            
            # store values for every iteration
            self.adict["coefficients_iterations"].append({"mean" : _mean, "standard_deviation" : _deviation, 
                                    "z_critical" : [np.abs(mean)/(deviation + 1e-15) for mean, deviation in zip(_mean, _deviation)], "distribution" : stack})

            coefficients_prev = coefficients_next # boolean array

            # update mask of small terms to zero
            self.adict["mask"] = [mask*coefficients_next[i] for i, mask in enumerate(self.adict["mask"])]
            self.adict["iterations"] += 1
        
        return _mean, _deviation

    def fit(self, features : list[np.ndarray], target : list[np.ndarray], include_column : Optional[list[np.ndarray]] = None, 
            constraints_dict : dict = {} , ensemble_iterations : int = 1, max_workers : Optional[int] = None, seed : int = 12345) -> None:

        # ensemble_iterations = 1 : do regular sindy else ensemble sindy
        # constraints_dict should be of the form {"consumption" : [], "formation" : [], 
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

        self.adict["coefficients_value"], self.adict["coefficients_deviation"] = self._stlsq(target, constraints_dict, ensemble_iterations, max_workers, seed)
        self._create_equations()

    # need to consider when stoichiometric in present
    def _create_equations(self) -> None:
        # stores the equations in adict to be used later
        self.adict["equations"] = []
        self.adict["equations_lambdify"] = []
        self.adict["coefficients_dict"] = []
        
        for i in range(self._n_states):
            expr = self._create_sympy_expressions(self.adict["stoichiometry"][i])
            self.adict["equations"].append(str(expr))
            self.adict["coefficients_dict"].append(expr.as_coefficients_dict())
            self.adict["equations_lambdify"].append(smp.lambdify(self.input_symbols, expr))


    def _create_sympy_expressions(self, stoichiometry_row : np.ndarray) -> str:

        coefficients_value : List[np.ndarray] = self.adict["coefficients_value"]
        library_labels : List[List[str]] = self.adict["library_labels"]
        expr = 0

        for j in range(len(library_labels)):
            zero_filter = filter(lambda x : x[0], zip(coefficients_value[j], library_labels[j]))
            expr += stoichiometry_row[j]*smp.sympify(reduce(lambda accum, value : 
                    accum + value[0] + " * " + value[1].replace(" ", "*") + " + ",   
                    map(lambda x : ("{:.2f}".format(x[0]), x[1]), zero_filter), "+").rstrip(" +")) 
        # replaced whitespaces with multiplication element wise library labels
        # simpify already handles xor operation
        return expr


    def plot_distribution(self, reaction_coefficients : bool = False, coefficients_iterations : bool = False) -> None:
        # plotting the distribution of casadi coefficients
        # create list of dictionary with symbols as keys and arrays as values
        _coefficients_list = [defaultdict(list) for _ in range(self._functional_library)]
        
        distribution = namedtuple("distribution", ("mean", "deviation"))
        _coefficients_distribution = [defaultdict(distribution) for _ in range(self._functional_library)]

        inclusion = namedtuple("probability", "inclusion")
        _coefficients_inclusion = [defaultdict(inclusion) for _ in range(self._functional_library)]

        for i, key in enumerate(self.adict["coefficients_casadi_ensemble"].keys()):
            for j, _symbol in enumerate(self.adict["library_labels"][i]):
                _coefficients_list[i][_symbol].extend(self.adict["coefficients_casadi_ensemble"][key][:, j])
                _coefficients_distribution[i][_symbol] = distribution(self.adict["coefficients_value"][i][j], self.adict["coefficients_deviation"][i][j])
                _coefficients_inclusion[i][_symbol] = inclusion(np.count_nonzero(_coefficients_list[i][_symbol])/self.adict["iterations_ensemble"])

        # ensemble_plot(_coefficients_list, _coefficients_distribution, _coefficients_inclusion)

        if reaction_coefficients:
            # plotting the distribution of reaction equations
            _reaction_coefficients_list = [defaultdict(list) for _ in range(self._n_states)]
            _reaction_coefficients_distribution = [defaultdict(distribution) for _ in range(self._n_states)]
            _reaction_coefficients_inclusion = [defaultdict(inclusion) for _ in range(self._n_states)]

            for i in range(self._n_states):
                # for each iteration
                for j in range(self.adict["iterations_ensemble"]):
                    _expr = self._create_sympy_expressions([self.adict["coefficients_casadi_ensemble"][key][j] for key in range(self._functional_library)], 
                                                            self.adict["library_labels"], self.adict["stoichiometry"][i])
                    _expr_coeff = _expr.as_coefficients_dict()

                    for key, value in _expr_coeff.items():
                        _reaction_coefficients_list[i][key].append(value)

                for key, value in _reaction_coefficients_list[i].items():
                    value.extend([0]*(self.adict["iterations_ensemble"] - len(value)))
                    # wrap into numpy array as it does not know how to deal with sympy objects
                    _reaction_coefficients_distribution[i][key] = distribution(np.mean(np.array(value, dtype=float)), np.std(np.array(value, dtype = float)))
                    _reaction_coefficients_inclusion[i][key] = inclusion(np.count_nonzero(np.array(value, dtype = float))/self.adict["iterations_ensemble"])

            ensemble_plot(_reaction_coefficients_list, _reaction_coefficients_distribution, _reaction_coefficients_inclusion)

        if coefficients_iterations :
            
            for key in range(self._functional_library):
                fig, ax = plt.subplots(self.adict["iterations"], 3, figsize = (10, 4))
                ax = np.ravel(ax)
                for i, _coefficients_iterations in enumerate(self.adict["coefficients_iterations"]):
                    
                    ax[3*i].bar(self.adict["library_labels"][key], _coefficients_iterations["mean"][key])
                    ax[3*i + 1].bar(self.adict["library_labels"][key], _coefficients_iterations["standard_deviation"][key])
                    ax[3*i + 2].bar(self.adict["library_labels"][key], _coefficients_iterations["z_critical"][key])
                    ax[3*i + 2].set(ylim = (-self.threshold, self.threshold))

                    if i == 0:
                        ax[3*i].set(title = "Mean")
                        ax[3*i + 1].set(title = "Sigma")
                        ax[3*i + 2].set(title = "z_critical")
                    
                    if i != self.adict["iterations"] - 1 : 
                        ax[3*i].set(xticklabels = [])
                        ax[3*i + 1].set(xticklabels = [])
                        ax[3*i + 2].set( xticklabels = [])
                    else:
                        ax[3*i].set_xticks(range(len(self.adict["library_labels"][key])), self.adict["library_labels"][key], rotation = 90)
                        ax[3*i + 1].set_xticks(range(len(self.adict["library_labels"][key])), self.adict["library_labels"][key], rotation = 90)
                        ax[3*i + 2].set_xticks(range(len(self.adict["library_labels"][key])), self.adict["library_labels"][key], rotation = 90)
                        
                plt.show()
            
    def _casadi_model(self, x : np.ndarray, t : np.ndarray, *args):

        return np.array([eqn(*x, *args) for eqn in self.adict["equations_lambdify"]])
    

    def predict(self, X : list[np.ndarray], *args) -> list:
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
    from utils import coefficient_difference_plot

    model = DynamicModel("kinetic_kosir", np.arange(0, 5, 0.01), n_expt = 1)
    features = model.integrate() # list of features
    target = model.approx_derivative # list of target value
    features = model.add_noise(0, 0.0)
    target = model.approx_derivative

    opti = Optimizer_casadi(FunctionalLibrary(2) , alpha = 0.0, threshold = 0.1, solver_dict={"ipopt.print_level" : 0, "print_time":0, "ipopt.sb" : "yes"}, 
                            max_iter = 20)
    stoichiometry = np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1) # chemistry constraints
    include_column = [[0, 2], [0, 3], [0, 1]]
    stoichiometry =  np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -0.5, -1]).reshape(4, -1) # mass balance constraints
    # stoichiometry = np.eye(4) # no constraints

    opti.fit(features, target, include_column = [], 
                constraints_dict= {"formation" : [], "consumption" : [], 
                                    "stoichiometry" : stoichiometry}, ensemble_iterations = 2, seed = 10, max_workers = 2)
    opti.print()
    print("--"*20)
    print("mean squared error :", opti.score(features, target))
    print("model complexity", opti.complexity)
    print("Total number of iterations", opti.adict["iterations"])
    print("--"*20)
    # print("coefficients at each iteration", opti.adict["coefficients_iterations"])
    print("--"*20)
    # opti.plot_distribution(reaction_coefficients = False, coefficients_iterations = True)

    coefficient_difference_plot(model.coefficients() , sigma = opti.adict["coefficients_dict"], sigma2 = opti.adict["coefficients_dict"])