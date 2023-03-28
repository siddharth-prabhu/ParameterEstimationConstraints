# type: ignore

from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple
from functools import reduce, partial
from collections import defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import numpy as np
import casadi as cd
import sympy as smp
import pickle
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import stats

from Base import Base
from FunctionalLibrary import FunctionalLibrary
from utils import ensemble_plot


@dataclass(frozen = False)
class Optimizer_casadi(Base):

    library : FunctionalLibrary = field(default = FunctionalLibrary())
    input_features : list[str] = field(default_factory=list)
    alpha : float = field(default = 0.0)
    num_points : float = field(default = 0.5)
    threshold : float = field(default = 0.01) # inverse of z_critical for boostrapping
    max_iter : int = field(default = 20)
    plugin_dict : dict = field(default_factory = dict)
    solver_dict : dict = field(default_factory = dict)

    _flag_fit : bool = field(default = False, init = False)
    adict : dict = field(default_factory = dict, init = False)

    def __post_init__(self):
        assert self.alpha >= 0 and self.threshold >= 0, "Regularization and thresholding parameter should be greater than equal to zero"
        assert self.max_iter >= 1, "Max iteration should be greater than zero"
        assert 0 <= self.num_points <= 1, "percent points to be considered as constraints should be in [0, 1]"

    def set_params(self, **kwargs) -> None:
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


    def _generate_library(self, data : np.ndarray, include_column : List[np.ndarray]) -> None:
        
        # given data creates list of matix of all possible combinations of terms 
        # returns a list of number of columns of each matrix

        # define input features if not given. Input features depend on the shape of data
        if not len(self.input_features):
            self.input_features = [f"x{i}" for i in range(self._input_states)]

        # define symbols that can be converted to equations later
        self.input_symbols = smp.symbols(reduce(lambda accum, value : accum + value + ", ", self.input_features, ""))

        # done using for loop instead of list comprehension becoz each fit_transform and get_features
        # share the same instance of the class
        self.adict["library"] = []
        self.adict["library_labels"] = []
        for i in range(self._reactions):
            self.adict["library"].append(self.library.fit_transform(data, include_column[i], False))
            self.adict["library_labels"].append(self.library.get_features(self.input_features))
        
        self.adict["library_dimension"] = [xi.shape for xi in self.adict["library"]]

    
    def _generate_library_derivative_free(self, data : List[np.ndarray], include_column : List[np.ndarray], time_span : np.ndarray) -> None:
        
        # given data as a list of np.ndarrays creates list of matix of all possible combinations of terms 
        # returns a list of number of columns of each matrix

        # define input features if not given. Input features depend on the shape of data
        if not len(self.input_features):
            self.input_features = [f"x{i}" for i in range(self._input_states)]

        # define symbols that can be converted to equations later
        self.input_symbols = smp.symbols(reduce(lambda accum, value : accum + value + ", ", self.input_features, ""))

        # done using for loop instead of list comprehension becoz each fit_transform and get_features
        # share the same instance of the class
        self.adict["library"] = []
        self.adict["library_labels"] = []
        for i in range(self._reactions):
            self.adict["library"].append(np.vstack([self.library.fit_transform(di, include_column[i], True, time_span) for di in data]))
            self.adict["library_labels"].append(self.library.get_features(self.input_features))
        
        self.adict["library_dimension"] = [xi.shape for xi in self.adict["library"]]
    

    def _create_decision_variables(self) -> None:
        # initializes the number of variables that will be used in casadi optimization 

        self.opti = cd.Opti() # create casadi instance
        # variables are defined individually because MX object cannot be indexed for jacobian/hessian
        self.adict["coefficients"] = [cd.vertcat(*(self.opti.variable(1) for _ in range(dimension[-1]))) for dimension in self.adict["library_dimension"]]
    
    # created separately becoz it needs to be run only once
    def _create_mask(self) -> None:
        self.adict["mask"] = [np.ones(dimension[-1]) for dimension in self.adict["library_dimension"]]

    def _update_cost(self, target : np.ndarray) -> None:
        # initialize the cost to zero
        self.adict["cost"] = 0
        
        # stoichiometric coefficient times the reactions 
        # need 2 for loops because of limitation of casadi 
        self.adict["reactions"] = [cd.mtimes(self.adict["library"][j], self.adict["coefficients"][j]) for j in range(self._reactions)]

        for i in range(self._states):
            asum = 0
            for j in range(self._reactions): 
                asum += self.adict["stoichiometry"][i, j]*self.adict["reactions"][j]

            self.adict["cost"] += cd.sumsqr(target[:, i] - asum)/2

        # normalize the cost by dividing by the number of data points
        # self.adict["cost"] /= self.adict["library_dimension"][0][0] # first array with first dimension

        # add regularization to the cost function
        for j in range(self._reactions):
            self.adict["cost"] += self.alpha*cd.sumsqr(self.adict["coefficients"][j])

    def _add_constraints(self, constraints_dict, seed : int = 12345) -> None:
        
        data_points = self.adict["library_dimension"][0][0]
        rng = np.random.default_rng(seed)
        chosen_rows = rng.choice(range(data_points), int(self.num_points*data_points), replace = False)

        # adding formation constraints
        state_formation = constraints_dict.get("formation", None) if constraints_dict else None
        if state_formation:
            for state in state_formation :
                asum = 0
                for j in range(self._reactions):
                    asum += self.adict["stoichiometry"][state, j]*self.adict["reactions"][j][chosen_rows]

                self.opti.subject_to(asum >= 0)

        # adding consumption constraints
        state_consumption = constraints_dict.get("consumption", None) if constraints_dict else None
        if state_consumption :
            for state in state_consumption:
                asum = 0
                for j in range(self._reactions):
                    asum += self.adict["stoichiometry"][state, j]*self.adict["reactions"][j][chosen_rows]
                
                self.opti.subject_to(asum <= 0)

    def _create_covariance(self) -> List[np.ndarray]:
        """
        returns the diagonal of covariance matrix
        """
        coefficients_value = self.opti.value(self.opti.x)

        cost_function = np.array(cd.Function("cost_function", [self.opti.x], [self.opti.f])(coefficients_value)).flatten()

        alist = []
        variable = []
        for dimension, coeff, coeff_mask in zip(self.adict["library_dimension"], self.adict["coefficients"], self.adict["mask"]):
            
            # zero coefficients have standard deviation of one by default
            alist.append(np.ones(dimension[1]))

            # select nonzero coefficients
            variable.extend([coeff[i] for i, mi in enumerate(coeff_mask.flatten()) if mi != 0])
        
        variable = cd.vertcat(*variable)
        hessian, _ = cd.hessian(self.opti.f, variable)
        hessian_function = cd.Function("hessian_function", [self.opti.x], [hessian])
        
        u, d, vh = np.linalg.svd(hessian_function(coefficients_value))
        d = np.where(d < 1e-10, 0, d)
        # d_inv = cost_function*2/(self.adict["library_dimension"][0][0] - variable.shape[0])/d
        d_inv = cost_function*2/(self.N)/d
        covariance_matrix_diag = np.diag(vh.T@np.diag(d_inv.flatten())@u.T)

        # map the variance to thier respective variables
        i = 0
        for j, coeff_mask in enumerate(self.adict["mask"]):      
            alist[j][coeff_mask.flatten().astype(bool)] = covariance_matrix_diag[i : i + sum(coeff_mask.flatten().astype(int))]
            i += sum(coeff_mask.flatten().astype(int))

        return alist

    def _minimize(self, plugin_dict : dict, solver_dict : dict):

        self.opti.minimize(self.adict["cost"])
        if solver_dict.get("solver", None):
            solver = solver_dict.pop("solver")
        else:
            solver = "ipopt"

        self.opti.solver(solver, plugin_dict, solver_dict)
        solution = self.opti.solve()
        
        assert solution.stats()["success"], "The solution did not converge" 
        return solution

    def _create_parameters(self) -> List:
        return [self.adict["coefficients"]]

    def _initialize_decision_variables(self) -> None:
        pass

    # function for multiprocessing
    def _stlsq_solve_optimization(self, library : List, target : np.ndarray, constraints_dict : dict, permutations : List, seed : int) -> List[List[np.ndarray]]:
        # create problem from scratch since casadi cannot run the same problem once optimized
        # steps should follow a sequence 
        # dont replace if there is only one ensemble iteration. Dataset rows are constant for all reactions 
        # parameters is added so that you can get the values of decision variables without having to rewrite this code
        self._create_decision_variables()
        parameters : List = self._create_parameters()
        self.adict["library"] = [value[permutations]*self.adict["mask"][ind] for ind, value in enumerate(library)]
        self._update_cost(target[permutations])
        if constraints_dict:
            self._add_constraints(constraints_dict, seed)
        self._initialize_decision_variables()
        _solution = self._minimize(self.plugin_dict, self.solver_dict) # no need to save for every iteration

        # list[np.ndarray]. additional layer of np.array and flatten to account for singular value, which casadi outputs as float
        return [[np.array([_solution.value(coeff)]).flatten() for coeff in params] for params in parameters]


    def _stlsq(self, target : np.ndarray, constraints_dict : dict, ensemble_iterations : int, variance_elimination : bool = False, 
                    max_workers : Optional[int] = None, seed : int = 12345) -> List[List[np.ndarray]]:
        
        # parameters is added so that you can get the values of additional decision variables without having to rewrite code
        # however thresholding is only done with respect to the original parameters
        rng = np.random.default_rng(seed)
        self._create_mask()
        # initial boolean array with all coefficients assumed to be less than the threshold value
        coefficients_prev = [np.zeros(dimension[-1]) for dimension in self.adict["library_dimension"]]
        # data structures compatible with multiprocessing : list, dict 
        self.adict["iterations"] = 0
        self.adict["iterations_ensemble"] = ensemble_iterations
        library = self.adict["library"] # use the original library terms

        for _ in tqdm(range(self.max_iter)):
            
            # keys are the reaction number while the values are np.ndarrays of coefficients
            # permutation without replacement when ensemble_iterations == 1
            self.adict["coefficients_casadi_ensemble"] : List[dict] = []
            permutations = [rng.choice(range(self.adict["library_dimension"][0][0]), self.adict["library_dimension"][0][0], replace = (ensemble_iterations > 1))
                                    for _ in range(self.adict["iterations_ensemble"])] 

            if variance_elimination and ensemble_iterations > 1: # use multiprocessing
                with ProcessPoolExecutor(max_workers = max_workers) as executor:           
                    _coefficients_ensemble = list(executor.map(self._stlsq_solve_optimization, repeat(library), repeat(target), repeat(constraints_dict), 
                                                                permutations, repeat(seed)))

                    _coefficients_ensemble = list(zip(*_coefficients_ensemble))
                    for _coefficients_ensemble_parameters in _coefficients_ensemble:
                        bdict = defaultdict(list)
                        for key in range(self._reactions):
                            bdict[key].extend(alist[key] for alist in _coefficients_ensemble_parameters)
                            
                        self.adict["coefficients_casadi_ensemble"].append(bdict)
            
            else : # Do not use multiprocessing.
                _coefficients_ensemble = [self._stlsq_solve_optimization(library, target, constraints_dict, permute, seed) for permute in permutations]
                
                # separate the values of parameters into a list of list of values
                _coefficients_ensemble = list(zip(*_coefficients_ensemble))
                for _coefficients_ensemble_parameters in _coefficients_ensemble: # for every params in parameters
                    bdict = defaultdict(list)
                    for key in range(self._reactions):
                        bdict[key].extend(alist[key] for alist in _coefficients_ensemble_parameters)
                    
                    self.adict["coefficients_casadi_ensemble"].append(bdict)
            
            # calculating mean and standard deviation 
            parameters_len = len(self.adict["coefficients_casadi_ensemble"])
            _mean, _deviation = [[] for _ in range(parameters_len)], [[] for _ in range(parameters_len)]
            
            for i in range(parameters_len):
                _coefficients_ensemble_adict = self.adict["coefficients_casadi_ensemble"][i]
                for key in _coefficients_ensemble_adict.keys():
                    stack = np.vstack(_coefficients_ensemble_adict[key])
                    _mean[i].append(np.mean(stack, axis = 0))
                    _deviation[i].append(np.std(stack, axis = 0))
                    self.adict["coefficients_casadi_ensemble"][i][key] = stack
            
            # list of boolean arrays
            # thresholding parameters are always the first entries of self.adict["coefficients_casadi_ensemble"]
            # thresholding is always performed on the original decision variables i.e. self.adict["coefficients"] and not the coefficients
            # that we get after multiplying with the stoichiometric matrix
            if not variance_elimination:
                print("Thresholding normal sindy")
                coefficients_next = [np.abs(self.adict["coefficients_casadi_ensemble"][0][key]) > self.threshold for key in self.adict["coefficients_casadi_ensemble"][0].keys()]
            else : # variance based thresholding
                if ensemble_iterations > 1:
                    print("Thresholding bootstrapping")
                    coefficients_next = [np.abs(mean/(deviation + 1e-15)) > self.threshold for mean, deviation in zip(_mean[0], _deviation[0])]
                else:
                    print("Thresholding covariance")
                    variance : List[np.ndarray] = self._create_covariance() 
                    _deviation[0] = [np.sqrt(var).flatten() for var in variance] 
                    coefficients_next  = [np.abs(mean/np.sqrt(deviation.reshape(1, -1))) > self.threshold for mean, deviation in zip(_mean[0], variance)]          

            # multiply with mask so that oscillating coefficients can be ignored.
            if np.array([np.allclose(coeff_prev, coeff_next*mask) for coeff_prev, coeff_next, mask in zip(coefficients_prev, coefficients_next, self.adict["mask"])]).all():
                print("Solution converged")
                break

            if not all([np.sum(coeff) for coeff in coefficients_next]):
                raise RuntimeError("Thresholding parameter eliminated all the coefficients")
            
            # store values for every iteration
            if not self.adict.get("coefficients_iterations", None):
                self.adict["coefficients_iterations"] : List[List[dict]] = [[] for _ in range(parameters_len)] # outer list is for each iteration, inner for each parameter
            
            for i in range(parameters_len):
                self.adict["coefficients_iterations"][i].append({"mean" : _mean[i], "standard_deviation" : _deviation[i], 
                                        "z_critical" : [np.abs(mean)/(deviation + 1e-15) for mean, deviation in zip(_mean[i], _deviation[i])], "distribution" : stack})

            coefficients_prev = coefficients_next # boolean array

            # update mask of small terms to zero
            self.adict["mask"] = [mask*coefficients_next[i] for i, mask in enumerate(self.adict["mask"])]
            self.adict["iterations"] += 1

        return _mean, _deviation

    def fit(self, features : List[np.ndarray], target : List[np.ndarray], time_span : np.ndarray, arguments : Optional[List[np.ndarray]] = None, 
            include_column : Optional[List[np.ndarray]] = None, constraints_dict : dict = {} , 
            ensemble_iterations : int = 1, variance_elimination : bool = False, derivative_free : bool = False,
            max_workers : Optional[int] = None, seed : int = 12345) -> None:

        """
        if variance_elimination = True and ensemble_iterations > 1 performs boostrapping
        if variance_elimination = True and ensemble_iterations <= 1 performs covariane matrix
        if variance_elimination = False performs normal thresholding (regular sindy)
        constraints_dict should be of the form {"consumption" : [], "formation" : [], 
                                                  "stoichiometry" : np.ndarray}

        features : list of np.ndarrays of states
        target : list of np.ndarrays. Different for normal sindy and derivative free sindy
        """
        
        # prevents errors in downstream
        if not variance_elimination:
            ensemble_iterations = 1

        self._flag_fit = True
        _output_states = np.shape(target)[-1]
        self._input_states = np.shape(features)[-1]
        self.N = len(features)

        if "stoichiometry" in constraints_dict and isinstance(constraints_dict["stoichiometry"], np.ndarray):
            states, reactions = constraints_dict["stoichiometry"].shape
            assert states == _output_states, "The rows of stoichiometry matrix should match the states of target"
            self.adict["stoichiometry"] = constraints_dict["stoichiometry"]
        else:
            self.adict["stoichiometry"] = np.eye(_output_states) 

        self._states, self._reactions = self.adict["stoichiometry"].shape

        if include_column:
            assert len(include_column) == self._reactions, "length of columns should match with the number of functional libraries"
            include_column = [list(range(self._input_states)) if len(alist) == 0 else alist for alist in include_column] 
        else:
            include_column = [list(range(self._input_states)) for _ in range(self._reactions)]

        if derivative_free:
            target = np.vstack([feat - feat[0] for feat in features])
            self._generate_library_derivative_free(features, include_column, time_span)
        else:
            features, target = np.vstack(features), np.vstack(target)
            self._generate_library(features, include_column)

        # _mean and _deviation : List[dict]
        # self.adict["coefficients_value"] and self.adict["coefficients_devaition"] : dict
        _mean, _deviation = self._stlsq(target, constraints_dict, ensemble_iterations, variance_elimination, max_workers, seed)
        self.adict["coefficients_value"], self.adict["coefficients_deviation"] = _mean[0], _deviation[0]
        self._create_equations()

    # need to consider when stoichiometric in present
    def _create_equations(self) -> None:
        # stores the equations in adict to be used later
        self.adict["equations"] = []
        self.adict["equations_lambdify"] = []
        self.adict["coefficients_dict"] = []
        
        for i in range(self._states):
            expr = self._create_sympy_expressions(self.adict["stoichiometry"][i]) # sympy expression

            # truncate coefficients less than 1e-5 to zero. These values can occur especially in mass balance formulation because of subtracting
            for atom in smp.preorder_traversal(expr):
                if atom.is_rational:
                    continue
                if atom.is_number and abs(atom) < 1e-5:
                    expr = expr.subs(atom, 0)
            
            self.adict["equations"].append(expr)
            self.adict["coefficients_dict"].append(expr.as_coefficients_dict())
            self.adict["equations_lambdify"].append(smp.lambdify(self.input_symbols, expr))


    def _create_sympy_expressions(self, stoichiometry_row : np.ndarray):
        """
        returns sympy expressions
        """
        # mask the coefficient values
        self.adict["coefficients_value_masked"] = []
        for coefficients, mask in zip(self.adict["coefficients_value"], self.adict["mask"]):
            self.adict["coefficients_value_masked"].append(coefficients*mask.flatten())

        coefficients_value : List[np.ndarray] = self.adict["coefficients_value_masked"]
        mask : List[np.ndarray] = self.adict["mask"]
        library_labels : List[List[str]] = self.adict["library_labels"]
        expr = 0

        for j in range(len(library_labels)):
            zero_filter = filter(lambda x : x[0], zip(coefficients_value[j], library_labels[j]))
            expr += stoichiometry_row[j]*smp.sympify(reduce(lambda accum, value : 
                    accum + value[0] + " * " + value[1].replace(" ", "*") + " + ",   
                    map(lambda x : (str(x[0]), x[1]), zero_filter), "+").rstrip(" +")) 
        # replaced whitespaces with multiplication element wise library labels
        # simpify already handles xor operation
        return expr


    def plot_distribution(self, coefficient_casadi_ensemble : Optional[dict] = None, mean : Optional[dict] = None, deviation : Optional[dict] = None,
                            coefficients_iterations : Optional[dict] = None) -> None:
        """
        # TODO updat this method to account for new data structure of self.adict["coefficients_casadi_ensemble"]
        parameter_ind gives the index of the coefficients/parameters defined in _create_parameters method to be plotted
        plots the distribution of casadi coefficients 
        """
        if not coefficient_casadi_ensemble :
            coefficients_casadi_ensemble = self.adict["coefficients_casadi_ensemble"][0]
        if not mean:
            mean = self.adict["coefficients_value"]
        if not deviation :
            deviation = self.adict["coefficients_deviation"]
        if not coefficients_iterations:
            coefficients_iterations = self.adict["coefficients_iterations"][0]

        # create list of dictionary with symbols as keys and arrays as values
        _coefficients_list = [defaultdict(list) for _ in range(self._reactions)]
        
        distribution = namedtuple("distribution", ("mean", "deviation"))
        _coefficients_distribution = [defaultdict(distribution) for _ in range(self._reactions)]

        inclusion = namedtuple("probability", "inclusion")
        _coefficients_inclusion = [defaultdict(inclusion) for _ in range(self._reactions)]

        for i, key in enumerate(coefficients_casadi_ensemble.keys()):
            for j, _symbol in enumerate(self.adict["library_labels"][i]):
                _coefficients_list[i][_symbol].extend(coefficients_casadi_ensemble[key][:, j])
                _coefficients_distribution[i][_symbol] = distribution(mean[i][j], deviation[i][j])
                _coefficients_inclusion[i][_symbol] = inclusion(np.count_nonzero(_coefficients_list[i][_symbol])/self.adict["iterations_ensemble"])

        ensemble_plot(_coefficients_list, _coefficients_distribution, _coefficients_inclusion)

        """
        #TODO create sympy equations for every value in coefficient_casadi_ensemble
        if reaction_coefficients:
            # plotting the distribution of reaction equations
            _reaction_coefficients_list = [defaultdict(list) for _ in range(self._n_states)]
            _reaction_coefficients_distribution = [defaultdict(distribution) for _ in range(self._n_states)]
            _reaction_coefficients_inclusion = [defaultdict(inclusion) for _ in range(self._n_states)]

            for i in range(self._n_states):
                # for each iteration
                for j in range(self.adict["iterations_ensemble"]):
                    _expr = self._create_sympy_expressions([coefficients_casadi_ensemble[key][j] for key in range(self._reactions)], 
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
        """
            
        for key in range(self._reactions):
            fig, ax = plt.subplots(self.adict["iterations"], 3, figsize = (10, 4))
            ax = np.ravel(ax)
            for i, _coefficients_iterations in enumerate(coefficients_iterations):
                
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
            
    def _casadi_model(self, x : np.ndarray, t : np.ndarray, model_args : np.ndarray):
        return np.array([eqn(*x, *model_args) for eqn in self.adict["equations_lambdify"]])
    

    def predict(self, X : List[np.ndarray], model_args : Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        
        assert self._flag_fit, "Fit the model before running predict"
        if model_args :
            # np.vectorize fails for adding arguments in partial that are placed before the vectorized arguments
            afunc = np.vectorize(self._casadi_model, signature = "(m),(),(k)->(n)")
            return [afunc(xi, 0, argi) for xi, argi in zip(X, model_args)]
        else:
            afunc = np.vectorize(partial(self._casadi_model, model_args = ()), signature = "(m),()->(n)")
            return [afunc(xi, 0) for xi in X]
        

    def score(self, X : list[np.ndarray], y : list[np.ndarray], metric : Callable = mean_squared_error, predict : bool = True, 
                model_args : Optional[List[np.ndarray]] = None) -> float:
        
        assert self._flag_fit, "Fit the model before running score"
        y_pred = self.predict(X, model_args) if predict else X

        return metric(np.vstack(y_pred), np.vstack(y))

    # integrate the model
    def simulate(self, X : list[np.ndarray], time_span : np.ndarray, model_args : Optional[np.ndarray] = None, **integrator_kwargs) -> list[np.ndarray]:
        
        """
        X is a list of features whose first row is extracted as initial conditions
        """
        
        assert self._flag_fit, "Fit the model before running score"
        x_init = [xi[0].flatten() for xi in X]
        result = []
        for i, xi in enumerate(x_init):
            assert len(xi) == self._reactions, "Initial conditions should be of right dimensions"

            try:
                _integration_solution = odeint(self._casadi_model, xi, time_span, args = (model_args[i], ) if model_args else ((), ), **integrator_kwargs)
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
        #TODO why not take lenght of dictionary elements of self.adict["coefficients_dict"]
        # self.adict["equations"] are now sympy equations and not strings
        # return sum(eqn.count("+") + eqn.lstrip("-").count("-") + 1 for eqn in self.adict["equations"])
        return sum(len(eqn) for eqn in self.adict["coefficients_dict"])

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
            # traverser throught the expression graph. Replace floating numbers with rounded values
            # round down all the numbers in the sympy expression so that they are easier to visualize
            # rounding down may substitute some coefficients to zero and therefore will be ignored while printing,
            # but are counted in the complexity
            for atom in smp.preorder_traversal(eqn):
                if atom.is_rational:
                    continue
                if atom.is_number:
                    eqn = eqn.subs(atom, round(atom, 2))

            print(f"{self.input_features[i]}' = ", eqn)


if __name__ == "__main__":

    from GenerateData import DynamicModel
    from utils import coefficient_difference_plot

    time_span = np.arange(0, 5, 0.01)
    model = DynamicModel("kinetic_kosir", time_span, arguments = [(373, 8.314)], n_expt = 2, seed = 20)
    features = model.integrate() # list of features
    target = model.approx_derivative # list of target value
    features = model.add_noise(0, 0)
    target = model.approx_derivative

    opti = Optimizer_casadi(FunctionalLibrary(1) , alpha = 0, threshold = 1, plugin_dict = {"ipopt.print_level" : 0, "print_time":0, "ipopt.sb" : "yes"}, 
                            max_iter = 20)
    # stoichiometry = np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1) # chemistry constraints
    # include_column = [[0, 2], [0, 3], [0, 1]]
    include_column = []
    # stoichiometry =  np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -0.5, -1]).reshape(4, -1) # mass balance constraints
    stoichiometry = np.eye(4) # no constraints
    
    derivative_free = True
    
    opti.fit(features, target, time_span = time_span, include_column = include_column, 
                constraints_dict= {"formation" : [], "consumption" : [], 
                                    "stoichiometry" : stoichiometry}, ensemble_iterations = 1000, seed = 10, max_workers = 2, 
                variance_elimination = False, derivative_free = derivative_free)
    opti.print()
    print("--"*20)
    print("mean squared error :", opti.score(features, target))
    print("model complexity", opti.complexity)
    print("Total number of iterations", opti.adict["iterations"])
    print("--"*20)
    # print("coefficients at each iteration", opti.adict["coefficients_iterations"])
    print("--"*20)
    # opti.plot_distribution(reaction_coefficients = False, coefficients_iterations = True)

    # coefficient_difference_plot(model.coefficients() , sigma = opti.adict["coefficients_dict"], sigma2 = opti.adict["coefficients_dict"])