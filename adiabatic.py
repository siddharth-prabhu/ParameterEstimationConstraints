# type: ignore
from typing import List, Optional, Callable, Any
from functools import reduce
import operator
import warnings
warnings.filterwarnings("ignore")

import casadi as cd
import numpy as np
import sympy as smp
from sklearn.metrics import mean_squared_error

from FunctionalLibrary import FunctionalLibrary
from energy import EnergySindy


class AdiabaticSindy(EnergySindy):

    def __init__(self, 
                library : FunctionalLibrary = FunctionalLibrary(2),
                input_features : List[str] = [],
                alpha : float = 0.0,
                num_points : float = 0.5,
                threshold : float = 0.01, # inverse of z_critical for boostrapping
                max_iter : int = 20,
                plugin_dict : dict = {},
                solver_dict : dict = {},
                initializer : str = "zeros"
                ):
        
        self.library = library
        self.input_features = input_features
        self.alpha = alpha
        self.num_points = num_points
        self.threshold = threshold
        self.max_iter = max_iter
        self.plugin_dict = plugin_dict
        self.solver_dict = solver_dict
        self.initializer = initializer

        super().__init__(
            self.library, 
            self.input_features,
            self.alpha,
            self.num_points,
            self.threshold,
            self.max_iter,
            self.plugin_dict,
            self.solver_dict,
            self.initializer,
            )
        

    def _generate_library_derivative_free(self, data : List[np.ndarray], include_column : List[np.ndarray], time_span : np.ndarray, 
                            output_time_span : np.ndarray, target : List[np.ndarray]) -> None:
        
        # given data as a list of np.ndarrays creates list of matix of all possible combinations of terms 
        # returns a list of number of columns of each matrix

        # define input features if not given. Input features depend on the shape of data
        if not len(self.input_features):
            self.input_features = [f"x{i}" for i in range(self._input_states)]

        # define symbols that can be converted to equations later
        self.input_symbols = smp.symbols(reduce(lambda accum, value : accum + value + ", ", self.input_features, ""))

        # done using for loop instead of list comprehension becoz each fit_transform and get_features
        # share the same instance of the class
        # library has a different shape here
        self.adict["library"] : List[List[np.ndarray]] = [] # shape = [#reactions, #initial conditions, #columns]
        self.adict["library_labels"] = []
        for i in range(self._reactions):
            self.adict["library"].append([np.diff(self.library.fit_transform(di, include_column[i], True, time_span, output_time_span, subtract_initial = False), axis = 0) for di in data])
            self.adict["library_labels"].append(self.library.get_features(self.input_features))
        
        # switch the dimensions of library for easy downstream operations, new shape = [#initial conditions, #reactions, #columns]
        self.adict["library"] = [[reaction[i] for reaction in self.adict["library"]] for i in range(self.N)]
        self.adict["library_dimension"] = [xi.shape for xi in self.adict["library"][0]]

        # Fit interpolation and get intermediate target values. Dont integrate the terms, just get their values at finer time points
        lib = FunctionalLibrary(1)
        self.adict["arguments_original"] = [
            lib.fit_transform(arg, derivative_free = True, time_span = time_span, output_time_span = output_time_span, subtract_initial = False, integrate_terms = False)[1:] 
            for arg in self.adict["arguments_original"]
            ]

        self.adict["target"] = [lib.fit_transform(tar, derivative_free = True, time_span = time_span, output_time_span = output_time_span, integrate_terms = False)[1:] for tar in target]
        self.input_symbols = (*self.input_symbols, *smp.symbols("T, R"))

    def _update_cost(self, target: List[np.ndarray]):
        # target here is list of np.ndarrays
        # add ode function here and for all experiments
        # initialize the cost to zero
        self.adict["cost"] = 0
        
        # stoichiometric coefficient times the reactions 
        # the reaction rate will be arhenius equation
        for _lib, _args, _tar in zip(self.adict["library"], self.adict["arguments"], target):
            # for each initial conditions
            _reactions = []
            for i, (_react_lib, A, B) in enumerate(zip(_lib, self.adict["coefficients"], self.adict["coefficients_energy"])):
                # for each reaction

                reaction_rate = cd.vertcat(*[(A*cd.exp(-B*10_000*(1/ti - 1/373)/R)).T for ti, R in _args])
                solution = []
                for j in range(len(_react_lib)):
                    if solution :
                        solution.append(solution[-1] + cd.dot(reaction_rate[j, :].T, _react_lib[j]))
                    else:
                        solution.append(cd.dot(reaction_rate[j, :].T, _react_lib[j]))

                _reactions.append(cd.vertcat(*solution))

            # multiply with stoichiometric matrix
            for k in range(self._states):
                asum = 0
                for j in range(self._reactions):
                    if self.adict["stoichiometry"][k, j] != 0:
                        asum += self.adict["stoichiometry"][k, j]*_reactions[j]
                
                self.adict["cost"] += cd.sumsqr(_tar[:, k] - asum)
        
        # adding regularization 
        for coeff in self.adict["coefficients"]:
            self.adict["cost"] += self.alpha*cd.sumsqr(coeff)

    # function for multiprocessing
    def _stlsq_solve_optimization(self, permutations : List, **kwargs) -> List[List[np.ndarray]]:
        # create problem from scratch since casadi cannot run the same problem once optimized
        # steps should follow a sequence 
        # dont replace if there is only one ensemble iteration. Dataset rows are constant for all reactions 
        # parameters is added so that you can get the values of decision variables without having to rewrite this code
        library = kwargs.get("library", None)
        target = kwargs.get("target", None)
        constraints_dict = kwargs.get("constraints_dict", None)
        seed = kwargs.get("seed", None)
        assert ((library is not None) and (target is not None) and (constraints_dict is not None) and 
                (seed is not None)), "library, target, constraints_dict and seed should be provided"

        self._create_decision_variables()
        self._parameters : List = self._create_parameters()
        self.adict["library"] = [[react_lib*self.adict["mask"][ind] for ind, react_lib in enumerate(lib)] for lib in library]
        self.adict["arguments"] = self.adict["arguments_original"]

        self._update_cost(target)
        if constraints_dict:
            self._add_constraints(constraints_dict, seed)
        self._initialize_decision_variables()
        _solution = self._minimize(self.plugin_dict, self.solver_dict) # no need to save for every iteration

        # list[np.ndarray]. additional layer of np.array and flatten to account for singular value, which casadi outputs as float
        return [[np.array([_solution.value(coeff)]).flatten() for coeff in params] for params in self._parameters]

    def fit(self, features : List[np.ndarray], target : List[np.ndarray], time_span : np.ndarray, arguments : List[np.ndarray],
                delta_t : float = 0.01, include_column : Optional[List[np.ndarray]] = None, constraints_dict : dict = {} , 
                max_workers : Optional[int] = None, seed : int = 12345, **kwargs) -> None:
        """
        target : the derivatives of states. depending on the formulation it will be used or replaced with states
        arguments is a list of arrays so that its compatible with vectorize
        if variance_elimination = True and ensemble_iterations > 1 performs boostrapping
        if variance_elimination = True and ensemble_iterations <= 1 performs covariane matrix
        if variance_elimination = False performs normal thresholding (regular sindy)
        constraints_dict should be of the form {"consumption" : [], "formation" : [], 
                                                   "stoichiometry" : np.ndarray}
        """

        self._flag_fit = True
        self._input_states = np.shape(features)[-1] - 1 # do not consider temperature
        _output_states = np.shape(target)[-1]
        self.N = len(features)

        assert len(arguments) == len(features), "Arguments and features should be consistent with the number of experiments"
        if arguments[0].ndim == 2 and len(arguments[0]) == len(features[0]):
            # There are as many arguments as there are data points. Just stack
            self.adict["arguments_original"] = arguments
        else:
            # There are not enough arguments as there are data points (each array in features can have varying data points)
            assert False, "There are not enought arguments as there are data points"

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

        # use the features as target values. In this case there cannot be a mismatch between features and target columns
        output_time_span = np.arange(time_span[0], time_span[-1], delta_t)
        features = [feat[:, :-1] for feat in features] # remove features
        self._generate_library_derivative_free(features, include_column, time_span, output_time_span, target)
        target = self.adict["target"]
    
        self._create_mask() # pulled outside optimization so that can be used in cost function
        _mean, _deviation = self._stlsq(target, constraints_dict, 1, False, max_workers, seed)
        (self.adict["coefficients_value"], self.adict["coefficients_energy_value"], self.adict["coefficients_deviation"], 
                                        self.adict["coefficients_energy_deviation"]) = (_mean[0], _mean[1], _deviation[0], _deviation[1])
        self._create_equations()

    def simulate(self, X : list[np.ndarray], time_span : np.ndarray, model_args : Optional[np.ndarray] = None, calculate_score : bool = False, 
                    metric : List[Callable] = [mean_squared_error], **integrator_kwargs) -> list[np.ndarray]:  
        
        # remove the temperature column from the data
        X = [xi[:, :-1] for xi in X]
        return super().simulate(X, time_span, model_args, calculate_score, metric, **integrator_kwargs)

    def predict(self, X : List[np.ndarray], time_span : np.ndarray, model_args : Optional[List[Any]] = None) -> List[np.ndarray]:
        
        # remove the temperature column from the data
        X = [xi[:, :-1] for xi in X]
        return super().predict(X, time_span, model_args)


if __name__ == "__main__":

    from GenerateData import DynamicModel
    from scipy.interpolate import CubicSpline

    time_span = np.arange(0, 5, 0.1)
    n_expt = 5
    model = DynamicModel("kinetic_kosir_temperature", time_span, n_expt = n_expt, seed = 20)
    features = model.integrate()
    target =  [feat[:, :-1] for feat in features]
    arguments = [np.column_stack((feat[:, -1], 8.314*np.ones(len(feat)))) for feat in features]

    plugin_dict = {"ipopt.print_level" : 5, "print_time":5, "ipopt.sb" : "yes", "ipopt.max_iter" : 2000, "ipopt.tol" : 1e-5}
    opti = AdiabaticSindy(FunctionalLibrary(2) , alpha = 0.01, threshold = 0.1, solver_dict={"solver" : "ipopt"}, 
                            plugin_dict = plugin_dict, max_iter = 1)
    
    include_column = [[0, 1], [0, 2], [0, 3]] 
    stoichiometry = np.array([-1, -1, -1, 2, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(4, -1)
    # include_column = None

    opti.fit(features, target, time_span, arguments = arguments, 
                include_column = include_column, 
                constraints_dict= {"formation" : [], "consumption" : [],
                                    "stoichiometry" : stoichiometry})

    opti.print()
    print("--"*20)
    
    interp = CubicSpline(time_span, features[0][:, -1])
    print("mean squared error :", opti.score(features, [deriv[:, :-1] for deriv in model.actual_derivative], time_span, model_args = arguments))
    print("model complexity", opti.complexity)
    print("Total number of stlsq iterations", opti.adict["iterations"])
    print("--"*20)
    