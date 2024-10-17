# type: ignore

from typing import List, Optional, Any, Callable, Tuple, Any
from functools import reduce

import casadi as cd
print("casadi version", cd.__version__)
import numpy as np
import sympy as smp
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from optimizer import Optimizer_casadi
from functional_library import FunctionalLibrary


class EnergySindy(Optimizer_casadi):

    def __init__(self, 
                    library : FunctionalLibrary = FunctionalLibrary(2),
                    input_features : List[str] = [],
                    alpha : float = 0.0,
                    threshold : float = 0.01,
                    max_iter : int = 20,
                    plugin_dict : dict = {},
                    solver_dict : dict = {},
                    initializer : str = "ones",
                    _dir : str = "",
                    logger : Optional[Callable] = None
                ):
        
        self.library = library
        self.input_features = input_features
        self.alpha = alpha
        self.threshold = threshold
        self.max_iter = max_iter
        self.plugin_dict = plugin_dict
        self.solver_dict = solver_dict
        self.initializer = initializer
        self._dir = _dir
        self.logger = logger

        super().__init__(
            self.library, 
            self.input_features,
            self.alpha,
            self.threshold,
            self.max_iter,
            self.plugin_dict,
            self.solver_dict,
            self.initializer,
            self._dir,
            self.logger
            )

    def _create_decision_variables(self):
        super()._create_decision_variables() # reaction rates at reference temperature of 373 K
        # adding activation energies
        # variables are defined individually because MX objects cannot be indexed for jacobian/hessian
        self.adict["coefficients_energy"] = [cd.vertcat(*(self.opti.variable(1) for _ in range(dimension[-1]))) for dimension in self.adict["library_dimension"]]
        
    def _generate_library(self, data : np.ndarray, include_column : List[np.ndarray]) -> None:
        
        super()._generate_library(data, include_column)
        # create new symbol for temperature and universal gas constant
        self.input_symbols = (*self.input_symbols, *smp.symbols("T, R"))

    def _generate_library_derivative_free(self, data : List[np.ndarray], include_column : List[np.ndarray], time_span : np.ndarray) -> None:

        super()._generate_library_derivative_free(data, include_column, time_span)
        self.input_symbols = (*self.input_symbols, *smp.symbols("T, R"))

    def _update_cost(self, target: np.ndarray):

        # initialize the cost to zero
        self.adict["cost"] = 0
        
        # stoichiometric coefficient times the reactions 
        # need 2 for loops because of limitation of casadi 
        # the reaction rate will be arhenius equation
        reaction_rate = [cd.vertcat(*[(A*cd.exp(-B*10_000*(1/ti - 1/373)/R)).T for ti, R in self.adict["arguments"]]) # for all temperatures
                                    for A, B in zip(self.adict["coefficients"], self.adict["coefficients_energy"])] # for all reactions
        
        self.adict["reactions"] = [cd.einstein(cd.vec(A), cd.vec(x), [*A.shape], [*x.shape], [A.shape[0]], [-1, -2], [-1, -2], [-1]) 
                                    for A, x in zip(self.adict["library"], reaction_rate)]

        # multiply with the stoichiometric matrix
        for i in range(self._states):
            asum = 0
            for j in range(self._reactions): 
                asum += self.adict["stoichiometry"][i, j]*self.adict["reactions"][j]

            self.adict["cost"] += cd.sumsqr(target[:, i] - asum)/2

        # normalize the cost by dividing by the number of data points
        # self.adict["cost"] /= self.adict["library_dimension"][0][0] # first array with first dimension

        # add regularization to the cost function
        for coefficients in self.adict["coefficients"]:
            self.adict["cost"] += self.alpha*cd.einstein(cd.vec(coefficients), cd.vec(coefficients), 
                                                    [*coefficients.shape], [*coefficients.shape], [], [-1, -2], [-1, -2], []) 

    def _add_constraints(self, constraints_dict, seed : int = 12345):

        if constraints_dict.get("activation_energy", False):
            # make all the activation energies positive
            for coefficient_energy in self.adict["coefficients_energy"]:
                self.opti.subject_to(cd.vec(coefficient_energy) >= 0)

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
        hessian_function = cd.Function("hessian_function", [self.opti.x], [cd.mtimes(hessian.T, hessian)])
        
        u, d, vh = np.linalg.svd(hessian_function(coefficients_value))
        d = np.where(d < 1e-10, 0, d)
        d_inv = cost_function*2/self.N/d
        covariance_matrix_diag = np.diag(vh.T@np.diag(d_inv.flatten())@u.T)

        # map the variance to thier respective variables
        i = 0
        for j, coeff_mask in enumerate(self.adict["mask"]):      
            alist[j][coeff_mask.flatten().astype(bool)] = covariance_matrix_diag[i : i + sum(coeff_mask.flatten().astype(int))]
            i += sum(coeff_mask.flatten().astype(int))

        return alist

    def _create_parameters(self) -> List:
        """
        List of decision variables for which mean and standard deviation needs to be traced. 
        The thresholding parameters always have to be the first one in the list
        """
        return [self.adict["coefficients"], self.adict["coefficients_energy"]]

    def _stlsq_solve_optimization(self, **kwargs) -> List[List[np.ndarray]]:
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
        self.adict["library"] = [value*self.adict["mask"][ind] for ind, value in enumerate(library)]
        # shuffle the arguments as well
        self.adict["arguments"] = self.adict["arguments_original"]
        self._update_cost(target)
        if constraints_dict:
            self._add_constraints(constraints_dict, seed)
        self._initialize_decision_variables()
        _solution = self._minimize(self.plugin_dict, self.solver_dict) # no need to save for every iteration

        # list[np.ndarray]. additional layer of np.array and flatten to account for singular value, which casadi outputs as float
        return [[np.array([_solution.value(coeff)]).flatten() for coeff in params] for params in self._parameters]

    def fit(self, features : List[np.ndarray], target : List[np.ndarray], time_span : np.ndarray, arguments : List[Any], 
                include_column : Optional[List[np.ndarray]] = None, constraints_dict : dict = {}, derivative_free : bool = False,
                max_workers : Optional[int] = None, seed : int = 12345, **kwargs) -> None:
        """
        target : the derivatives of states. depending on the formulation it will be used or replaced with states
        arguments is a list of arrays so that its compatible with vectorize
        constraints_dict should be of the form {"stoichiometry" : np.ndarray}
        """

        self._flag_fit = True
        self._input_states = np.shape(features)[-1]
        _output_states = np.shape(target)[-1] if not derivative_free else self._input_states
        self.N = len(features)

        assert len(arguments) == len(features), "Arguments and features should be consistent with the number of experiments"
        if arguments[0].ndim == 2 and len(arguments[0]) == len(features[0]):
            # There are as many arguments as there are data points. Just stack
            self.adict["arguments_original"] = np.squeeze(np.vstack(arguments))
        else:
            # There are not enough arguments as there are data points (each array in features can have varying data points)
            self.adict["arguments_original"] = np.squeeze(np.vstack([np.tile(args, (len(feat), 1)) for args, feat in zip(arguments, features)]))

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
            # use the features as target values. In this case there cannot be a mismatch between features and target columns
            target = np.vstack([feat - feat[0] for feat in features])
            self._generate_library_derivative_free(features, include_column, time_span)
        else:
            features, target = np.vstack(features), np.vstack(target)
            self._generate_library(features, include_column)

        self._create_mask() # pulled outside optimization so that can be used in cost function
        _mean = self._stlsq(target, constraints_dict, max_workers, seed)
        (self.adict["coefficients_value"], self.adict["coefficients_energy_value"]) = (_mean[0], _mean[1])
        self._create_equations()

    def simulate(self, X : list[np.ndarray], time_span : np.ndarray, model_args : Optional[np.ndarray] = None, calculate_score : bool = False, 
                metric : List[Callable] = [mean_squared_error], **integrator_kwargs) -> list[np.ndarray]:
        
        """
        X is a list of features whose first row is extracted as initial conditions
        """
        if model_args is not None:
        
            assert len(model_args) == len(X), "Arguments should be same the number of initial conditions"
            
            def afunc(xi, argi):
                if isinstance(argi, np.ndarray):
                    if argi.ndim == 2 and len(argi) == len(xi):
                        return [CubicSpline(time_span, argi[:, 0]), argi[0, -1]]
                
                return argi

            model_args = list(map(afunc, X, model_args))

        return super().simulate(X, time_span, model_args, calculate_score, metric, **integrator_kwargs)

    @property
    def coefficients(self):
        return self.adict["coefficients_value"], self.adict["coefficients_energy_value"]

    def _create_sympy_expressions(self) -> str:

        # mask the coefficient values
        self.adict["coefficients_value_masked"] = []
        for coefficients, mask in zip(self.adict["coefficients_value"], self.adict["mask"]):
            self.adict["coefficients_value_masked"].append(coefficients*mask.flatten())

        # Do not round the coefficients here (Rounding may compromise accuracy while prediciton or scoring). 
        # Round them only when printing
        coefficients_value : List[np.ndarray] = self.adict["coefficients_value_masked"]
        coefficients_energy : List[np.ndarray] = self.adict["coefficients_energy_value"]
        library_labels : List[List[str]] = self.adict["library_labels"]
        _expr = []
        
        # modify expr to include arhenius equation (R and T are additional symbols that are defined)
        # replaced whitespaces with multiplication element wise library labels
        # simpify already handles xor operation
        for j in range(len(library_labels)):
            zero_filter = filter(lambda x : x[0], zip(coefficients_value[j], coefficients_energy[j], library_labels[j]))
            _astring = reduce(lambda accum, value : 
                    accum + value[0] + "*exp(-(" + value[1] + "/R)*(1/T - Rational(1, 373)))* " + value[-1].replace(" ", "*") + " + ",   
                    map(lambda x : (str(x[0]), str(x[1]*10_000), x[-1]), zero_filter), "+").rstrip(" +")
            _expr.append(smp.sympify(_astring))
        return _expr


if __name__ == "__main__":

    import logging

    from generate_data import DynamicModel
    from utils import coefficients_plot

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logfile = logging.FileHandler("energy.txt")
    logger.addHandler(logfile)

    """
    ## Running temperature dependant df-sindy (integral based) formulation 
    time_span = np.arange(0, 5, 0.01)
    arguments = [(360, 8.314), (370, 8.314), (380, 8.314), (390, 8.314), (373, 8.314), (385, 8.314)][:4]
    model = DynamicModel("kinetic_kosir", time_span, n_expt = len(arguments) if arguments else 4, arguments = arguments, seed = 20)
    features = model.integrate() # list of features
    target = model.approx_derivative # list of target value
    features = model.add_noise(0, 0.2)
    target = model.approx_derivative
    arguments = model.arguments
     
    plugin_dict = {"ipopt.print_level" : 5, "print_time":5, "ipopt.sb" : "yes", "ipopt.max_iter" : 1000}
    # plugin_dict = {}
    opti = EnergySindy(FunctionalLibrary(1) , alpha = 0.1, threshold = 0.5, solver_dict={"solver" : "ipopt"}, 
                            plugin_dict = plugin_dict, max_iter = 20, logger = logger)
    
    stoichiometry = np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1) # chemistry constraints
    include_column = [[0, 2], [0, 3], [0, 1]] # chemistry constraints

    opti.fit(features, target, time_span, arguments, include_column = include_column, 
                constraints_dict= {"formation" : [], "consumption" : [], "stoichiometry" : stoichiometry}, 
                derivative_free = True, seed = 20, max_workers = 2)

    opti.print()
    print("--"*20)
    print("mean squared error :", opti.score(features, target, time_span, model_args = arguments))
    print("model complexity", opti.complexity)
    print("Total number of iterations", opti.adict["iterations"])
    print("--"*20)
    print("coefficients energy at each iteration", opti.adict["coefficients_energy_value"])
    print("--"*20)
    print("model simulation", opti.simulate(features, time_span, arguments))
    print("--"*20)
    # opti.plot_distribution()

    # coefficient_difference_plot(model.coefficients , sigma = opti.adict["coefficients_dict"], sigma2 = opti.adict["coefficients_dict"])
    """

    ## Running temperature dependent sindy (derivative based) formulation 
    time_span = np.arange(0, 5, 0.01)
    model = DynamicModel("kinetic_kosir_temperature", time_span, n_expt = 2)
    integration = model.integrate()
    features = [feat[:, :-1] for feat in integration]
    target = [tar[:, :-1] for tar in model.approx_derivative]
    arguments = [np.column_stack((feat[:, -1], np.ones(len(feat))*8.314)) for feat in integration]

    plugin_dict = {"ipopt.print_level" : 5, "print_time":5, "ipopt.sb" : "yes", "ipopt.max_iter" : 1000}
    opti = EnergySindy(FunctionalLibrary(2) , alpha = 0.1, threshold = 0.5, solver_dict={"solver" : "ipopt"}, 
                            plugin_dict = plugin_dict, max_iter = 1, logger = logger)
    
    stoichiometry = np.array([-1, -1, -1, 2, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(4, -1) # chemistry constraints
    include_column = [[0, 1], [0, 2], [0, 3]] # chemistry constraints

    opti.fit(features, target, time_span, arguments, include_column = include_column, 
                constraints_dict= {"stoichiometry" : stoichiometry}, 
                derivative_free = False, seed = 20, max_workers = 2)
    
    opti.print()
    print("--"*20)
    print("mean squared error :", opti.score(features, target, time_span, model_args = arguments))
    print("model complexity", opti.complexity)
    print("Total number of iterations", opti.adict["iterations"])
    print("--"*20)
    print("coefficients energy at each iteration", opti.adict["coefficients_energy_value"])
    print("--"*20)
    # print("simulate", opti.simulate(features, time_span, arguments))
    
    # preprocess coefficients before plotting
    def replace_keys(alist):
        
        def _replace(adict):

            keys = list(adict.keys())
            for key in keys:
                value = adict[key]
                new_key = smp.Mul(*key.args[:-1])
                adict[new_key] = value
                del adict[key]
            return adict

        return list(map(_replace, alist))

    actual_coeff = replace_keys(model.coefficients(args_as_symbols = True))
    discovered_coeff = replace_keys(opti.adict["coefficients_pre_stoichiometry_dict"])
    coefficients_plot(actual_coeff, discovered_coeff, title = "Dynamics of reaction")