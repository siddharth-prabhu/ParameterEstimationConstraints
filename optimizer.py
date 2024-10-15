# type: ignore

from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Any
from functools import reduce, partial
from collections import defaultdict, namedtuple

import numpy as np
import casadi as cd
import sympy as smp
import pickle
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.integrate import odeint

from base import Base
from functional_library import FunctionalLibrary


@dataclass(frozen = False)
class Optimizer_casadi(Base):

    library : FunctionalLibrary = field(default = FunctionalLibrary())
    input_features : list[str] = field(default_factory=list)
    alpha : float = field(default = 0.0)
    threshold : float = field(default = 0.01)
    max_iter : int = field(default = 20)
    plugin_dict : dict = field(default_factory = dict)
    solver_dict : dict = field(default_factory = dict)
    initializer : str = field(default = "ones")

    _flag_fit : bool = field(default = False, init = False)
    adict : dict = field(default_factory = dict, init = False)

    def __post_init__(self) : self.check_params()

    def check_params(self):
        assert isinstance(self.alpha, Callable) or (isinstance(self.alpha, float) and self.alpha >= 0), "Regularization parameter (alpha) should either be callable or greater than 0"
        assert self.threshold >= 0, "Thresholding parameter should be greater than equal to zero"
        assert self.max_iter >= 1, "Max iteration should be greater than zero"

    def set_params(self, **kwargs) -> None:
        # sets the values of various parameter for gridsearchcv

        if "optimizer__alpha" in kwargs:
            setattr(self, "alpha", kwargs["optimizer__alpha"])
        
        if "optimizer__threshold" in kwargs:
            setattr(self, "threshold", kwargs["optimizer__threshold"])
        
        if "optimizer__max_iter" in kwargs:
            setattr(self, "max_iter", kwargs["optimizer__max_iter"])

        self.check_params()
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
                if self.adict["stoichiometry"][i, j] != 0:
                    asum += self.adict["stoichiometry"][i, j]*self.adict["reactions"][j]

            self.adict["cost"] += cd.sumsqr(target[:, i] - asum)/2

        # add regularization to the cost function
        _alpha = self.alpha(self.adict.get("iterations", 0)) if isinstance(self.alpha, Callable) else self.alpha
        for j in range(self._reactions):
            if _alpha > 0 :
                self.adict["cost"] += _alpha*cd.sumsqr(self.adict["coefficients"][j])

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
        """ Parameters used for thresholding """
        return [self.adict["coefficients"]]

    def _initialize_decision_variables(self) -> None:
        """
        Sets the initial guess for decision varibles
        Default guess value chosen by casadi is zero
        """
        iteration = self.adict.get("iterations", 0)
        if iteration == 0:
            # use the specified initializer to set the initial conditions
            if self.initializer == "zeros":
                pass # default initializer in casadi is zeros
            elif self.initializer == "ones":
                self.opti.set_initial(self.opti.x, np.ones(self.opti.x.shape))
            else:
                assert False, f"Initializer {self.initializer} not recognized"
        else:
            # use the previous optimal solution as the starting point
            for i, param in enumerate(self._parameters):
                for key, value in zip(param, self.adict["initial_conditions"][i]):
                    self.opti.set_initial(key, cd.reshape(value, key.shape))

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
        self._update_cost(target)
        self._initialize_decision_variables()
        _solution = self._minimize(self.plugin_dict, self.solver_dict) # no need to save for every iteration

        # optimal values are passed because casadi objects cannot be pickled for multiprocessing
        # list[np.ndarray]. additional layer of np.array and flatten to account for singular value, which casadi outputs as float
        return [[np.array([_solution.value(coeff)]).flatten() for coeff in params] for params in self._parameters]

    def _stlsq(self, target : np.ndarray, constraints_dict : dict, time_span : Optional[np.ndarray] = None, 
               max_workers : Optional[int] = None, seed : int = 12345) -> List[List[np.ndarray]]:
        
        # parameters is added so that you can get the values of additional decision variables without having to rewrite code
        # however thresholding is only done with respect to the original parameters
        # initial boolean array with all coefficients assumed to be less than the threshold value
        coefficients_prev = [np.zeros(dimension[-1]) for dimension in self.adict["library_dimension"]]
        # data structures compatible with multiprocessing : list, dict 
        self.adict["iterations"] = 0
        library = self.adict["library"] # use the original library terms

        for iteration in tqdm(range(self.max_iter)):
            
            # keys are the reaction number while the values are np.ndarrays of coefficients
            self.adict["coefficients_casadi"] : List[dict] = []
            _coefficients = self._stlsq_solve_optimization(library = library, target = target, constraints_dict = constraints_dict, seed = seed, time_span = time_span)
            
            # separate the values of parameters into a list of list of values
            for _coefficients_parameters in _coefficients: # for every params in parameters
                bdict = defaultdict(list)
                for key in range(self._reactions):
                    bdict[key] = _coefficients_parameters[key]
                
                self.adict["coefficients_casadi"].append(bdict)
                     
            # list of boolean arrays
            # thresholding parameters are always the first entries of self.adict["coefficients_casadi"]
            # thresholding is always performed on the original decision variables i.e. self.adict["coefficients"] and not the coefficients
            # that we get after multiplying with the stoichiometric matrix
            print("Thresholding normal sindy")
            coefficients_next = [np.abs(self.adict["coefficients_casadi"][0][key]) > self.threshold for key in self.adict["coefficients_casadi"][0].keys()]
        
            # check for full elimination before checking for convergence. Because we assume all coefficients initially are below threshold. 
            # if new coefficients indeed come out to be less than threshold, then they can be caught by the following check
            if sum(np.sum(coeff) for coeff in coefficients_next) == 0:
                raise RuntimeError("Thresholding parameter eliminated all the coefficients")

            # multiply with mask so that oscillating coefficients can be ignored.
            if np.array([np.allclose(coeff_prev, coeff_next*mask) for coeff_prev, coeff_next, mask in zip(coefficients_prev, coefficients_next, self.adict["mask"])]).all():
                print("Solution converged")
                break
            
            # store values for every iteration
            if iteration == 0 : self.adict["coefficients_iterations"] : List[List[dict]] = [] # outer list is for each iteration, inner for each parameter
            self.adict["coefficients_iterations"].extend(self.adict["coefficients_casadi"])

            coefficients_prev = coefficients_next # boolean array

            # update mask of small terms to zero, and initial conditions for next iteration
            self.adict["mask"] = [mask*coefficients_next[i] for i, mask in enumerate(self.adict["mask"])]
            self.adict["initial_conditions"] = _coefficients
            self.adict["iterations"] += 1

        return _coefficients

    def fit(self, features : List[np.ndarray], target : List[np.ndarray], time_span : np.ndarray, arguments : Optional[List[np.ndarray]] = None, 
            include_column : Optional[List[np.ndarray]] = None, constraints_dict : dict = {} , 
            derivative_free : bool = False, max_workers : Optional[int] = None, seed : int = 12345) -> None:

        """
        constraints_dict should be of the form {"stoichiometry" : np.ndarray}

        features : list of np.ndarrays of states
        target : list of np.ndarrays. Different for normal sindy and derivative free sindy
        """

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

        self._create_mask() # pulled outside optimization so that can be used in cost function
        _mean = self._stlsq(target, constraints_dict, max_workers, seed)
        self.adict["coefficients_value"] = _mean[0]
        self._create_equations()

    def _create_equations(self) -> None:
        # stores the equations in adict to be used later
        self.adict["equations"] = []
        self.adict["equations_lambdify"] = []
        self.adict["coefficients_dict"] = []
        
        exprs = self._create_sympy_expressions()
        self.adict["equations_pre_stoichiometry"] = exprs
        self.adict["coefficients_pre_stoichiometry_dict"] = [expr.as_coefficients_dict() for expr in exprs]
        exprs = np.dot(self.adict["stoichiometry"], exprs) # sympy expressions
        
        for expr in exprs:
            # truncate coefficients less than 1e-5 to zero. These values can occur especially in mass balance formulation because of subtracting
            for atom in smp.preorder_traversal(expr):
                if atom.is_rational:
                    continue
                if atom.is_number and abs(atom) < 10 * self.solver_dict.get("tol", 1e-5):
                    expr = expr.subs(atom, 0)
            
            self.adict["equations"].append(expr)
            self.adict["coefficients_dict"].append(expr.as_coefficients_dict())
            self.adict["equations_lambdify"].append(smp.lambdify(self.input_symbols, expr))

    def _create_sympy_expressions(self):
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
        _expr = []

        for j in range(len(library_labels)):
            zero_filter = filter(lambda x : x[0], zip(coefficients_value[j], library_labels[j]))
            _astring = reduce(lambda accum, value : 
                    accum + value[0] + " * " + value[1].replace(" ", "*") + " + ",   
                    map(lambda x : (str(x[0]), x[1]), zero_filter), "+").rstrip(" +")
            _expr.append(smp.sympify(_astring))
        # replaced whitespaces with multiplication element wise library labels
        # simpify already handles xor operation
        return _expr
        
    def _casadi_model(self, x : np.ndarray, t : np.ndarray, model_args : Any):

        if model_args is not None and len(model_args) > 0 and isinstance(model_args[0], Callable):
            return np.array([eqn(*x, *(model_args[0](t), *model_args[1:])) for eqn in self.adict["equations_lambdify"]])

        return np.array([eqn(*x, *model_args) for eqn in self.adict["equations_lambdify"]])
    
    def predict(self, X : List[np.ndarray], time_span : np.ndarray, model_args : Optional[List[Any]] = None) -> List[np.ndarray]:
        
        assert self._flag_fit, "Fit the model before running predict"
        if model_args is not None:
        
            def afunc(x, argi):
                if isinstance(argi[0], Callable):
                    # callable single list
                    return np.vectorize(partial(self._casadi_model, model_args = argi), signature= "(m),()->(n)")(x, time_span)
                elif isinstance(argi, np.ndarray):
                    if argi.ndim == 2 and argi.shape[0] == len(time_span):
                        return np.vectorize(self._casadi_model, signature = "(m),(),(k)->(n)")(x, time_span, argi)
                    else:
                        return np.vectorize(partial(self._casadi_model, model_args = argi), signature= "(m),()->(n)")(x, time_span)
                else:
                    assert False, f"Unrecognized arguments {argi}"

            # np.vectorize fails for adding arguments in partial that are placed before the vectorized arguments
            return list(map(afunc, X, model_args))
        else:
            afunc = np.vectorize(partial(self._casadi_model, model_args = ()), signature = "(m),()->(n)")
            return [afunc(xi, 0) for xi in X]
        
    def score(self, X : list[np.ndarray], y : list[np.ndarray], time_span : np.ndarray, metric : Callable = mean_squared_error, predict : bool = True, 
                model_args : Optional[List[np.ndarray]] = None) -> float:
        
        assert self._flag_fit, "Fit the model before running score"
        y_pred = self.predict(X, time_span, model_args) if predict else X

        return metric(np.vstack(y_pred), np.vstack(y))

    def simulate(self, X : list[np.ndarray], time_span : np.ndarray, model_args : Optional[np.ndarray] = None, calculate_score : bool = False, 
                metric : List[Callable] = [mean_squared_error], **integrator_kwargs) -> list[np.ndarray]:
        
        """
        Integrate the model
        X is a list of features whose first row is extracted as initial conditions
        """
        assert self._flag_fit, "Fit the model before running score"
        x_init = [xi[0].flatten() for xi in X]
        result = []
        for i, xi in enumerate(x_init):
            assert len(xi) == self._states, "Initial conditions should be of right dimensions"
            try:
                _integration_solution = odeint(self._casadi_model, xi, time_span, args = (model_args[i], ) if model_args else ((), ), **integrator_kwargs)
            except Exception as error:
                print(error)
                raise ValueError(f"Integration failed with error {error}") 
            else:
                if np.isnan(_integration_solution).any() or np.isinf(_integration_solution).any():
                    raise ValueError("Integration failed becoz nan or inf detected")
                result.append(_integration_solution)
        
        if not calculate_score :
            return result
        else:
            return result, tuple(self.score(X, result, time_span, metric = met, predict = False, model_args = model_args) for met in metric)
    
    @property
    def complexity(self):
        #TODO why not take length of dictionary elements of self.adict["coefficients_dict"]
        # self.adict["equations"] are now sympy equations and not strings
        # return sum(eqn.count("+") + eqn.lstrip("-").count("-") + 1 for eqn in self.adict["equations"])
        return sum(len(eqn) for eqn in self.adict["coefficients_dict"])

    @property
    def coefficients(self):
        return self.adict["coefficients_value"]

    def save_model(self, pathname : str = "saved_data\casadi_model") -> None:
        # saves the equations using pickle

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

    from generate_data import DynamicModel
    from utils import coefficients_plot

    """
    time_span = np.arange(0, 5, 0.01)
    model = DynamicModel("kinetic_kosir", time_span, arguments = [(373, 8.314)], n_expt = 6, seed = 20)
    features = model.integrate() # list of features
    target = model.approx_derivative # list of target value
    # features = model.add_noise(0, 0.2)
    target = model.approx_derivative

    opti = Optimizer_casadi(FunctionalLibrary(1), alpha = 0.0, threshold = 0.1, plugin_dict = {"ipopt.print_level" : 0, "print_time":0, "ipopt.sb" : "yes"}, 
                            max_iter = 20)
    # stoichiometry = np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1) # chemistry constraints
    # include_column = [[0, 2], [0, 3], [0, 1]]
    include_column = []
    stoichiometry =  np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -0.5, -1]).reshape(4, -1) # mass balance constraints
    # stoichiometry = np.eye(4) # no constraints
    
    derivative_free = True
    
    opti.fit(features, target, time_span = time_span, include_column = include_column, 
                constraints_dict= {"stoichiometry" : stoichiometry}, seed = 10, max_workers = 1, 
                derivative_free = derivative_free)
    opti.print()
    print("--"*20)
    print("mean squared error :", opti.score(features, target, time_span))
    print("model complexity", opti.complexity)
    print("Total number of iterations", opti.adict["iterations"])
    print("--"*20)
    # print("coefficients at each iteration", opti.adict["coefficients_iterations"])
    print("--"*20)

    # coefficients_plot(model.coefficients() , [opti.adict["coefficients_dict"], opti.adict["coefficients_dict"]])
    """

    # Testing Menten problem 

    time_span = np.arange(0, 20, 0.01)
    model = DynamicModel("kinetic_menten", time_span, n_expt = 10, arguments = [(0.1, 200, 3)], seed = 20)
    features = model.integrate() # list of features
    target = model.approx_derivative # list of target value

    opti = Optimizer_casadi(FunctionalLibrary(2), alpha = lambda iteration : 0.001, threshold = 0.05, plugin_dict = {"ipopt.print_level" : 5, "print_time": 5, "ipopt.sb" : "no"}, 
                            max_iter = 20)
    include_column = [[0, 1, 2], [0, 1, 2], [1, 2, 3]]
    stoichiometry = np.array([-1, 1, 0, -1, 1, 1, 1, -1, -1, 0, 0, 1]).reshape(4, -1) # no constraints
    
    opti.fit(features, target, time_span = time_span, include_column = include_column, 
                constraints_dict= {"stoichiometry" : stoichiometry}, seed = 10, max_workers = 1, derivative_free = True)
    opti.print()
    print("--"*20)
    print("mean squared error :", opti.score(features, target, time_span))
    print("model complexity", opti.complexity)
    print("Total number of iterations", opti.adict["iterations"])
    print("--"*20)
    coefficients_plot(model.coefficients() , [opti.adict["coefficients_dict"], opti.adict["coefficients_dict"]])
    