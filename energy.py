# type: ignore

from typing import List, Optional, Any, Callable, Tuple
from functools import reduce

import casadi as cd
print("casadi version", cd.__version__)
import numpy as np
import sympy as smp
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from Optimizer import Optimizer_casadi
from FunctionalLibrary import FunctionalLibrary


class EnergySindy(Optimizer_casadi):

    def __init__(self, 
                    library : FunctionalLibrary = FunctionalLibrary(2),
                    input_features : List[str] = [],
                    alpha : float = 0.0,
                    num_points : float = 0.5,
                    threshold : float = 0.01, # inverse of z_critical for boostrapping
                    max_iter : int = 20,
                    plugin_dict : dict = {},
                    solver_dict : dict = {}):
        
        self.library = library
        self.input_features = input_features
        self.alpha = alpha
        self.num_points = num_points
        self.threshold = threshold
        self.max_iter = max_iter
        self.plugin_dict = plugin_dict
        self.solver_dict = solver_dict

        super().__init__(self.library, 
                        self.input_features,
                        self.alpha,
                        self.num_points,
                        self.threshold,
                        self.max_iter,
                        self.plugin_dict,
                        self.solver_dict)

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
        for i in range(self._output_states):
            asum = 0
            for j in range(self._functional_library): 
                asum += self.adict["stoichiometry"][i, j]*self.adict["reactions"][j]

            self.adict["cost"] += cd.sumsqr(target[:, i] - asum)/2

        # normalize the cost by dividing by the number of data points
        # self.adict["cost"] /= self.adict["library_dimension"][0][0] # first array with first dimension

        # add regularization to the cost function
        for coefficients in self.adict["coefficients"]:
            self.adict["cost"] += self.alpha*cd.einstein(cd.vec(coefficients), cd.vec(coefficients), 
                                                    [*coefficients.shape], [*coefficients.shape], [], [-1, -2], [-1, -2], []) 

    def _add_constraints(self, constraints_dict, seed : int = 12345):

        super()._add_constraints(constraints_dict, seed)

        if constraints_dict.get("energy", False):
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
        The thresholding parameters always have to be the first ones
        """
        return [self.adict["coefficients"], self.adict["coefficients_energy"]]


    def _initialize_decision_variables(self) -> None:
        """
        Sets the initial guess for decision varibles
        Default guess value chosen by casadi is zero
        """
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
        # shuffle the arguments as well
        self.adict["arguments"] = self.adict["arguments_original"][permutations]
        self._update_cost(target[permutations])
        if constraints_dict:
            self._add_constraints(constraints_dict, seed)
        self._initialize_decision_variables()
        _solution = self._minimize(self.plugin_dict, self.solver_dict) # no need to save for every iteration

        # list[np.ndarray]. additional layer of np.array and flatten to account for singular value, which casadi outputs as float
        return [[np.array([_solution.value(coeff)]).flatten() for coeff in params] for params in parameters]


    def fit(self, features : List[np.ndarray], target : List[np.ndarray], time_span : np.ndarray, arguments : List[np.ndarray], 
                include_column : Optional[List[np.ndarray]] = None, constraints_dict : dict = {} , 
                ensemble_iterations : int = 1, variance_elimination : bool = False, derivative_free : bool = False,
                max_workers : Optional[int] = None, seed : int = 12345) -> None:
    
        # arguments is a list of arrays so that its compatible with vectorize
        # if variance_elimination = True and ensemble_iterations > 1 performs boostrapping
        # if variance_elimination = True and ensemble_iterations <= 1 performs covariane matrix
        # if variance_elimination = False performs normal thresholding (regular sindy)
        # constraints_dict should be of the form {"consumption" : [], "formation" : [], 
        #                                           "stoichiometry" : np.ndarray}
        self._flag_fit = True
        self._output_states = np.shape(target)[-1]
        self._input_states = np.shape(features)[-1]
        self.N = len(features)

        assert len(arguments) == len(features), "Arguments and features should be consistent with the number of experiments"
        # match the arguments with the number of data points (each array in features can have varying data points)
        self.adict["arguments_original"] = np.squeeze(np.vstack([np.tile(args, (len(feat), 1)) for args, feat in zip(arguments, features)]))

        if "stoichiometry" in constraints_dict and isinstance(constraints_dict["stoichiometry"], np.ndarray):
            rows, cols = constraints_dict["stoichiometry"].shape
            assert rows == self._output_states, "The rows should match the number of states"
            self._functional_library = cols
            self.adict["stoichiometry"] = constraints_dict["stoichiometry"]
        else:
            self._functional_library = self._output_states
            self.adict["stoichiometry"] = np.eye(self._output_states) 

        if include_column:
            assert len(include_column) == self._functional_library, "length of columns should match with the number of functional libraries"
            include_column = [list(range(self._input_states)) if len(alist) == 0 else alist for alist in include_column] 
        else:
            include_column = [list(range(self._input_states)) for _ in range(self._functional_library)]

        if derivative_free:
            target = np.vstack(target)
            self._generate_library_derivative_free(features, include_column, time_span)
        else:
            features, target = np.vstack(features), np.vstack(target)
            self._generate_library(features, include_column)

        _mean, _deviation = self._stlsq(target, constraints_dict, ensemble_iterations, variance_elimination, max_workers, seed)
        (self.adict["coefficients_value"], self.adict["coefficients_energy_value"], self.adict["coefficients_deviation"], 
                                        self.adict["coefficients_energy_deviation"]) = (_mean[0], _mean[1], _deviation[0], _deviation[1])
        self._create_equations()

    @property
    def coefficients(self):
        return self.adict["coefficients_value"], self.adict["coefficients_energy_value"]

    def _create_sympy_expressions(self, stoichiometry_row : np.ndarray) -> str:

        # mask the coefficient values
        self.adict["coefficients_value_masked"] = []
        for coefficients, mask in zip(self.adict["coefficients_value"], self.adict["mask"]):
            self.adict["coefficients_value_masked"].append(coefficients*mask.flatten())

        # Do not round the coefficients here (Rounding may compromise accuracy while prediciton or scoring). 
        # Round them only when printing
        coefficients_value : List[np.ndarray] = self.adict["coefficients_value_masked"]
        coefficients_energy : List[np.ndarray] = self.adict["coefficients_energy_value"]
        library_labels : List[List[str]] = self.adict["library_labels"]
        expr = 0
        
        for j in range(len(library_labels)):
            zero_filter = filter(lambda x : x[0], zip(coefficients_value[j], coefficients_energy[j], library_labels[j]))
            # modify expr to include arhenius equation (R and T are additional symbols that are defined)
            expr += stoichiometry_row[j]*smp.sympify(reduce(lambda accum, value : 
                    accum + value[0] + "*exp(-(" + value[1] + "/R)*(1/T - Rational(1, 373)))* " + value[-1].replace(" ", "*") + " + ",   
                    map(lambda x : (str(x[0]), str(x[1]*10_000), x[-1]), zero_filter), "+").rstrip(" +"))
        # replaced whitespaces with multiplication element wise library labels
        # simpify already handles xor operation
        return expr


if __name__ == "__main__":

    from GenerateData import DynamicModel
    from utils import coefficient_difference_plot

    time_span = np.arange(0, 5, 0.01)
    arguments = [(360, 8.314), (370, 8.314), (380, 8.314), (390, 8.314)]
    model = DynamicModel("kinetic_kosir", time_span, n_expt = len(arguments), arguments = arguments, seed = 20)
    features = model.integrate() # list of features
    target = model.approx_derivative # list of target value
    features = model.add_noise(0, 0.0)
    target = model.approx_derivative
    arguments = model.arguments
     
    plugin_dict = {"ipopt.print_level" : 5, "print_time":5, "ipopt.sb" : "yes", "ipopt.max_iter" : 3000}
    # plugin_dict = {}
    opti = EnergySindy(FunctionalLibrary(2) , alpha = 0, threshold = 2, solver_dict={"solver" : "ipopt"}, 
                            plugin_dict = plugin_dict, max_iter = 20)
    
    stoichiometry = np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1) # chemistry constraints
    # include_column = [[0, 2], [0, 3], [0, 1]] # chemistry constraints
    include_column = []

    opti.fit(features, [feat - feat[0] for feat in features], time_span, arguments, include_column = include_column, 
                constraints_dict= {"formation" : [], "consumption" : [], "energy" : False,
                                    "stoichiometry" : stoichiometry}, ensemble_iterations = 100, variance_elimination = True, derivative_free = True, seed = 20, max_workers = 2)
    opti.print()
    print("--"*20)
    print("mean squared error :", opti.score(features, target, model_args = arguments))
    print("model complexity", opti.complexity)
    print("Total number of iterations", opti.adict["iterations"])
    print("--"*20)
    print("coefficients energy at each iteration", opti.adict["coefficients_energy_value"])
    print("--"*20)
    # print("model simulation", opti.simulate(features, time_span, arguments))
    print("--"*20)
    # opti.plot_distribution()

    # coefficient_difference_plot(model.coefficients , sigma = opti.adict["coefficients_dict"], sigma2 = opti.adict["coefficients_dict"])
 