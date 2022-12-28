import casadi as cd
import numpy as np
import sympy as smp
from sklearn.metrics import mean_squared_error

from Optimizer import Optimizer_casadi
from FunctionalLibrary import FunctionalLibrary

from typing import List, Optional, Any, Callable, Tuple
from functools import reduce

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
        self.adict["coefficients_energy"] = [self.opti.variable(dimension[-1], 1) for dimension in self.adict["library_dimension"]] # activation energies

    def _generate_library(self, data : np.ndarray, include_column : List[np.ndarray]):
        
        super()._generate_library(data, include_column)
        # create new symbol for temperature and universal gas constant
        self.input_symbols = (*self.input_symbols, *smp.symbols("T, R"))

    def _update_cost(self, target: np.ndarray):

        # initialize the cost to zero
        self.adict["cost"] = 0
        
        # stoichiometric coefficient times the reactions 
        # need 2 for loops because of limitation of casadi 
        # the reaction rate will be arhenius equation
        reaction_rate = [cd.vertcat(*[(A*cd.exp(-B*1000*(1/ti - 1/373)/R)).T for ti, R in self.adict["arguments"]]) 
                                    for A, B in zip(self.adict["coefficients"], self.adict["coefficients_energy"])]

        # self.adict["reactions"] = [cd.mtimes(self.adict["library"][j], reaction_rate[j]) for j in range(self._functional_library)]
        self.adict["reactions"] = [cd.einstein(cd.vec(A), cd.vec(x), [*A.shape], [*x.shape], [A.shape[0]], [-1, -2], [-1, -2], [-1]) 
                                    for A, x in zip(self.adict["library"], reaction_rate)]

        for i in range(self._n_states):
            asum = 0
            for j in range(self._functional_library): 
                asum += self.adict["stoichiometry"][i, j]*self.adict["reactions"][j]

            self.adict["cost"] += cd.sumsqr(target[:, i] - asum)

        # normalize the cost by dividing by the number of data points
        self.adict["cost"] /= self.adict["library_dimension"][0][0] # first array with first dimension

        # add regularization to the cost function
        for j in range(self._functional_library):
            coefficients = self.adict["coefficients"][j]
            self.adict["cost"] += self.alpha*cd.einstein(cd.vec(coefficients), cd.vec(coefficients), 
                                                    [*coefficients.shape], [*coefficients.shape], [], [-1, -2], [-1, -2], [])

    def _create_parameters(self) -> List:
        """
        List of decision variables for which mean and standard deviation needs to be traced. 
        The thresholding parameters always have to be the first ones
        """
        return [self.adict["coefficients"], self.adict["coefficients_energy"]]


    def fit(self, features : List[np.ndarray], target : List[np.ndarray], arguments : List[np.ndarray], include_column : Optional[List[np.ndarray]] = None, 
            constraints_dict : dict = {} , ensemble_iterations : int = 1, max_workers : Optional[int] = None, seed : int = 12345) -> None:
    
        # arguments is a list of arrays so that its compatible with vectorize
        # ensemble_iterations = 1 : do regular sindy else ensemble sindy
        # constraints_dict should be of the form {"consumption" : [], "formation" : [], 
        #                                           "stoichiometry" : np.ndarray}
        self._flag_fit = True
        self._n_states = np.shape(features)[-1]

        assert len(arguments) == len(features), "Arguments and features should be consistent with the number of experiments"
        # match the arguments with the number of data points (each array in features can have varying data points)
        self.adict["arguments"] = np.squeeze(np.vstack([np.tile(args, (len(feat), 1)) for args, feat in zip(arguments, features)]))

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

        _mean, _deviation = self._stlsq(target, constraints_dict, ensemble_iterations, max_workers, seed)
        (self.adict["coefficients_value"], self.adict["coefficients_energy_value"], self.adict["coefficients_deviation"], 
                                        self.adict["coefficients_energy_deviation"]) = (_mean[0], _mean[1], _deviation[0], _deviation[1])
        self._create_equations()

    @property
    def coefficients(self):
        return self.adict["coefficients_value"], self.adict["coefficients_energy_value"]

    def _create_sympy_expressions(self, stoichiometry_row : np.ndarray) -> str:

        # Do not round the coefficients here (Rounding may compromise accuracy while prediciton or scoring). 
        # Round them only when printing
        coefficients_value : List[np.ndarray] = self.adict["coefficients_value"]
        coefficients_energy : List[np.ndarray] = self.adict["coefficients_energy_value"]
        library_labels : List[List[str]] = self.adict["library_labels"]
        expr = 0
        
        for j in range(len(library_labels)):
            zero_filter = filter(lambda x : x[0], zip(coefficients_value[j], coefficients_energy[j], library_labels[j]))
            # modify expr to include arhenius equation (R and T are additional symbols that are defined)
            expr += stoichiometry_row[j]*smp.sympify(reduce(lambda accum, value : 
                    accum + value[0] + "*exp(-(" + value[1] + "/R)*(1/T - Rational(1, 373)))* " + value[-1].replace(" ", "*") + " + ",   
                    map(lambda x : (str(x[0]), str(x[1]), x[-1]), zero_filter), "+").rstrip(" +"))
        # replaced whitespaces with multiplication element wise library labels
        # simpify already handles xor operation
        return expr


if __name__ == "__main__":

    from GenerateData import DynamicModel
    from utils import coefficient_difference_plot

    time_span = np.arange(0, 5, 0.01)
    model = DynamicModel("kinetic_kosir", time_span, n_expt = 20)
    features = model.integrate() # list of features
    target = model.approx_derivative # list of target value
    features = model.add_noise(0, 0.0)
    target = model.approx_derivative
    arguments = model.arguments

    opti = EnergySindy(FunctionalLibrary(1) , alpha = 0.1, threshold = 0.5, solver_dict={"max_iter" : 5000}, 
                            plugin_dict = {"ipopt.print_level" : 5, "print_time":5, "ipopt.sb" : "yes", "ipopt.max_iter" : 10000, "ipopt.tol" : 1e-6}, max_iter = 20)
    
    stoichiometry = np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1) # chemistry constraints
    # include_column = [[0, 2], [0, 3], [0, 1]]
    # stoichiometry =  np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -0.5, -1]).reshape(4, -1) # mass balance constraints
    # stoichiometry = np.eye(4) # no constraints

    opti.fit(features, target, arguments, include_column = [], 
                constraints_dict= {"formation" : [], "consumption" : [], 
                                    "stoichiometry" : stoichiometry}, ensemble_iterations = 2, seed = 10, max_workers = 2)
    opti.print()
    print("--"*20)
    arguments = [np.array([373, 8.314])]*len(features)
    print("mean squared error :", opti.score(features, target, model_args = arguments))
    print("model complexity", opti.complexity)
    print("Total number of iterations", opti.adict["iterations"])
    print("--"*20)
    print("coefficients at each iteration", opti.adict["coefficients_iterations"])
    print("--"*20)
    # print("model simulation", opti.simulate(features, time_span, arguments))
    # print("--"*20)
    opti.plot_distribution()

    # coefficient_difference_plot(model.coefficients , sigma = opti.adict["coefficients_dict"], sigma2 = opti.adict["coefficients_dict"])
    