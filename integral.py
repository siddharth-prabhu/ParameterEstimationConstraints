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


class IntegralSindy(EnergySindy):

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

    def _generate_interpolation(self, data: List[np.ndarray], include_column: List[np.ndarray], time_span: np.ndarray) -> None:
        """
        given data as a list of np.ndarrays, fits an interpolation and creates functional library
        A matrix is not created but a callable (x, t) functional library is returned
        data : should always have temperature as last column
        """
        self.adict["initial"] = [di[0, :-1] for di in data] # saving the initial conditions of states

        # define input features if not given. Input features depend on the shape of data
        if not len(self.input_features):
            self.input_features = [f"x{i}" for i in range(self._input_states)]

        # define symbols that can be converted to equations later
        self.input_symbols = smp.symbols(reduce(lambda accum, value : accum + value + ", ", self.input_features, ""))

        # done using for loop instead of list comprehension becoz each fit_transform and get_features
        # share the same instance of the class
        self.adict["interpolation_library"] : List[List[Callable]]= [] # for all states, for all experiments, function
        self.adict["interpolation_temperature"] : List[List[Callable]] = []
        self.adict["library_labels"] : List[List]= []
        for i in range(self._reactions):
            self.adict["interpolation_library"].append([self.library.fit_transform(
                di[:, :-1], 
                include_column[i], 
                True, 
                time_span, 
                get_function = True, 
                interpolation_scheme = "casadi"
                ) for di in data])
            
            self.adict["interpolation_temperature"].append([FunctionalLibrary(1).fit_transform(
                di[:, -1].reshape(-1, 1), 
                derivative_free = True, 
                time_span = time_span, 
                get_function= True,
                interpolation_scheme = "casadi"
                ) for di in data])
            
            self.adict["library_labels"].append(self.library.get_features(self.input_features))
        
        # rows of data is number of experiments*time_span, cols of data is the dimension of the returned callable matrix
        self.adict["library"] = None # just to make things compatible with downstream ops
        self.adict["library_dimension"] = [(len(time_span), len(xi[0](0, 0))) for xi in self.adict["interpolation_library"]]
        self.input_symbols = (*self.input_symbols, *smp.symbols("T, R"))

    def _create_decision_variables(self):
        
        self.opti = cd.Opti() # create casadi instance
        variables = reduce(operator.add, [dim[-1] for dim in self.adict["library_dimension"]])
        variables = self.opti.variable(variables*2) # reaction rates and activation energies

        reference, activation, start = [], [], 0
        for dim in self.adict["library_dimension"]:
            reference.append(variables[start : start + dim[-1]])
            activation.append(variables[start + dim[-1] : start + 2*dim[-1]]) 
            start += 2*dim[-1]

        self.adict["coefficients"] = reference
        self.adict["coefficients_energy"] = activation
        self.adict["coefficients_combined"] = variables
        # self.adict["multiple_shooting"] 

    def _update_cost(self, target: List[np.ndarray], time_span : np.ndarray):
        # target here is list of np.ndarrays
        # add ode function here and for all experiments
        self.adict["cost"] = 0

        # define symbolic equations
        time = cd.MX.sym("t", 1)
        parameters = cd.MX.sym("p", 2*reduce(operator.add, [dim[-1] for dim in self.adict["library_dimension"]])) 
        states = cd.MX.sym("x", self._states)

        for i in range(self.N): # for all experiments

            def feature_ode(t, reference, activation, reaction):
                # do it separately for each reaction since number of parameters can vary
                # reference, activation : shape (feature columns, 1)
                # interpolation : shape (1, feature columns)
                # multiple by mask before returning so that there are less terms to integrate
                interpolation_states = cd.vertcat(*self.adict["interpolation_library"][reaction][i](0, t))
                interpolation_temp = cd.vertcat(*self.adict["interpolation_temperature"][reaction][i](0, t))
                rate_temp = reference*cd.exp(-10_000*activation*(1/interpolation_temp - 1/373)/8.314)
                return cd.dot(interpolation_states, rate_temp*(self.adict["mask"][reaction].flatten()))

            def ode(x, p, t):
                # p : shape = sum of all decision variable X 1
                # multiply with stoichiometric coefficients !
                reference, activation, start = [], [], 0
                for dim in self.adict["library_dimension"]:
                    reference.append(p[start : start + dim[-1]])
                    activation.append(p[start + dim[-1] : start + 2*dim[-1]]) 
                    start += 2*dim[-1]
                
                return cd.mtimes(
                    self.adict["stoichiometry"], 
                    cd.vertcat(*[feature_ode(t, ref, act, reaction) for reaction, ref, act in zip(range(self._reactions), reference, activation)])
                            )         
            
            # simulate the ode forward in time
            casadi_ode = {"x" :  states, "p" : parameters, "t" : time, "ode" : ode(states, parameters, time)}
            
            # auto differentiation fails with grid options
            # https://github.com/casadi/casadi/issues/2619
            # casadi_integrator = cd.integrator("integral", "cvodes", casadi_ode, {"grid" : time_span, "output_t0" : True})
            
            concentration = self.opti.variable(len(time_span)//self.horizon, self._states)
            solution = [self.adict["initial"][i].reshape(1, -1)] # initial conditions
            x_initial = solution[-1]
            for j in range(len(time_span) - 1):

                if j%self.horizon == 0:
                    x_initial = concentration[j//self.horizon, :]

                casadi_integrator = cd.integrator("kosir", "cvodes", casadi_ode, {"t0" : time_span[j], "tf" : time_span[j + 1]})
                integration_solution = casadi_integrator(x0 = x_initial, p = self.adict["coefficients_combined"])["xf"]
                x_initial = integration_solution.T
                solution.append(x_initial)
            
            # update cost
            self.adict["cost"] += cd.sumsqr(target[i] - cd.vertcat(*solution))/len(target)
            self.opti.subject_to(cd.vertcat(*solution)[::self.horizon, :] == concentration)

        # adding regularization 
        for coeff in self.adict["coefficients"]:
            self.adict["cost"] += self.alpha*cd.sumsqr(coeff)

    # function for multiprocessing
    def _stlsq_solve_optimization(self, permutations : List, **kwargs) -> List[List[np.ndarray]]:
        # create problem from scratch since casadi cannot run the same problem once optimized
        # steps should follow a sequence 
        # dont replace if there is only one ensemble iteration. Dataset rows are constant for all reactions 
        # parameters is added so that you can get the values of decision variables without having to rewrite this code
        target = kwargs.get("target", False)
        time_span = kwargs.get("time_span", False)
        assert ((time_span is not None) and (target is not None)), "time_span and target should be provided"

        self._create_decision_variables()
        self._parameters : List = self._create_parameters()
        self._update_cost(target, time_span)
        self._initialize_decision_variables()
        _solution = self._minimize(self.plugin_dict, self.solver_dict) # no need to save for every iteration

        # list[np.ndarray]. additional layer of np.array and flatten to account for singular value, which casadi outputs as float
        return [[np.array([_solution.value(coeff)]).flatten() for coeff in params] for params in self._parameters]

    def fit(self, features: List[np.ndarray], target: List[np.ndarray], time_span: np.ndarray, arguments: List[np.ndarray] = None, 
            include_column: Optional[List[np.ndarray]] = None, constraints_dict: dict = {}, shooting_horizon : int = 20, max_workers : Optional[int] = None, 
            seed: int = 12345, **kwargs) -> None:
        """
        features : temperature is always the last column
        target : the derivatives of states. In this formulation it will be replaced with states
        arguments is a list of arrays so that its compatible with vectorize
        constraints_dict should be of the form {"consumption" : [], "formation" : [], 
                                                   "stoichiometry" : np.ndarray}
        """
        
        self._flag_fit = True
        self._input_states = np.shape(features)[-1] - 1 # do not consider temperature
        _output_states = np.shape(target)[-1] # target do not have temperature
        self.N = len(features)
        self.horizon = shooting_horizon

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

        self._generate_interpolation(features, include_column, time_span)
        target = [feat[:, :-1] for feat in features]
        # mask cannot be part of optimization because it cannot be reset every otpmization loop
        self._create_mask() # pulled outside optimization so that can be used in cost function

        _mean, _deviation = self._stlsq(
            target, 
            constraints_dict, 
            ensemble_iterations = 1, 
            variance_elimination = False,
            time_span = time_span, 
            max_workers = max_workers, 
            seed = seed)
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

    time_span = np.arange(0, 5, 0.01)
    n_expt = 5
    model = DynamicModel("kinetic_kosir_temperature", time_span, n_expt = n_expt)
    features = model.integrate()
    target =  [feat[:, :-1] for feat in features]

    plugin_dict = {"ipopt.print_level" : 5, "print_time":5, "ipopt.sb" : "yes", "ipopt.max_iter" : 3000, "ipopt.tol" : 1e-5}
    opti = IntegralSindy(FunctionalLibrary(1) , alpha = 0.01, threshold = 0.1, solver_dict={"solver" : "ipopt"}, 
                            plugin_dict = plugin_dict, max_iter = 20)
    
    stoichiometry = np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0], dtype = int).reshape(4, -1) # chemistry constraints
    # include_column = [[0, 2], [0, 3], [0, 1]] # chemistry constraints
    include_column = []
    
    opti.fit(features, target, time_span, include_column = include_column, 
                constraints_dict= {"formation" : [], "consumption" : [],
                                    "stoichiometry" : stoichiometry})

    opti.print()
    print("--"*20)
    interp = CubicSpline(time_span, features[0][:, -1])
    arguments = [[interp, 8.314] for _ in range(n_expt)]
    print("mean squared error :", opti.score(features, target, time_span, model_args = arguments))
    print("model complexity", opti.complexity)
    print("Total number of iterations", opti.adict["iterations"])
    print("--"*20)