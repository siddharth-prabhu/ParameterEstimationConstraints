# type: ignore
from typing import List, Optional, Callable
from functools import reduce
import operator

import casadi as cd
import numpy as np
import sympy as smp

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
                solver_dict : dict = {}
                ):
        
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
        self.adict["library_dimension"] = [(len(xi)*len(time_span), len(xi[0](0, 0))) for xi in self.adict["interpolation_library"]]
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
                interpolation_states = cd.horzcat(*self.adict["interpolation_library"][reaction][i](0, t))
                interpolation_temp = cd.vertcat(*self.adict["interpolation_temperature"][reaction][i](0, t))
                rate_temp = reference*cd.exp(-10_000*activation*(1/interpolation_temp - 1/373)/8.314)
                return cd.mtimes(interpolation_states, rate_temp)

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

            casadi_ode = {"x" :  states, "p" : parameters, "t" : time, "ode" : ode(states, parameters, time)}

            # simulate the ode forward in time
            solution = [self.adict["initial"][i].reshape(1, -1)] # initial conditions
            for j in range(len(time_span) - 1):
                casadi_integrator = cd.integrator("integral", "cvodes", casadi_ode, {"t0" : time_span[j], "tf" : time_span[j + 1]})
                integration_solution = casadi_integrator(x0 = solution[-1], p = self.adict["coefficients_combined"])["xf"]
                solution.append(integration_solution.T[-1, :])

            self.adict["cost"] += cd.sumsqr(target[i] - cd.vertcat(*solution))

        # adding regularization 
        self.adict["cost"] += self.alpha*reduce(lambda value, accum : accum + cd.sumsqr(value), self.adict["coefficients"])


    def fit(self, features: List[np.ndarray], target: List[np.ndarray], time_span: np.ndarray, arguments: List[np.ndarray] = None, 
            include_column: Optional[List[np.ndarray]] = None, constraints_dict: dict = {}, seed: int = 12345) -> None:
        """
        arguments is a list of arrays so that its compatible with vectorize
        constraints_dict should be of the form {"consumption" : [], "formation" : [], 
                                                   "stoichiometry" : np.ndarray}
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

        
        self._generate_interpolation(features, include_column, time_span)
        self._create_decision_variables()
        self._update_cost(target, time_span)
        self.opti.minimize(self.adict["cost"])
        self.opti.solver("ipopt")
        self.opti.solve()

if __name__ == "__main__":

    from scipy.integrate import odeint

    def kinetic_kosir(x, t, args) -> np.ndarray:
        # A -> 2B; A <-> C; A <-> D
        R = args
        rates = reaction_rate_kosir(x[-1], R)
        return np.array([-(rates[0] + rates[1] + rates[3])*x[0] + rates[2]*x[2] + rates[4]*x[3],
                        2*rates[0]*x[0],
                        rates[1]*x[0] - rates[2]*x[2],
                        rates[3]*x[0] - rates[4]*x[3], 
                        373*(np.pi*np.cos(np.pi*t)/50)])

    def reaction_rate_kosir(T, R):

        # original values are at reference temperature of 373 K
        # This function is called several times. Consider defining constants outside the function

        if T == 373:
            return [8.566/2, 1.191, 5.743, 10.219, 1.535]

        Eab = 30*10**3
        Eac = 40*10**3
        Eca = 45*10**3
        Ead = 50*10**3
        Eda = 60*10**3

        return [8.566/2*np.exp(-(Eab/R)*(1/T - 1/373)), 1.191*np.exp(-(Eac/R)*(1/T - 1/373)), 5.743*np.exp(-(Eca/R)*(1/T - 1/373)), 
                10.219*np.exp(-(Ead/R)*(1/T - 1/373)), 1.535*np.exp(-(Eda/R)*(1/T - 1/373))]


    n_expt = 2
    delta_time = 0.01
    time_span = np.arange(0, 5, delta_time)
    np.random.seed(20)
    x_init = [np.random.normal(5, 20, size = (4, )) for _ in range(n_expt)]
    temperature = [373]
    features = [odeint(kinetic_kosir, [*xi, temperature[-1]], time_span, args = (8.314, )) for xi in x_init] # list of features
     
    plugin_dict = {"ipopt.print_level" : 5, "print_time":5, "ipopt.sb" : "yes", "ipopt.max_iter" : 1000}
    opti = IntegralSindy(FunctionalLibrary(2) , alpha = 0.1, threshold = 0.5, solver_dict={"solver" : "ipopt"}, 
                            plugin_dict = plugin_dict, max_iter = 20)
    
    stoichiometry = np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1) # chemistry constraints
    include_column = [[0, 2], [0, 3], [0, 1]] # chemistry constraints

    opti.fit(features, [feat[:, :-1] for feat in features], time_span, include_column = include_column, 
                constraints_dict= {"formation" : [], "consumption" : [], "energy" : False,
                                    "stoichiometry" : stoichiometry})
