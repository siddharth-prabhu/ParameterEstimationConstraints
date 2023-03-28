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

    def _generate_library_derivative_free(self, data: List[np.ndarray], include_column: List[np.ndarray], time_span: np.ndarray) -> None:
        """
        given data as a list of np.ndarrays, fits an interpolation and creates functional library
        A matrix is not created but a callable (x, t) functional library is returned
        """

        # define input features if not given. Input features depend on the shape of data
        if not len(self.input_features):
            self.input_features = [f"x{i}" for i in range(self._input_states)]

        # define symbols that can be converted to equations later
        self.input_symbols = smp.symbols(reduce(lambda accum, value : accum + value + ", ", self.input_features, ""))

        # done using for loop instead of list comprehension becoz each fit_transform and get_features
        # share the same instance of the class
        self.adict["library"] : List[List[Callable]]= [] # for all states, for all experiments, function
        self.adict["library_labels"] : List[List]= []
        for i in range(self._functional_library):
            self.adict["library"].append([self.library.fit_transform(di, include_column[i], True, time_span, get_function = True) for di in data])
            self.adict["library_labels"].append(self.library.get_features(self.input_features))
        
        # rows of data is number of experiments*time_span, cols of data is the dimension of the returned callable matrix
        self.adict["library_dimension"] = [(len(xi)*len(time_span), len(xi[0](0, 0))) for xi in self.adict["library"]]
        self.input_symbols = (*self.input_symbols, *smp.symbols("T, R"))

    def _update_cost(self, target: List[np.ndarray], time_span : np.ndarray):
        # target here is list of np.ndarrays
        # add ode function here and for all experiments

        # define symbolic equations
        time = cd.MX.sym("t", 1)
        parameters = cd.MX.sym("p", 2*reduce(operator.add, [dim[-1] for dim in self.adict["library_dimension"]]))
        states = cd.MX.sym("x", self._output_states)

        for i in range(self.N):

            
            def feature_ode(x, p, t):
                # p : shape (feature columns, self._input_states)
                # interpolation : shape (self._input_states, feature columns)
                # cannot pass matrix with casadi. Use vector instead
                p = cd.reshape(p, (-1, self._functional_library))
                interpolation = cd.vertcat([cd.horzcat(xi[i](0, t)) for xi in self.adict["library"]])
                rate_temp = 

                return 1



        


    def fit(self, features: List[np.ndarray], target: List[np.ndarray], time_span: np.ndarray, arguments: List[np.ndarray], 
            include_column: Optional[List[np.ndarray]] = None, constraints_dict: dict = {}, seed: int = 12345) -> None:
        """
        arguments is a list of arrays so that its compatible with vectorize
        constraints_dict should be of the form {"consumption" : [], "formation" : [], 
                                                   "stoichiometry" : np.ndarray}
        """

        self._flag_fit = True
        self._output_states = np.shape(target)[-1] # input and output states matrix can be different
        self._input_states = np.shape(features)[-1]
        self.N = len(features)
        self.arguments = arguments

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