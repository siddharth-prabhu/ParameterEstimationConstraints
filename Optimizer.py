from dataclasses import dataclass, field
from Base import Base
from typing import Optional, Callable
from functools import reduce

from FunctionalLibrary import FunctionalLibrary
import numpy as np
import casadi as cd
import sympy as smp
import pickle
from sklearn.metrics import mean_squared_error
from scipy.integrate import odeint


@dataclass(frozen = False)
class Optimizer_casadi(Base):

    library : FunctionalLibrary = field(default = FunctionalLibrary())
    input_features : list[str] = field(default_factory=list)
    alpha : float = field(default = 0.0)
    alpha_mass : float = field(default = 0.01)
    num_points : float = field(default = 0.1) # 10 % of total data
    threshold : float = field(default = 0.01)
    max_iter : int = field(default = 20)
    solver_dict : dict = field(default_factory = dict)

    _flag_fit : bool = field(default = False, init = False)
    adict : dict = field(default_factory = dict, init = False)

    def __post_init__(self):
        pass

    def set_params(self, **kwargs):
        # sets the values of various parameter for gridsearchcv

        if "optimizer__alpha" in kwargs:
            setattr(self, "alpha", kwargs["optimizer__alpha"])
        
        if "optimizer__threshold" in kwargs:
            setattr(self, "threshold", kwargs["optimizer__threshold"])
        
        if "optimizer__max_iter" in kwargs:
            setattr(self, "max_iter", kwargs["optimizer__max_iter"])
        
        if "optimizer__alpha_mass" in kwargs:
            setattr(self, "alpha_mass", kwargs["optimizer__alpha_mass"])

        if "optimize__num_points" in kwargs:
            setattr(self, "num_points", kwargs["optimize__num_points"])

        self.library.set_params(**kwargs)


    def _generate_library(self, data : np.ndarray, include_column = list[np.ndarray]):
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

    def _update_cost(self, target : np.ndarray, method : str = "stlsq"):
        
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

        if method == "stlsq":
            # add regularization to the cost function
            for j in range(self._functional_library):
                self.adict["cost"] += self.alpha*cd.sumsqr(self.adict["coefficients"][j])
        else: # implement l1 norm
            norm_variables = [self.opti.variable(dimension[-1], 1) for dimension in self.adict["library_dimension"]]
            for j in range(self._functional_library):
                self.adict["cost"] += self.alpha*cd.sum1(norm_variables[j])
                self.opti.subject_to(self.adict["coefficients"][j] <= norm_variables[j])
                self.opti.subject_to(self.adict["coefficients"][j] >= -norm_variables[j])

    def _add_constraints(self, constraints_dict, seed : int = 12345):
        
        data_points = self.adict["library_dimension"][0][0]
        rng = np.random.default_rng(seed)
        chosen_rows = rng.choice(range(data_points), int(self.num_points*data_points), replace = False)
        
        # adding mass balance constraints 
        state_mass = constraints_dict["mass_balance"]
        if state_mass and getattr(self, "_flag_chemcon", False):
            asum = 0
            for i in range(self._n_states):
                asum += state_mass[i]*cd.mtimes(self.adict["library"][i][chosen_rows], self.adict["coefficients"][i])

            self.adict["cost"] += self.alpha_mass*cd.sum1(asum)**2/len(chosen_rows)

        # adding formation constraints
        state_formation = constraints_dict["formation"]
        if state_formation:
            for state in state_formation :
                asum = 0
                for j in range(self._functional_library):
                    asum += self.adict["stoichiometry"][state, j]*self.adict["reactions"][j]

                self.opti.subject_to(asum >= 0)

        # adding consumption constraints
        state_consumption = constraints_dict["consumption"]
        if state_consumption :
            for state in state_consumption:
                asum = 0
                for j in range(self._functional_library):
                    asum += self.adict["stoichiometry"][state, j]*self.adict["reactions"][j]
                
                self.opti.subject_to(asum <= 0)


    def _minimize(self, solver_dict : dict):

        self.opti.minimize(self.adict["cost"])
        self.opti.solver("ipopt", solver_dict, {"max_iter" : 30})
        solution = self.opti.solve()
        # assert solution.success, "The solution did not converge" add assertion 
        return solution

    def _stlsq(self, target : np.ndarray, constraints_dict : dict) -> list[np.ndarray]:

        self._create_mask()
        coefficients_prev = [np.ones(dimension[-1]) for dimension in self.adict["library_dimension"]]
        self.adict["iterations"] = 0
 
        for i in range(self.max_iter):
            # create problem from scratch since casadi cannot run the same problem once optimized
            # steps should follow a sequence 
            self._create_decision_variables()  
            self.adict["library"] = [value*self.adict["mask"][i] for i, value in enumerate(self.adict["library"])]   
            self._update_cost(target, method = "stlsq") 
            if constraints_dict:
                self._add_constraints(constraints_dict)
            self.adict["solution"] = self._minimize(self.solver_dict) # no need to save for every iteration

            # list[np.ndarray]. additional layer of np.array and flatten to account for singular value, which casadi outputs as float
            coefficients = [np.array([self.adict["solution"].value(coeff)]).flatten() for coeff in self.adict["coefficients"]] 
            
            coefficients_next = [np.abs(coeff) > self.threshold for coeff in coefficients] # list of boolean arrays

            if np.array([np.allclose(coeff_prev, coeff_next) for coeff_prev, coeff_next in zip(coefficients_prev, coefficients_next)]).all():
                print("Solution converged")
                break

            if not sum([np.sum(coeff) for coeff in coefficients_next]):
                raise RuntimeError("Thresholding parameter eliminated all the coefficients")
                break
            
            coefficients_prev = coefficients_next # boolean array

            # update mask of small terms to zero
            self.adict["mask"] = [mask*coefficients_next[i] for i, mask in enumerate(self.adict["mask"])]
            self.adict["iterations"] += 1
        
        return coefficients

    def _least_square(self, target : np.ndarray, constraints_dict : dict) -> list[np.ndarray]:
        
        self._create_mask()
        self.adict["iterations"] = 1
        print("running least square")
        # steps should follow a sequence 
        self._create_decision_variables()  
        self.adict["library"] = [value*self.adict["mask"][i] for i, value in enumerate(self.adict["library"])]   
        self._update_cost(target, method = "least_square")
        self.adict["solution"] = self._minimize(self.solver_dict)

        # list[np.ndarray]. additional layer of np.array and flatten to account for singular value, which casadi outputs as float
        coefficients = [np.array([self.adict["solution"].value(coeff)]).flatten() for coeff in self.adict["coefficients"]] 

        return coefficients

    def fit(self, features : list[np.ndarray], target : list[np.ndarray], include_column : Optional[list[np.ndarray]] = None, 
            constraints_dict : dict = {}, method : str = "stlsq" ) -> None:

        # constraints_dict should be of the form {"mass_balance" : [], "consumption" : [], "formation" : [], 
        #                                           "stoichiometry" : np.ndarray}
        self._flag_fit = True
        self._n_states = np.shape(features)[-1]

        if "stoichiometry" in constraints_dict and isinstance(constraints_dict["stoichiometry"], np.ndarray):
            rows, cols = constraints_dict["stoichiometry"].shape
            assert rows == self._n_states, "The rows should match the number of states"
            self._functional_library = cols
            self.adict["stoichiometry"] = constraints_dict["stoichiometry"]
            self._flag_chemcon = True
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

        if method == "stlsq":
            self.adict["coefficients_value"] = self._stlsq(target, constraints_dict)
        elif method == "lsq":
            self.adict["coefficients_value"] = self._least_square(target, constraints_dict)
        else:
            raise ValueError(f"Method {method} not yet implemented")

        self._create_equations()

    # need to consider when stoichiometric in present
    def _create_equations(self) -> None:
        # stores the equations in adict to be used later
        self.adict["equations"] = []
        self.adict["equations_lambdify"] = []
        for i in range(self._n_states):
            expr = 0
            for j in range(self._functional_library):
                zero_filter = filter(lambda x : x[0], zip(self.adict["coefficients_value"][j], self.adict["library_labels"][j]))
                expr += self.adict["stoichiometry"][i, j]*smp.sympify(reduce(lambda accum, value : 
                        accum + value[0] + " * " + value[1].replace(" ", "*") + " + ",   
                        map(lambda x : ("{:.2f}".format(x[0]), x[1]), zero_filter), "+").rstrip(" +")) 
            # replaced whitespaces with multiplication element wise library labels
            # simpify already handles xor operation

            self.adict["equations"].append(str(expr))
            self.adict["equations_lambdify"].append(smp.lambdify(self.input_symbols, expr))


    def _casadi_model(self, x : np.ndarray, t : np.ndarray):

        return np.array([eqn(*x) for eqn in self.adict["equations_lambdify"]])
    

    def predict(self, X : list[np.ndarray]) -> list:
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

    model = DynamicModel("kinetic_kosir", np.arange(0, 5, 0.01), n_expt = 15)
    features = model.integrate() # list of features
    target = model.approx_derivative # list of target value

    opti = Optimizer_casadi(FunctionalLibrary(3) , alpha = 0.01, threshold = 0.1, solver_dict={"ipopt.print_level" : 0, "print_time":0})
    stoichiometry = np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1)
    # stoichiometry = None
    # opti.fit(features, target, include_column = [], 
    #         constraints_dict = {}, method = "lsq")
    opti.fit(features, target, include_column = [[0, 2], [0, 3], [0, 1]], 
                constraints_dict= {"mass_balance" : [], "formation" : [1], "consumption" : [], 
                                    "stoichiometry" : stoichiometry}, method = "stlsq")
    opti.print()
    print("--"*20)
    print("mean squared error :", opti.score(features, target))
    print(opti.complexity)
    print("Total number of iterations", opti.adict["iterations"])