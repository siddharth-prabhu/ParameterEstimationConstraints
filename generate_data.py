# type: ignore

from dataclasses import dataclass, field
from typing import ClassVar, Optional, List, Tuple
from functools import reduce, partial

import numpy as np
import sympy as smp
import pickle
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pysindy as ps


@dataclass
class DynamicModel():

    model : str 
    time_span : np.ndarray
    initial_condition : Optional[List[np.ndarray]] = field(default_factory = list)
    arguments : Optional[List[np.ndarray]] = field(default_factory = list)
    n_expt : int = field(default = 1)
    seed : int = field(default = 12345)

    _model_dict : ClassVar 
    _solution_flag : bool = field(init = False, default = False)

    def __post_init__(self):
        self._model_dict = {"kinetic_simple" : {"function" : DynamicModel.kinetic_simple, "n_states" : 4, "low" : [5, 5, 5, 5], "high" : [20, 20, 20, 20]},
                            "kinetic_rober" : {"function" : DynamicModel.kinetic_rober, "n_states" : 3, "low" : [5, 5, 5], "high" : [20, 20, 20]},
                            "kinetic_menten" : {"function" : DynamicModel.kinetic_menten, "n_states" : 4, "low" : [5, 5, 5, 5], "high" : [20, 20, 20, 20]},
                            "kinetic_kosir" : {"function" : DynamicModel.kinetic_kosir, "n_states" : 4, "low" : [5, 5, 5, 5], "high" : [20, 20, 20, 20]}, 
                            "kinetic_kosir_temperature" : {"function" : DynamicModel.kinetic_kosir_temperature, "n_states" : 5,  
                            "low" : [5, 5, 5, 5, 373], "high" : [10, 10, 10, 10, 373]}}
        assert self.model in self._model_dict, "Dynamic model is not defined yet"
        
        rng = np.random.default_rng(self.seed)
        if not self.initial_condition:
            self.initial_condition = [np.concatenate([rng.uniform(low, high, size = (1, )) 
                                        for low, high in zip(self._model_dict[self.model]["low"], self._model_dict[self.model]["high"])])
                                        for _ in range(self.n_expt)]

        if not self.arguments:
            # By default the data is generated for varying temperature values
            self.arguments : List[np.ndarray] = [[rng.uniform(360, 390), 8.314] for _ in range(self.n_expt)]
        else:
            # use the same arguments for all the experiments
            if not len(self.arguments) == self.n_expt and len(self.arguments) == 1:
                # convert arguments to numpy array
                if not isinstance(self.arguments[0], np.ndarray):
                    self.arguments[0] = np.array(self.arguments[0])
                self.arguments = self.arguments*self.n_expt

        self.arguments = [np.array(argi) for argi in self.arguments]
        assert len(self.initial_condition[-1]) == self._model_dict[self.model]["n_states"], "Incorrect number of states"
        assert len(self.initial_condition) == self.n_expt, "Initial conditions should match the number of experiments"
        assert len(self.arguments) == self.n_expt, "List of arguments should match the number of experiments"

        self._n_states = self._model_dict[self.model]["n_states"]
        self.model_func = self._model_dict[self.model]["function"] 

    @staticmethod
    def kinetic_simple(x, t, args : Optional[Tuple] = None) -> np.ndarray:
        # A + B <--> C --> D + B
        # k1 = 1, kr1 = 0.5, k2 = 2, kr2 = 1
        k1, kr1, k2, kr2 = [1, 0.5, 2, 1]
        
        return np.array([-k1*x[0]*x[1] + kr1*x[2],
                -k1*x[0]*x[1] + (kr1 + k2)*x[2],
                k1*x[0]*x[1] - (kr1 + k2)*x[2],
                k2*x[2]])
    
    @staticmethod
    def kinetic_rober(x, t, args) -> np.ndarray:
        # A -> B; 2B -> C + B -> A + C
        # k1, k2, k3 = 0.04, 3e7, 1e4
        k1, k2, k3 = args
        return np.array([
            -k1 * x[0] + k3 * x[1] * x[2], 
            k1 * x[0] - k2 * x[1]**2 - k3 * x[1] * x[2],
            k2 * x[1]**2
        ])

    @staticmethod
    def kinetic_menten(x, t, args) -> np.ndarray:
        # A + B -> C; C -> A + B; C -> B + D
        # k1, k2, k3 = 110, 100, 1
        k1, k2, k3 = args
        reactions = np.array([k1 * x[0] * x[1], k2 * x[2], k3 * x[2]])
        stoichiometric = np.array([-1, 1, 0, -1, 1, 1, 1, -1, -1, 0, 0, 1]).reshape(4, -1)
        return np.dot(stoichiometric, reactions)
    
    @staticmethod
    def _reactions(x, k) -> np.ndarray:
        return np.array([
            k[0]*x[0], 
            k[1]*x[0] - k[2]*x[2],
            k[3]*x[0] - k[4]*x[3]
        ])

    @staticmethod
    def kinetic_kosir(x, t, args) -> np.ndarray:
        # A -> 2B; A <-> C; A <-> D
        T, R = args
        rates = DynamicModel.reaction_rate_kosir(T, R)
        stoichiometry = np.array([-1, -1, -1, 2, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(4, -1)
        return np.dot(stoichiometry, DynamicModel._reactions(x, rates))

    @staticmethod
    def reaction_rate_kosir(T, R) -> List:
        """
        Original values are at reference temperature of 373 K
        This function is called several times. Consider defining constants outside the function
        """
        if T == 373:
            return [8.566/2, 1.191, 5.743, 10.219, 1.535]

        Eab = 30*10**3
        Eac = 40*10**3
        Eca = 45*10**3
        Ead = 50*10**3
        Eda = 60*10**3

        if isinstance(T, smp.Symbol):
            return [8.566/2*smp.exp(-(Eab/R)*(1/T - 1/373)), 1.191*smp.exp(-(Eac/R)*(1/T - 1/373)), 5.743*smp.exp(-(Eca/R)*(1/T - 1/373)), 
                    10.219*smp.exp(-(Ead/R)*(1/T - 1/373)), 1.535*smp.exp(-(Eda/R)*(1/T - 1/373))]
        else :
            return [8.566/2*np.exp(-(Eab/R)*(1/T - 1/373)), 1.191*np.exp(-(Eac/R)*(1/T - 1/373)), 5.743*np.exp(-(Eca/R)*(1/T - 1/373)), 
                    10.219*np.exp(-(Ead/R)*(1/T - 1/373)), 1.535*np.exp(-(Eda/R)*(1/T - 1/373))]

    @staticmethod
    def kinetic_kosir_temperature(x, t, args) -> np.ndarray:
        # A -> 2B; A <-> C; A <-> D
        _, R  = args
        rates = DynamicModel.reaction_rate_kosir(x[-1], R)
        stoichiometry = np.array([-1, -1, -1, 2, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(4, -1)
        
        return np.append(
            np.dot(stoichiometry, DynamicModel._reactions(x, rates)),
            373*(np.pi*np.cos(np.pi*t)/50)if not isinstance(x[0], smp.Symbol) else 373*(smp.pi*smp.cos(smp.pi*t)/50)
            )

    def coefficients(self, x : Optional[Tuple[smp.symbols]] = None, t : Optional[np.ndarray] = None, args_as_symbols : bool = False, 
                     pre_stoichiometry : bool = False) -> List[dict]:

        # if symbols are not provided
        if not x:
            x = smp.symbols(reduce(lambda accum, value : accum + value + ",", [f"x{i}" for i in range(self._n_states)], ""))

        # if arguments are not specified then take the first from the list of arguments
        if args_as_symbols :
            args = smp.symbols("T, R") 
        else:
            args = self.arguments[0]

        if pre_stoichiometry :
            rates = DynamicModel.reaction_rate_kosir(*args)
            equations = DynamicModel._reactions(x, rates)
        else :
            equations = self.model_func(x, 0, args)
            
        return [eqn.as_coefficients_dict() for eqn in equations]

    # forward simulates the chosen model using scipy odeint
    def integrate(self, **odeint_kwargs) -> List:
        
        self._solution_flag = True
        self.solution = [odeint(self.model_func, xi, 
                        self.time_span, args = (args, ), **odeint_kwargs) for xi, args in zip(self.initial_condition, self.arguments)]

        return self.solution 

    # plots the integrated solution with respect to time
    @staticmethod
    def plot(y_value : np.ndarray, x_value : np.ndarray, xlabel : str = "Time", 
            ylabel : str = "Concentration", legend : Optional[List[str]] = None, title : str = "Concentration vs time", **kwargs) -> None:
        
        plt.plot(x_value, y_value, **kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if legend :
            plt.legend(legend)
        plt.show()

    # calculates the actual derivative of the model
    @property
    def actual_derivative(self) -> List:
        assert self._solution_flag, "Integrate the model before calling this method"
        # passing tuples in vectorize can be an issue
        
        derivatives = []
        for xi, args in zip(self.solution, self.arguments):
            derivatives.append(np.vstack([self.model_func(_xi, _ti, args) for _xi, _ti in zip(xi, self.time_span)]))

        return derivatives
        """
        return [np.vectorize(partial(self.model_func, args = args), signature = "(m),(n)->(m)")(xi, self.time_span) for 
                xi, args in zip(self.solution, self.arguments)]
        """
    
    # calculates the approximate derivative using finite difference
    @property
    def approx_derivative(self) -> List:
        assert self._solution_flag, "Integrate the model before calling the method"
        
        # middle = np.zeros_like(self.solution)
        # second order central difference on the middle elements
        # middle[1:-1] = (self.solution[2:] -2*self.solution[1:-1] + self.solution[:-2])/((self.time_span[2:] - self.time_span[1:-1])**2).reshape(-1, 1)
        # second order forward difference on the first element
        # middle[0] = (self.solution[2] - 2*self.solution[1] + self.solution[0])/(self.time_span[1] - self.time_span[0])**2
        # second order backward difference on the last element
        # middle[-1] = (self.solution[-1] - 2*self.solution[-2] + self.solution[-3])/(self.time_span[-1] - self.time_span[0])**2
        
        return [ps.FiniteDifference()._differentiate(xi, self.time_span) for xi in self.solution]
    
    # adds gaussian noise the the datapoints
    def add_noise(self, mean : float = 0, variance : float  = 0.1, multiplicative = False) -> List:

        rng = np.random.default_rng(self.seed)
        if multiplicative:
            self.solution = [value * (1 + rng.normal(mean, variance, size = value.shape)) for value in self.solution]
            return self.solution
        else:
            self.solution = [value + rng.normal(mean, variance, size = value.shape) for value in self.solution]
            return self.solution

    @staticmethod
    def save_data(data : dict, path : str) -> None:
        
        with open(path, "wb") as file:
            pickle.dump(data, file)

if __name__ == "__main__":

    t_span = np.arange(0, 10, 0.1)
    # x_init = np.array([1, 2, 3, 4])
    model = DynamicModel("kinetic_kosir", t_span, arguments = [(373, 8.314)] ,n_expt = 2)
    solution = model.integrate()
    # model.plot(solution, t_span, "Time", "Concentration", ["A", "B", "C", "D"])

    print("actual derivatives shape", model.actual_derivative[0].shape)
    print("--"*20)
    print("noise added shape", model.add_noise()[0].shape)
    print("--"*20)
    

    # getting data for kinetic_arhenius
    model = DynamicModel("kinetic_kosir", t_span, n_expt = 10)
    solution = model.integrate()

    print("actual derivatives shape", model.actual_derivative[0].shape)
    print("--"*20)
    print("noise added shape", model.add_noise()[0].shape)
    print("--"*20)
    print(model.arguments)
    print("--"*20)
    print("coefficients", model.coefficients())

    model = DynamicModel("kinetic_kosir_temperature", t_span, n_expt = 2)
    integration = model.integrate()
    print("coefficients", model.coefficients(args_as_symbols = True))

    # testing menten kinetics
    t_span = np.arange(0, 20, 0.01)
    model = DynamicModel("kinetic_menten", t_span, n_expt = 5, arguments = [(0.1, 2, 3)])
    solution = model.integrate()