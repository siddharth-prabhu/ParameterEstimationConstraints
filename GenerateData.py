from dataclasses import dataclass, field
import numpy as np
np.random.seed(10)

import pickle
from typing import ClassVar, Optional
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pysindy as ps

@dataclass
class DynamicModel():

    model : str 
    time_span : np.ndarray
    initial_condition : list[np.ndarray] = field(default_factory=list)
    n_expt : int = field(default = 1)

    _model_dict : ClassVar 
    _solution_flag : bool = field(init = False, default = False)

    def __post_init__(self):
        self._model_dict = {"kinetic_simple" : {"function" : DynamicModel.kinetic_simple, "n_states" : 4},
                            "kinetic_kosir" : {"function" : DynamicModel.kinetic_kosir, "n_states" : 4}}
        assert self.model in self._model_dict, "Dynamic model is not defined yet"
        
        if not self.initial_condition:
            self.initial_condition = [np.random.uniform(0, 20, size = self._model_dict[self.model]["n_states"]) for _ in range(self.n_expt)]
        else:
            assert len(self.initial_condition[-1]) == self._model_dict[self.model]["n_states"], "Incorrect number of states"
            assert len(self.initial_condition) == len(self.n_expt), "Initial conditions should match the number of experiments"

        self.model = self._model_dict[self.model]["function"]

    @staticmethod
    def kinetic_simple(x, t, *args) -> np.ndarray:
        # A + B <--> C --> D + B
        # k1 = 1, kr1 = 0.5, k2 = 2, kr2 = 1
        k1, kr1, k2, kr2 = args
        
        return np.array([-k1*x[0]*x[1] + kr1*x[2],
                -k1*x[0]*x[1] + (kr1 + k2)*x[2],
                k1*x[0]*x[1] - (kr1 + k2)*x[2],
                k2*x[2]])
    
    @staticmethod
    def kinetic_kosir(x, t, *args) -> np.ndarray:

        return np.array([-15.693*x[0] + 5.743*x[2] + 1.534*x[3],
            8.566*x[0],
            1.191*x[0] - 5.743*x[2],
            10.219*x[0] - 1.535*x[3]])

    # forward simulates the chosen model using scipy odeint
    def integrate(self, model_args: tuple = (), **odeint_kwargs) -> list:
        
        self._solution_flag = True
        self._model_args = model_args
        self.solution = [odeint(self.model, xi, 
                        self.time_span, args = self._model_args, **odeint_kwargs) for xi in self.initial_condition]

        return self.solution 

    # plots the integrated solution with respect to time
    @staticmethod
    def plot(y_value : np.ndarray, x_value : np.ndarray, xlabel : str = "Time", 
            ylabel : str = "Concentration", legend : Optional[list[str]] = None, **kwargs) -> None:
        
        plt.plot(x_value, y_value, **kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if legend :
            plt.legend(legend)
        plt.show()

    # calculates the actual derivative of the model
    @property
    def actual_derivative(self) -> list:
        assert self._solution_flag, "Integrate the model before calling this method"
        return [np.vectorize(self.model, signature = "(m),(n),(k)->(m)")(xi, self.time_span, self._model_args) for 
                xi in self.solution]

    # calculates the approximate derivative using finite difference
    @property
    def approx_derivative(self) -> list:
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
    @staticmethod
    def add_noise(data : list, mean : float = 0, variance : float  = 0.1, multiplicative = "False") -> list:
        if multiplicative:
            return [value * (1 + np.random.normal(mean, variance, size = value.shape)) for value in data]
        else:
            return [value + np.random.normal(mean, variance, shape = value.shape) for value in data]

    @staticmethod
    def save_data(data : dict, path : str) -> None:
        
        with open(path, "wb") as file:
            pickle.dump(data, file)

if __name__ == "__main__":

    t_span = np.arange(0, 10, 0.1)
    # x_init = np.array([1, 2, 3, 4])
    model = DynamicModel("kinetic_kosir", t_span, [], 2)
    solution = model.integrate()
    # model.plot(solution, t_span, "Time", "Concentration", ["A", "B", "C", "D"])

    print(model.actual_derivative)
    print("--"*20)
    print(model.add_noise(model.actual_derivative))
    print("--"*20)
    