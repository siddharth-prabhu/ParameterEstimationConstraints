from dataclasses import dataclass, field
import numpy as np
np.random.seed(10)

from typing import ClassVar
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import pysindy as ps

@dataclass
class DynamicModel():

    model : str 
    initial_condition : np.ndarray 
    time_span : np.ndarray 

    _model_dict : ClassVar 
    _solution_flag : bool = field(init = False, default = False)

    def __post_init__(self):
        self._model_dict = {"kinetic_simple" : {"function" : DynamicModel.kinetic_simple, "n_states" : 4},
                            "kinetic_kosir" : {"function" : DynamicModel.kinetic_kosir, "n_states" : 4}}
        assert self.model in self._model_dict, "Dynamic model is not defined yet"
        assert len(self.initial_condition) == self._model_dict[self.model]["n_states"], "Incorrect number of states"
        
        self.model = self._model_dict[self.model]["function"]

    @staticmethod
    def kinetic_simple(x, t, *args):
        # A + B <--> C --> D + B
        # k1 = 1, kr1 = 0.5, k2 = 2, kr2 = 1
        k1, kr1, k2, kr2 = args
        
        return np.array([-k1*x[0]*x[1] + kr1*x[2],
                -k1*x[0]*x[1] + (kr1 + k2)*x[2],
                k1*x[0]*x[1] - (kr1 + k2)*x[2],
                k2*x[2]])
    
    @staticmethod
    def kinetic_kosir(x, t, *args):

        return np.array([-15.693*x[0] + 5.743*x[2] + 1.534*x[3],
            8.566*x[0],
            1.191*x[0] - 5.743*x[2],
            10.219*x[0] - 1.535*x[3]])

    # forward simulates the chosen model using scipy odeint
    def integrate(self, model_args: tuple, **odeint_kwargs):
        
        self._solution_flag = True
        self._model_args = model_args
        self.solution = odeint(self.model, self.initial_condition, 
                        self.time_span, args = self._model_args, **odeint_kwargs)

        return self.solution 

    # plots the integrated solution with respect to time
    @staticmethod
    def plot(y_value : np.ndarray, x_value : np.ndarray, xlabel : str, ylabel : str, legend : list[str], **kwargs):
        
        plt.plot(x_value, y_value, **kwargs)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(legend)
        plt.show()

    # calculates the actual derivative of the model
    @property
    def actual_derivative(self):
        assert self._solution_flag, "Integrate the model before calling this method"
        return np.vectorize(self.model, signature = "(m),(n),(k)->(m)")(self.solution, self.time_span, self._model_args)

    # calculates the approximate derivative using finite difference
    @property
    def approx_derivative(self):
        assert self._solution_flag, "Integrate the model before calling the method"
        
        middle = np.zeros_like(self.solution)
        # second order central difference on the middle elements
        middle[1:-1] = (self.solution[2:] -2*self.solution[1:-1] + self.solution[:-2])/((self.time_span[2:] - self.time_span[1:-1])**2).reshape(-1, 1)
        # second order forward difference on the first element
        middle[0] = (self.solution[2] - 2*self.solution[1] + self.solution[0])/(self.time_span[1] - self.time_span[0])**2
        # second order backward difference on the last element
        middle[-1] = (self.solution[-1] - 2*self.solution[-2] + self.solution[-3])/(self.time_span[-1] - self.time_span[0])**2
        
        return middle
    
    # adds gaussian noise the the datapoints
    @staticmethod
    def add_noise(data : np.ndarray, mean : float = 0, variance : float  = 0.1, multiplicative = "False"):
        if multiplicative:
            return data * (1 + np.random.normal(mean, variance, size = data.shape))
        else:
            return data + np.random.normal(mean, variance, shape = data.shape)



if __name__ == "__main__":

    t_span = np.arange(0, 10, 0.1)
    x_init = np.array([1, 2, 3, 4])
    model = DynamicModel("kinetic_kosir", x_init, t_span)
    solution = model.integrate(())
    # model.plot(solution, t_span, "Time", "Concentration", ["A", "B", "C", "D"])

    model.initial_condition = np.array([5, 6, 7, 8])
    model.integrate(()) # call integrate eveytime data is changed
    
    print(model.actual_derivative)
    print("--"*20)
    print(model.add_noise(model.actual_derivative))
    print("--"*20)
    print(solution.shape)
    
    print(np.isclose(ps.FiniteDifference()._differentiate(solution, t_span), model.approx_derivative, atol = 10**-5))