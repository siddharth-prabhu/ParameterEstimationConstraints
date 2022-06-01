from dataclasses import dataclass, field
import numpy as np
np.random.seed(10)

from typing import ClassVar
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# add checks on the args, lenght of the inputs variables, etc

@dataclass
class DynamicModel():

    model : str 
    initial_condition : np.ndarray 
    time_span : np.ndarray 

    _model_dict : ClassVar 

    def __post_init__(self):
        self._model_dict = {"kinetic_simple" : DynamicModel.kinetic_simple,
                            "kinetic_kosir" : DynamicModel.kinetic_kosir}
        assert self.model in self._model_dict, "Dynamic model is not defined yet"

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
        self.solution = odeint(self._model_dict[self.model], self.initial_condition, 
                        self.time_span, args = self._model_args, **odeint_kwargs)

        return self.solution 

    # plots the integrated solution with respect to time
    def plot(self, **kwargs):
        assert self.solution, "Integrate the model before calling this method"
        
        plt.plot(self.time_span, self.solution, **kwargs)
        plt.show()

    # calculates the actual derivative of the model
    @property
    def actual_derivative(self):
        assert self.solution, "Integrate the model before calling this method"
        return np.vectorize(self._model_dict[self.model], signature = "(m),(n),(k)->(m)")(self.solution, self.time_span, self._model_args)

    # calculates the approximate derivative using finite difference
    def approx_derivative(self):
        # (self.solution[1:, :] -2*self.solution[] + self.solution[:-1, :])/(self.time_span[1:] - self.time_span[:-1])**2
        pass
    
    # adds gaussian noise the the datapoints
    def add_noise(self, mean, variance, mulitplicative = "False"):
        assert self.solution, "Integrate the model before calling this method"

        if multiplicative:
            self.solution_noise = self.solution * (1 + np.random.normal(mean, variance, size = self.solution.shape))
        else:
            self.solution_noise = self.solution + np.random.normal(mean, variance, shape = self.solution.shape)

        return self.solution_noise



if __name__ == "__main__":

    t_span = np.arange(0, 10, 0.1)
    x_init = np.array([1, 2, 3, 4])
    model = DynamicModel("kinetic_kosir", x_init, t_span)
    model.integrate(())
    # model.plot()

    model.initial_condition = np.array([5, 6, 7, 8])
    model.integrate(()) # call integrate eveytime data is changed
    
    print(model.actual_derivative)