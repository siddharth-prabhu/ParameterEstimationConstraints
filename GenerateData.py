from dataclasses import dataclass, field
import numpy as np
np.random.seed(10)

from typing import Optional
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# add features : change initial conditions, solution should automatically change
# noise should also (property). add timespam, initial in init
# add checks on the args, lenght of the inputs variables, etc
# 
@dataclass
class DynamicModel():

    model : str 
    initial_condition : np.ndarray 
    time_span : np.ndarray 

    model_dict : dict = field(init = False)
    solution_flag : bool = field(init = False, default = False)

    def __post_init__(self):
        self.model_dict = {"kinetic_simple" : DynamicModel.kinetic_simple,
                            "kinetic_kosir" : DynamicModel.kinetic_kosir}
        assert self.model in self.model_dict, "Dynamic model is not defined yet"

    @staticmethod
    def kinetic_simple(x, t, *args):
        # A + B <--> C --> D + B
        # k1 = 1, kr1 = 0.5, k2 = 2, kr2 = 1
        k1, kr1, k2, kr2 = args
        
        return [-k1*x[0]*x[1] + kr1*x[2],
                -k1*x[0]*x[1] + (kr1 + k2)*x[2],
                k1*x[0]*x[1] - (kr1 + k2)*x[2],
                k2*x[2]]
    
    @staticmethod
    def kinetic_kosir(x, t):

        return [-15.693*x[0] + 5.743*x[2] + 1.534*x[3],
            8.566*x[0],
            1.191*x[0] - 5.743*x[2],
            10.219*x[0] - 1.535*x[3]]

    # forward simulates the chosen model using scipy odeint
    def integrate(self, model_args: tuple, **odeint_kwargs):
        
        self.solution = odeint(self.model_dict[self.model], self.initial_condition, 
                        self.time_span, args = model_args, **odeint_kwargs)
        self.solution_flag = True

        return self.solution

    # plots the integrated solution with respect to time
    def plot(self):
        assert self.solution_flag, "Integrate the model before calling this method"
        
        plt.plot(self.time_span, self.solution)
        plt.show()

    # calculates the actual derivative of the model
    def actual_derivatives(self):
        assert self.solution_flag, "Integrate the model before calling this method"
        

    # adds gaussian noise the the datapoints
    def add_noise(self, mean, variance, mulitplicative = "False"):
        assert self.solution_flag, "Integrate the model before calling this method"

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
    model.plot()