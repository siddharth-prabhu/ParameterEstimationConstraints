from GenerateData import DynamicModel
from FunctionalLibrary import FunctionalLibrary
from HyperOpt import HyperOpt
from Optimizer import Optimizer_casadi

import numpy as np

# without hyperparameter optimization
t_span = np.arange(0, 5, 0.01)
model = DynamicModel("kinetic_kosir", t_span, n_expt = 15)
features = model.integrate() # list of features
# features = model.add_noise(features, 0, 0.1)
target = model.approx_derivative # list of target value
include_column = [[0, 2, 3], [0], [0, 2], [0, 3]]
constraints_dict = {"mass_balance" : [56.108, 28.05, 56.106, 56.108], "consumption" : [], "formation" : []}
# constraints_dict = None
# include_column = None

opti = Optimizer_casadi(alpha = 0.0, threshold = 0.01, solver_dict={"ipopt.print_level" : 0, "print_time":0})
# opti.fit(features, target, include_column = [[], [], [], []],)
# opti.print() # without hyperopt
print("--"*20)

def run_gridsearch(features : list[np.ndarray], target : list[np.ndarray], parameters : dict, add_noise : bool = False, add_constraints : bool = False):
    
    if add_constraints:
        include_column = [[0, 2, 3], [0], [0, 2], [0, 3]]
        constraints_dict = {"mass_balance" : [56.108, 28.05, 56.106, 56.108], "consumption" : [], "formation" : []}
    else:
        include_column = None
        constraints_dict = None

    if add_noise:
        features = opti.add_noise(features, 0, 0.1)

    opt = HyperOpt(features, target, t_span, parameters, Optimizer_casadi(solver_dict = {"ipopt.print_level" : 0, "print_time":0}), 
                include_column = include_column, constraints_dict = constraints_dict)

    opt.gridsearch()
    opt.plot()


# with hyperparameter optmization
params = {"optimizer__threshold": [0.01, 0.1, 1], 
    "optimizer__alpha": [0, 0.01, 0.1, 1], 
    "feature_library__include_bias" : [False],
    "feature_library__degree": [1, 2, 3]}

run_gridsearch(features, target, params, add_constraints = True)