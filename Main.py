from GenerateData import DynamicModel
from FunctionalLibrary import FunctionalLibrary
from HyperOpt import HyperOpt
from Optimizer import Optimizer_casadi

import numpy as np

# without hyperparameter optimization
t_span = np.arange(0, 5, 0.01)
model = DynamicModel("kinetic_kosir", t_span, n_expt = 15)
features = model.integrate() # list of features
target = model.approx_derivative # list of target value

opti = Optimizer_casadi(alpha = 0.0, threshold = 0.1, solver_dict={"ipopt.print_level" : 0, "print_time":0})
opti.fit(features, target, include_column = [[], [0, 1], [], []], 
        constraints_dict = {"mass_balance" : [56.108, 28.05, 56.106, 56.108], "consumption" : [], "formation" : []})
opti.print()

# with hyperparameter optmization
params = {"optimizer__threshold": [0.01, 0.1, 1, 10], 
    "optimizer__alpha": [0, 0.01, 0.1, 1, 10], 
    "feature_library__include_bias" : [False],
    "feature_library__degree": [1, 2, 3]}

opt = HyperOpt(features, target, t_span, params, Optimizer_casadi(solver_dict = {"ipopt.print_level" : 0, "print_time":0}))
opt.gridsearch()
opt.plot()