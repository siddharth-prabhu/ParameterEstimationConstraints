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


opti = Optimizer_casadi(alpha = 0.0, threshold = 0.01, solver_dict={"ipopt.print_level" : 0, "print_time":0})
# opti.fit(features, target, include_column = [[], [], [], []], 
#          constraints_dict = {"mass_balance" : [56.108, 28.05, 56.106, 56.108], "consumption" : [], "formation" : [1]})
# opti.print() # without hyperopt
print("--"*20)


def run_gridsearch(features : list[np.ndarray], target : list[np.ndarray], parameters : dict, add_noise : bool = False, 
                    add_constraints : bool = False, filename : str = "saved_data\Gridsearch_results.html", title : str = "Conc vs time"):
    
    if add_constraints == "mass_balance":
        include_column = [[0, 2, 3], [0], [0, 2], [0, 3]]
        constraints_dict = {"mass_balance" : [56.108, 28.05, 56.106, 56.108], "consumption" : [], "formation" : [1]}
        # mass balance : equality constraint; formation/consumption : inequality constraint
    elif add_constraints == "stoichiometry":
        include_column = [[0, 1], [0, 2], [0, 3]]
        constraints_dict = {"mass_balance" : [], "consumption" : [], "formation" : [3], 
                                "stoichiometry" : np.array([-1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0, 2]).reshape(4, -1)}
    else:
        include_column = None
        constraints_dict = {}

    if add_noise:
        features = model.add_noise(features, 0, 0.01)
        
    model.plot(features[-1], t_span, legend=["A", "B", "C", "D"])
    opt = HyperOpt(features, target, t_span, parameters, Optimizer_casadi(solver_dict = {"ipopt.print_level" : 0, "print_time":0}), 
                include_column = include_column, constraints_dict = constraints_dict)

    opt.gridsearch()
    opt.plot(filename, title)


# with hyperparameter optmization
params = {"optimizer__threshold": [0.01, 0.1, 1, 10], 
    "optimizer__alpha": [0, 0.01, 0.1, 1, 10], 
    "feature_library__include_bias" : [False],
    "feature_library__degree": [1, 2, 3]}

noise_sd = 0.1
run_gridsearch(features, target, params, add_noise = noise_sd, add_constraints = False, filename = "saved_data\Gridsearch_no_con.html", 
                title = f"Without constraints {noise_sd} noise")
run_gridsearch(features, target, params, add_noise = noise_sd, add_constraints = "mass_balance", filename = "saved_data\Gridsearch_con.html",
                title = f"With constraints {noise_sd} noise")
run_gridsearch(features, target, params, add_noise = noise_sd, add_constraints = "stoichiometry", filename = "saved_data\Gridsearch_con.html",
                title = f"With constraints {noise_sd} noise")