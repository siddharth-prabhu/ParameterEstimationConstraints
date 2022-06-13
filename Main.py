from GenerateData import DynamicModel
from FunctionalLibrary import FunctionalLibrary
from HyperOpt import HyperOpt
from Optimizer import Optimizer_casadi

import numpy as np
np.random.seed(10)

# without hyperparameter optimization
t_span = np.arange(0, 5, 0.01)
model = DynamicModel("kinetic_kosir", t_span, n_expt = 15)
features = model.integrate() # list of features
target = model.approx_derivative # list of target value
noise_sd = 0.1

# without hyperparameter optimization
# features = model.add_noise(features, 0, noise_sd)
# opti = Optimizer_casadi(FunctionalLibrary(1), alpha = 0.0, threshold = 0.01, solver_dict={"ipopt.print_level" : 0, "print_time":0})
""" opti.fit(features, target, include_column = [[], [], [], []], 
          constraints_dict = {"mass_balance" : [56.108, 28.05, 56.106, 56.108], "consumption" : [], "formation" : [1]}) """

""" opti.fit(features, target, include_column = [[0, 2], [0, 3], [0, 1]],
        constraints_dict = {"mass_balance" : [], "consumption" : [], "formation" : [1], 
                                "stoichiometry" : np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1)}) """
""" opti.print()  
 """
print("--"*20)


def run_gridsearch(parameters : dict, add_noise : float = 0.0, 
                    add_constraints : bool = False, filename : str = "saved_data\Gridsearch_results.html", title : str = "Conc vs time"):
    
    if add_constraints == "mass_balance":
        include_column = []
        constraints_dict = {"mass_balance" : [56.108, 28.05, 56.106, 56.108], "consumption" : [], "formation" : [1]}
        # mass balance : equality constraint; formation/consumption : inequality constraint
    elif add_constraints == "stoichiometry":
        include_column = [[0, 2], [0, 3], [0, 1]]
        constraints_dict = {"mass_balance" : [], "consumption" : [], "formation" : [1], 
                                "stoichiometry" : np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1)}
    else:
        include_column = None
        constraints_dict = {}

    t_span = np.arange(0, 5, 0.01)
    model = DynamicModel("kinetic_kosir", t_span, n_expt = 15)
    features = model.integrate() 
    if add_noise:
        features = model.add_noise(0, add_noise)
        
    target = model.approx_derivative
    
    # model.plot(features[-1], t_span, legend=["A", "B", "C", "D"], title = title)
    opt = HyperOpt(features, target, t_span, parameters, Optimizer_casadi(solver_dict = {"ipopt.print_level" : 0, "print_time":0}), 
                include_column = include_column, constraints_dict = constraints_dict)

    opt.gridsearch()
    opt.plot(filename, title)


# with hyperparameter optmization
params = {"optimizer__threshold": [0.01, 0.1], 
    "optimizer__alpha": [0, 0.01, 0.1, 1], 
    "feature_library__include_bias" : [False],
    "feature_library__degree": [1, 2, 3]}


noise_level = [0.0, 0.01, 0.1, 0.2, 0.4]
for noise in noise_level:
    run_gridsearch(params, add_noise = noise, add_constraints = False, 
                filename = f"saved_data\Gridsearch_no_con_{noise}.html", title = f"Without constraints {noise} noise") 
    run_gridsearch(params, add_noise = noise, add_constraints = "mass_balance", 
                filename = f"saved_data\Gridsearch_con_{noise}.html", title = f"With constraints {noise} noise") 
    run_gridsearch(params, add_noise = noise, add_constraints = "stoichiometry", 
                filename = f"saved_data\Gridsearch_stoichiometry_{noise}.html", title = f"With stoichiometry {noise} noise") 

