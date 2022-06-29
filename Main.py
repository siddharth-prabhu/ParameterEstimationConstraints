from GenerateData import DynamicModel
from FunctionalLibrary import FunctionalLibrary
from HyperOpt import HyperOpt
from Optimizer import Optimizer_casadi

import numpy as np


def run_gridsearch(features : list[np.ndarray], target : list[np.ndarray], features_clean : list[np.ndarray], target_clean : list[np.ndarray], 
                    time_span : np.ndarray, parameters : dict, add_constraints : bool = False, 
                    filename : str = "saved_data\Gridsearch_results.html", title : str = "Conc vs time"):
    
    if add_constraints == "mass_balance":
        include_column = []
        constraints_dict = {"mass_balance" : [56.108, 28.05, 56.106, 56.108], "consumption" : [], "formation" : []}
        # mass balance : equality constraint; formation/consumption : inequality constraint
    elif add_constraints == "stoichiometry":
        include_column = [[0, 2], [0, 3], [0, 1]]
        constraints_dict = {"mass_balance" : [], "consumption" : [], "formation" : [], 
                                "stoichiometry" : np.array([-1, -1, -1, 0, 0, 2, 1, 0, 0, 0, 1, 0]).reshape(4, -1)}
    else:
        include_column = None
        constraints_dict = {}
    
    # model.plot(features[-1], t_span, legend=["A", "B", "C", "D"], title = title)
    opt = HyperOpt(features, target, features_clean, target_clean, time_span, parameters, Optimizer_casadi(solver_dict = {"ipopt.print_level" : 0, "print_time": 0}), 
                include_column = include_column, constraints_dict = constraints_dict)

    opt.gridsearch()
    opt.plot(filename, title)

def run_all(noise_level : float, parameters : dict, iterate = True):

    t_span = np.arange(0, 5, 0.01)
    model = DynamicModel("kinetic_kosir", t_span, n_expt = 15)
    features_clean = model.integrate()
    target_clean = model.approx_derivative
    features = model.add_noise(0, noise_level)
    target = model.approx_derivative

    if iterate:
        print(f"Running simulation for {noise_level} noise and without constraints")
        run_gridsearch(features, target, features_clean, target_clean, t_span, parameters, add_constraints = False, 
                    filename = f"saved_data\Gridsearch_no_con_{noise_level}.html", title = f"Without constraints {noise_level} noise")

        print(f"Running simulation for {noise_level} noise and with constraints")
        run_gridsearch(features, target, features_clean, target_clean, t_span, parameters, add_constraints = "mass_balance", 
                    filename = f"saved_data\Gridsearch_con_{noise_level}.html", title = f"With constraints {noise_level} noise") 
                    
        print(f"Running simulation for {noise_level} noise and with stoichiometry")
        run_gridsearch(features, target, features_clean, target_clean, t_span, parameters, add_constraints = "stoichiometry", 
                    filename = f"saved_data\Gridsearch_stoichiometry_{noise_level}.html", title = f"With stoichiometry {noise_level} noise") 

# with hyperparameter optmization
params = {"optimizer__threshold": [0.01, 0.1], 
    "optimizer__alpha": [0, 0.01, 0.1, 1], 
    "feature_library__include_bias" : [False],
    "feature_library__degree": [1, 2, 3]}


noise_level = [0.0, 0.01, 0.1, 0.2, 0.4]
for noise in noise_level:
    run_all(noise, params, iterate=True)

""" 
# without hyperparameter optimization
t_span = np.arange(0, 5, 0.01)
model = DynamicModel("kinetic_kosir", t_span, n_expt = 15)
features = model.integrate() # list of features
target_clean = model.approx_derivative # list of target value

features = model.add_noise(0, 0.0)
target = model.approx_derivative
print(f"Features value", features[-1][-1])
print(f"Target value", target[-1][-1])

# model.plot(features[-1], t_span, legend = ["A", "B", "C", "D"])

model = Optimizer_casadi(FunctionalLibrary(2) , alpha = 0, threshold = 0.1, solver_dict={"ipopt.print_level" : 0, "print_time":0})
model.fit(features[:12], target[:12], include_column = [], 
            constraints_dict= {})
model.print()
print(model.complexity) """