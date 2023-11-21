from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import sympy as smp
    
    
def ensemble_plot(coefficients_list : List[dict], distribution : List[dict], inclusion : List[dict]) -> None:
    """
    coefficients_list : list of dictionaries containing symbols as keys and values at different bootstrapping samples
    distribution : list of dictionaries that have symbols as keys and namedtuple(mean, deviation) as values
    inclusion : list of dictionaries that have symbols as keys and namedtuple(inclusion probability) as values

    plots histogram of the distribution of coefficients with mean and standard deviaiton in the tile. 
    Bar plots for inclusion probabilities
    """

    assert len(coefficients_list) == len(distribution) == len(inclusion), "arguments should be of same length"
    
    for i, (coefficients_dict, distribution_dict) in enumerate(zip(coefficients_list, distribution)):
        fig = plt.figure(figsize = (5, 8))
        fig.subplots_adjust(hspace = 0.5)
        for j, key in enumerate(coefficients_dict.keys()):
            ax = fig.add_subplot(len(coefficients_dict)//3 + 1, 3, j + 1)
            ax.hist(np.array(coefficients_dict[key], dtype=float), bins = 10)
            _mean, _deviation = distribution_dict[key].mean, distribution_dict[key].deviation
            ax.set_title(f"{key}, mean : {round(_mean, 2)}, sd : {round(_deviation, 2)}, cv : {round(_deviation / (_mean + 1e-15), 2)}")
    
        plt.show()

    fig, ax = plt.subplots(-(-len(inclusion)//2), 2, figsize = (10, 15))
    fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
    ax = np.ravel(ax)
    for i, inclusion_dict in enumerate(inclusion):
        inclusion_dict_keys = inclusion_dict.keys()
        ax[i].barh(list(map(str, list(inclusion_dict_keys))), [inclusion_dict[key].inclusion for key in inclusion_dict_keys])
        ax[i].set(title = f"Inclusion probability x{i}", xlim = (0, 1))
        
    plt.show()

def coefficients_plot(original_coefficients_list : List[dict], discovered_coefficients_list : List[dict], labels : Optional[List[list]] = None, 
                      figname : str = "coefficients_plot", **kwargs):
    """
    Function that plots the parameters in discovered_coefficients_list and the parameters in original_coefficient_list
    as horizontal bar plots.
    Dictionary keys should be symbols. 
    """ 
    assert len(original_coefficients_list) == len(discovered_coefficients_list), "Length of provided lists should be same"

    if labels is None:
        # get unique values of keys from both dictionaries
        labels = [list(set((*orig_dict.keys(), *dis_dict.keys()))) for orig_dict, dis_dict in zip(original_coefficients_list, discovered_coefficients_list)]

    def string_to_symbol(x):
        # convert string to symbols
        if isinstance(x, smp.Symbol):
            return x
        
        return smp.sympify(x.replace(" ", "*"))

    rows = len(original_coefficients_list)
    with plt.style.context(["science", "notebook", "light"]):
        fig, ax =  plt.subplots(rows, 1, figsize = (15, 20))
        fig.subplots_adjust(hspace = 0.8, wspace = 0.5)
        ax = np.ravel(ax)
        
        for i, orig_dict, dis_dict, label in zip(range(len(labels)), original_coefficients_list, discovered_coefficients_list, labels):
            
            x = np.arange(len(label))
            width = 1/5

            label = list(map(string_to_symbol, label))
            orig_values = list(map(lambda x : orig_dict.get(x, 0), label))
            dis_values = list(map(lambda x : dis_dict.get(x, 0), label))

            ax[i].bar(x - 0.5*width, orig_values, label = "original", width = width, color = "blue")
            ax[i].bar(x + 0.5*width, dis_values, label = "discovered", width = width, color = "red")
            ax[i].set(ylabel = "coefficients", title = f"Dynamic equation of x{i}")
            ax[i].set_xticks(x, labels = label, rotation = 90)
            ax[i].legend()

        plt.savefig(figname)
        plt.close()