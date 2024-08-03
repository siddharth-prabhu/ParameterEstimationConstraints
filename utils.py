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

def coefficients_plot(original_coefficients_list : List[dict], discovered_coefficients_list : List[List[dict]], labels : Optional[List[list]] = None, 
                      expt_names : List[str] = [], path : str = "coefficients_plot", title : Optional[str] = None, **kwargs):
    """
    Function that plots the parameters in discovered_coefficients_list and the parameters in original_coefficient_list
    as horizontal bar plots.
    Dictionary keys should be symbols. 

    discovered_coefficients_list = List[coefficients for various models]
    """ 
    num_mod = len(discovered_coefficients_list)
    num_coeff_list = len(discovered_coefficients_list[0])
    assert all(map(lambda coeff_list : len(coeff_list) == num_coeff_list, discovered_coefficients_list)), "Length of coefficient list of different models should be same"
    assert len(original_coefficients_list) == num_coeff_list, "Length of provided lists should be same"
    
    if len(expt_names) > 0:
        assert len(expt_names) == num_mod, "Experiment names should be equal to the number of different models provided"
    else : 
        expt_names = ["discovered"]*num_mod

    title = "Coefficients in dynamic equation of r" if title is None else title

    if labels is None:
        # get unique values of keys from both dictionaries
        labels = []

        for i, orig_dict in enumerate(original_coefficients_list):

            _keys = []
            for _dict in discovered_coefficients_list :
                _keys.extend([*_dict[i].keys()])

            labels.append(set((*orig_dict.keys(), *_keys)))

    def string_to_symbol(x):
        # convert string to symbols to match the keys of dictionary
        if isinstance(x, str):
            return smp.sympify(x.replace(" ", "*"))
        return x
    
    def symbol_to_string(x):
        if isinstance(x, str):
            return x
        return str(x).replace("**", "^").replace("*", " ")

    max_x = max(map(len, labels))
    rows = len(original_coefficients_list)
    with plt.style.context(["science", "notebook", "light"]):
        fig, ax =  plt.subplots(rows, 1, figsize = (15, 20))
        fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
        ax = np.ravel(ax)
        
        for i, orig_dict, label in zip(range(len(labels)), original_coefficients_list, labels):
            
            x = np.arange(len(label))
            width = 1/max(5, num_mod + 2)
            start = num_mod / 2

            label_sym = list(map(string_to_symbol, label))
            orig_values = list(map(lambda x : orig_dict.get(x, 0), label_sym))
            dis_values = [list(map(lambda x : dis_dict[i].get(x, 0), label_sym)) for dis_dict in discovered_coefficients_list]
            label_str = list(map(symbol_to_string, label_sym))

            ax[i].bar(x - start*width, orig_values, label = "original", width = width, edgecolor = "k", **kwargs)
            for j in range(num_mod):
                ax[i].bar(x + (- start + j + 1)*width, dis_values[j], label = expt_names[j], width = width, edgecolor = "k", **kwargs)
            
            ax[i].set(xlabel = "coefficients of", ylabel = "value", title = f"{title}{i}")
            ax[i].set_xticks(x, labels = label_str, rotation = 90)
            ax[i].set_xlim(left = -2*width, right = max_x)
            ax[i].hlines(0, -2*width, max_x, "k", alpha = 1, linestyles = "solid", linewidth = 1)
            ax[i].legend()
            ax[i].grid(axis = "x", color = "k", alpha = 1, linestyle = "solid", linewidth = 1)

        plt.savefig(path)
        plt.close()