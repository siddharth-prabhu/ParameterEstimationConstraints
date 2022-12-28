import numpy as np
import matplotlib.pyplot as plt
    
from typing import List
    
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

def coefficient_difference_plot(original_coefficients_list : List[dict], **kwargs):
    """
    kwargs value should be of type : List[dict] and len == len(original_coefficients_list)

    Function that takes the difference of parameters in kwargs with the parameters in original_coefficient_list
    and plots them as horizontal bar plots 
    """
    #  

    # calculate the difference betweeen the coefficients
    def take_difference(ind, adict):
        for key, value in original_coefficients_list[ind].items():
            adict[key] = value - adict.get(key, 0)

    rows = -(-len(original_coefficients_list)//2)
    with plt.style.context(["science", "notebook", "light"]):
        fig, ax =  plt.subplots(rows, 2, figsize = (10, 15))
        fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
        ax = np.ravel(ax)
        
        for key, coefficients_list in kwargs.items():
            for i, adict in enumerate(coefficients_list):        
                take_difference(i, adict)
                labels, value = zip(*adict.items())
                ax[i].barh(list(map(str, labels)), value, label = f"{key}")
                ax[i].set(ylabel = "coefficients", xlabel = "", title = f"dx{i}/dt")
                ax[i].legend()

        plt.show()
        plt.close()