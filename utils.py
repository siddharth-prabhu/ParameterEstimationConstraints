import numpy as np
import matplotlib.pyplot as plt
    
    
def ensemble_plot(coefficients_list : list[dict], distribution : list[dict], inclusion : list[dict]) -> None:

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

def coefficient_difference_plot(original_coefficients_list : list[dict], coefficients_list : list[dict]):

    # calculate the difference betweeen the coefficients
    def take_difference(ind, adict):
        for key, value in original_coefficients_list[ind].items():
            adict[key] = value - adict.get(key, 0)
        
    rows = -(-len(original_coefficients_list)//2)
    fig, ax =  plt.subplots(rows, 2, figsize = (10, 15))
    fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
    ax = np.ravel(ax)
    for i, adict in enumerate(coefficients_list):
        
        take_difference(i, adict)
        labels, value = zip(*adict.items())
        ax[i].bar(list(map(str, labels)), value)
        ax[i].set(xlabel = "coefficients", ylabel = "", title = f"dx{i}/di")

    plt.show()
