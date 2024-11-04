from typing import NamedTuple, List

import numpy as np

class ReactionData(NamedTuple):
    arguments : List[tuple] # The first entry in the list is used for the LS problem formulation
    include_column : List[list]
    stoichiometry_unconstrained : np.ndarray
    stoichiometry_mass_balance : np.ndarray
    stoichiometry : np.ndarray



reaction_data = {
    "kinetic_kosir" : ReactionData(
        arguments = [(373, 8.314), (365, 8.314), (370, 8.314), (380, 8.314), (390, 8.314), (385, 8.314)],
        include_column = [[0, 1], [0, 2], [0, 3]],
        stoichiometry_unconstrained = np.eye(4),
        stoichiometry_mass_balance = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, -1, -0.5, -1]).reshape(4, -1), # "species_mass" : [56.108, 28.05, 56.106, 56.108]
        stoichiometry = np.array([-1, -1, -1, 2, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(4, -1), 
    ),

    "kinetic_menten" : ReactionData(
        arguments = [(0.1, 0.2, 0.3)],
        include_column = [[0, 1, 2], [1, 2, 3]],
        stoichiometry_unconstrained = np.eye(4),
        stoichiometry_mass_balance = np.array([1, 0, 0, 0, 1, 0, -1, 0, -1, 0, 0, 1]).reshape(4, -1), # mass balance A + C + D = 0 ; replaced C
        stoichiometry = np.array([-1, 0, -1, 1, 1, -1, 0, 1]).reshape(4, -1),
    ),

    "kinetic_carb" : ReactionData(
        arguments = [()],
        include_column = [[3, 4, 5, 10], [1, 3, 4, 6], [0, 2, 4, 6], [1, 3, 4, 7], [0, 3, 4, 8], [3, 8, 9, 10]],
        stoichiometry_unconstrained = np.eye(11),
        stoichiometry_mass_balance = np.array([
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        ]).reshape(11, -1), # mass balance sum(all species) = 0 : replaced D
        stoichiometry = np.array([
            0, 0, -1, 0, -1, 0, 0, -1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0,
            -1, 1, 0, -1, 1, 1, 1, -1, 1, 1, -1, 0, -1, 0, 0, 0, 0, 0, 
            0, 1, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, -1, 
            0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, -1
        ]).reshape(-1, 6),
    )
}