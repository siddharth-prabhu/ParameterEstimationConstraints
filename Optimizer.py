from dataclasses import dataclass, field
from Base import Base

from FunctionalLibrary import FunctionalLibrary
import numpy as np
import casadi as cd

@dataclass(frozen = False)
class Optimizer_casadi(Base):

    library : FunctionalLibrary = field(default_factory = FunctionalLibrary)
    input_features : list[str] = field(default_factory=list)
    alpha : float = field(default = 0.0)
    threshold : float = field(default = 0.1)

    opti = field(default = cd.Opti, init = False)
    adict = field(default_factory = dict)

    def __post_init__(self):
        pass

    
    def set_params(self, **kwargs):
        # sets the values of various parameter for gridsearchcv

        if kwargs.get("optimizer__alpha", False):
            setattr(self, "alpha", kwargs["optimizer__alpha"])
        
        if kwargs.get("optimizer__threshold", False):
            setattr(self, "threshold", kwargs["optimizer__threshold"])
            
        self.library.set_params(**kwargs)

    
    def _generate_library(self, data : list[np.ndarray], include_column = list[np.ndarray]):
        # given data creates list of matix of all possible combinations of terms 
        # returns a list of number of columns of each matrix

        self.adict["library"] = [self.library.fit_transform(data, include_column[i]) for i in range(self._n_states)]
        
        return [xi.shape for xi in library]

    def _create_decision_variables(self, library_dimension):
        # initializes the number of variables that will be used in casadi optimization 
        # returns adictionary for future value retrieval

        for i, dimension in enumerate(library_dimension):
            self.adict[f"coefficients{i}"] = self.opti.variable(dimension[1], 1)

        self.adict["mask"] = [np.ones_like(lib) for lib in self.adict["library"]]
        
    
    def _update_cost(target : np.ndarray, library_dimension : list):

        for i in range(self._n_states):
            self.adict["cost"] += (cd.sumsqr(target[:, i] - cd.mtimes(self.adict["library"][i], self.adict[f"coefficients{i}"]))/
                            cd.sum1(list(map(lambda x : x[0], library_dimension))))

    
    def _minimize(self, solver_dict):

        self.opti.minimize(self.adict["cost"])
        self.opti.solver("ipopt", solver_dict)
        return self.opti.solve()


    def fit(self, features : list[np.ndarray], target : list[np.ndarray], include_column : list[np.ndarray]):
        self._n_states = np.shape(features)[-1]
        features, target = np.vstack(features), np.vstack(target)
        library_dimension = self._generate_library(fetures, include_column)

        for i in range(10):
            self._create_decision_variables(library_dimension)
            # update library using mask 
            self.adict["library"] = [value*self.adict["mask"][i] for i, value in enumerate(self.adict["library"])]
            self._update_cost(target, library_dimension)
            self.adict[f"solution_{i}"] = self._minimize({"ipopt.print_level" : 1})

        return self.library.degree, self.alpha, self.threshold

    def predict(self):
        pass

    def score(self):
        pass

    def print(self):
        pass

if __name__ == "__main__":

    opti = Optimizer_casadi(FunctionalLibrary(2))
    print(opti.fit())
    kwargs = {"feature_library__degree" : 1, "optimizer__threshold" : 1}
    opti.set_params(**{"feature_library__degree" : 1, "optimizer__threshold" : 1, "optimizer__alpha" : 2.})
    print(opti.fit())

    another = Optimizer_casadi()
    print(another.library.degree)