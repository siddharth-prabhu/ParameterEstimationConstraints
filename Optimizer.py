from dataclasses import dataclass, field

from FunctionalLibrary import FunctionalLibrary
import numpy as np
import casadi as cd

@dataclass(frozen = False)
class Optimizer:

    library : FunctionalLibrary = field(default_factory = FunctionalLibrary)
    # library_names : list[str]
    # method : str = field(default = "SINDy")
    alpha : float = field(default = 0.0)
    threshold : float = field(default = 0.1)

    def __post_init__(self):
        # assert np.shape(X) == np.shape(y), "features and targets should be of same dimention"
        pass

    def set_params(self, **kwargs):
        
        if kwargs.get("optimizer__alpha", False):
            setattr(self, "alpha", kwargs["optimizer__alpha"])
        
        if kwargs.get("optimizer__threshold", False):
            setattr(self, "threshold", kwargs["optimizer__threshold"])
            
        self.library.set_params(**kwargs)

    @staticmethod
    def generate_library():
        pass

    def fit(self):
        return self.library.degree, self.alpha, self.threshold

    def predict(self):
        pass

    def score(self):
        pass

    def print(self):
        pass

if __name__ == "__main__":

    opti = Optimizer(FunctionalLibrary(2))
    print(opti.fit())
    kwargs = {"feature_library__degree" : 1, "optimizer__threshold" : 1}
    opti.set_params(**{"feature_library__degree" : 1, "optimizer__threshold" : 1, "optimizer__alpha" : 2.})
    print(opti.fit())

    another = Optimizer()
    print(another.library.degree)