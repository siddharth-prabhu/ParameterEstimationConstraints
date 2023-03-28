# type: ignore
from dataclasses import dataclass, field
import itertools as it
from typing import Optional, List, Tuple, Any, Iterable, Callable
import operator
from functools import reduce

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint


@dataclass(frozen = False)
class FunctionalLibrary():
    
    degree : int = field(default = 2)
    include_bias : bool = field(default = False)
    interaction_only : bool = field(default = False)
    include_interaction : bool = field(default = True)
    interpolation_scheme : Callable = field(default = CubicSpline)

    def __post_init__(self):

        assert self.degree >= 1, "Polynomial degree has to be >= 1"

    def set_params(self, **kwargs):
        if "feature_library__degree" in kwargs:
            setattr(self, "degree", kwargs["feature_library__degree"])
        
        if "feature_library__include_bias" in kwargs:
            setattr(self, "include_bias", kwargs["feature_library__include_bias"])
        
        if "feature_library__interaction_only" in kwargs:
            setattr(self, "interaction_only", kwargs["feature_library__interaction_only"])
        
        if "feature_library__include_interaction" in kwargs:
            setattr(self, "include_interaction", kwargs["feature_library__include_interaction"])
                
    def fit_transform(self, features : np.ndarray, include_feature: Optional[list[int]] = None, 
                        derivative_free : bool = False, time_span : Optional[np.ndarray] = None, get_function : Optional[bool] = False) -> np.ndarray:
        """
        include_feature is zero indexed list of indices eg : [0, 1, 2, 3]
        get_function returns the interpolation matrix as a function of time. (Only used in derivative_free case)
        """

        if derivative_free:
            assert isinstance(time_span, np.ndarray), "time_span should be specified for interpolation" 

        if include_feature:
            self._feature_included = include_feature
            features = features[:, self._feature_included]

        alist = []
        
        if derivative_free:
            # create interpolation scheme for all columns in features
            interpolation = [self.interpolation_scheme(time_span, feat) for feat in features.T]
            for i in range(self.degree):
                combinations = self._get_combinations(interpolation, i + 1)
                alist.extend(combinations)

            combinations_integration = lambda x, t : list(map(lambda a : reduce(lambda accum, value : value(t)*accum, a, 1), alist))
            if get_function:
                return combinations_integration
            else:
                return odeint(combinations_integration, combinations_integration(0, 0), time_span) - combinations_integration(0, 0)
        
        else:
            for i in range(self.degree):
                combinations = self._get_combinations(features.T, i + 1)
                alist.extend(list(map(lambda x : reduce(operator.mul, x), combinations)))

            return np.column_stack(alist)

    def _get_combinations(self, x : Any, repeat : int) -> Iterable:

        if self.include_interaction:
            return it.combinations_with_replacement(x, repeat)
        else:
            return it.combinations(x, repeat)


    def get_features(self, input_features : List[str]) -> List:
        
        if getattr(self, "_feature_included", False):
            input_features = [feature for anum, feature in enumerate(input_features) if anum in self._feature_included]

        alist = []
        for i in range(self.degree):
            combinations = self._get_combinations(input_features, i + 1)
            alist.extend(list(map(" ".join, combinations)))

        return alist


if __name__ == "__main__":

    a = np.random.normal(size = (10, 3))
    time_span = np.arange(0, 10, 1)
    lib = FunctionalLibrary(4)
    b = lib.fit_transform(a, [0, 1, 2], True, time_span)
    print(lib.get_features(["x0", "x1", "x2"]))
    print(b.shape)

