from dataclasses import dataclass, field
from xml.etree.ElementInclude import include
import numpy as np
import pysindy as ps

from typing import Optional

@dataclass(frozen = False)
class FunctionalLibrary():
    
    degree : int = field(default = 2)
    include_bias : bool = field(default = False)
    interaction_only : bool = field(default = False)
    include_interaction : bool = field(default = True)

    def set_params(self, **kwargs):
        if kwargs.get("feature_library__degree", False):
            setattr(self, "degree", kwargs["feature_library__degree"])
        
        if kwargs.get("feature_library__include_bias", False):
            setattr(self, "include_bias", kwargs["feature_library__include_bias"])
        
        if kwargs.get("feature_library__interaction_only", False):
            setattr(self, "interaction_only", kwargs["feature_library__interaction_only"])
        
        if kwargs.get("feature_library__include_interaction", False):
            setattr(self, "include_interaction", kwargs["feature_library__include_interaction"])
                
    def fit_transform(self, features : np.ndarray, include_feature: Optional[list[int]] = None) -> np.ndarray:
        # include_feature is zero indexed list of indices eg : [[0, 1, 2, 3]]

        if include_feature:
            self._feature_included = include_feature
            features = features[:, self._feature_included]
        
        self.poly = ps.PolynomialLibrary(degree = self.degree, include_bias = self.include_bias, 
                                include_interaction = self.include_interaction, interaction_only = self.interaction_only)

        return self.poly.fit_transform(features)

    def get_features(self, input_features : list[str]):
        
        if getattr(self, "_feature_included", False):
            input_features = [feature for anum, feature in enumerate(input_features) if anum in self._feature_included]

        return self.poly.get_feature_names(input_features)


if __name__ == "__main__":

    a = np.random.normal(size = (10, 3))
    lib = FunctionalLibrary()
    lib.fit_transform(a)
    print(lib.get_features(["x1", "x2", "x3"]))
    lib.set_params(**{"feature_library__degree" : 3})
    print(lib.get_features(["x1", "x2", "x3"]))


