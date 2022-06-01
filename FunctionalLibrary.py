from dataclasses import dataclass, field
import numpy as np
import pysindy as ps


@dataclass(frozen = True)
class FunctionalLibrary():
    
    degree : int = field(default = 2)
    include_bias : bool = field(default = False)
    interaction_only : bool = field(default = False)
    include_interaction : bool = field(default = True)

    def fit_transform(self, feature : np.ndarray, remove_feature: list = []):
        self._feature_included = [feature for feature in range(feature.shape[1]) if feature not in remove_feature]
        feature = feature[:, self._feature_included]
        self.poly = ps.PolynomialLibrary(degree = self.degree, include_bias = self.include_bias, 
                                include_interaction = self.include_interaction, interaction_only = self.interaction_only)

        return self.poly.fit_transform(feature)

    def get_features(self, input_features : list[str]):
        input_features = [feature for anum, feature in enumerate(input_features) if anum in self._feature_included]

        return self.poly.get_feature_names(input_features)


if __name__ == "__main__":

    a = np.random.normal(size = (10, 3))
    lib = FunctionalLibrary(3)
    lib.fit_transform(a, [0])
    print(lib.get_features(["x1", "x2", "x3"]))


