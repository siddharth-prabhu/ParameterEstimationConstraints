import numpy as np
np.random.seed(10)

from GenerateData import DynamicModel


t_span = np.arange(0, 5, 0.01)
model = DynamicModel("kinetic_kosir", t_span, n_expt = 1)
features = model.integrate() # list of features

features_noise = model.add_noise(features, 0, 0.1)
features_manual = [xi + np.random.normal(0, 0.1) for xi in features]
print(np.isclose(features_noise, features_manual).all())