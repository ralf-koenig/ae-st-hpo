from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np

# X has duplicates:
X = np.array([[1,1],[1,1],[1,1],[2,2],[2,2],[2,2]])
y = np.array([ 2, -2,  3,  2, -3, 4])

kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)
gpr.fit(X, y)

# Predict at the same input:
mu, std = gpr.predict(np.array([[1,1]]), return_std=True)
print(mu)
print(std)

mu, std = gpr.predict(np.array([[2,2]]), return_std=True)
print(mu)
print(std)


