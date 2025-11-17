import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import diptest  # Assuming you have diptest installed

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_histogram_with_gaussians(data, mean_1, mean_2, std_1, std_2, weight_1, weight_2, title):
    # Create histogram
    count, bins, ignored = plt.hist(data, bins=10, edgecolor='black', density=True, alpha=0.6)

    # Generate x-values for plotting the Gaussian curves
    x_vals = np.linspace(min(data) - 1, max(data) + 1, 1000)

    # Add the two normal distributions
    plt.plot(x_vals.reshape(-1, 1), weight_1 * norm.pdf(x_vals, loc=mean_1, scale=std_1).reshape(-1, 1), label=f'{weight_1:.2f} N({mean_1:.2f},{std_1:.2f})', linewidth=2)
    plt.plot(x_vals.reshape(-1, 1), weight_2 * norm.pdf(x_vals, loc=mean_2, scale=std_2).reshape(-1, 1), label=f'{weight_2:.2f} N({mean_2:.2f},{std_2:.2f})', linewidth=2)
    plt.plot(x_vals.reshape(-1, 1), weight_1 * norm.pdf(x_vals, loc=mean_1, scale=std_1).reshape(-1, 1) +
             weight_2 * norm.pdf(x_vals, loc=mean_2, scale=std_2).reshape(-1, 1), label=f'sum', linewidth=2)

    plt.title(f"Histogram - {title}")
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


# generate some bimodal random draws
N = 1000
hN = N // 2

# mean_1, mean_2 = -1, 1
# std_1, std_2  = 1, 1

x = np.empty(N, dtype=np.float64)
# x[:hN] = np.random.normal(mean_1, std_1, hN)
x[:N] = np.random.negative_binomial(5, .6, N)
# x[hN:] = np.random.normal(mean_2, std_2, hN)
# x[hN:] = np.random.negative_binomial(5, .3, hN)

# x = np.log1p(x)

# Step 2: Fit GMM
n_components = 2  # Number of Gaussians
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
gmm.fit(x.reshape(-1, 1))

mean_1, mean_2 = gmm.means_.flatten()
std_1, std_2  = gmm.covariances_.flatten()
weight_1, weight_2 = gmm.weights_.flatten()

# both the dip statistic and p-value
dip, pval = diptest.diptest(x)
print(f"dip statistic: {dip}, p-value: {pval}")

plot_histogram_with_gaussians(x, mean_1, mean_2, std_1, std_2, weight_1, weight_2,f"x - dip ({dip}) - p-value ({pval})")

