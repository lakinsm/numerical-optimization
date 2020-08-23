from numopt.functions import ScalarFunction
from numopt.optimizers import UnconstrainedOptimizer
import numpy as np

# Create symmetric data centered at 0 (fixed data points to which we will fit parameter values using MLE)
my_data = np.array([-3, -1, -5, -4, 0, 0, 1, 3, 4, 5], dtype=np.float64)

# Initial guess for mu/sigma (not optimal on purpose)
mu = 2.
sigma = 1.
initial_guess = np.array([mu, sigma])


# Negative log-likelihood function for the normal distribution
def normNegLogLikelihood(params, data):
	mu = params[0]
	sigma = params[1]
	N = len(data)
	log_likelihood = -((N/2) * np.log(2 * np.pi)) - ((N/2) * np.log(sigma**2)) - ((1 / (2 * sigma**2)) * np.sum((data - mu)**2))
	return -log_likelihood


sf = ScalarFunction(normNegLogLikelihood)
opt = UnconstrainedOptimizer(sf, params=my_data)
result = opt.fit(initial_guess, return_all_values=True)
print("Optimal Values:")  # Should be the optimal value, mu ~= 0, sigma ~= 3.2
print("mu: {}\tsigma: {}".format(result[0][0], result[0][1]))
print("log-likelihood: {}".format(-result[1]))
print("\n\nParameter values at each iteration:")
for i, opt_points in enumerate(result[2]):
	print("Iter: {}\tmu: {}\tsigma: {}\tlog-likelihood: {}".format(
		i,
		opt_points[0][0],
		opt_points[0][1],
		opt_points[1]
	))
