"""
Base classes for numerical optimization routines and statistical operations
"""


class ProbabilityDistribution:
	def __init__(self, params, bounds=None):
		self.params = params
		self.bounds = bounds


class Kernel:
	def __init__(self, kernel_function, params):
		self.K = kernel_function
		self.params = params
		self.data = None
		self.kernel_data = None

	def transform(self, data, store=False):
		transformed_data = self.K(data, self.params)
		if store:
			self.data = data
			self.kernel_data = transformed_data
		return transformed_data


class Optimizer:
	def __init__(self, opt_function, data, max_iterations=50, tolerance=1e-8, minimize=True):
		self.obj_function = opt_function
		self.data = data
		self.max_iterations = max_iterations
		self.tolerance = tolerance
		self.minimize = minimize

	def fit(self, method='bfgs'):
		x = 1
