import numpy as np
import sys
from numopt.functions import ScalarFunction


class UnconstrainedOptimizer:
	def __init__(self, objective_function, params, max_iteration=100, tolerance=1e-6):
		assert(isinstance(objective_function, ScalarFunction))
		self.obj_fun = objective_function
		self.params = params
		self.max_iters = max_iteration
		self.tol = tolerance

	def fit(self, x0):
		return True


if __name__ == '__main__':
	def parabola(x):
		return x ** 2

	my_func = ScalarFunction(parabola, None, function_bounds=[-np.inf, np.inf])
	my_opt = UnconstrainedOptimizer(my_func, [])
