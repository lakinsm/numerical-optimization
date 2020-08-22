import numpy as np
import sys
from numopt.functions import ScalarFunction


class UnconstrainedOptimizer:
	def __init__(self, objective_function, params, max_iteration=50, tolerance=1e-6):
		assert(isinstance(objective_function, ScalarFunction))
		self.params = params
		self.f = self._wrapFunction(objective_function.function)
		self.g = self._wrapFunction(objective_function.grad)
		self.max_iters = max_iteration
		self.tol = tolerance

	def fit(self, x0):
		"""
		Find the optimum value for the objective function given a starting guess x0
		:param x0: double array or scalar, initial point
		:return: double array or scalar, input point that optimizes the objective function
		"""
		# Ensure x0 is an array
		x0 = np.atleast_1d(x0)

		# Initial values
		gnorm = np.inf  # the 2-norm of the gradient is our measure of gradient delta
		old_fval = self.f(x0)  # initial value of the function at the initial guess
		gfk = self.g(x0)  # initial gradient of the function at the initial guess
		k = 0  # the current iteration
		N = len(x0)  # the number of dimensions in x0
		I = np.eye(N, dtype=np.float64)  # identity matrix of dimension NxN
		Hk = I  # Initial guess at point k for the Hessian matrix (we just use the identity here as a starting point)
		old_old_fval = old_fval + (np.linalg.norm(gfk) / 2)  # Initial step guess (to initialize new value line search)
		xk = x0  # xk will be the variable in which we store the current x values at iteration k

		while (k < self.max_iters) & (gnorm > self.tol):
			k += 1

	def _wrapFunction(self, fun):
		"""
		Create a function definition that implicitly passes the fixed parameters so we don't have to keep passing them
		:param fun: Callable function
		:return: The callable function with the parameters implied, if there are any parameters
		"""
		assert(callable(fun))
		if self.params:
			params = self.params
			def newFunction(x):
				return fun(x, params)
		else:
			def newFunction(x):
				return fun(x)
		return newFunction

if __name__ == '__main__':
	def parabola(x):
		return x ** 2

	sf = ScalarFunction(parabola)
	opt = UnconstrainedOptimizer(sf, None)
	opt.fit(4)
