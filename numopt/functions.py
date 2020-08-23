import numpy as np


class ScalarFunction:
	"""
	Maps input in f(x): R^n -> R
	Example: f(x, y) -> x^2 + y^2
	"""

	def __init__(self, callable_function, input_bounds=None):
		"""
		:param callable_function: Callable function
		:param input_bounds: list, [(lower, upper), (lower, upper), ...]
		"""
		assert(callable(callable_function))
		self.function = callable_function
		self.fun_value = None
		self.grad_value = None
		self.hes_value = None
		if input_bounds:
			self.lb = np.array([x[0] for x in input_bounds])
			self.ub = np.array([x[1] for x in input_bounds])
		else:
			self.lb = -np.inf
			self.ub = np.inf
		self.eps = np.finfo(np.float64).eps ** 0.5

	def fun(self, *args):
		local_value = self.function(*args)
		if self.fun_value != local_value:
			self.fun_value = local_value
		return self.fun_value

	def grad(self, x0, *args):
		local_x0 = np.atleast_1d(x0)
		fx0 = self.function(x0, *args)
		x_eps = self._adjustEpsToBounds(local_x0)
		x1 = x0 + x_eps
		gradient = np.array([0. for _ in range(len(x0))], dtype=np.float64)
		# The approximate derivative is the rise over run, perturbing one dimension at a time
		for i in range(len(x1)):
			x1i = local_x0.copy()
			x1i[i] = x1[i]
			fxi = self.function(x1i, *args)
			gradient[i] = fxi - fx0
		gradient /= x_eps
		return gradient

	def hes(self):
		return None

	def _adjustEpsToBounds(self, x):
		"""
		This function calculates the direction and magnitude of eps such that we remain within the input (x) bounds.
		:param x0: Double array, the original point to perturb
		:return: Double array, eps values that, when added to x0, remain within the input bounds
		"""
		# Store x as an array (if it isn't already)
		x0 = np.atleast_1d(x)

		# Determine the "space" we have to work with between x_values and bounds
		lower_dist = x0 - self.lb
		upper_dist = self.ub - x0

		# Try to increment in the forward direction
		x1 = x0 + self.eps

		# Set up an array to store the forward deltas
		h = np.array([self.eps for _ in range(len(x0))])

		# Check if the new (forward adjusted) values violate the bounds, store boolean results in array
		violated = (x1 < self.lb) | (x1 > self.ub)

		# Check if our eps value "fits" into the "space" we calculated earlier, store boolean results in array
		fitting = self.eps <= np.maximum(lower_dist, upper_dist)

		# For those observations that are violating the bounds AND where we have "space", flip the sign
		# so that we aren't violating the bounds anymore
		h[violated & fitting] *= -1

		# For those that aren't fitting, try adjusting up or down depending on which (up/down) value is larger.
		# Note that in the case where the value is so tight that we have no space in either direction, this just adds 0
		# to the value of x, ensuring that we remain in the bounded space
		forward = (upper_dist >= lower_dist) & ~fitting
		h[forward] = upper_dist[forward]
		backward = (upper_dist < lower_dist) & ~fitting
		h[backward] = -lower_dist[backward]

		# Return an array of eps that respects the input bounds
		return h


if __name__ == '__main__':
	def parabola(x):
		return x ** 2  # derivative is 2x

	def linear(x):
		return sum(x)

	def normPDF(x, mu, sigma):
		return (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * (((x - mu) / sigma)**2))

	x0 = np.array([1, 4, 5, 3])

	test1 = ScalarFunction(linear, input_bounds=[(0, 1), (4, 6), (4, 6), (0, 10)])
	print(test1.fun(x0))  # should return 13
	print(test1.grad(x0))  # should return <-2, 2, 2, 2>

	test2 = ScalarFunction(parabola)
	print(test2.fun(4))  # should return 16
	print(test2.grad(4))  # should return 8

	test3 = ScalarFunction(parabola, input_bounds=[(0, 4)])
	print(test2.fun(-4))  # should return 16
	print(test2.grad(-4))  # should return -8
