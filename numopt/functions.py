import numpy as np
import sys


class ScalarFunction:
	"""
	Maps input in f(x): R^n -> R
	Example: f(x, y) -> x^2 + y^2
	"""

	def __init__(self, callable_function, input_bounds=None, function_bounds=None):
		"""
		:param callable_function: Callable function
		:param input_bounds: list, [(lower, upper), (lower, upper), ...]
		:param function_bounds: tuple, (lower, upper)
		"""
		assert(callable(callable_function))
		self.function = callable_function
		self.fun_value = None
		self.grad_value = None
		self.hes_value = None
		self.fun_lb = function_bounds[0]
		self.fun_ub = function_bounds[1]
		self.eps = np.finfo(np.float64).eps ** 0.5

	def fun(self, *args):
		local_value = self.function(*args)
		if self.fun_value != local_value:
			self.fun_value = local_value
		return self.fun_value

	def grad(self, *args):
		x = np.array(*args)
		local_result = self.function(x)
		upper_eps = self.function(x + self.eps)
		lower_eps = self.function(x - self.eps)
		if upper_eps > self.fun_ub:
			if lower_eps < self.fun_lb:
				print("Perturbation is outside of all bounds")
				sys.exit(1)
			return (local_result - lower_eps) / self.eps
		return (upper_eps - local_result) / self.eps

	def hes(self):
		return None


if __name__ == '__main__':
	def parabola(x):
		return x ** 2  # derivative is 2x

	x0 = np.array([1, 4, 5, 3])

	my_para = ScalarFunction(parabola, function_bounds=[-np.inf, np.inf])
	print(my_para.fun(4))  # should return 16
	print(my_para.grad(4))  # should be 8
	print()
