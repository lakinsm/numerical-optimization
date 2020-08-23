import numpy as np
from warnings import warn
from numopt.functions import ScalarFunction


# The following is based on the Python scipy package.
def lineSearchWolfe(f, myfprime, xk, pk, gfk=None, old_fval=None,
					   old_old_fval=None, args=(), c1=1e-4, c2=0.9, amax=None,
					   extra_condition=None, max_iter=10):
	"""Find alpha that satisfies strong Wolfe conditions.
	Parameters
	----------
	f : callable f(x,*args)
		Objective function.
	myfprime : callable f'(x,*args)
		Objective function gradient.
	xk : ndarray
		Starting point.
	pk : ndarray
		Search direction.
	gfk : ndarray, optional
		Gradient value for x=xk (xk being the current parameter
		estimate). Will be recomputed if omitted.
	old_fval : float, optional
		Function value for x=xk. Will be recomputed if omitted.
	old_old_fval : float, optional
		Function value for the point preceding x=xk.
	args : tuple, optional
		Additional arguments passed to objective function.
	c1 : float, optional
		Parameter for Armijo condition rule.
	c2 : float, optional
		Parameter for curvature condition rule.
	amax : float, optional
		Maximum step size
	extra_condition : callable, optional
		A callable of the form ``extra_condition(alpha, x, f, g)``
		returning a boolean. Arguments are the proposed step ``alpha``
		and the corresponding ``x``, ``f`` and ``g`` values. The line search
		accepts the value of ``alpha`` only if this
		callable returns ``True``. If the callable returns ``False``
		for the step length, the algorithm will continue with
		new iterates. The callable is only called for iterates
		satisfying the strong Wolfe conditions.
	max_iter : int, optional
		Maximum number of iterations to perform.
	Returns
	-------
	alpha : float or None
		Alpha for which ``x_new = x0 + alpha * pk``,
		or None if the line search algorithm did not converge.
	fc : int
		Number of function evaluations made.
	gc : int
		Number of gradient evaluations made.
	new_fval : float or None
		New function value ``f(x_new)=f(x0+alpha*pk)``,
		or None if the line search algorithm did not converge.
	old_fval : float
		Old function value ``f(x0)``.
	new_slope : float or None
		The local slope along the search direction at the
		new value ``<myfprime(x_new), pk>``,
		or None if the line search algorithm did not converge.
	Notes
	-----
	Uses the line search algorithm to enforce strong Wolfe
	conditions. See Wright and Nocedal, 'Numerical Optimization',
	1999, pp. 59-61.
	"""
	# Lists are used here for their persistence by reference assignment, since Python does not allow us to
	# pass by reference manually
	fc = [0]
	gc = [0]
	gval = [None]
	gval_alpha = [None]
	fprime = myfprime

	def phi(alpha):
		fc[0] += 1
		return f(xk + (alpha * pk), *args)

	def derphi(alpha):
		gc[0] += 1
		gval[0] = fprime(xk + (alpha * pk), *args)
		gval_alpha[0] = alpha
		return np.dot(gval[0], pk)

	if gfk is None:
		gfk = fprime(xk, *args)
	derphi0 = np.dot(gfk, pk)

	if extra_condition is not None:
		# Add the current gradient as argument
		def extra_condition2(alpha, phi):
			if gval_alpha[0] != alpha:
				derphi(alpha)
			x = xk + (alpha * pk)
			return extra_condition(alpha, x, phi, gval[0])
	else:
		extra_condition2 = None

	alpha_star, phi_star, old_fval, derphi_star = scalarSearchWolfe(
		phi,
		derphi,
		old_fval,
		old_old_fval,
		derphi0,
		c1,
		c2,
		amax,
		extra_condition2,
		max_iter=max_iter
	)

	if derphi_star is None:
		warn('The line search algorithm did not converge', Warning)
	else:
		# derphi_star is a scalar, so use the most recently calculated gardient used in computing it
		# derphi = gfk*pk.  This is the gradient at the next step; no need to compute it again in the outer loop
		derphi_star = gval[0]

	return alpha_star, fc[0], gc[0], phi_star, old_fval, derphi_star


def scalarSearchWolfe(phi, derphi, phi0=None,
						 old_phi0=None, derphi0=None,
						 c1=1e-4, c2=0.9, amax=None,
						 extra_condition=None, max_iter=10):
	"""Find alpha that satisfies strong Wolfe conditions.
	alpha > 0 is assumed to be a descent direction.
	Parameters
	----------
	phi : callable phi(alpha)
		Objective scalar function.
	derphi : callable phi'(alpha)
		Objective function derivative. Returns a scalar.
	phi0 : float, optional
		Value of phi at 0.
	old_phi0 : float, optional
		Value of phi at previous point.
	derphi0 : float, optional
		Value of derphi at 0
	c1 : float, optional
		Parameter for Armijo condition rule.
	c2 : float, optional
		Parameter for curvature condition rule.
	amax : float, optional
		Maximum step size.
	extra_condition : callable, optional
		A callable of the form ``extra_condition(alpha, phi_value)``
		returning a boolean. The line search accepts the value
		of ``alpha`` only if this callable returns ``True``.
		If the callable returns ``False`` for the step length,
		the algorithm will continue with new iterates.
		The callable is only called for iterates satisfying
		the strong Wolfe conditions.
	max_iter : int, optional
		Maximum number of iterations to perform.
	Returns
	-------
	alpha_star : float or None
		Best alpha, or None if the line search algorithm did not converge.
	phi_star : float
		phi at alpha_star.
	phi0 : float
		phi at 0.
	derphi_star : float or None
		derphi at alpha_star, or None if the line search algorithm
		did not converge.
	Notes
	-----
	Uses the line search algorithm to enforce strong Wolfe
	conditions. See Wright and Nocedal, 'Numerical Optimization',
	1999, pp. 59-61.
	"""
	if phi0 is None:
		phi0 = phi(0.)

	if derphi0 is None:
		derphi0 = derphi(0.)

	alpha0 = 0
	if old_phi0 is not None and derphi0 != 0:
		alpha1 = min(1.0, 1.01 * 2 * (phi0 - old_phi0) / derphi0)
	else:
		alpha1 = 1.0

	if alpha1 < 0:
		alpha1 = 1.0

	if amax is not None:
		alpha1 = min(alpha1, amax)

	phi_a1 = phi(alpha1)

	phi_a0 = phi0
	derphi_a0 = derphi0

	if extra_condition is None:
		extra_condition = lambda alpha, phi: True

	for i in range(max_iter):
		if alpha1 == 0 or (amax is not None and alpha0 == amax):
			# alpha1 should not be 0, perhaps due to numerical underflow
			alpha_star = None
			phi_star = phi0
			phi0 = old_phi0
			derphi_star = None

			if alpha1 == 0:
				msg = 'Rounding errors prevent line search convergence'
			else:
				msg = 'The line search algorithm could not find a solution <= amax: {}'.format(amax)

			warn(msg, Warning)
			break

		if (phi_a1 > phi0 + (c1 * alpha1 * derphi0)) or ((phi_a1 >= phi_a0) and (i > 1)):
			# Bracketed minimum found
			alpha_star, phi_star, derphi_star = _zoom(alpha0, alpha1, phi_a0, phi_a1, derphi_a0, phi,
													  derphi, phi0, derphi0, c1, c2, extra_condition)
			break

		derphi_a1 = derphi(alpha1)
		if np.abs(derphi_a1) <= -c2 * derphi0:
			# Wolfe/curvature condition met
			if extra_condition(alpha1, phi_a1):
				alpha_star = alpha1
				phi_star = phi_a1
				derphi_star = derphi_a1

		if derphi_a1 >= 0:
			# Bracketed minimum found / back search
			alpha_star, phi_star, derphi_star = _zoom(alpha1, alpha0, phi_a1, phi_a0, derphi_a1, phi,
													  derphi, phi0, derphi0, c1, c2, extra_condition)
			break

		alpha2 = 2 * alpha1  # increase by factor of 2 each iteration
		if amax is not None:
			alpha2 = min(alpha2, amax)
		alpha0 = alpha1
		alpha1 = alpha2
		phi_a0 = phi_a1
		phi_a1 = phi(alpha1)
		derphi_a0 = derphi_a1

	# For/else
	else:
		# max_iter reached
		alpha_star = alpha1
		phi_star = phi_a1
		derphi_star = None
		warn('The line search algorithm did not converge', Warning)

	return alpha_star, phi_star, phi0, derphi_star


def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo, phi, derphi, phi0, derphi0, c1, c2, extra_condition):
	"""Zoom stage of approximate linesearch satisfying strong Wolfe conditions.

		Part of the optimization algorithm in `scalarSearchWolfe`.

		Notes
		-----
		Implements Algorithm 3.6 (zoom) in Wright and Nocedal,
		'Numerical Optimization', 1999, pp. 61.
		"""
	max_iter = 10
	i = 0
	delta1 = 0.2  # cubic interpolant check
	delta2 = 0.1  # quadratic interpolant check
	phi_rec = phi0
	a_rec = 0
	while True:
		# interpolate to find a trial step length between a_lo and
		# a_hi Need to choose interpolation here. Use cubic
		# interpolation and then if the result is within delta *
		# dalpha or outside of the interval bounded by a_lo or a_hi
		# then use quadratic interpolation, if the result is still too
		# close, then use bisection

		dalpha = a_hi - a_lo
		if dalpha < 0:
			a, b = a_hi, a_lo
		else:
			a, b = a_lo, a_hi

		# minimizer of cubic interpolant
		# (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
		#
		# if the result is too close to the end points (or out of the
		# interval), then use quadratic interpolation with phi_lo,
		# derphi_lo and phi_hi if the result is still too close to the
		# end points (or out of the interval) then use bisection

		if i > 0:
			cchk = delta1 * dalpha
			a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
							a_rec, phi_rec)
		if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
			qchk = delta2 * dalpha
			a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
			if (a_j is None) or (a_j > b - qchk) or (a_j < a + qchk):
				a_j = a_lo + 0.5 * dalpha

		# Check new value of a_j

		phi_aj = phi(a_j)
		if (phi_aj > phi0 + c1 * a_j * derphi0) or (phi_aj >= phi_lo):
			phi_rec = phi_hi
			a_rec = a_hi
			a_hi = a_j
			phi_hi = phi_aj
		else:
			derphi_aj = derphi(a_j)
			if abs(derphi_aj) <= -c2 * derphi0 and extra_condition(a_j, phi_aj):
				a_star = a_j
				val_star = phi_aj
				valprime_star = derphi_aj
				break
			if derphi_aj * (a_hi - a_lo) >= 0:
				phi_rec = phi_hi
				a_rec = a_hi
				a_hi = a_lo
				phi_hi = phi_lo
			else:
				phi_rec = phi_lo
				a_rec = a_lo
			a_lo = a_j
			phi_lo = phi_aj
			derphi_lo = derphi_aj
		i += 1
		if i > max_iter:
			# Failed to find a conforming step size
			a_star = None
			val_star = None
			valprime_star = None
			break

	return a_star, val_star, valprime_star


def _cubicmin(a, fa, fpa, b, fb, c, fc):
	"""
	Finds the minimizer for a cubic polynomial that goes through the
	points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
	If no minimizer can be found, return None.
	"""
	# f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

	with np.errstate(divide='raise', over='raise', invalid='raise'):
		try:
			C = fpa
			db = b - a
			dc = c - a
			denom = (db * dc) ** 2 * (db - dc)
			d1 = np.empty((2, 2))
			d1[0, 0] = dc ** 2
			d1[0, 1] = -db ** 2
			d1[1, 0] = -dc ** 3
			d1[1, 1] = db ** 3
			[A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
											fc - fa - C * dc]).flatten())
			A /= denom
			B /= denom
			radical = B * B - 3 * A * C
			xmin = a + (-B + np.sqrt(radical)) / (3 * A)
		except ArithmeticError:
			return None
	if not np.isfinite(xmin):
		return None
	return xmin


def _quadmin(a, fa, fpa, b, fb):
	"""
	Finds the minimizer for a quadratic polynomial that goes through
	the points (a,fa), (b,fb) with derivative at a of fpa.
	"""
	# f(x) = B*(x-a)^2 + C*(x-a) + D
	with np.errstate(divide='raise', over='raise', invalid='raise'):
		try:
			D = fa
			C = fpa
			db = b - a * 1.0
			B = (fb - D - C * db) / (db * db)
			xmin = a - C / (2.0 * B)
		except ArithmeticError:
			return None
	if not np.isfinite(xmin):
		return None
	return xmin


class UnconstrainedOptimizer:
	def __init__(self, objective_function, params, max_iteration=50, tolerance=1e-6):
		assert (isinstance(objective_function, ScalarFunction))
		self.params = params
		self.f = self._wrapFunction(objective_function.function)
		self.g = self._wrapFunction(objective_function.grad)
		self.max_iters = max_iteration
		self.tol = tolerance

	def fit(self, x0):
		"""
		Find the optimum value for the objective function given a starting guess x0.
		Uses BFGS algorithm and line search with strong Wolfe conditions.
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

		print("Begin .fit()")
		while (k < self.max_iters) & (gnorm > self.tol):
			print("Fit Iteration: {}".format(k + 1))
			pk = -np.dot(Hk, gfk)

			print("Pre LineSearch: xk: {}\tHk: {}\tgfk: {}\tpk: {}".format(xk, Hk, gfk, pk))
			# alpha_star, fc[0], gc[0], phi_star, old_fval, derphi_star
			alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
				lineSearchWolfe(self.f, self.g, xk, pk, gfk, old_fval, old_old_fval)
			if alpha_k is None:
				alpha_k = 1.0
				gfkp1 = None
			xkp1 = xk + (alpha_k * pk)
			sk = xkp1 - xk
			xk = xkp1
			if gfkp1 is None:
				gfkp1 = self.g(xkp1)

			yk = gfkp1 - gfk
			gfk = gfkp1

			gnorm = np.linalg.norm(gfk)

			try:
				rhok = 1.0 / (np.dot(yk, sk))
			except ZeroDivisionError:
				rhok = 1000.0
			if np.isinf(rhok):
				rhok = 1000.0
			A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
			A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
			Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] * sk[np.newaxis, :])
			k += 1

		fval = self.f(xk)
		print("End .fit()")
		return xk, fval

	def _wrapFunction(self, fun):
		"""
		Create a function definition that implicitly passes the fixed parameters so we don't have to keep passing them
		:param fun: Callable function
		:return: The callable function with the parameters implied, if there are any parameters
		"""
		assert (callable(fun))
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
	opt = UnconstrainedOptimizer(sf, None, max_iteration=50)
	res = opt.fit(4)
	print(res)  # Should return approximately (0, 0)

	def paraboloid(x):
		return ((x[0] / 3)**2) + (x[1]**2)

	sf2 = ScalarFunction(paraboloid)
	opt2 = UnconstrainedOptimizer(sf2, None, max_iteration=50)
	res2 = opt2.fit([1, 1])
	print(res2)
