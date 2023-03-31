import unittest

import jax.numpy as jnp

from diffrax import NewtonNonlinearSolver


class TestNewtonRaphson(unittest.TestCase):
    def test_multivariate_newtonraphson_rootfinding(self):

        x_guess = [0.0, 0.0]
        atol, rtol = 1e-5, 1e-7

        def func(x, args=None):
            x1, x2 = x

            f1 = x1**2 + x2 - 3
            f2 = x1 * 3 - x2 * 4 + 5

            return jnp.array([f1, f2])

        solver = NewtonNonlinearSolver(atol=atol, rtol=rtol)
        solution = solver(func, x_guess, args=None)
        print(solution.root)
        print(func(solution.root))
