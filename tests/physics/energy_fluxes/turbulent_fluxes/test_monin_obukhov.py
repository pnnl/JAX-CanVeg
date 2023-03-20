import unittest

from jaxopt import Bisection

from diffrax import NewtonNonlinearSolver

from jax_watershed.physics.energy_fluxes.turbulent_fluxes.monin_obukhov import func_most, calculate_ustar
from jax_watershed.physics.energy_fluxes.turbulent_fluxes.monin_obukhov import calculate_ψm, calculate_ψc
from jax_watershed.physics.energy_fluxes.turbulent_fluxes.monin_obukhov import calculate_Tstar, calculate_qstar
from jax_watershed.shared_utilities.constants import C_TO_K


class TestMoninObukhov(unittest.TestCase):

    def test_estimate_obukhov_length(self):
        print("Performing test_estimate_obukhov_length()...")
        # tol = 1e-7
        L_guess = -1.
        atol, rtol = 1e-5, 1e-7
        uz, z, d = 3., 2., 0.5
        z0m, z0c = 0.05, 0.05
        ts, tz = 12. + C_TO_K, 10. + C_TO_K
        qs, qz = 10., 8.
        kwarg = dict(uz=uz, Tz=tz, qz=qz, Ts=ts, qs=qs, z=z, d=d, z0m=z0m, z0c=z0c)
        # args = dict(uz=uz, tz=tz, qz=qz, ts=ts, qs=qs, z=z, d=d, z0m=z0m, z0c=z0c)

        func = lambda L, args: func_most(L, **args)
        print(func(L_guess, kwarg))

        solver = NewtonNonlinearSolver(atol=atol, rtol=rtol)
        solution = solver(func, L_guess, args=kwarg)
        # solution = solver(func_most, L_guess, args=kwarg)
        # print(solution.result, solution.num_steps)
        L = solution.root
        # bisec = Bisection(optimality_fun=func_most, lower=-99, upper=90, tol=tol)
        # result = bisec.run(**kwarg)
        # L = result.params

        # Calculate z-d at the reference height
        z_minus_d = z - d

        # Evaluate ψ for momentum at the reference height (z-d) and surface (z0m)
        ψm_z   = calculate_ψm(ζ=z_minus_d / L)
        ψm_z0m = calculate_ψm(ζ=z0m / L)

        # Evaluate ψ for scalars at the reference height (z-d) and surface (z0m)
        ψc_z   = calculate_ψc(ζ=z_minus_d / L)
        ψc_z0c = calculate_ψc(ζ=z0c / L)

        # Calculate the friction velocity
        ustar = calculate_ustar(u1=0., u2=uz, z1=d+z0m, z2=z, d=d, ψm1=ψm_z0m, ψm2=ψm_z)
        tstar = calculate_Tstar(T1=ts, T2=tz, z1=d+z0c, z2=z, d=d, ψc1=ψc_z0c, ψc2=ψc_z)
        qstar = calculate_qstar(q1=qs, q2=uz, z1=d+z0c, z2=z, d=d, ψc1=ψc_z0c, ψc2=ψc_z)

        print("The estimated Obukhov length is: {}".format(L))
        print("The estimated friction velocity is: {}".format(ustar))
        print("The estimated temperature scale is: {}".format(tstar))
        print("The estimated water vapor scale is: {}".format(qstar))
        print(ψc_z, ψc_z0c, ψm_z, ψm_z0m)
        print("")
        # self.assertTrue(result.state.error <= tol)
        self.assertTrue(func(L, kwarg) <= atol)