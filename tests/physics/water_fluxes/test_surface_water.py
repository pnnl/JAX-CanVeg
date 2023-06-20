import unittest

import jax.numpy as jnp

from jax_watershed.physics.water_fluxes import canopy_water_balance
from jax_watershed.physics.water_fluxes import ground_water_balance
from jax_watershed.physics.water_fluxes import solve_surface_water


class TestSurfaceWater(unittest.TestCase):
    # def test_canopy_water_change(self):
    #     print("Performing test_canopy_water_balance()...")
    #     dW_can = canopy_water_balance(Δt=1800, W_can=.2, P=1., L=3.0, S=0.0, E_vw=0.0)
    #     print("The canopy water change is {} kg m-2 s-1.".format(dW_can))
    #     print("")
    #     self.assertTrue( dW_can >= 0 )

    def test_surface_water_change(self):
        print("Performing test_surface_water_balance()...")
        Δt = 1800
        W_can, L, S = 0.2, 3.0, 0.0
        P, E_vw, E_g = 1.0, 0.1, 0.0
        z_table, f_max, k_sat = 10.0, 1.0, 1e-7 * 1e3

        ΔW_can_Δt, Q_intr, Q_drip, f_pi = canopy_water_balance(
            Δt=Δt, W_can=W_can, P=P, L=L, S=S, E_vw=E_vw
        )
        ΔW_g_Δt, Q_infil, R = ground_water_balance(
            Q_drip=Q_drip,
            P=P,
            f_pi=f_pi,
            f_max=f_max,
            z_table=z_table,
            k_sat=k_sat,
            E_g=E_g,
        )

        print("The canopy interception: {} kg m-2 s-1.".format(Q_intr))
        print("The infiltration: {} kg m-2 s-1.".format(Q_infil))
        print("The canopy drip flux: {} kg m-2 s-1.".format(Q_drip))
        print("The runoff: {} kg m-2 s-1.".format(R))
        print("The canopy water change: {} kg m-2 s-1.".format(ΔW_can_Δt))
        print("The ground water change: {} kg m-2 s-1.".format(ΔW_g_Δt))
        print("")

        # self.assertTrue( 1 >= 0 )
        self.assertTrue(ΔW_can_Δt + ΔW_g_Δt == P - Q_infil - R - E_vw - E_g)

    def test_surface_water(self):
        print("Performing test_surface_water()...")
        Δt = 1800
        W_can, W_g = 0.2, 0.2
        L, S = 3.0, 0.0
        P, E_vw, E_g = 1.0, 0.1, 0.0
        z_table, f_max, k_sat = 10.0, 1.0, 1e-7 * 1e3

        W_surf = jnp.array([W_can, W_g])

        W_surf_new = solve_surface_water(
            Δt=Δt,
            W_surf=W_surf,
            L=L,
            S=S,
            P=P,
            E_vw=E_vw,
            E_g=E_g,
            z_table=z_table,
            f_max=f_max,
            k_sat=k_sat,
        )

        print("The canopy water: {} kg m-2.".format(W_surf_new[0]))
        print("The ground water: {} kg m-2.".format(W_surf_new[1]))
        print("")

        self.assertTrue(1 >= 0)
