import unittest

import jax
import numpy as np

import jax.numpy as jnp
from jax import config

import canoak  # noqa: E402

from jax_canoak.physics.energy_fluxes import uz  # type: ignore
from jax_canoak.physics.energy_fluxes import boundary_resistance  # type: ignore
from jax_canoak.physics.energy_fluxes import friction_velocity  # type: ignore
from jax_canoak.shared_utilities.constants import lleaf, nnu  # type: ignore
from jax_canoak.shared_utilities.constants import nuvisc, dh  # type: ignore
from jax_canoak.shared_utilities.constants import dv, dc  # type: ignore

config.update("jax_enable_x64", True)

jtot = 3
jtot3 = 5
sze = jtot + 2
sze3 = jtot3 + 2
soilsze = 12
szeang = 19


class Test(unittest.TestCase):
    def test_uz(self):
        print("Performing test_uz()...")
        # Inputs
        zzz, ht, wnd = 1.2, 2.3, 4.2

        # CANOAK
        uz_np = canoak.uz(zzz, ht, wnd)  # type: ignore

        # JAX
        uz_jit = jax.jit(uz)
        uz_jnp = uz_jit(zzz, ht, wnd)

        # print(uz)
        print("")
        self.assertTrue(np.allclose(uz_np, uz_jnp))

    def test_friction_velocity(self):
        print("Performing test_friction_velocity()...")
        # Inputs
        ustar, H_old = 1.2, 0.5
        sensible_heat_flux = 2.3
        air_density, T_Kelvin = 1.034, 285.3

        # CANOAK
        H_old_np, zl_np = canoak.friction_velocity(  # type: ignore
            ustar, H_old, sensible_heat_flux, air_density, T_Kelvin
        )

        # JAX
        friction_velocity_jit = jax.jit(friction_velocity)
        H_old_jnp, zl_jnp = friction_velocity_jit(
            ustar, H_old, sensible_heat_flux, air_density, T_Kelvin
        )

        # print(zl_jnp, zl_np)
        # print(H_old_jnp, H_old_np)
        print("")
        self.assertTrue(np.allclose(zl_np, zl_jnp))
        self.assertTrue(np.allclose(H_old_np, H_old_jnp))

    def test_boundary_resistance(self):
        print("Performing test_boundary_resistance()...")
        # Inputs
        wnd, zzz, ht = 1.2, 1.5, 2.0
        delz, grasshof = ht / jtot, 9.8 * np.power(lleaf, 3) / np.power(nnu, 2)
        TLF, press_kPa = 280.34, 101.234
        pr33 = np.power(nuvisc / dh, 0.33)
        sc33 = np.power(nuvisc / dv, 0.33)
        scc33 = np.power(nuvisc / dc, 0.33)
        tair_filter_np = 278.2 + np.zeros(sze3)

        # CANOAK
        heat_np, vapor_np, co2_np = canoak.boundary_resistance(  # type: ignore
            delz,
            zzz,
            ht,
            TLF,
            grasshof,
            press_kPa,
            wnd,
            pr33,
            sc33,
            scc33,
            tair_filter_np,
        )

        # JAX
        ind = int(zzz / delz)
        tair_filter_z = jnp.array(tair_filter_np)[ind]
        boundary_resistance_jit = jax.jit(boundary_resistance)
        heat_jnp, vapor_jnp, co2_jnp = boundary_resistance_jit(
            zzz, ht, TLF, grasshof, press_kPa, wnd, pr33, sc33, scc33, tair_filter_z
        )

        print(heat_np, vapor_np, co2_np)
        print(heat_jnp, vapor_jnp, co2_jnp)
        print("")
        self.assertTrue(np.allclose(heat_np, heat_jnp))
        self.assertTrue(np.allclose(vapor_np, vapor_jnp))
        self.assertTrue(np.allclose(co2_np, co2_jnp))
