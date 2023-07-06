import unittest

import jax
import numpy as np

import jax.numpy as jnp
from jax import config

import canoak  # noqa: E402

from jax_canoak.physics.carbon_fluxes import stomata  # type: ignore
from jax_canoak.physics.carbon_fluxes import tboltz  # type: ignore
from jax_canoak.physics.carbon_fluxes import temp_func  # type: ignore
from jax_canoak.physics.carbon_fluxes import soil_respiration  # type: ignore

config.update("jax_enable_x64", True)

jtot = 3
jtot3 = 5
sze = jtot + 2
sze3 = jtot3 + 2
soilsze = 12
szeang = 19


class TestPhotosynthesisStomata(unittest.TestCase):
    def test_stomata(self):
        print("Performing test_stomata()...")
        # Inputs
        lai, pai, rcuticle = 3.0, 0.0, 10.0
        par_sun_np = np.random.random(sze) * 10
        par_shade_np = np.random.random(sze)
        sun_rs_np = np.zeros(sze)
        shd_rs_np = np.zeros(sze)

        # CANOAK
        canoak.stomata(  # type: ignore
            jtot, lai, pai, rcuticle, par_sun_np, par_shade_np, sun_rs_np, shd_rs_np
        )

        # JAX
        stomata_jit = jax.jit(stomata)
        sun_rs_jnp, shd_rs_jnp = stomata_jit(
            lai,
            pai,
            rcuticle,
            jnp.array(par_sun_np),
            jnp.array(par_shade_np),
        )

        print("")
        self.assertTrue(np.allclose(sun_rs_jnp, sun_rs_np))
        self.assertTrue(np.allclose(shd_rs_jnp, shd_rs_np))

    def test_tboltz(self):
        print("Performing test_tboltz()...")
        # Inputs
        rate, eakin = 10.0, 23.3
        topt, tl = 50.0, 10.4

        # CANOAK
        t_np = canoak.tboltz(rate, eakin, topt, tl)  # type: ignore

        # JAX
        tboltz_jit = jax.jit(tboltz)
        t_jnp = tboltz_jit(rate, eakin, topt, tl)

        # print(t_np, t_jnp)
        print("")
        self.assertTrue(np.allclose(t_np, t_jnp))

    def test_temp_func(self):
        print("Performing test_temp_func()...")
        # Inputs
        rate, eact = 10.0, 23.3
        tprime, tref, t_lk = 50.0, -10, 89.3

        # CANOAK
        t_np = canoak.temp_func(rate, eact, tprime, tref, t_lk)  # type: ignore

        # JAX
        temp_func_jit = jax.jit(temp_func)
        t_jnp = temp_func_jit(rate, eact, tprime, tref, t_lk)

        # print(t_np, t_jnp)
        print("")
        self.assertTrue(np.allclose(t_np, t_jnp))

    def test_soil_respiration(self):
        print("Performing test_temp_func()...")
        # Inputs
        Ts, base_respiration = 21.0, 8.0

        # CANOAK
        r_np = canoak.soil_respiration(Ts, base_respiration)  # type: ignore

        # JAX
        soil_respiration_jit = jax.jit(soil_respiration)
        r_jnp = soil_respiration_jit(Ts, base_respiration)

        # print(r_np, r_jnp)
        print("")
        self.assertTrue(np.allclose(r_np, r_jnp))
