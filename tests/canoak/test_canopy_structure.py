import unittest

import jax
import numpy as np

import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

import canoak  # noqa: E402

# from jax_canoak.physics.carbon_fluxes import angle  # noqa: E402
# from jax_canoak.physics.carbon_fluxes import freq  # noqa: E402
from jax_canoak.physics.carbon_fluxes import angle  # noqa: E402
from jax_canoak.physics.carbon_fluxes import lai_time  # noqa: E402
from jax_canoak.physics.energy_fluxes import freq  # noqa: E402
from jax_canoak.physics.energy_fluxes import gammaf  # noqa: E402

jtot = 3
jtot3 = 5
sze = jtot + 2
sze3 = jtot3 + 2
soilsze = 12
szeang = 19


class TestCanopyStructure(unittest.TestCase):
    def test_angle(self):
        print("Performing test_angle()...")
        # Inputs
        latitude, longitude, zone = 38.1, -121.65, -8.0
        year, day_local, hour_local = (
            2019,
            194,
            12.0,
        )

        # CANOAK
        beta_rad, sin_beta, beta_deg = canoak.angle(  # type: ignore
            latitude, longitude, zone, year, day_local, hour_local
        )

        # JAX
        angle_jit = jax.jit(angle)
        beta_rad_j, sin_beta_j, beta_deg_j = angle_jit(  # type: ignore
            latitude, longitude, zone, year, day_local, hour_local
        )

        # print(beta_rad, sin_beta, beta_deg)
        print("")
        self.assertTrue(np.allclose(beta_rad, beta_rad_j))
        self.assertTrue(np.allclose(sin_beta, sin_beta_j))
        self.assertTrue(np.allclose(beta_deg, beta_deg_j))
        # self.assertTrue(1==1)

    def test_gammaf(self):
        print("Performing test_irflux()...")
        # Inputs
        x = 2.0

        # CANOAK
        y_np = canoak.gammaf(x)  # type: ignore

        # JAX
        gammaf_jit = jax.jit(gammaf)
        y_jnp = gammaf_jit(x)

        # print(y_jnp, y_np)
        print("")
        self.assertTrue(np.allclose(y_np, y_jnp))

    def test_lai_time(self):
        print("Performing test_lai_time()...")
        # Inputs
        tsoil, lai, ht = 285.0, 4.0, 1.0
        par_reflect, par_trans, par_soil_refl = 0.0377, 0.072, 0.0
        par_absorbed = 1 - par_reflect - par_trans - par_soil_refl
        nir_reflect, nir_trans, nir_soil_refl = 0.60, 0.26, 0.0
        nir_absorbed = 1 - nir_reflect - nir_trans - nir_soil_refl
        ht_midpt_np = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        # lai_freq_np = np.array([0.05, .3,.3, .3, .05]) * lai
        lai_freq_np = np.array([0.6, 0.6, 0.6, 0.6, 0.6]) * lai
        dLAIdz_np, exxpdir_np = np.zeros(5), np.zeros(5)
        bdens_np = np.zeros(9)
        Gfunc_sky_np = np.zeros([sze, szeang])

        # CANOAK
        canoak.lai_time(  # type: ignore
            jtot,
            sze,
            tsoil,
            lai,
            ht,
            par_reflect,
            par_trans,
            par_soil_refl,
            par_absorbed,
            nir_reflect,
            nir_trans,
            nir_soil_refl,
            nir_absorbed,
            ht_midpt_np,
            lai_freq_np,
            bdens_np,
            Gfunc_sky_np,
            dLAIdz_np,
            exxpdir_np,
        )

        # JAX
        # lai_time_jit = jax.jit(lai_time)
        exxpdir_jnp, dLAIdz_jnp, Gfunc_sky_jnp = lai_time(
            sze, lai, ht, jnp.array(ht_midpt_np), jnp.array(lai_freq_np)
        )

        # print(dLAIdz_np)
        # print(dLAIdz_jnp, dLAIdz_np)
        # print(Gfunc_sky_jnp, Gfunc_sky_np)
        # print(exxpdir_jnp, exxpdir_np)
        print("")
        self.assertTrue(np.allclose(exxpdir_jnp, exxpdir_np, atol=1e-4))
        self.assertTrue(np.allclose(dLAIdz_jnp, dLAIdz_np))
        self.assertTrue(np.allclose(Gfunc_sky_jnp, Gfunc_sky_np, atol=1e-4))

    def test_freq(self):
        print("Performing test_freq()...")
        # Inputs
        lflai = 4.0
        # bdens_np = np.zeros(soilsze)
        bdens_np = np.zeros(9)

        # CANOAK
        canoak.freq(lflai, bdens_np)  # type: ignore

        # JAX
        freq_jit = jax.jit(freq)
        bdens_jnp = freq_jit(lflai)

        # print(bdens_np, bdens_jnp)
        print("")
        self.assertTrue(np.allclose(bdens_np, bdens_jnp))
