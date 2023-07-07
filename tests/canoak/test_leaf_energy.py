import unittest

import jax
import numpy as np

# import jax.numpy as jnp
from jax import config

import canoak  # noqa: E402

from jax_canoak.physics.energy_fluxes import es  # type: ignore
from jax_canoak.physics.energy_fluxes import sfc_vpd  # type: ignore

config.update("jax_enable_x64", True)

jtot = 3
jtot3 = 5
sze = jtot + 2
sze3 = jtot3 + 2
soilsze = 12
szeang = 19


class TestLeafEnergy(unittest.TestCase):
    def test_es(self):
        print("Performing test_es()...")
        # Inputs
        tk = 292.0

        # CANOAK
        es1 = canoak.es(tk)  # type: ignore

        # JAX
        es_jit = jax.jit(es)
        es2 = es_jit(tk)

        print("")
        self.assertTrue(np.allclose(es1, es2))

    def test_sfc_vpd(self):
        print("Performing test_sfc_vpd()...")
        # Inputs
        Z, hz, tlk = 1.5, 2.0, 292.0
        delz = hz / jtot
        leleafpt, latent, vapor = 2.3, 1.4, 5.6
        rhov_air_np = np.random.random(sze3)
        ind_z = int(Z / delz) - 1
        rhov_air_z = rhov_air_np[ind_z]

        # CANOAK
        vpd_np = canoak.sfc_vpd(  # type: ignore
            delz, tlk, Z, leleafpt, latent, vapor, rhov_air_np
        )

        # JAX
        sfc_vpd_jit = jax.jit(sfc_vpd)
        vpd_jnp = sfc_vpd_jit(tlk, leleafpt, latent, vapor, rhov_air_z)

        # print(vpd_np, vpd_jnp)
        print("")
        self.assertTrue(np.allclose(vpd_np, vpd_jnp))
