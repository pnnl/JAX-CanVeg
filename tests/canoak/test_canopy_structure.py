import unittest

import jax
import numpy as np

# import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

import canoak  # noqa: E402
from jax_canoak.physics.carbon_fluxes import freq  # noqa: E402
from jax_canoak.physics.carbon_fluxes import gammaf  # noqa: E402

jtot = 3
jtot3 = 5
sze = jtot + 2
sze3 = jtot3 + 2
soilsze = 12


class TestCanopyStructure(unittest.TestCase):
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

        print(bdens_np, bdens_jnp)
        print("")
        self.assertTrue(np.allclose(bdens_np, bdens_jnp))
