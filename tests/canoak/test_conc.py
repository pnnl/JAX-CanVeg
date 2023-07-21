import unittest

import jax
import numpy as np
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

import canoak  # noqa: E402
from jax_canoak.shared_utilities import conc  # noqa: E402

sze3 = 152
jtot = 30
jtot3 = 150
# jtot = 3
# jtot3 = 5
met_zl = 1.5
delz = 0.5
izref = 1
cref = 2.2
soilflux = 5.4
factor = 4.5
ustar_ref = 0.5
ustar = 1.8


class TestConc(unittest.TestCase):
    def test_conc(self):
        print("Performing test_conc()...")
        # inputs
        source_np = np.random.random(jtot)
        cncc_np = np.ones(jtot3)
        dispersion_np = np.random.rand(jtot3, jtot)
        source_jnp = jnp.array(source_np)
        dispersion_jnp = jnp.array(dispersion_np)

        # canoak conc
        # source_np = np.ones(jtot, dtype='float64')
        # cncc_np = np.ones(jtot3, dtype='float64') + 0.2
        # dispersion_np = np.ones([jtot3,jtot], dtype='float64') + 0.5
        # print("canoak...")
        izref_cpp = izref + 1
        canoak.conc(  # type: ignore
            cref,
            soilflux,
            factor,
            sze3,
            jtot,
            jtot3,
            met_zl,
            delz,
            izref_cpp,
            ustar_ref,
            ustar,
            source_np,
            cncc_np,
            dispersion_np,
        )

        # jax conc
        # print("jax...")
        conc_jit = jax.jit(conc)
        cncc_jnp = conc_jit(
            cref,
            soilflux,
            factor,
            # sze3, jtot, jtot3,
            met_zl,
            delz,
            izref,
            ustar_ref,
            ustar,
            source_jnp,
            dispersion_jnp,
        )

        print(cncc_jnp)
        # print(cncc_np)
        # print(dispersion_np)
        print("")
        # self.assertTrue(1 > 0)
        # print(cncc_jnp[cncc_jnp != cncc_np])
        # print(cncc_np[cncc_jnp != cncc_np])
        self.assertTrue(np.allclose(cncc_jnp, cncc_np))
