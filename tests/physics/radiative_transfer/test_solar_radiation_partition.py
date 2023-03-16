import unittest

# import sys
# sys.path.append('/Users/jian449/Library/CloudStorage/OneDrive-PNNL/Codes/jax-watershed/src')

import jax
from jax import jit, vmap
import jax.numpy as jnp
from jax_watershed.physics.energy_flux.radiative_transfer import partition_solar_radiation
from jax_watershed.physics.energy_flux.radiative_transfer import calculate_solar_elevation


class PartitionSolarRadiation(unittest.TestCase):

    def test_partition_solar_radiation(self):
        print("Performing test_partition_solar_radiation()...")
        radiations = partition_solar_radiation(
            # solar_rad=1064., solar_elev_angle=35., pres=101.976, 
            solar_rad=452., solar_elev_angle=42., pres=101.976, 
        )
        print("Direct beam PAR: {}".format(radiations[0]))
        print("Diffuse PAR: {}".format(radiations[1]))
        print("Direct beam NIR: {}".format(radiations[2]))
        print("Diffuse NIR: {}".format(radiations[3]))
        print("")
        # self.assertIsInstance(radiations, jnp.ndarray)
        par_prop = (radiations[0]+radiations[1]) / radiations.sum()
        self.assertTrue(par_prop<0.47 and par_prop>0.44)
    

    def test_partition_solar_radiation_vmap(self):
        print("Performing test_partition_solar_radiation_vmap()...")
        n = 10
        rads, elevs, press = 452.*jnp.ones(n), 42.*jnp.ones(n), 101.976*jnp.ones(n)
        # results = vmap(partition_solar_radiation)(rads, elevs, press)
        jitted_func = jit(vmap(partition_solar_radiation))
        results = jitted_func(rads, elevs, press)
        print(results)
        # inputs = jnp.stack([rads, elevs, press]).T
        self.assertTrue(results.shape == (n, 4))