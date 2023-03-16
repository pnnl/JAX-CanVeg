import unittest

from jax import jit
import jax.numpy as jnp
from jax_watershed.physics.energy_flux.radiative_transfer import calculate_ground_albedos, calculate_ground_vegetation_emissivity

solar_elev_angle, f_snow = 30., 0.3
L, S = 2., 1,
rad_type, pft_ind = 'PAR', 10

class TestAlbedoEmissivity(unittest.TestCase):

    def test_calculate_ground_albedos(self):
        print("Performing test_calculate_ground_albedos()...")
        α_g_db, α_g_dif = calculate_ground_albedos(f_snow=f_snow, rad_type=rad_type)
        print("Albedos for direct beam and diffuse fluxes: {}".format(jnp.array([α_g_db, α_g_dif])))
        print("")
        self.assertIsInstance(α_g_db, float)

    def test_calculate_ground_vegetation_emissivity(self):
        print("Performing test_calculate_ground_vegetation_emissivity()...")
        ε_g, ε_v = calculate_ground_vegetation_emissivity(
            solar_elev_angle=solar_elev_angle, f_snow=f_snow, L=L, S=S, pft_ind=pft_ind
        )
        print("Emissivities of ground and vegetation: {}".format(jnp.array([ε_g, ε_v])))
        print("")
        self.assertIsInstance(ε_g, float)