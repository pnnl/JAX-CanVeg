import unittest

import sys
sys.path.append('/Users/jian449/Library/CloudStorage/OneDrive-PNNL/Codes/jax-watershed/src')

from jax import jit
import jax.numpy as jnp
from jax_watershed.physics.radiative_transfer import calculate_solar_elevation, calculate_solar_elevation_Walraven

# latitude, longitude =46.264305, -119.533354 
latitude, longitude =47, -122
year, day, hour, zone = 2023, 68, 17.5, 8
# year, day, hour, zone = 2023, 150, 20., 8
is_day_saving = False

class TestSolarAngle(unittest.TestCase):

    def test_solar_angle(self):
        print("Performing test_solar_angle()...")
        solar_elevation = calculate_solar_elevation(
            latitude=latitude, longitude=longitude,
            year=year, day=day, hour=hour, zone=zone,
            is_day_saving=is_day_saving
        )
        print("Solar elevation from calculate_solar_elevation: {}".format(solar_elevation))
        print("")
        self.assertIsInstance(solar_elevation, jnp.ndarray)

    def test_solar_angle_walraven(self):
        print("Performing test_solar_angle_walraven()...")
        solar_elevation = calculate_solar_elevation_Walraven(
            latitude=latitude, longitude=longitude,
            year=year, day=day, hour=hour, zone=zone,
            is_day_saving=is_day_saving
        )
        print("Solar elevation from calculate_solar_elevation_Walraven: {}".format(solar_elevation))
        print("")
        self.assertIsInstance(solar_elevation, jnp.ndarray)