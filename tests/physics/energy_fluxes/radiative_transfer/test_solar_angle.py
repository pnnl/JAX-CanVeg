import unittest

from jax import jit
import jax.numpy as jnp
from jax_watershed.physics.energy_fluxes.radiative_transfer import calculate_solar_elevation
from jax_watershed.physics.energy_fluxes.radiative_transfer import calculate_solar_elevation_Walraven
from jax_watershed.physics.energy_fluxes.radiative_transfer import calculate_solar_elevation_Walraven_CANOAK

# latitude, longitude =46.264305, -119.533354 
# year, day, hour, zone = 2023, 68, 15.5, 8
# latitude, longitude   = 38.538, -121.758 
# year, day, hour, zone = 1977, 120, 15, 8
# is_day_saving = False

latitude, longitude   = 31.31, 120.77
year, day, hour, zone = 2023, 68, 10., -8
# year, day, hour, zone = 1977, 120, 15, 8
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