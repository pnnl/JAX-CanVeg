import unittest

from jax import jit
import jax.numpy as jnp
from jax_watershed.physics.energy_fluxes.radiative_transfer import main_func
from jax_watershed.shared_utilities.constants import C_TO_K as c2k


class TestMainFunc(unittest.TestCase):

    def test_main_func(self):
        solar_rad, L_down, pres        = 352., 200., 101.976
        L, S, pft_ind                  = 2., 1., 10
        latitude, longitude            = 31.31, 120.77
        year, day, hour, zone          = 2023, 68, 10., -8
        # T_v_t1, T_v_t2, T_g_t1, T_g_t2 = 15., 16., 10., 11.
        T_v_t1, T_v_t2, T_g_t1, T_g_t2 = 15.+c2k, 16.+c2k, 10.+c2k, 11.+c2k

        f_snow, f_cansno = 0.1, 0.1

        print("Performing test_partition_solar_radiation()...")
        Rnet, S_v, S_g, L_v, L_g, solar_rad_balanced, longwave_balanced = main_func(
        # Rnet, S_v, S_g, L_v, L_g, solar_rad_balanced = main_func(
            solar_rad=solar_rad, L_down=L_down, pres=pres,
            f_snow=f_snow, f_cansno=f_cansno, L=L, S=S, pft_ind=pft_ind,
            T_v_t1=T_v_t1, T_v_t2=T_v_t2, T_g_t1=T_g_t1, T_g_t2=T_g_t2,
            latitude=latitude, longitude=longitude, 
            year=year, day=day, hour=hour,
            zone=zone, is_day_saving=False,
        )
        print("The net radiation components: {}".format(jnp.array([
            Rnet, S_v, S_g, L_v, L_g
        ])))
        print("")
        self.assertTrue(solar_rad_balanced)
        self.assertTrue(longwave_balanced)
        self.assertTrue(Rnet==S_v+S_g-L_v-L_g)