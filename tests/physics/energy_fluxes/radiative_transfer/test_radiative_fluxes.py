# import unittest

# from jax import jit
# import jax.numpy as jnp
# from jax_watershed.physics.radiative_transfer import calculate_solar_fluxes, calculate_longwave_fluxes, calculate_canopy_sunlit_shaded_par  # noqa: E501

# solar_elev_angle, f_snow = 30., 0.3
# L, S = 2., 1,
# rad_type, pft_ind = 'PAR', 10

# class TestRadiativeFluxes(unittest.TestCase):

#     def test_solar_fluxes(self):
#         print("Performing test_solar_fluxes()...")
#         S_v, S_g = calculate_solar_fluxes(
#             S_db_par: float, S_dif_par: float, S_db_nir: float, S_dif_nir: float,
#             I_can_db_par: float, I_can_dif_par: float, I_can_db_nir: float, I_can_dif_nir: float,  # noqa: E501
#             I_down_db_par: float, I_down_dif_par: float, I_down_db_nir: float, I_down_dif_nir: float,  # noqa: E501
#             I_down_trans_can_par: float, I_down_trans_can_nir: float,
#             α_g_db_par: float, α_g_dif_par: float, α_g_db_nir: float, α_g_dif_nir: float,  # noqa: E501
#         )
#         print("Albedos for direct beam and diffuse fluxes: {}".format(jnp.array([α_g_db, α_g_dif])))  # noqa: E501
#         print("")
#         self.assertIsInstance(α_g_db, float)

#     def test_longwave_fluxes(self):
#         print("Performing test_longwave_fluxes()...")
#         L_v, L_g, L_up, L_up_g, L_down_v = calculate_longwave_fluxes(
#             L_down: float, ε_v: float, ε_g: float,
#             T_v_t1: float, T_v_t2: float, T_g_t1: float, T_g_t2: float,
#             L: float, S: float
#         )
#         print("Emissivities of ground and vegetation: {}".format(jnp.array([ε_g, ε_v])))  # noqa: E501
#         print("")
#         self.assertIsInstance(ε_g, float)

#     def test_canopy_sunlit_shaded_par(self):
#         pass
