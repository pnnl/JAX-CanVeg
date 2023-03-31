import unittest

from jax_watershed.shared_utilities.constants import pft_clm5
from jax_watershed.physics.energy_fluxes.radiative_transfer import (
    calculate_canopy_fluxes_per_unit_incident,
)

import jax.numpy as jnp


class TestCanopyRadiation(unittest.TestCase):
    def test_canopy_radiative_transfer(self):
        print("Performing test_canopy_radiative_transfer()...")
        pft = "C3 grass"
        α_g_db, α_g_dif = 0.3, 0.3
        pft_ind = pft_clm5.index(pft)
        fluxes = calculate_canopy_fluxes_per_unit_incident(
            solar_elev_angle=30,
            α_g_db=α_g_db,
            α_g_dif=α_g_dif,
            L=2.0,
            S=0.5,
            f_snow=0.0,
            f_cansno=0.0,
            rad_type="PAR",
            pft_ind=pft_ind,
        )
        print("Upward fluxes above the canopy:  {}".format(fluxes[:2]))
        # TODO: check the negative values of downward fluxes below the canopy
        print("Downward fluxes below the canopy:  {}".format(fluxes[2:4]))
        print("Direct beam flux transmitted through the canopy:  {}".format(fluxes[4]))
        print(
            "Absorption of direct beam radiation by sunlit and shaded leaves:  {}".format(  # noqa: E501
                fluxes[5:7]
            )
        )
        print(
            "Absorption of diffuse radiation by sunlit and shaded leaves:  {}".format(
                fluxes[7:9]
            )
        )
        print("")
        # print(2 + α_g_db * fluxes[2] + α_g_dif * fluxes[4] + α_g_dif * fluxes[3])
        # print(jnp.sum(jnp.array(fluxes)))
        self.assertAlmostEqual(
            float(fluxes.sum()),
            # jnp.sum(jnp.array(fluxes)),
            float(2 + α_g_db * fluxes[2] + α_g_dif * fluxes[4] + α_g_dif * fluxes[3]),
            places=3,
            msg="The sum of the fluxes is {}, not 2".format(jnp.sum(jnp.array(fluxes))),
        )
