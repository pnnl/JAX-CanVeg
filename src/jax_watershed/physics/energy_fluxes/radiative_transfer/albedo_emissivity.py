"""
Calculating the albedos and emissivities.

Author: Peishi Jiang
Date: 2023.03.13.
"""
# TODO: A ice albedo scheme is needed.

from typing import Tuple

import jax.numpy as jnp

from ....shared_utilities.constants import α_soil_clm5, ε_soil_clm5, ε_snow_clm5, χl_clm5
from ....shared_utilities.constants import RADIANS as RADD

from .canopy_radiative_transfer import G

# Warning: The following snow albedos are made up.
α_snow = {"PAR": 0.8, "NIR": 0.8}


def calculate_ground_albedos(f_snow: float, rad_type: str) -> Tuple[float, float]:
    """Calculat ground albedos.

    Args:
        f_snow (float): The fraction of ground covered by snow [-]
        rad_type (str): The radiation type or band, either PAR or NIR

    Returns:
        Tuple[float, float]: The overall direct beam and diffuse ground albedos
    """
    α_soil_db, α_soil_dif = α_soil_clm5[rad_type], α_soil_clm5[rad_type]
    α_snow_db, α_snow_dif = α_snow[rad_type], α_snow[rad_type]

    # The overall direct beam ground albedo
    # based on Eq(3.58)
    α_g_db  = α_soil_db * (1-f_snow) + α_snow_db * f_snow

    # The overall diffuse ground albedo
    # based on Eq(3.59)
    α_g_dif = α_soil_dif * (1-f_snow) + α_snow_dif * f_snow

    return α_g_db, α_g_dif


def calculate_ground_vegetation_emissivity(
    solar_elev_angle: float, f_snow: float, L: float, S: float, pft_ind: int
    # f_snow: float, L: float, S: float, μ_bar: float
) -> Tuple[float, float]:
    """Calculate the ground and vegetation emissivities.

    Args:
        solar_elev_angle (float): Solar elevation angle [degree]
        f_snow (float): The fraction of ground covered by snow [-]
        L (float): The leaf area index [m2 m2-1]
        S (float): The stem area index [m2 m2-1]
        pft_ind (int): The index of plant functional type based on the pft_clm5
        μ_bar (float): The average inverse diffuse optical depth per unit leaf and stem area 

    Returns:
        Tuple[float, float]: The ground and vegetation emissivities
    """
    χl      = χl_clm5[pft_ind]
    ε_soil, ε_snow = ε_soil_clm5, ε_snow_clm5

    # Calculate the cosine of the solar zenith angle of the incident beam
    # , or the sine of the solar elevation angle
    solar_elev_rad = solar_elev_angle * RADD
    μ              = jnp.sin(solar_elev_rad)

    # Calculate G, K, and μ_bar
    # TODO: the calculation of φ1 and φ2 might be incorrect
    #       the following is only applicable to -0.4 <= χl <= 0.6
    φ1    = 0.5 - 0.633*χl - 0.33*χl**2
    φ2    = 0.877 * (1-2*φ1)
    K     = G(φ1, φ2, μ) / μ
    μ_bar = 1./φ2 * (1 - φ1/φ2*jnp.log((φ1+φ2)/φ1))  # Eq(3.4) in CLM5

    # Calculate the emissivities
    ε_g = ε_soil * (1-f_snow) + ε_snow * f_snow
    ε_v = 1 - jnp.exp(-(L+S)/μ_bar)

    return ε_g, ε_v
