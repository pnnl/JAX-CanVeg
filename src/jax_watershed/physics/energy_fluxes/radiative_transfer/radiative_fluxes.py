"""
Calculating radiative fluxes based on Chapter 4 of CLM5 documentation.

Author: Peishi Jiang
Date: 2023.03.13.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from ....shared_utilities.constants import STEFAN_BOLTZMANN_CONSTANT as σ
from ....shared_utilities.constants import C_TO_K


def calculate_net_radiation_at_the_surface(
    S_v: float, S_g: float, L_v: float, L_g: float,
) -> float:
    """Calcuate the net radiation at the surface.

    Args:
        S_v (float): The total solar radiation absorbed by the vegetation [W m-2]
        S_g (float): The total solar radiation absorbed by the ground [W m-2]
        L_v (float): The net longwave radiation from the canopy back to the atmosphere [W m-2]
        L_g (float): The net longwave radiation from the canopy back to the atmosphere [W m-2]

    Returns:
        float: The net radaition at the surface [W m-2]
    """
    return (S_v+S_g) - (L_v+L_g)


def calculate_solar_fluxes(
    S_db_par: float, S_dif_par: float, S_db_nir: float, S_dif_nir: float,
    I_can_db_par: float, I_can_dif_par: float, I_can_db_nir: float, I_can_dif_nir: float,
    I_down_db_par: float, I_down_dif_par: float, I_down_db_nir: float, I_down_dif_nir: float, 
    I_down_trans_can_par: float, I_down_trans_can_nir: float,
    α_g_db_par: float, α_g_dif_par: float, α_g_db_nir: float, α_g_dif_nir: float, 
) -> Tuple[float, float]:
    """Calcuate different components of solar fluxes.

    Args:
        S_db_par (float): The incident PAR direct beam solar fluxes [W m-2]
        S_dif_par (float): The incident PAR diffuse solar fluxes [W m-2]
        S_db_nir (float): The incident NIR direct beam solar fluxes [W m-2]
        S_dif_nir (float): The incident NIR diffuse solar fluxes [W m-2]
        I_can_db_par (float): The PAR direct beam fluxes absorbed by the vegetation per unit incident flux [-]
        I_can_dif_par (float): The PAR diffuse fluxes absorbed by the vegetation per unit incident flux [-]
        I_can_db_nir (float): The NIR direct beam fluxes absorbed by the vegetation per unit incident flux [-]
        I_can_dif_nir (float): The NIR diffuse fluxes absorbed by the vegetation per unit incident flux [-]
        I_down_db_par (float): The PAR downward direct beam radiation per unit incident flux [-]
        I_down_dif_par (float): The PAR downward diffuse radiation per unit incident flux [-]
        I_down_db_nir (float): The NIR downward direct beam radiation per unit incident flux [-]
        I_down_dif_nir (float): The NIR downward diffuse radiation per unit incident flux [-]
        I_down_trans_can_par (float): The PAR direct beam flux transmitted through the canopy per unit incident flux [-]
        I_down_trans_can_nir (float): The NIR direct beam flux transmitted through the canopy per unit incident flux [-]

    Returns:
        Tuple[float, float]: The tota solar radiation absorbed by the vegetation and the ground [W m-2]
    """
    # The total solar radiation absorbed by the vegetation
    S_v = S_db_par * I_can_db_par + S_dif_par * I_can_dif_par + \
          S_db_nir * I_can_db_nir + S_dif_nir * I_can_dif_nir     # Eq(4.1) in CLM5
    
    # The total solar radiation absorbed by the ground
    S_g = S_db_par * I_down_trans_can_par * (1-α_g_db_par) + \
          (S_db_par * I_down_db_par + S_dif_par * I_down_dif_par) * (1-α_g_dif_par) + \
          S_db_nir * I_down_trans_can_nir * (1-α_g_db_nir) + \
          (S_db_nir * I_down_db_nir + S_dif_nir * I_down_dif_nir) * (1-α_g_dif_nir)     # Eq(4.2) in CLM5

    return S_v, S_g
         

def calculate_longwave_fluxes(
    L_down: float, ε_v: float, ε_g: float,
    T_v_t1: float, T_v_t2: float, T_g_t1: float, T_g_t2: float,
    L: float, S: float
) -> Tuple:
    """Calculate longwave fluxes.

    Args:
        L_down (float): The downward atmospheric longwave radiation [W m-2]
        T_v_t1 (float): The vegetation temperature at the previous time step [degK]
        T_v_t2 (float): The vegetation temperature at the current time step [degK]
        T_g_t1 (float): The snow/soil surface temperature at the previous time step [degK]
        T_g_t2 (float): The snow/soil surface temperature at the current time step [degK]
        L (float): The exposed leaf area index [m2 m2-1]
        S (float): The exposed stem area index [m2 m2-1]

    Returns:
        Tuple: The different components of longwave fluxes.
    """
    δ_veg = jnp.heaviside(L+S-0.05, 1.)
    # δ_veg = 0.
    # δ_veg = 1.

    # # Convert the temperature units form degC to degK
    # T_v_t1, T_v_t2 = T_v_t1 + C_TO_K, T_v_t2 + C_TO_K
    # T_g_t1, T_g_t2 = T_g_t1 + C_TO_K, T_g_t2 + C_TO_K

    # The upward longwave radiation from the vegetation/soi system for exposed leaf and stem area
    # based on Eq(4.14) in CLM5
    L_up_v = (1-ε_g) * (1-ε_v) * (1-ε_v) * L_down + \
               ε_v * σ * T_v_t1**4 + \
               ε_v * (1-ε_g) * (1-ε_g) * σ * (T_v_t1)**4 + \
               4 * ε_v * σ * T_v_t1**3 * (T_v_t2-T_v_t1) + \
               4 * ε_v * (1-ε_g) * (1-ε_v) * σ * T_v_t1**3 * (T_v_t2-T_v_t1) + \
               ε_g * (1-ε_v) * σ * T_g_t1**4
    # print((1-ε_g) * (1-ε_v) * (1-ε_v) * L_down,
    #            ε_v * σ * T_v_t1**4,
    #            ε_v * (1-ε_g) * (1-ε_g) * σ * (T_v_t1)**4,
    #            4 * ε_v * σ * T_v_t1**3 * (T_v_t2-T_v_t1),
    #            4 * ε_v * (1-ε_g) * (1-ε_v) * σ * T_v_t1**3 * (T_v_t2-T_v_t1),
    #            ε_g * (1-ε_v) * σ * T_g_t1**4)

    # The downward longwave radiation below the vegetation
    # based on Eq(4.16) in CLM5
    L_down_v = (1-ε_v) * L_down + ε_v * σ * T_v_t1**4 + 4 * ε_v * σ * T_v_t1**3 * (T_v_t2-T_v_t1)
    # print((1-ε_v) * L_down, ε_v * σ * T_v_t1**4, 4 * ε_v * σ * T_v_t1**3 * (T_v_t2-T_v_t1))

    # The upward longwave radiation from the ground
    # based on Eq(4.15) in CLM5
    L_up_g = (1-ε_g) * L_down_v + ε_g * σ * T_g_t1**4

    # The upward longwave radiation from the surface to the atmosphere
    # based on Eq(4.11) in CLM5
    L_up = δ_veg * L_up_v + (1-δ_veg) * (1-ε_g) * L_down + \
           (1-δ_veg) * ε_g * σ * T_g_t1**4 + 4 * ε_g * σ * T_g_t1**3 * (T_g_t2-T_g_t1)
    # print(L_up)
    # print(δ_veg * L_up_v, (1-δ_veg) * (1-ε_g) * L_down, (1-δ_veg) * ε_g * σ * T_g_t1**4, 4 * ε_g * σ * T_g_t1**3 * (T_g_t2-T_g_t1))


    # The net longwave radiation flux for the ground (positive toward the atmosphere)
    # based on Eq(4.17) in CLM5
    L_g = ε_g * σ * T_g_t1**4 - δ_veg * ε_g * L_down_v - (1-δ_veg) * ε_g * L_down \
          + 4 * ε_g * σ * T_g_t1**3 * (T_g_t2-T_g_t1) 
    # print(L_g)

    # The net longwave radiation flux for vegetation (positive toward the atmosphere)
    # based on Eq(4.18) in CLM5
    L_v = (2-ε_v*(1-ε_g)) * ε_v * σ * T_v_t1**4 - \
          ε_v * ε_g * σ * T_g_t1 ** 4 - \
          ε_v * (1+(1-ε_g)*(1-ε_v)) * L_down
    
    # print(ε_g * σ * T_g_t1**4, δ_veg * ε_g * L_down_v, (1-δ_veg) * ε_g * L_down)
    # print(L_v, L_g, L_up, L_up_g, L_down_v)

    return L_v, L_g, L_up, L_up_g, L_down_v, L_up_v, δ_veg


def calculate_canopy_sunlit_shaded_par(
    S_db_par: float, S_dif_par: float,
    I_can_sun_db_par: float, I_can_sun_dif_par: float,
    I_can_sha_db_par: float, I_can_sha_dif_par: float,
    L_sun: float, L_sha: float
) -> Tuple[float, float]:
    """Calculate the canopy sunlit and sun shaded PAR.

    Args:
        S_db_par (float): The incident PAR direct beam solar fluxes [W m-2]
        S_dif_par (float): The incident PAR diffuse solar fluxes [W m-2]
        I_can_sun_db_par (float): The absorption of direct beam radiation by sunlit leaves per unit incident flux [-]
        I_can_sun_dif_par (float): The absorption of diffuse radiation by sunlit leaves per unit incident flux [-]
        I_can_sha_db_par (float): The absorption of direct beam radiation by shaded leaves per unit incident flux [-]
        I_can_sha_dif_par (float): The absorption of diffuse radiation by shaded leaves per unit incident flux [-]
        L_sun (float): The sunlit plant area index [m2 m2-1]
        L_sha (float): The sun shaded plant area index [m2 m2-1]

    Returns:
        Tuple[float, float]: The absorbed PAR over the sunlit and shaded canopy [W m-2]
    """
    φ_sun = (I_can_sun_db_par * S_db_par + I_can_sun_dif_par * S_dif_par) / L_sun  # Eq(4.5) in CLM5
    φ_sha = (I_can_sha_db_par * S_db_par + I_can_sha_dif_par * S_dif_par) / L_sha  # Eq(4.6) in CLM5

    return φ_sun, φ_sha


def check_solar_energy_conservation(
    S_db_par: float, S_dif_par: float, S_db_nir: float, S_dif_nir: float,
    S_v: float, S_g: float,
    I_up_db_par: float, I_up_dif_par: float, I_up_db_nir: float, I_up_dif_nir: float,
) -> bool:
    # Eq(4.4) in CLM5
    lhs = S_db_par + S_dif_par + S_db_nir + S_dif_nir
    rhs = S_v + S_g + S_db_par*I_up_db_par + S_dif_par*I_up_dif_par + \
          S_db_nir*I_up_db_nir + S_dif_nir*I_up_dif_nir       
    return lhs == rhs


# TODO: Check the longwave energy balance/conservation
def check_longwave_energy_conservation(
    L_down: float, L_v: float, L_g: float, L_up: float, 
    L_up_g: float, L_down_v: float, L_up_v: float, δ_veg: float
) -> bool:
    # Eq(4.9) in CLM5
    # lhs = L_up - L_down
    # rhs = L_g + L_v 
    # print(L_up, L_down, L_g, L_v)
    # lhs = L_down + L_down_v
    # rhs = L_v + L_g + L_up_v + L_up_g
    # print(lhs, rhs)
    # return lhs == rhs
    # if δ_veg == 0:
    #     result = L_up == L_g + L_down
    # elif δ_veg == 1:
    #     result = L_up == L_g + L_v + L_down
    # print(L_up, L_g + L_down)       # When δ_veg = 0 
    # print(L_up, L_g + L_v + L_down) # When δ_veg = 1
    return True
    
    # The total solar radiation absorbed by the vegetation
    S_v = S_db_par * I_can_db_par + S_dif_par * I_can_dif_par + \
          S_db_nir * I_can_db_nir + S_dif_nir * I_can_dif_nir     # Eq(4.1) in CLM5
    
    # The total solar radiation absorbed by the ground
    S_g = S_db_par * I_down_trans_can_par * (1-α_g_db_par) + \
          (S_db_par * I_down_db_par + S_dif_par * I_down_dif_par) * (1-α_g_dif_par) + \
          S_db_nir * I_down_trans_can_nir * (1-α_g_db_nir) + \
          (S_db_nir * I_down_db_nir + S_dif_nir * I_down_dif_nir) * (1-α_g_dif_nir)     # Eq(4.2) in CLM5