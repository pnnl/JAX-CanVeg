"""
Several commonly-used relations among atmospheric pressure, temperature, humidity are put here.

Author: Peishi Jiang
Date: 2023.03.23

"""  # noqa: E501

import jax.numpy as jnp
from ...shared_utilities.types import Float_0D
from ...shared_utilities.constants import C_TO_K as c2k
from ...shared_utilities.constants import R_DA as Rda


def ρ_from_e_pres_temp(pres: Float_0D, e: Float_0D, T: Float_0D) -> Float_0D:
    """Calculate the air density from water vapor pressure and atmospheric pressure

    Args:
        pres (Float_0D): Atmospheric pressure [Pa]
        e (Float_0D): Water vapor pressure [Pa]
        T (Float_0D): Atmospheric temperature [degK]

    Returns:
        Float_0D: Air density [kg m-3]
    """
    ρ = (pres - 0.378 * e) / (Rda * T)
    return ρ


def qsat_from_temp_pres(T: Float_0D, pres: Float_0D) -> Float_0D:
    """Calculate the saturated specific humidity from temperature and pressure.

    Args:
        T (Float_0D): Temperature [degK]
        pres (Float_0D): Pressure [Pa]

    Returns:
        Float_0D: Saturated specific humidity [kg kg-1]
    """
    # Saturated water vapor pressure given T
    esat = esat_from_temp(T)  # [Pa]

    # Saturated specific humidity
    qsat = q_from_e_pres(e=esat, pres=pres)  # [kg kg-1]

    return qsat


def esat_from_temp(T: Float_0D) -> Float_0D:
    """Calculate saturated water vapor pressure from air temperature based on Tetens equation.

    Args:
        T (Float_0D): Air temperature [degK]

    Returns:
        Float_0D: Saturated water vapor pressure [Pa]
    """  # noqa: E501
    a, b = 17.2693882, 35.86
    esat = 610.78 * jnp.exp(a * (T - c2k) / (T - b))  # [Pa]
    # jax.debug.print("esat terms: {}",  jnp.array([T, T - c2k, T - b]))
    return esat


def e_from_q_pres(pres: Float_0D, q: Float_0D) -> Float_0D:
    """Calculate water vapor pressure from specific humidity and atmospheric pressure

    Args:
        pres (Float_0D): Atmospheric pressure [Pa]
        q (Float_0D): Specific humidity [kg kg-1]

    Returns:
        Float_0D: Water vapor pressure [Pa]
    """
    e = q * pres / (0.622 + 0.378 * q)  # [kg kg-1]
    return e


def q_from_e_pres(pres: Float_0D, e: Float_0D) -> Float_0D:
    """Calculate specific humidity from vapor pressure and atmospheric pressure

    Args:
        pres (Float_0D): Atmospheric pressure [Pa]
        e (Float_0D): Water vapor pressure [Pa]

    Returns:
        Float_0D: Specific humidity [kg kg-1]
    """
    q = (0.622 * e) / (pres - 0.378 * e)  # [kg kg-1]
    return q
