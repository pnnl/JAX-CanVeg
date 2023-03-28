"""
Several commonly-used relations among atmospheric pressure, temperature, humidity are put here.

Author: Peishi Jiang
Date: 2023.03.23

"""

import jax
import jax.numpy as jnp
from ...shared_utilities.constants import C_TO_K as c2k
from ...shared_utilities.constants import R_DA as Rda


def ρ_from_e_pres_temp(pres: float, e: float, T: float) -> float:
    """Calculate the air density from water vapor pressure and atmospheric pressure

    Args:
        pres (float): Atmospheric pressure [Pa]
        e (float): Water vapor pressure [Pa]
        T (float): Atmospheric temperature [degK]

    Returns:
        float: Air density [kg m-3]
    """
    ρ = (pres - 0.378*e) / (Rda*T)
    return ρ


def qsat_from_temp_pres(T: float, pres: float) -> float:
    """Calculate the saturated specific humidity from temperature and pressure.

    Args:
        T (float): Temperature [degK]
        pres (float): Pressure [Pa]

    Returns:
        float: Saturated specific humidity [kg kg-1]
    """
    # Saturated water vapor pressure given T
    esat = esat_from_temp(T) # [Pa]

    # Saturated specific humidity
    qsat = q_from_e_pres(e=esat, pres=pres)  # [kg kg-1]

    return qsat


def esat_from_temp(T: float) -> float:
    """Calculate saturated water vapor pressure from air temperature based on Tetens equation.

    Args:
        T (float): Air temperature [degK]

    Returns:
        float: Saturated water vapor pressure [Pa]
    """
    a, b = 17.2693882, 35.86
    esat = 610.78 * jnp.exp(a * (T - c2k) / (T - b)) # [Pa]
    # jax.debug.print("esat terms: {}",  jnp.array([T, T - c2k, T - b]))
    return  esat


def e_from_q_pres(pres: float, q: float) -> float:
    """Calculate water vapor pressure from specific humidity and atmospheric pressure

    Args:
        pres (float): Atmospheric pressure [Pa]
        q (float): Specific humidity [kg kg-1]

    Returns:
        float: Water vapor pressure [Pa]
    """
    e = q * pres / (0.622 + 0.378 * q) # [kg kg-1]
    return e


def q_from_e_pres(pres: float, e: float) -> float:
    """Calculate specific humidity from vapor pressure and atmospheric pressure

    Args:
        pres (float): Atmospheric pressure [Pa]
        e (float): Water vapor pressure [Pa]

    Returns:
        float: Specific humidity [kg kg-1]
    """
    q = (0.622 * e) / (pres - 0.378 * e) # [kg kg-1]
    return q