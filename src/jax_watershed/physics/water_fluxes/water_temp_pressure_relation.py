"""
Several commonly-used relations among atmospheric pressure, temperature, humidity are put here.

Author: Peishi Jiang
Date: 2023.03.23

"""  # noqa: E501

# import jax
import jax.numpy as jnp
from ...shared_utilities.types import Float_0D
from ...shared_utilities.constants import C_TO_K as c2k
from ...shared_utilities.constants import R_DA as Rda
from ...shared_utilities.constants import G as g
from ...shared_utilities.constants import R_WV as Rwv


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


def calculate_ground_specific_humidity(
    Tg: Float_0D,
    pres: Float_0D,
    θg: Float_0D = 0.01,
) -> Float_0D:
    # Eqs(5.73)-(5.75) in CLM5
    q_g_sat = qsat_from_temp_pres(T=Tg, pres=pres)

    # TODO: revise the calculation of s1 based on Eq(5.75) in CLM5
    # TODO: revise the calculation of B1 and ψsat1 based on Section 7.3 in CLM5
    # s1, ψsat1, B1 = 0.5, -10., 3.
    # ψ1 = ψsat1 * s1 ** (-B1)
    ψ1 = -1.0e8
    α = jnp.exp(ψ1 * g / (1e3 * Rwv * Tg))
    # jax.debug.print('alpha coefficient: {}', jnp.array([α]))
    q_g = α * q_g_sat

    return q_g
