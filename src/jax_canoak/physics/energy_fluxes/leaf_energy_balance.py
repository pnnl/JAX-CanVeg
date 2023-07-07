"""
Leaf energy balance subroutines and runctions, including:
- energy_and_carbon_fluxes()
- energy_balance_amphi()
- sfc_vpd()
- llambda()
- es()
- desdt()
- des2dt()

Author: Peishi Jiang
Date: 2023.07.07.
"""

# import jax
import jax.numpy as jnp

# from typing import Tuple

from ...shared_utilities.types import Float_0D


def energy_and_carbon_fluxes():
    pass


def energy_balance_amphi():
    pass


def llambda():
    pass


def es(tk: Float_0D) -> Float_0D:
    """Calculate saturated vapor pressure given temperature in Kelvin.

    Args:
        tk (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    tc = tk - 273.15
    return 613.0 * jnp.exp(17.502 * tc / (240.97 + tc))


def sfc_vpd(
    tlk: Float_0D,
    leleafpt: Float_0D,
    latent: Float_0D,
    vapor: Float_0D,
    rhov_air_z: Float_0D,
) -> Float_0D:
    """This function computes the relative humidity at the leaf surface
       for application int he Ball Berry Equation.
       Latent heat flux, LE, is passed through the function, mol m-2 s-1
       and it solves for the humidity at the leaf surface.

    Args:
        tlk (Float_0D): _description_
        leleafpt (Float_0D): _description_
        latent (Float_0D): _description_
        vapor (Float_0D): _description_
        rhov_air_z (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    # Saturation vapor pressure at leaf temperature
    es_leaf = es(tlk)

    # Water vapor density at leaf [kg m-3]
    rhov_sfc = (leleafpt / latent) * vapor + rhov_air_z
    # jax.debug.print("rhov_air_z: {x}", x=rhov_air_z)

    e_sfc = 1000 * rhov_sfc * tlk / 2.165  # Pa
    vpd_sfc = es_leaf - e_sfc  # Pa
    rhum_leaf = 1.0 - vpd_sfc / es_leaf  # 0 to 1.0

    return rhum_leaf


def desdt():
    pass


def des2dt():
    pass
