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

import jax
import jax.numpy as jnp

# from typing import Tuple

from ...shared_utilities.types import Float_0D
from ...shared_utilities.constants import rgc1000


def energy_and_carbon_fluxes():
    pass


def energy_balance_amphi():
    pass


def llambda(tak: Float_0D) -> Float_0D:
    """Latent heat vaporization, J kg-1.

    Args:
        tak (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    y = 3149000.0 - 2370.0 * tak
    # add heat of fusion for melting ice
    y = jax.lax.cond(tak < 273.0, lambda x: x + 333.0, lambda x: x, y)
    return y


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


def desdt(t: Float_0D, latent18: Float_0D) -> Float_0D:
    """Calculate the first derivative of es with respect to t.

    Args:
        t (Float_0D): _description_
        latent18 (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    return es(t) * latent18 / (rgc1000 * t * t)


def des2dt(t: Float_0D, latent18: Float_0D) -> Float_0D:
    """Calculate the second derivative of the saturation vapor pressure
       temperature curve.

    Args:
        t (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """

    return -2.0 * es(t) * llambda(t) * 18.0 / (rgc1000 * t * t * t) + desdt(
        t, latent18
    ) * llambda(t) * 18.0 / (rgc1000 * t * t)
