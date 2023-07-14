"""
Soil energy balance functions and subroutines, including:
- set_soil()
- set_soil_temp()
- soil_energy_balance()
- soil_sfc_resistance()

Author: Peishi Jiang
Date: 2023.07.14.
"""

# import jax
import jax.numpy as jnp

# from typing import Tuple

from ...shared_utilities.types import Float_0D


def set_soil():
    pass


def set_soil_temp():
    pass


def soil_energy_balance():
    pass


def soil_sfc_resistance(wg: Float_0D):
    """Calculate the soil surface resistance.

    Args:
        wg (Float_0D): _description_

    Returns:
        _type_: _description_
    """
    # Camillo and Gurney model for soil resistance
    # Rsoil= 4104 (ws-wg)-805, ws=.395, wg=0
    # ws= 0.395
    # wg is at 10 cm, use a linear interpolation to the surface, top cm, mean
    # between 0 and 2 cm
    wg0 = 1.0 * wg / 10.0
    # y=4104.* (0.395-wg0)-805.;

    # model of Kondo et al 1990, JAM for a 2 cm thick soil
    y = 3.0e10 * jnp.power((0.395 - wg0), 16.6)

    return y
