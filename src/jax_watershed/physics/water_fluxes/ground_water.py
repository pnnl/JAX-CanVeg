"""
Ground water balance.

Author: Peishi Jiang
Date: 2023.06.13

"""  # noqa: E501

import jax.numpy as jnp

from typing import Tuple
from ...shared_utilities.types import Float_0D


def ground_water_balance(
    # Δt: Float_0D, W_surf: Float_0D, Q_drip: Float_0D,
    Q_drip: Float_0D,
    P: Float_0D,
    f_pi: Float_0D,
    f_max: Float_0D,
    z_table: Float_0D,
    k_sat: Float_0D,
    E_g: Float_0D
    # ) -> Float_0D:
) -> Tuple[Float_0D, Float_0D, Float_0D]:
    """Calculate the components of the surface water balance.

    Args:
        Δt (Float_0D): The time interval [s]
        W_surf (Float_0D): The surface water at the current step [kg m-2 s-1]
        Q_drip (Float_0D): The canopy drip flux [kg m-2 s-1]
        P (Float_0D): The rain precipitation [kg m-2 s-1]
        f_pi (Float_0D): The fraction of intercepted rain precipitation [-]
        f_max (Float_0D): The potential maximum value of f_sat
        z_table (Float_0D): The water table depth [m]
        k_sat (Float_0D): The saturated hydraulic conductivity [kg m-2 s-1]
        E_g (Float_0D): The ground evaporation [kg m-2 s-1]

    Returns:
        Float_0D: The surface water components.
    """
    # The moisture input to the surface [kg m-2 s-1]
    Q_thru = (1 - f_pi) * P
    Q_g = Q_drip + Q_thru

    # The Dunne runoff R_D [kg m-2 s-1]
    # Eqs.(7.26)-(7.27) in CLM5
    f_over = 0.5
    f_sat = f_max * jnp.exp(-0.5 * f_over * z_table)
    R_D = f_sat * Q_g

    # The infiltration [kg m-2 s-1]
    # TODO: double check the calculation of infiltration
    Q_infilsoil_max = (1 - f_sat) * k_sat
    Q_infilsoil = (1 - f_sat) * Q_g
    Q_infilh2o = f_sat * k_sat
    Q_infil = jnp.min(jnp.array([Q_infilsoil, Q_infilsoil_max])) + Q_infilh2o

    # The Horton runoff R_H, [kg m-2 s-1]
    R_H = jnp.max(jnp.array([0.0, Q_infilsoil - Q_infilsoil_max]))

    # The runoff [kg m-2 s-1]
    # print(R_D, R_H, f_sat, f_pi)
    R = R_D + R_H

    # Change of the surface water balance [kg m-2 s-1]
    ΔW_g_Δt = Q_g - Q_infil - R - E_g

    return ΔW_g_Δt, Q_infil, R
