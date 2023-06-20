"""
Surface water balance.

Author: Peishi Jiang
Date: 2023.06.16

"""  # noqa: E501


import jax.numpy as jnp

from ...shared_utilities.types import Float_0D, Float_1D
from .ground_water import ground_water_balance
from .canopy_water import canopy_water_balance


def surface_water_vector_field(
    Δt: Float_0D,
    W_surf: Float_1D,
    L: Float_0D,
    S: Float_0D,
    P: Float_0D,
    E_vw: Float_0D,
    E_g: Float_0D,
    z_table: Float_0D,
    f_max: Float_0D,
    k_sat: Float_0D,
) -> Float_1D:
    """Calculate the vector field of the surface water.

    Args:
        Δt (Float_0D): The time interval [s]
        W_surf (Float_1D): The canopy and ground water [kg m-2]
        L (Float_0D): The leaf area index [-]
        S (Float_0D): The stem area index [-]
        P (Float_0D): The rain precipitation [kg m-2 s-1]
        E_vw (Float_0D): The canopy evaporation [kg m-2 s-1]
        E_g (Float_0D): The ground evaporation [kg m-2 s-1]
        z_table (Float_0D): The water table depth [m]
        k_sat (Float_0D): The saturated hydraulic conductivity [kg m-2 s-1]
        E_g (Float_0D): The ground evaporation [kg m-2 s-1]

    Returns:
        Float_1D: The change of canopy and ground water content [kg m-2 s-1]
    """
    # W_can, W_g = W_surf[0], W_surf[1]
    W_can = W_surf[0]

    ΔW_can_Δt, Q_intr, Q_drip, f_pi = canopy_water_balance(
        Δt=Δt, W_can=W_can, P=P, L=L, S=S, E_vw=E_vw
    )
    ΔW_g_Δt, Q_infil, R = ground_water_balance(
        Q_drip=Q_drip,
        P=P,
        f_pi=f_pi,
        f_max=f_max,
        z_table=z_table,
        k_sat=k_sat,
        E_g=E_g,
    )

    return jnp.array([ΔW_can_Δt, ΔW_g_Δt])
