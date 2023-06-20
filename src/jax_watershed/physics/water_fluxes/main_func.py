"""
These are some main functions for solve water balance.

Author: Peishi Jiang
Date: 2023.06.18.

"""

import jax.numpy as jnp

from ...shared_utilities.types import Float_0D, Float_1D
from .ground_water import ground_water_balance
from .canopy_water import canopy_water_balance

# from .surface_water import surface_water_vector_field

# Aboveground water balance including surface and
# canopy water balance.
def solve_surface_water(
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
    """Calculate the surface water contents at the next time step.

    Args:
        Δt (Float_0D): The time interval [s]
        W_surf (Float_1D): The canopy and ground water at the current time step [kg m-2]
        L (Float_0D): The leaf area index [-]
        S (Float_0D): The stem area index [-]
        P (Float_0D): The rain precipitation [kg m-2 s-1]
        E_vw (Float_0D): The canopy evaporation [kg m-2 s-1]
        E_g (Float_0D): The ground evaporation [kg m-2 s-1]
        z_table (Float_0D): The water table depth [m]
        k_sat (Float_0D): The saturated hydraulic conductivity [kg m-2 s-1]
        E_g (Float_0D): The ground evaporation [kg m-2 s-1]

    Returns:
        Float_1D: The canopy and ground water content at the next time step [kg m-2 s-1]
    """
    W_can, W_g = W_surf[0], W_surf[1]

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

    # We use explicit Euler method to solve the ODE
    W_can_new = W_can + ΔW_can_Δt * Δt
    W_g_new = W_g + ΔW_g_Δt * Δt

    # print(ΔW_can_Δt, ΔW_g_Δt)

    # Make sure the water contents are non-negative
    W_can_new = jnp.max(jnp.array([0, W_can_new]))
    W_g_new = jnp.max(jnp.array([0, W_g_new]))

    return jnp.array([W_can_new, W_g_new])


# Belowground water balance solved by Richards equation
def solve_subsurface_water():
    pass
