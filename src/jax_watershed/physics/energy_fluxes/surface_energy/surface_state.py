"""
Calculating the following surface atmosphere states in a dual-source model from 
vegetation/ground/reference/atmosphere states:
- temperature;
- specific humidity

Author: Peishi Jiang
Date: 2023.03.20.
"""

from ....shared_utilities.types import Float_0D


def calculate_Ts_from_TvTgTa(
    Tv: Float_0D,
    Tg: Float_0D,
    Ta: Float_0D,
    gam: Float_0D,
    gvm: Float_0D,
    ggm: Float_0D,
) -> Float_0D:
    """Calculating the surface/canopy air temperature.

    Args:
        Tv (Float_0D): The vegetation temperature [degK]
        Tg (Float_0D): The ground temperature [degK]
        Ta (Float_0D): The reference atmospheric temperature [degK]
        gam (Float_0D): The heat conductance from the surface to the atmosphere [m s-1]
        gvm (Float_0D): The heat conductance from the vegetation to the surface [m s-1]
        ggm (Float_0D): The heat conductance from the ground to the surface [m s-1]

    Returns:
        Float_0D: The surface/canopy air temperatuere [degK]
    """
    # Revised from Eq(15.12) in Bonan(2019) with leaf area index L imbedded in gbh
    # jax.debug.print("Conductances: {}", jnp.array([gam, gvm, ggm]))
    return (gam * Ta + 2 * gvm * Tv + ggm * Tg) / (gam + 2 * gvm + ggm)
    # return (gam*Ta + gvm*Tv + ggm*Tg) / (gam + gvm + ggm)


def calculate_qs_from_qvqgqa(
    qv_sat: Float_0D,
    qg: Float_0D,
    qa: Float_0D,
    gaw: Float_0D,
    gvw: Float_0D,
    ggw: Float_0D,
):
    """Calculating the surface/canopy air specific humidity.

    Args:
        qv_sat (Float_0D): The vegetation saturated specific humidity [kg kg-1]
        qg (Float_0D): The ground specific humidity [kg kg-1]
        qa (Float_0D): The reference atmospheric specific humidity [kg kg-1]
        gaw (Float_0D): The water vapor conductance from the surface to the atmosphere [m s-1]
        gvw (Float_0D): The water vapor conductance from the vegetation to the surface [m s-1]
        ggw (Float_0D): The water vapor conductance from the ground to the surface [m s-1]

    Returns:
        Float_0D: The surface/canopy air temperatuere [degK]
    """  # noqa: E501
    # Based on Eq(5.107) in CLM5
    return (gaw * qa + gvw * qv_sat + ggw * qg) / (gaw + gvw + ggw)
