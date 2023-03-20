"""
Calculating the following surface atmosphere states in a dual-source model from vegetation/ground/reference/atmosphere states:
- temperature;
- specific humidity

Author: Peishi Jiang
Date: 2023.03.20.
"""


def calculate_Ts_from_TvTgTa(
    Tv:float, Tg:float, Ta:float, 
    gam:float, gvm:float, ggm:float
) -> float:
    """Calculating the surface/canopy air temperature.

    Args:
        Tv (float): The vegetation temperature [degK]
        Tg (float): The ground temperature [degK]
        Ta (float): The reference atmospheric temperature [degK]
        gam (float): The heat conductance from the surface to the atmosphere [m s-1]
        gvm (float): The heat conductance from the vegetation to the surface [m s-1]
        ggm (float): The heat conductance from the ground to the surface [m s-1]

    Returns:
        float: The surface/canopy air temperatuere [degK]
    """
    # Revised from Eq(15.12) in Bonan(2019) with leaf area index L imbedded in gbh
    return (gam*Ta + 2*gvm*Tv + ggm*Tg) / (gam + 2*gvm + ggm)


def calculate_qs_from_qvqgqa(
    qv_sat:float, qg: float, qa:float,
    gaw:float,  gvw:float, ggw:float
):
    """Calculating the surface/canopy air specific humidity.

    Args:
        qv_sat (float): The vegetation saturated specific humidity [g kg-1]
        qg (float): The ground specific humidity [g kg-1]
        qa (float): The reference atmospheric specific humidity [g kg-1]
        gaw (float): The water vapor conductance from the surface to the atmosphere [m s-1]
        gvw (float): The water vapor conductance from the vegetation to the surface [m s-1]
        ggw (float): The water vapor conductance from the ground to the surface [m s-1]

    Returns:
        float: The surface/canopy air temperatuere [degK]
    """
    # Based on Eq(5.107) in CLM5
    return (gaw*qa + gvw*qv_sat + ggw*qg) / (gaw + gvw + ggw)