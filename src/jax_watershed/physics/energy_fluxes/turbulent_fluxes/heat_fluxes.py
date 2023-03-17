"""
Calculation of sensible heat and water vapor fluxes for a dual source canopy model, based on:
(1) CLM5 documentation;
(2) Chapter 6 in "Climate Change and Terrestrial Ecosystem Modeling" by Bonan (2019).

Author: Peishi Jiang
Date: 2023.03.17.
"""

from ....shared_utilities.constants import C_P as Cp

def calculate_H(
    T_2: float, T_1: float, ρ_atm: float, gh: float
) -> float:
    """Calculating sensible heat flux

    Args:
        T_2 (float): The temperature at the top or towards negative h direction [degK]
        T_1 (float): The temperature at the bottom or towards positive h direction [degK]
        ρ_atm (float): The density of atmospheric air [kg m-3]
        gh (float): The conductance of heat flux [m s-1]

    Returns:
        float: The sensible heat flux [W m-2]
    """
    return ρ_atm * Cp * (T_2 - T_1) * gh


def calculate_E(
    q_2: float, q_1: float, ρ_atm: float, ge: float
) -> float:
    """Calculating the water vapor flux.

    Args:
        q_2 (float): The specific humidity at the top or towards negative E direction [kg kg-1]
        q_1 (float): The specific humidity at the bottom or towards positive E direction [kg kg-1]
        ρ_atm (float): The density of atmospheric air [kg m-3]
        ge (float): The conductance of water vapor [m s-1]

    Returns:
        float: The water vapor flux [kg m-2 s-1]
    """
    return ρ_atm * (q_2 - q_1) * ge