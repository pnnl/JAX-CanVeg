"""
Calculation of sensible heat and water vapor fluxes for a dual source canopy model, based on:
(1) CLM5 documentation;
(2) Chapter 6 in "Climate Change and Terrestrial Ecosystem Modeling" by Bonan (2019).

Author: Peishi Jiang
Date: 2023.03.17.
"""  # noqa: E501

from ....shared_utilities.types import Float_0D
from ....shared_utilities.constants import C_P as Cp


def calculate_H(
    T_2: Float_0D, T_1: Float_0D, ρ_atm: Float_0D, gh: Float_0D
) -> Float_0D:
    """Calculating sensible heat flux

    Args:
        T_2 (Float_0D): The temperature at the top or towards positive H direction [degK]
        T_1 (Float_0D): The temperature at the bottom or towards negative H direction [degK]
        ρ_atm (Float_0D): The density of atmospheric air [kg m-3]
        gh (Float_0D): The conductance of heat flux [m s-1]

    Returns:
        Float_0D: The sensible heat flux [W m-2]
    """  # noqa: E501
    return -ρ_atm * Cp * (T_2 - T_1) * gh


def calculate_E(
    q_2: Float_0D, q_1: Float_0D, ρ_atm: Float_0D, ge: Float_0D
) -> Float_0D:
    """Calculating the water vapor flux.

    Args:
        q_2 (Float_0D): The specific humidity at the top or towards positive E direction [kg kg-1]
        q_1 (Float_0D): The specific humidity at the bottom or towards negative E direction [kg kg-1]
        ρ_atm (Float_0D): The density of atmospheric air [kg m-3]
        ge (Float_0D): The conductance of water vapor [m s-1]

    Returns:
        Float_0D: The water vapor flux [kg m-2 s-1]
    """  # noqa: E501
    return -ρ_atm * (q_2 - q_1) * ge


def calculate_G(T_g: Float_0D, T_s1: Float_0D, κ: Float_0D, dz: Float_0D) -> Float_0D:
    """Calculating the ground heat flux based on the temperature difference between the
       ground and first soil layer.

    Args:
        T_g (Float_0D): The ground temperature [degK]
        T_s1 (Float_0D): The temperature of the first soil layer [degK]
        κ (Float_0D): the thermal conductivity [W m-1 K-1]
        dz (Float_0D): The soil depth of the first soil layer [m]

    Returns:
        Float_0D: The ground heat flux [W m-2]
    """
    # Based on Eq(7.8) in Bonan (2019)
    return -κ * (T_g - T_s1) / dz
