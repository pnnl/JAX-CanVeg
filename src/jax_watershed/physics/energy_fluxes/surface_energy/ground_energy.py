"""
Solving the ground energy balance to estimate:
- sensible/latent/longwave fluxes
- ground temperature

Source:
- Chapter 5 in CLM5 documentation
- Chapter 7 in "Climate Change and Terrestrial Ecosystem Modeling" by Bonan (2019)

Author: Peishi Jiang
Date: 2023.03.20.

"""

from ....shared_utilities.constants import λ_VAP as λ

from ..turbulent_fluxes import calculate_H, calculate_E, calculate_G


def ground_energy_balance(
    T_g: float, T_s: float, T_s1: float, κ:float, dz: float, q_g: float, q_s: float, 
    gh: float, ge: float, S_g: float, L_g: float, ρ_atm: float, 
) -> float:
    """Leaf energy balance equation, used for solving the vegetation temperature, T_g.
       Note: use calculate_conductance_ground_canopy to calcuate gh and
                 calculate_conductance_ground_canopy_water_vapo to calculate ge.

    Args:
        T_g (float): The ground temperature [degK]
        T_s (float): The surface atmosphere temperature [degK]
        T_s1 (float): The temperature of the first soil layer [degK]
        κ (float): the thermal conductivity [W m-1 K-1]
        dz (float): The soil depth of the first soil layer [m]
        q_g (float): The specific humidity of the ground [g kg-1]
        q_s (float): The surface specific humidity [g kg-1]
        gh (float): The heat conductance from ground to surface [m s-1]
        ge (float): The water vapor conductance from ground to surface [m s-1]
        S_g (float): The net solar fluxes absorbed by the ground [W m-2]
        L_g (float): The net longwave flux leaving from the ground to the atmosphere [W m-2]
        ρ_atm (float): The density of atmospheric air [kg m-3]

    Returns:
        float: The difference among net solar/longwave radiation and sensible/latent heat fluxes in the canopy.
    """

    # Calculate the sensible heat flux
    # Eq(5.88) in CLM5
    H = calculate_H(T_1=T_g, T_2=T_s, ρ_atm=ρ_atm, gh=gh)
    
    # Calculate the latent heat flux
    # Eq(5.101) in CLM5
    E = calculate_E(q_1=q_g, q_2=q_s, ρ_atm=ρ_atm, ge=ge)  # [kg m-2 s-1]
    λE = λ * E # [W m-2]

    # Calculate the ground heat flux
    G = calculate_G(T_g=T_g, T_s1=T_s1, κ=κ, dz=dz) 

    # leaf energy balance
    return S_g - L_g - H - λE - G