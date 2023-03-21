"""
Solving the leaf energy balance, by using a dual source model,to estimate:
- sensible/latent/longwave fluxes
- leaf temperature

Source: 
- Chapter 5 in CLM5 documentation
- Chapter 10 in "Climate Change and Terrestrial Ecosystem Modeling" by Bonan (2019)

Author: Peishi Jiang
Date: 2023.03.20.
"""

from ....shared_utilities.constants import λ_VAP as λ

from ..radiative_transfer import calculate_longwave_fluxes
from ..turbulent_fluxes import calculate_H, calculate_E
from ..turbulent_fluxes import calculate_conductance_leaf_boundary


# TODO: Assume that the change of biomass energy in the canopy is negligible for now!
def leaf_energy_balance(
    T_v: float, T_s: float, q_v_sat: float, q_s: float, 
    gh: float, ge: float, S_v: float, L_v: float, ρ_atm: float, 
) -> float:
    """Leaf energy balance equation, used for solving the vegetation temperature, T_v

    Args:
        T_v (float): The vegetation/leaf temperature [degK]
        T_s (float): The surface atmosphere temperature [degK]
        q_v_sat (float): The saturated specific humidity of the surface atmosphere temperature [g kg-1]
        q_s (float): The surface specific humidity [g kg-1]
        gh (float): The total leaf boundary conductance [m s-1]
        ge (float): The total leaf conductance to water vapor [m s-1]
        S_v (float): The net solar fluxes absorbed by the vegetation [W m-2]
        L_v (float): The net longwave flux leaving from canopy to the atmosphere [W m-2]
        ρ_atm (float): The density of atmospheric air [kg m-3]

    Returns:
        float: The difference among net solar/longwave radiation and sensible/latent heat fluxes in the canopy.
    """

    # Calculate the sensible heat flux
    # Eq(5.88) in CLM5
    # here 2*gh is to account for the adaxial and abaxial sides of the leaf
    H = calculate_H(T_1=T_v, T_2=T_s, ρ_atm=ρ_atm, gh=2*gh)
    
    # Calculate the latent heat flux
    # Eq(5.101) in CLM5
    E = calculate_E(q_1=q_v_sat, q_2=q_s, ρ_atm=ρ_atm, ge=ge)  # [kg m-2 s-1]
    λE = λ * E # [W m-2]

    # print(T_v, T_s, q_s, q_v_sat)
    # print(S_v, L_v, H, E, λE)

    # leaf energy balance
    return S_v - L_v - H - λE


# def leaf_energy_balance(
#     # T_v: float, T_g: float, T_s: float, T_a: float, gs: float, S_v: float, L_v: float, 
#     T_v: float, T_s: float, q_v_sat: float, q_s: float, gs: float, S_v: float, L_v: float, 
#     ustar: float, L: float, S: float, ρ_atm: float, 
# ) -> float:
#     """Leaf energy balance equation, used for solving the vegetation temperature, T_v

#     Args:
#         T_v (float): The vegetation/leaf temperature [degK]
#         T_s (float): The surface atmosphere temperature [degK]
#         q_v_sat (float): The saturated specific humidity of the surface atmosphere temperature [g kg-1]
#         q_s (float): The surface specific humidity [g kg-1]
#         gs (float): The stomatal conductance [m s-1]
#         S_v (float): The net solar fluxes absorbed by the vegetation [W m-2]
#         L_v (float): The net longwave flux leaving from canopy to the atmosphere [W m-2]
#         ustar (float): The friction velocity [m s-1]
#         L (float): The exposed leaf area index [m2 m2-1]
#         S (float): The exposed stem area index [m2 m2-1]

#     Returns:
#         float: The difference among net solar/longwave radiation and sensible/latent heat fluxes in the canopy.
#     """

#     # # Calculate the net longwave radiation in the canopy
#     # calculate_longwave_fluxes(L_down=L_down, ε_v=ε_v, ε_g=ε_g, T_v_t1=T_v_t1, T_v_t2=T_v_t2, T_g_t1=T_g_t1, T_g_t2=T_g_t2, L=L, S=S)

#     # Calculate the resistance/conductance terms
#     gh = calculate_conductance_leaf_boundary(ustar) * (L+S)
#     ge = 1./gh + 1./gs # gs: stomatal conductance, based on Eq(7.6) in Bonan (2019)

#     # Calculate the sensible heat flux
#     # Eq(5.88) in CLM5
#     # here 2*gh is to account for the adaxial and abaxial sides of the leaf
#     H = calculate_H(T_1=T_v, T_2=T_s, ρ_atm=ρ_atm, gh=2*gh)
    
#     # Calculate the latent heat flux
#     # Eq(5.101) in CLM5
#     E = calculate_E(q_1=q_v_sat, q_2=q_s, ρ_atm=ρ_atm, ge=ge)  # [kg m-2 s-1]
#     λE = λ * E # [W m-2]

#     # leaf energy balance
#     return S_v - L_v - H - λE

# def calculate_conductance_conductance