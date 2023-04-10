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

from ....shared_utilities.types import Float_0D

from ....shared_utilities.constants import λ_VAP as λ

from ..turbulent_fluxes import calculate_H, calculate_E


# TODO: Assume that the change of biomass energy in the canopy is negligible for now!
def leaf_energy_balance(
    T_v: Float_0D,
    T_s: Float_0D,
    q_v_sat: Float_0D,
    q_s: Float_0D,
    gh: Float_0D,
    ge: Float_0D,
    S_v: Float_0D,
    L_v: Float_0D,
    ρ_atm: Float_0D,
) -> Float_0D:
    """Leaf energy balance equation, used for solving the vegetation temperature, T_v

    Args:
        T_v (Float_0D): The vegetation/leaf temperature [degK]
        T_s (Float_0D): The surface atmosphere temperature [degK]
        q_v_sat (Float_0D): The saturated specific humidity of the surface atmosphere
                         temperature [kg kg-1]
        q_s (Float_0D): The surface specific humidity [kg kg-1]
        gh (Float_0D): The total leaf boundary conductance [m s-1]
        ge (Float_0D): The total leaf conductance to water vapor [m s-1]
        S_v (Float_0D): The net solar fluxes absorbed by the vegetation [W m-2]
        L_v (Float_0D): The net longwave flux leaving from canopy to the atmosphere [W m-2]
        ρ_atm (Float_0D): The density of atmospheric air [kg m-3]

    Returns:
        Float_0D: The difference among net solar/longwave radiation and sensible/latent
               heat fluxes in the canopy.
    """  # noqa: E501

    # Calculate the sensible heat flux
    # Eq(5.88) in CLM5
    # here 2*gh is to account for the adaxial and abaxial sides of the leaf
    H = calculate_H(T_1=T_v, T_2=T_s, ρ_atm=ρ_atm, gh=2 * gh)

    # Calculate the latent heat flux
    # Eq(5.101) in CLM5
    E = calculate_E(q_1=q_v_sat, q_2=q_s, ρ_atm=ρ_atm, ge=ge)  # [kg m-2 s-1]
    λE = λ * E  # [W m-2]

    # print(T_v, T_s, q_s, q_v_sat)
    # print(S_v, L_v, H, E, λE)

    # leaf energy balance
    # jax.debug.print("canopy energy balance: {}", jnp.array([T_v, T_s, S_v, L_v, H, λE]))  # noqa: E501
    return S_v - L_v - H - λE