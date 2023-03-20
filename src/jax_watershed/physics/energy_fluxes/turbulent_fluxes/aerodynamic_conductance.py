"""
Calculation of aerodynamic conductance of momentum and scaler for a dual source canopy model, based on:
(1) CLM5 documentation;
(2) Chapter 6 in "Climate Change and Terrestrial Ecosystem Modeling" by Bonan (2019).

Author: Peishi Jiang
Date: 2023.03.17.
"""

import jax.numpy as jnp

from ....shared_utilities.constants import VON_KARMAN_CONSTANT as k


def calculate_momentum_conduct_surf_atmos(
    uref: float, zref: float, d: float, z0m: float, 
    ψmref: float, ψms: float
) -> float:
    """Calculating the aerodynamic conductance for momentum between the ground and the atmosphere based on Eq(5.55) in CLM5

    Args:
        uref (float)  : The velocity at the reference height zref [m s-1]
        zref (float)  : The reference height where uref is measured [m]
        d (float)     : The displacement height [m]
        z0m (float)   : The surface roughness length for momentum [m]
        ψmref (float) : The momentum value at the reference height from Monin Obukhov similarity theory
        ψms (float)   : The momentum value at the roughness length for momentum from Monin Obukhov similarity theory

    Returns:
        float: the aerodynamic conductance for momentum [m s-1]
    """

    gam = uref * k ** 2 / ( (jnp.log((zref-d)/z0m) - (ψmref-ψms)) ** 2 )

    return gam


def calculate_scalar_conduct_surf_atmos(
    uref: float, zref: float, d: float, z0m: float, z0c: float,
    ψmref: float, ψms: float, ψcref: float, ψcs: float
) -> float:
    """Calculating the aerodynamic conductance for scalar between the ground and the atmosphere based on Eq(5.55) in CLM5

    Args:
        uref (float)  : The velocity at the reference height zref [m s-1]
        zref (float)  : The reference height where uref is measured [m]
        d (float)     : The displacement height [m]
        z0m (float)   : The surface roughness length for momentum [m]
        z0c (float)   : The surface roughness length for scalar [m]
        ψmref (float) : The momentum value at the reference height from Monin Obukhov similarity theory
        ψms (float)   : The momentum value at the roughness length for momentum from Monin Obukhov similarity theory
        ψcref (float) : The scalar value at the reference height from Monin Obukhov similarity theory
        ψcs (float)   : The scalar value at the roughness length for momentum from Monin Obukhov similarity theory

    Returns:
        float: the aerodynamic conductance for scalar [m s-1]
    """

    gac = uref * k ** 2 / ( (jnp.log((zref-d)/z0m) - (ψmref-ψms)) * (jnp.log((zref-d)/z0c) - (ψcref-ψcs)) )
    # gac = (u2 - u1) * k ** 2 / ( (jnp.log((z2-d)/(z1-d)) - (ψm2-ψm1)) * (jnp.log((z2-d)/(z1-d)) - (ψc2-ψc1)) )

    return gac


def calculate_conductance_ground_canopy(
    L: float, S: float, ustar: float, z0m: float
) -> float:
    """Calculating the aerodynamic conductance to heat/moisture transfer between the ground and the canopy surface/air.

    Args:
        L (float): The exposed leaf area index [m2 m2-1]
        S (float): The exposed stem area index [m2 m2-1]
        ustar (float): The friction velocity [m s-1]
        z0m (float)   : The surface roughness length for momentum [m]

    Returns:
        float: The aerodynamic conductance between the ground and the canopy air [m s-1].
    """

    # The dense canopy and bare soil turbulent transfer coefficients
    # Eqs(5.120) and (5.121)
    cs_bare  = k / 0.13 * (z0m * ustar / 1.5e-5) ** (-0.45)
    cs_dense = 0.004

    # The turbulent transfer coefficient between the underlying soil and the canopy air
    # Eq(5.118) in CLM5
    W = jnp.exp(-(L+S))
    cs = cs_bare * W + cs_dense * (1-W)

    ga = cs * ustar

    return ga


def calculate_conductance_ground_canopy_water_vapo(
    L: float, S: float, ustar: float, z0m: float, gsoil: float 
) -> float:
    """Calculating the total conductance of water vapor transfer between the ground and the canopy surface/air.

    Args:
        L (float): The exposed leaf area index [m2 m2-1]
        S (float): The exposed stem area index [m2 m2-1]
        ustar (float): The friction velocity [m s-1]
        z0m (float) : The surface roughness length for momentum [m]
        gsoil (float): The soil water vapor conductance from the top soil layer to [m s-1]

    Returns:
        float: The total conductance of water vapor transfer between the ground and the canopy air [m s-1].
    """

    gh = calculate_conductance_ground_canopy(L=L, S=S, ustar=ustar, z0m=z0m)

    # Based on Eq(5.110) in CLM5
    return 1. / (1./gh + 1./gsoil)


def calculate_conductance_leaf_boundary(ustar: float) -> float:
    """Calculating the leaf boundary layer conduntance based on Eq(5.122) in CLM5.

    Args:
        ustar (float): The friction velocity [m s-1]

    Returns:
        float: The leaf boundary layer conductance [m s-1]
    """
    cv = 0.01 # [m s-1]
    dleaf = 0.04 # [m], based on Table 5.1 in CLM5
    glb = cv * (ustar / dleaf) ** 0.5

    return glb


def calculate_total_conductance_leaf_boundary(ustar: float, L: float, S: float) -> float:
    """Calculating the total leaf boundary layer conductance

    Args:
        ustar (float): The friction velocity [m s-1]
        L (float): The exposed leaf area index [m2 m2-1]
        S (float): The exposed stem area index [m2 m2-1]

    Returns:
        float: The total leaf boundary layer conductance [m s-1]
    """
    # Eq(5.96) in CLM5
    return calculate_conductance_leaf_boundary(ustar) * (L+S)


def calculate_total_conductance_leaf_water_vapor(gh: float, gs: float) -> float:
    """Calculating the total water vapor conductance of leaf [m s-1]

    Args:
        gh (float): The total leaf boundary layer conductance [m s-1]
        gs (float): The stomatal conductance [m s-1]

    Returns:
        float: The total water vapor conductance of leaf [m s-1]
    """
    # Eq(10.8) in Bonan (2019)
    return 1. / (1./gh + 1./gs)