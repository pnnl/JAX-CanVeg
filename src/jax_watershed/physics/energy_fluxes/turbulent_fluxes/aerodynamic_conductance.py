"""
Calculation of aerodynamic conductance of momentum and scaler for a dual source canopy model, based on:
(1) CLM5 documentation;
(2) Chapter 6 in "Climate Change and Terrestrial Ecosystem Modeling" by Bonan (2019).

Author: Peishi Jiang
Date: 2023.03.17.
"""  # noqa: E501

# import jax
import jax.numpy as jnp

from ....shared_utilities.types import Float_0D
from ....shared_utilities.constants import VON_KARMAN_CONSTANT as k


def calculate_momentum_conduct_surf_atmos(
    uref: Float_0D,
    zref: Float_0D,
    d: Float_0D,
    z0m: Float_0D,
    ψmref: Float_0D,
    ψms: Float_0D,
) -> Float_0D:
    """Calculating the aerodynamic conductance for momentum between the ground and
       the atmosphere based on Eq(5.55) in CLM5

    Args:
        uref (Float_0D)  : The velocity at the reference height zref [m s-1]
        zref (Float_0D)  : The reference height where uref is measured [m]
        d (Float_0D)     : The displacement height [m]
        z0m (Float_0D)   : The surface roughness length for momentum [m]
        ψmref (Float_0D) : The momentum value at the reference height from Monin Obukhov
                        similarity theory
        ψms (Float_0D)   : The momentum value at the roughness length for momentum from
                        Monin Obukhov similarity theory

    Returns:
        Float_0D: the aerodynamic conductance for momentum [m s-1]
    """

    gam = uref * k**2 / ((jnp.log((zref - d) / z0m) - (ψmref - ψms)) ** 2)

    return gam


def calculate_scalar_conduct_surf_atmos(
    uref: Float_0D,
    zref: Float_0D,
    d: Float_0D,
    z0m: Float_0D,
    z0c: Float_0D,
    ψmref: Float_0D,
    ψms: Float_0D,
    ψcref: Float_0D,
    ψcs: Float_0D,
) -> Float_0D:
    """Calculating the aerodynamic conductance for scalar between the ground and the
       atmosphere based on Eq(5.56) in CLM5

    Args:
        uref (Float_0D)  : The velocity at the reference height zref [m s-1]
        zref (Float_0D)  : The reference height where uref is measured [m]
        d (Float_0D)     : The displacement height [m]
        z0m (Float_0D)   : The surface roughness length for momentum [m]
        z0c (Float_0D)   : The surface roughness length for scalar [m]
        ψmref (Float_0D) : The momentum value at the reference height from Monin Obukhov
                        similarity theory
        ψms (Float_0D)   : The momentum value at the roughness length for momentum from
                        Monin Obukhov similarity theory
        ψcref (Float_0D) : The scalar value at the reference height from Monin Obukhov
                        similarity theory
        ψcs (Float_0D)   : The scalar value at the roughness length for momentum from
                        Monin Obukhov similarity theory

    Returns:
        Float_0D: the aerodynamic conductance for scalar [m s-1]
    """

    gac = (
        uref
        * k**2
        / (
            (jnp.log((zref - d) / z0m) - (ψmref - ψms))
            * (jnp.log((zref - d) / z0c) - (ψcref - ψcs))
        )
    )
    # gac = (u2 - u1) * k ** 2 / ( (jnp.log((z2-d)/(z1-d)) - (ψm2-ψm1)) * (jnp.log((z2-d)/(z1-d)) - (ψc2-ψc1)) )  # noqa: E501

    return gac


def calculate_conductance_ground_canopy(
    L: Float_0D, S: Float_0D, ustar: Float_0D, z0m: Float_0D
) -> Float_0D:
    """Calculating the aerodynamic conductance to heat/moisture transfer between the ground and the canopy surface/air.

    Args:
        L (Float_0D): The exposed leaf area index [m2 m2-1]
        S (Float_0D): The exposed stem area index [m2 m2-1]
        ustar (Float_0D): The friction velocity [m s-1]
        z0m (Float_0D)   : The surface roughness length for momentum [m]

    Returns:
        Float_0D: The aerodynamic conductance between the ground and the canopy air [m s-1].
    """  # noqa: E501

    # The dense canopy and bare soil turbulent transfer coefficients
    # Eqs(5.120) and (5.121)
    cs_bare = k / 0.13 * (z0m * ustar / 1.5e-5) ** (-0.45)
    cs_dense = 0.004

    # The turbulent transfer coefficient between the underlying soil and the canopy air
    # Eq(5.118) in CLM5
    W = jnp.exp(-(L + S))
    cs = cs_bare * W + cs_dense * (1 - W)

    ga = cs * ustar

    return ga


def calculate_conductance_ground_canopy_water_vapo(
    L: Float_0D, S: Float_0D, ustar: Float_0D, z0m: Float_0D, gsoil: Float_0D
) -> Float_0D:
    """Calculating the total conductance of water vapor transfer between the ground and
       the canopy surface/air.

    Args:
        L (Float_0D): The exposed leaf area index [m2 m2-1]
        S (Float_0D): The exposed stem area index [m2 m2-1]
        ustar (Float_0D): The friction velocity [m s-1]
        z0m (Float_0D) : The surface roughness length for momentum [m]
        gsoil (Float_0D): The soil water vapor conductance from the top soil layer to [m s-1]

    Returns:
        Float_0D: The total conductance of water vapor transfer between the ground and the
               canopy air [m s-1].
    """  # noqa: E501

    gh = calculate_conductance_ground_canopy(L=L, S=S, ustar=ustar, z0m=z0m)

    # Based on Eq(5.110) in CLM5
    return 1.0 / (1.0 / gh + 1.0 / gsoil)


def calculate_conductance_leaf_boundary(ustar: Float_0D) -> Float_0D:
    """Calculating the leaf boundary layer conduntance based on Eq(5.122) in CLM5.

    Args:
        ustar (Float_0D): The friction velocity [m s-1]

    Returns:
        Float_0D: The leaf boundary layer conductance [m s-1]
    """
    cv = 0.01  # [m s-1]
    dleaf = 0.04  # [m], based on Table 5.1 in CLM5
    glb = cv * (ustar / dleaf) ** 0.5
    # jax.debug.print("ustar: {}", jnp.array([ustar, cv, dleaf, glb]))
    return glb


def calculate_total_conductance_leaf_boundary(
    ustar: Float_0D, L: Float_0D, S: Float_0D
) -> Float_0D:
    """Calculating the total leaf boundary layer conductance

    Args:
        ustar (Float_0D): The friction velocity [m s-1]
        L (Float_0D): The exposed leaf area index [m2 m2-1]
        S (Float_0D): The exposed stem area index [m2 m2-1]

    Returns:
        Float_0D: The total leaf boundary layer conductance [m s-1]
    """
    # Eq(5.96) in CLM5
    return calculate_conductance_leaf_boundary(ustar) * (L + S)


def calculate_total_conductance_leaf_water_vapor(
    gh: Float_0D, gs: Float_0D
) -> Float_0D:
    """Calculating the total water vapor conductance of leaf [m s-1]

    Args:
        gh (Float_0D): The total leaf boundary layer conductance [m s-1]
        gs (Float_0D): The stomatal conductance [m s-1]

    Returns:
        Float_0D: The total water vapor conductance of leaf [m s-1]
    """
    # Eq(10.8) in Bonan (2019)
    return 1.0 / (1.0 / gh + 1.0 / gs)
