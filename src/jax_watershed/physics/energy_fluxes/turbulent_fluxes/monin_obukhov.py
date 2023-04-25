"""
Calculating the components of the Monin-Obukhov similarity theory (MOST), including
(1) the Monin-Obukhov length;
(2) the Monin-Obukhov psi function for momentum and scalars;
(3) ustar, tstar, and qstar.

Source:
Chapter 6 in "Climate Change and Terrestrial Ecosystem Modeling" by Bonan (2019)

Author: Peishi Jiang
Date: 2023.03.16.
"""

# TODO: Note that the current implementation of MOST assume short canopy (e.g., grassland).  # noqa: E501
#       For large canopy (e.g., tall trees), sublayer roughness parameterization is needed. (see Chapter 6 in Bonan2019)  # noqa: E501

import jax
import jax.numpy as jnp

from ....shared_utilities.types import Float_0D
from ....shared_utilities.constants import PI as π
from ....shared_utilities.constants import G as g
from ....shared_utilities.constants import VON_KARMAN_CONSTANT as k


def func_most(
    L_guess: Float_0D,
    uz: Float_0D,
    Tz: Float_0D,
    qz: Float_0D,
    Ts: Float_0D,
    qs: Float_0D,
    z: Float_0D,
    d: Float_0D,
    z0m: Float_0D,
    z0c: Float_0D,
) -> Float_0D:
    """This is the function to solve for the Obukhov length. Given the current estimate of the Obukhov length (x),
       calcuate ustar, tstar, and qstar and then the new length. The function is the change in Obukhov length.
       It is modifed from a matlab implementation: https://github.com/gbonan/bonanmodeling/blob/master/sp_07_01/most.m.

    Args:
        L_guess (Float_0D): The initial guess of the Obukhov length [m].
        uz (Float_0D): The wind velocity at the reference height [m s-1].
        Tz (Float_0D): The temperature at the reference height [degK].
        qz (Float_0D): The specific humidity at the reference height [kg kg-1]
        Ts (Float_0D): The surface temperature [degK]
        qs (Float_0D): The surface specific humidity [kg kg-1]
        z (Float_0D): The reference height [m]
        d (Float_0D): The displacement height [m]
        z0m (Float_0D): The roughness length for momentum [m]
        z0c (Float_0D): The roughness length for scalars [m]

    Returns:
        Float_0D: the change in Obukhov length [m]
    """  # noqa: E501
    # Calculate z-d at the reference height
    z_minus_d = z - d

    # Evaluate ψ for momentum at the reference height (z-d) and surface (z0m)
    ψm_z = calculate_ψm(ζ=z_minus_d / L_guess)
    ψm_z0m = calculate_ψm(ζ=z0m / L_guess)

    # Evaluate ψ for scalars at the reference height (z-d) and surface (z0m)
    ψc_z = calculate_ψc(ζ=z_minus_d / L_guess)
    ψc_z0c = calculate_ψc(ζ=z0c / L_guess)

    # Calculate ustar, tstar, qstar, tzv, and tvstar
    ustar = calculate_ustar(
        u1=0.0, u2=uz, z1=d + z0m, z2=z, d=d, ψm1=ψm_z0m, ψm2=ψm_z
    )  # [m s-1]
    Tstar = calculate_Tstar(
        T1=Ts, T2=Tz, z1=d + z0c, z2=z, d=d, ψc1=ψc_z0c, ψc2=ψc_z
    )  # [degK]
    qstar = calculate_qstar(
        q1=qs, q2=qz, z1=d + z0c, z2=z, d=d, ψc1=ψc_z0c, ψc2=ψc_z
    )  # [kg kg-1]

    Tzv = Tz * (1 + 0.608 * qz)
    Tvstar = Tstar * (1 + 0.608 * qz) + 0.608 * Tz * qstar  # Eq(5.17) in CLM5

    # jax.debug.print("{}", jnp.array([ustar, tstar, qstar, Tzv, Tvstar]))
    # jax.debug.print("{}", jnp.array([qs, qz, qstar]))

    L_est = calculate_L(ustar=ustar, T2v=Tzv, Tvstar=Tvstar)

    return L_guess - L_est


def calculate_L(ustar: Float_0D, T2v: Float_0D, Tvstar: Float_0D) -> Float_0D:
    """Calculating the Monin Obukhov length based on Eq(6.31) in Bonan (2019).

    Args:
        ustar (Float_0D): the friction velocity [m s-1]
        T2v (Float_0D): the potential virtual temperature at [degK]
        Tvstar (Float_0D): the characteristic scale for the virtual temperature [degK]

    Returns:
        Float_0D: the Monin Obukhov length [m]
    """
    L = ustar**2 * T2v / (k * g * Tvstar)
    # jax.debug.print("L components: {}", jnp.array([ustar,T2v,Tvstar]))
    return L


def calculate_ustar(
    u1: Float_0D,
    u2: Float_0D,
    z1: Float_0D,
    z2: Float_0D,
    d: Float_0D,
    ψm1: Float_0D,
    ψm2: Float_0D,
) -> Float_0D:
    """Calculating the friction velocity based on Eq(6.39) in Bonan(2019)

    Args:
        u1 (Float_0D): The velocity at height z1 [m s-1]
        u2 (Float_0D): The velocity at height z2 [m s-1]
        z1 (Float_0D): The height where u1 is measured [m]
        z2 (Float_0D): The height where u2 is measured [m]
        d (Float_0D): The displacement height [m]
        ψm1 (Float_0D): The momentum value at z1 from Monin Obukhov similarity theory
        ψm2 (Float_0D): The momentum value at z2 from Monin Obukhov similarity theory

    Returns:
        Float_0D: The friction velocity [m s-1]
    """
    ustar = (u2 - u1) * k / (jnp.log((z2 - d) / (z1 - d)) - (ψm2 - ψm1))
    # jax.debug.print("ustar: {}", jnp.array([ustar, u2-u1, jnp.log((z2-d)/(z1-d)), ψm2, ψm1]))  # noqa: E501
    return ustar


def calculate_Tstar(
    T1: Float_0D,
    T2: Float_0D,
    z1: Float_0D,
    z2: Float_0D,
    d: Float_0D,
    ψc1: Float_0D,
    ψc2: Float_0D,
) -> Float_0D:
    """Calculating the characteristic scale of temperature based on Eq(6.41) in Bonan(2019).

    Args:
        T1 (Float_0D): The temperature at height z1 [degK]
        T2 (Float_0D): The temperature at height z2 [degK]
        z1 (Float_0D): The height where u1 is measured [m]
        z2 (Float_0D): The height where u2 is measured [m]
        d (Float_0D): The displacement height [m]
        ψc1 (Float_0D): The scalar value at z1 from Monin Obukhov similarity theory
        ψc2 (Float_0D): The scalar value at z2 from Monin Obukhov similarity theory

    Returns:
        Float_0D: The friction velocity [m s-1]
    """  # noqa: E501
    Tstar = (T2 - T1) * k / (jnp.log((z2 - d) / (z1 - d)) - (ψc2 - ψc1))
    # print(T2, T1, jnp.log((z2 - d) / (z1 - d)), ψc2, ψc1)
    return Tstar


def calculate_qstar(
    q1: Float_0D,
    q2: Float_0D,
    z1: Float_0D,
    z2: Float_0D,
    d: Float_0D,
    ψc1: Float_0D,
    ψc2: Float_0D,
) -> Float_0D:
    """Calculating the characteristic scale of water content based on Eq(6.42) in Bonan(2019).

    Args:
        q1 (Float_0D): The specific humidity at height z1 [kg kg-1]
        q2 (Float_0D): The specific humidity at height z2 [kg kg-1]
        z1 (Float_0D): The height where u1 is measured [m]
        z2 (Float_0D): The height where u2 is measured [m]
        d (Float_0D): The displacement height [m]
        ψc1 (Float_0D): The scalar value at z1 from Monin Obukhov similarity theory
        ψc2 (Float_0D): The scalar value at z2 from Monin Obukhov similarity theory

    Returns:
        Float_0D: The friction velocity [m s-1]
    """  # noqa: E501
    qstar = (q2 - q1) * k / (jnp.log((z2 - d) / (z1 - d)) - (ψc2 - ψc1))
    return qstar


def calculate_ψc(ζ: Float_0D) -> Float_0D:
    """Calcuate ψ for the scalar value based on Eq(6.47) in Bonan (2019).

    Args:
        ζ (Float_0D): A dimensionless parameter accounting for the effect of buoyancy in
                   the Monin-Obukhov similarity theory [-]

    Returns:
        Float_0D:ψc
    """

    def ψc_cond1(ζ):
        χ = (1 - 16 * ζ) ** (0.25)
        # return 2 * jnp.log((1 + χ**2) / χ)
        return 2 * jnp.log((1 + χ**2) / 2)

    def ψc_cond2(ζ):
        return -5 * ζ

    cond = ζ < 0
    ψc = jax.lax.cond(cond, ψc_cond1, ψc_cond2, ζ)
    return ψc


def calculate_ψm(ζ: Float_0D) -> Float_0D:
    """Calcuate ψ for the momentum value based on Eq(6.46) in Bonan (2019).

    Args:
        ζ (Float_0D): A dimensionless parameter accounting for the effect of buoyancy in
                   the Monin-Obukhov similarity theory [-]

    Returns:
        Float_0D:ψm
    """

    def ψm_cond1(ζ):
        χ = (1 - 16 * ζ) ** (0.25)
        return (
            # 2 * jnp.log((1 + χ) / χ)
            # + jnp.log((1 + χ**2) / χ)
            2 * jnp.log((1 + χ) / 2.0)
            + jnp.log((1 + χ**2) / 2.0)
            - 2.0 * jnp.arctan(χ)
            # - 2.0 / jnp.tan(χ)
            + π / 2.0
        )

    def ψm_cond2(ζ):
        return -5 * ζ

    cond = ζ < 0
    ψm = jax.lax.cond(cond, ψm_cond1, ψm_cond2, ζ)
    return ψm
