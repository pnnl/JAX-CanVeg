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

# TODO: Note that the current implementation of MOST assume short canopy (e.g., grassland).
#       For large canopy (e.g., tall trees), sublayer roughness parameterization is needed. (see Chapter 6 in Bonan2019)

import jax
import jax.numpy as jnp

from jaxopt import Bisection

from ....shared_utilities.constants import PI as π
from ....shared_utilities.constants import G as g
from ....shared_utilities.constants import VON_KARMAN_CONSTANT as k

# def estimate_L_bisect(
#     L_guess: float, uz: float, tz: float, qz: float, ts: float, qs: float,
#     z: float, d: float, z0m: float, z0c: float, 
# ) -> float:
#     pass

def func_most(
    L_guess: float, uz: float, Tz: float, qz: float, Ts: float, qs: float,
    z: float, d: float, z0m: float, z0c: float, 
) -> float:
    """This is the function to solve for the Obukhov length. Given the current estimate of the Obukhov length (x),
       calcuate ustar, tstar, and qstar and then the new length. The function is the change in Obukhov length.
       It is modifed from a matlab implementation: https://github.com/gbonan/bonanmodeling/blob/master/sp_07_01/most.m.

    Args:
        L_guess (float): The initial guess of the Obukhov length [m].
        uz (float): The wind velocity at the reference height [m s-1].
        Tz (float): The temperature at the reference height [degK].
        qz (float): The specific humidity at the reference height [kg kg-1]
        Ts (float): The surface temperature [degK]
        qs (float): The surface specific humidity [kg kg-1]
        z (float): The reference height [m]
        d (float): The displacement height [m]
        z0m (float): The roughness length for momentum [m]
        z0c (float): The roughness length for scalars [m]

    Returns:
        float: the change in Obukhov length [m]
    """
    # Calculate z-d at the reference height
    z_minus_d = z - d

    # Evaluate ψ for momentum at the reference height (z-d) and surface (z0m)
    ψm_z   = calculate_ψm(ζ=z_minus_d / L_guess)
    ψm_z0m = calculate_ψm(ζ=z0m / L_guess)

    # Evaluate ψ for scalars at the reference height (z-d) and surface (z0m)
    ψc_z   = calculate_ψc(ζ=z_minus_d / L_guess)
    ψc_z0c = calculate_ψc(ζ=z0c / L_guess)

    # Calculate ustar, tstar, qstar, tzv, and tvstar
    ustar = calculate_ustar(u1=0., u2=uz, z1=d+z0m, z2=z, d=d, ψm1=ψm_z0m, ψm2=ψm_z) # [m s-1]
    tstar = calculate_Tstar(T1=Ts, T2=Tz, z1=d+z0c, z2=z, d=d, ψc1=ψc_z0c, ψc2=ψc_z) # [degK]
    qstar = calculate_qstar(q1=qs, q2=qz, z1=d+z0c, z2=z, d=d, ψc1=ψc_z0c, ψc2=ψc_z) # [kg kg-1]
    # qstar = qstar * 1e-3  # [kg kg-1]

    Tzv = Tz * (1 + 0.608 * qz)
    # Tvstar = tstar * (1 + 0.608 * qz * 1e-3) + 0.608 * Tz * qstar * 1e-3  # Eq(5.17) in CLM5
    Tvstar = tstar * (1 + 0.608 * qz) + 0.608 * Tz * qstar # Eq(5.17) in CLM5

    # jax.debug.print("{}", jnp.array([ustar, tstar, qstar, Tzv, Tvstar]))

    L_est = calculate_L(ustar=ustar, T2v=Tzv, Tvstar=Tvstar)

    return L_guess - L_est


def calculate_L(ustar: float, T2v: float, Tvstar: float) -> float:
    """Calculating the Monin Obukhov length based on Eq(6.31) in Bonan (2019).

    Args:
        ustar (float): the friction velocity [m s-1]
        T2v (float): the potential virtual temperature at [degK]
        Tvstar (float): the characteristic scale for the temperature [degK]

    Returns:
        float: the Monin Obukhov length [m]
    """
    L = ustar**2 * T2v / (k*g*Tvstar)
    return L


def calculate_ustar(u1: float, u2: float, z1: float, z2: float, d: float, ψm1: float, ψm2: float) -> float:
    """Calculating the friction velocity based on Eq(6.39) in Bonan(2019)

    Args:
        u1 (float): The velocity at height z1 [m s-1]
        u2 (float): The velocity at height z2 [m s-1]
        z1 (float): The height where u1 is measured [m]
        z2 (float): The height where u2 is measured [m]
        d (float): The displacement height [m]
        ψm1 (float): The momentum value at z1 from Monin Obukhov similarity theory
        ψm2 (float): The momentum value at z2 from Monin Obukhov similarity theory

    Returns:
        float: The friction velocity [m s-1]
    """
    ustar = (u2 - u1) * k / (jnp.log((z2-d)/(z1-d)) - (ψm2-ψm1))
    return ustar


def calculate_Tstar(T1: float, T2: float, z1: float, z2: float, d: float, ψc1: float, ψc2: float) -> float:
    """Calculating the characteristic scale of temperature based on Eq(6.41) in Bonan(2019).

    Args:
        T1 (float): The temperature at height z1 [degK]
        T2 (float): The temperature at height z2 [degK]
        z1 (float): The height where u1 is measured [m]
        z2 (float): The height where u2 is measured [m]
        d (float): The displacement height [m]
        ψc1 (float): The scalar value at z1 from Monin Obukhov similarity theory
        ψc2 (float): The scalar value at z2 from Monin Obukhov similarity theory

    Returns:
        float: The friction velocity [m s-1]
    """
    Tstar = (T2 - T1) * k / (jnp.log((z2-d)/(z1-d)) - (ψc2-ψc1))
    return Tstar


def calculate_qstar(q1: float, q2: float, z1: float, z2: float, d: float, ψc1: float, ψc2: float) -> float:
    """Calculating the characteristic scale of water content based on Eq(6.42) in Bonan(2019).

    Args:
        q1 (float): The specific humidity at height z1 [kg kg-1]
        q2 (float): The specific humidity at height z2 [kg kg-1]
        z1 (float): The height where u1 is measured [m]
        z2 (float): The height where u2 is measured [m]
        d (float): The displacement height [m]
        ψc1 (float): The scalar value at z1 from Monin Obukhov similarity theory
        ψc2 (float): The scalar value at z2 from Monin Obukhov similarity theory

    Returns:
        float: The friction velocity [m s-1]
    """
    qstar = (q2 - q1) * k / (jnp.log((z2-d)/(z1-d)) - (ψc2-ψc1))
    return qstar


def calculate_ψc(ζ: float) -> float:
    """Calcuate ψ for the scalar value based on Eq(6.47) in Bonan (2019).

    Args:
        ζ (float): A dimensionless parameter accounting for the effect of buoyancy in the Monin-Obukhov similarity theory [-]

    Returns:
        float:ψc 
    """

    def ψc_cond1(ζ):
        χ = (1-16*ζ) **(0.25)
        return 2*jnp.log((1+χ**2)/χ) 
    
    def ψc_cond2(ζ):
        return -5*ζ

    cond = ζ < 0
    ψc = jax.lax.cond(
        cond, ψc_cond1, ψc_cond2, ζ
    )
    return ψc


def calculate_ψm(ζ: float) -> float:
    """Calcuate ψ for the momentum value based on Eq(6.46) in Bonan (2019).

    Args:
        ζ (float): A dimensionless parameter accounting for the effect of buoyancy in the Monin-Obukhov similarity theory [-]

    Returns:
        float:ψm 
    """

    def ψm_cond1(ζ):
        χ = (1-16*ζ) **(0.25)
        return 2*jnp.log((1+χ)/χ) + jnp.log((1+χ**2)/χ) - 2.*jnp.arctan(χ) + π/2.
    
    def ψm_cond2(ζ):
        return -5*ζ

    cond = ζ < 0
    ψm = jax.lax.cond(
        cond, ψm_cond1, ψm_cond2, ζ
    )
    return ψm
