"""
Calculating the components of the Monin-Obukhov theory, including
(1) the Monin-Obukhov length;
(2) the Monin-Obukhov psi function for momentum and scalars;
(3) ustar, tstar, and qstar.

Source:
Chapter 6 in "Climate Change and Terrestrial Ecosystem Modeling" by Bonan (2019)

Author: Peishi Jiang
Date: 2023.03.16.
"""

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
    L_guess: float, uz: float, tz: float, qz: float, ts: float, qs: float,
    z: float, d: float, z0m: float, z0c: float, 
) -> float:
    """This is the function to solve for the Obukhov length. For the current estimate of the Obukhov length (x),
       calcuate ustar, tstar, and qstar and then the new length. The function is the change in Obukhov length.
       It is modifed from a matlab implementation: https://github.com/gbonan/bonanmodeling/blob/master/sp_07_01/most.m.

    Args:
        L_guess (float): The initial guess of the Obukhov length [m].
        uz (float): The wind velocity at the reference height [m s-1].
        tz (float): The temperature at the reference height [degK].
        qz (float): The specific humidity at the reference height [g kg-1]
        ts (float): The surface temperature [degK]
        qs (float): The surface specific humidity [g kg-1]
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
    ustar = calculate_ustar(u1=0., u2=uz, z1=d+z0m, z2=z, d=d, ψm1=ψm_z0m, ψm2=ψm_z)
    tstar = calculate_tstar(t1=ts, t2=tz, z1=d+z0c, z2=z, d=d, ψc1=ψc_z0c, ψc2=ψc_z)
    qstar = calculate_qstar(q1=qs, q2=uz, z1=d+z0c, z2=z, d=d, ψc1=ψc_z0c, ψc2=ψc_z)

    tzv = tz * (1 + 0.608 * qz)
    tvstar = tstar * (1 + 0.608 * qz) + 0.608 * tz * qstar  # Eq(5.17) in CLM5

    L_est = calculate_L(ustar=ustar, t2v=tzv, tvstar=tvstar)

    return L_guess - L_est


def calculate_L(ustar: float, t2v: float, tvstar: float) -> float:
    """Calculating the Monin Obukhov length based on Eq(6.31) in Bonan (2019).

    Args:
        ustar (float): the friction velocity [m s-1]
        t2v (float): the potential virtual temperature at [degK]
        tvstar (float): the characteristic scale for the temperature [degK]

    Returns:
        float: the Monin Obukhov length [m]
    """
    L = ustar**2 * t2v / (k*g*tvstar)
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


def calculate_tstar(t1: float, t2: float, z1: float, z2: float, d: float, ψc1: float, ψc2: float) -> float:
    """Calculating the characteristic scale of temperature based on Eq(6.41) in Bonan(2019).

    Args:
        t1 (float): The temperature at height z1 [degK]
        t2 (float): The temperature at height z2 [degK]
        z1 (float): The height where u1 is measured [m]
        z2 (float): The height where u2 is measured [m]
        d (float): The displacement height [m]
        ψc1 (float): The scalar value at z1 from Monin Obukhov similarity theory
        ψc2 (float): The scalar value at z2 from Monin Obukhov similarity theory

    Returns:
        float: The friction velocity [m s-1]
    """
    tstar = (t2 - t1) * k / (jnp.log((z2-d)/(z1-d)) - (ψc2-ψc1))
    return tstar


def calculate_qstar(q1: float, q2: float, z1: float, z2: float, d: float, ψc1: float, ψc2: float) -> float:
    """Calculating the characteristic scale of water content based on Eq(6.42) in Bonan(2019).

    Args:
        q1 (float): The specific humidity at height z1 [g kg-1]
        q2 (float): The specific humidity at height z2 [g kg-1]
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
