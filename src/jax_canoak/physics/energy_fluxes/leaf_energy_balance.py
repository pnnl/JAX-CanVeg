"""
Leaf energy balance subroutines and runctions, including:
- energy_balance_amphi()
- sfc_vpd()
- llambda()
- es()
- desdt()
- des2dt()

Author: Peishi Jiang
Date: 2023.07.07.
"""

import jax
import jax.numpy as jnp

from typing import Tuple

from ...shared_utilities.types import Float_0D
from ...shared_utilities.constants import rgc1000, epsigma, cp
from ...shared_utilities.constants import epsigma2, epsigma4, epsigma8, epsigma12


def energy_balance_amphi(
    qrad: Float_0D,
    tkta: Float_0D,
    rhovva: Float_0D,
    rvsfc: Float_0D,
    stomsfc: Float_0D,
    air_density: Float_0D,
    latent: Float_0D,
    press_Pa: Float_0D,
    heat: Float_0D,
) -> Tuple[Float_0D, Float_0D, Float_0D, Float_0D]:
    """ENERGY BALANCE COMPUTATION for Amphistomatous leaves
       A revised version of the quadratic solution to the leaf energy balance
       relationship is used: Paw U, KT. 1987. J. Thermal Biology. 3: 227-233.
       H is sensible heat flux density on the basis of both sides of a leaf
       J m-2 s-1 (W m-2).  Note KC includes a factor of 2 here for heat flux
       because it occurs from both sides of a leaf.

    Args:
        qrad (Float_0D): _description_
        tkta (Float_0D): _description_
        rhovva (Float_0D): _description_
        rvsfc (Float_0D): _description_
        stomsfc (Float_0D): _description_
        air_density (Float_0D): _description_
        latent (Float_0D): _description_
        press_Pa (Float_0D): _description_
        heat (Float_0D): _description_

    Returns:
        Tuple[Float_0D, Float_0D, Float_0D, Float_0D]: _description_
    """
    # jax.debug.print(
    #     "Energy balance inputs (jax): {a} {b} {c} {d} {e} {f} {g} {h} {i}",
    #     a=qrad, b=tkta, c=rhovva, d=rvsfc, e=stomsfc,
    #     f=air_density, g=latent, h=press_Pa, i=heat
    # )
    # tkta=taa
    latent18 = latent * 18.0
    est = es(tkta)  # Pa
    # vapor pressure above leaf, Pa rhov is kg m-3
    ea = 1000 * rhovva * tkta / 2.1650

    # Vapor pressure deficit, Pa
    vpd_leaf = est - ea
    vpd_leaf = jnp.clip(vpd_leaf, a_min=0.0)

    # Slope of the vapor pressure-temperature curve, Pa/C
    # evaluate as function of Tk
    dest = desdt(tkta, latent18)

    # Second derivative of the vapor pressure-temperature curve, Pa/C
    # Evaluate as function of Tk
    d2est = des2dt(tkta, latent18)

    # Compute products of air temperature, K
    tk2 = tkta * tkta
    tk3 = tk2 * tkta
    tk4 = tk3 * tkta

    # Longwave emission at air temperature, W m-2
    llout = epsigma * tk4

    # Coefficient for latent heat flux
    # Oaks evaporate from only one side. They are hypostomatous.
    # Cuticle resistance is included in STOM.
    # stomsfc is already for top and bottom from Photosynthesis_amphi
    #  ke = 1./ (rvsfc + stomsfc) # hypostomatous
    ke = 2.0 / (
        rvsfc + 2.0 * stomsfc
    )  # amphistomatous..to add the Rb, need to assess rstop = rsbottom and add
    lecoef = air_density * 0.622 * latent * ke / press_Pa

    # Coefficients for sensible heat flux
    hcoef = air_density * cp / heat
    hcoef2 = 2 * hcoef

    # The quadratic coefficients for the a LE^2 + b LE +c =0
    repeat = hcoef + epsigma4 * tk3
    acoeff = lecoef * d2est / (2.0 * repeat)
    acoef = acoeff / 4.0
    bcoef = -(repeat) - lecoef * dest / 2.0 + acoeff * (-qrad / 2.0 + llout)
    ccoef = (
        repeat * lecoef * vpd_leaf
        + lecoef * dest * (qrad / 2.0 - llout)
        + acoeff * ((qrad * qrad) / 4.0 + llout * llout - qrad * llout)
    )
    product = bcoef * bcoef - 4.0 * acoef * ccoef
    lept = (-bcoef - jnp.sqrt(product)) / (2.0 * acoef)

    # solve for Ts using quadratic solution
    # coefficients to the quadratic solution
    atlf = epsigma12 * tk2 + d2est * lecoef / 2.0
    btlf = epsigma8 * tk3 + hcoef2 + lecoef * dest
    ctlf = -qrad + 2 * llout + lecoef * vpd_leaf
    product = btlf * btlf - 4 * atlf * ctlf
    tsfckpt = jax.lax.cond(
        product >= 0,
        lambda: tkta + (-btlf + jnp.sqrt(product)) / (2 * atlf),
        lambda: tkta,
    )
    tsfckpt = jax.lax.cond(
        (tsfckpt < -230.0) | (tsfckpt > 325.0), lambda: tkta, lambda: tsfckpt
    )

    # long wave emission of energy
    lout_leafpt = epsigma2 * jnp.power(tsfckpt, 4)

    # H is sensible heat flux
    H_leafpt = hcoef2 * (tsfckpt - tkta)

    # jax.debug.print(
    #     "Energy balance (jax): {a} {b} {c} {d} {e}",
    #     a=qrad, b=tsfckpt, c=lept, d=H_leafpt, e=lout_leafpt
    # )
    return tsfckpt, lept, H_leafpt, lout_leafpt


def llambda(tak: Float_0D) -> Float_0D:
    """Latent heat vaporization, J kg-1.

    Args:
        tak (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    y = 3149000.0 - 2370.0 * tak
    # add heat of fusion for melting ice
    y = jax.lax.cond(tak < 273.0, lambda x: x + 333.0, lambda x: x, y)
    return y


def es(tk: Float_0D) -> Float_0D:
    """Calculate saturated vapor pressure given temperature in Kelvin.

    Args:
        tk (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    tc = tk - 273.15
    return 613.0 * jnp.exp(17.502 * tc / (240.97 + tc))


def sfc_vpd(
    tlk: Float_0D,
    leleafpt: Float_0D,
    latent: Float_0D,
    vapor: Float_0D,
    rhov_air_z: Float_0D,
) -> Float_0D:
    """This function computes the relative humidity at the leaf surface
       for application int he Ball Berry Equation.
       Latent heat flux, LE, is passed through the function, mol m-2 s-1
       and it solves for the humidity at the leaf surface.

    Args:
        tlk (Float_0D): _description_
        leleafpt (Float_0D): _description_
        latent (Float_0D): _description_
        vapor (Float_0D): _description_
        rhov_air_z (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    # Saturation vapor pressure at leaf temperature
    es_leaf = es(tlk)

    # Water vapor density at leaf [kg m-3]
    rhov_sfc = (leleafpt / latent) * vapor + rhov_air_z
    # jax.debug.print("rhov_air_z: {x}", x=rhov_air_z)

    e_sfc = 1000 * rhov_sfc * tlk / 2.165  # Pa
    vpd_sfc = es_leaf - e_sfc  # Pa
    rhum_leaf = 1.0 - vpd_sfc / es_leaf  # 0 to 1.0

    return rhum_leaf


def desdt(t: Float_0D, latent18: Float_0D) -> Float_0D:
    """Calculate the first derivative of es with respect to t.

    Args:
        t (Float_0D): _description_
        latent18 (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    return es(t) * latent18 / (rgc1000 * t * t)


def des2dt(t: Float_0D, latent18: Float_0D) -> Float_0D:
    """Calculate the second derivative of the saturation vapor pressure
       temperature curve.

    Args:
        t (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """

    return -2.0 * es(t) * llambda(t) * 18.0 / (rgc1000 * t * t * t) + desdt(
        t, latent18
    ) * llambda(t) * 18.0 / (rgc1000 * t * t)
