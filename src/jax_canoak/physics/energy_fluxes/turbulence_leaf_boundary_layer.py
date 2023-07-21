"""
Turbulence and leaf boundary layer functions, including:
- uz()
- boundary_resistance()
- friction_velocity()

Author: Peishi Jiang
Date: 2023.07.11.
"""

import jax
import jax.numpy as jnp

from typing import Tuple

from ...shared_utilities.types import Float_0D
from ...shared_utilities.constants import nnu, lleaf, betfact
from ...shared_utilities.constants import ddh, ddv, ddc


def uz(zzz: Float_0D, ht: Float_0D, wnd: Float_0D) -> Float_0D:
    """U(Z) inside the canopy during the day is about 1.09 u*
       This simple parameterization is derived from turbulence
       data measured in the WBW forest by Baldocchi and Meyers, 1988.

    Args:
        zzz (Float_0D): _description_
        ht (Float_0D): _description_
        wnd (Float_0D): _description_
    """
    zh = zzz / ht

    # use Cionco exponential function
    uh = wnd * jnp.log((0.55 - 0.33) / 0.055) / jnp.log((2.8 - 0.333) / 0.055)
    y = uh * jnp.exp(-2.5 * (1 - zh))

    return y


def boundary_resistance(
    zzz: Float_0D,
    ht: Float_0D,
    TLF: Float_0D,
    grasshof: Float_0D,
    press_kPa: Float_0D,
    wnd: Float_0D,
    pr33: Float_0D,
    sc33: Float_0D,
    scc33: Float_0D,
    tair_filter_z: Float_0D,
) -> Tuple[Float_0D, Float_0D, Float_0D]:
    """This subroutine computes the leaf boundary layer
       resistances for heat, vapor and CO2 (s/m).
       Flat plate theory is used, as discussed in Schuepp (1993) and
       Grace and Wilson (1981).
       We consider the effects of turbulent boundary layers and sheltering.
       Schuepp's review shows a beta factor multiplier is necessary for SH in
       flows with high turbulence.

    Args:
        zzz (Float_0D): _description_
        ht (Float_0D): _description_
        TLF (Float_0D): _description_
        grasshof (Float_0D): _description_
        press_kPa (Float_0D): _description_
        wnd (Float_0D): _description_
        pr33 (Float_0D): _description_
        sc33 (Float_0D): _description_
        scc33 (Float_0D): _description_
        tair_filter_z (Float_0D): _description_

    Returns:
        Tuple[Float_0D, Float_0D, Float_0D]: _description_
    """
    # Difference between leaf and air temperature
    deltlf = TLF - tair_filter_z
    T_kelvin = tair_filter_z + 273.16
    graf = jax.lax.cond(deltlf > 0, lambda: grasshof * deltlf / T_kelvin, lambda: 0.0)
    nnu_T_P = nnu * (101.3 / press_kPa) * jnp.power((T_kelvin / 273.16), 1.81)
    Re = lleaf * uz(zzz, ht, wnd) / nnu_T_P
    Re5 = jax.lax.cond(Re > 0.0, lambda: jnp.sqrt(Re), lambda: 100.0)
    Re8 = jnp.power(Re, 0.8)
    Res_factor = jax.lax.cond(
        Re > 14000.0,
        lambda: 0.036 * Re8 * betfact,  # turbulent boundary layer
        lambda: 0.66 * Re5 * betfact,  # laminar boundary layer
    )
    # Sh_heat = Res_factor * pr33
    # Sh_vapor = Res_factor * sc33
    # Sh_CO2 = Res_factor * scc33
    # If there is free convection
    # jax.debug.print(
    #     "{a} {b} {c}", a=graf, b=T_kelvin, c=nnu_T_P
    # )
    conds = jnp.array(
        [
            (graf / (Re * Re) > 1.0) & (graf < 100000.0),
            (graf / (Re * Re) > 1.0) & (graf >= 100000.0),
            (graf / (Re * Re) <= 1.0),
        ]
    )
    index = jnp.where(conds, size=1)[0][0]
    coef = jax.lax.switch(
        index,
        [
            lambda: 0.5 * jnp.power(graf, 0.25),
            lambda: 0.13 * jnp.power(graf, 0.33),
            lambda: Res_factor,
        ],
    )
    Sh_heat = pr33 * coef
    Sh_vapor = sc33 * coef
    Sh_CO2 = scc33 * coef

    # Correct diffusivities for temperature and pressure
    ddh_T_P = ddh * (101.3 / press_kPa) * jnp.power((T_kelvin / 273.16), 1.81)
    ddv_T_P = ddv * (101.3 / press_kPa) * jnp.power((T_kelvin / 273.16), 1.81)
    ddc_T_P = ddc * (101.3 / press_kPa) * jnp.power((T_kelvin / 273.16), 1.81)

    heat = lleaf / (ddh_T_P * Sh_heat)
    vapor = lleaf / (ddv_T_P * Sh_vapor)
    co2 = lleaf / (ddc_T_P * Sh_CO2)

    vapor = jnp.clip(vapor, a_max=9999)

    return heat, vapor, co2


def friction_velocity(
    ustar: Float_0D,
    H_old: Float_0D,
    sensible_heat_flux: Float_0D,
    air_density: Float_0D,
    T_Kelvin: Float_0D,
) -> Tuple[Float_0D, Float_0D]:
    """this subroutine updates ustar and stability corrections
       based on the most recent H and z/L values

    Args:
        ustar (Float_0D): _description_
        H_old (Float_0D): _description_
        sensible_heat_flux (Float_0D): _description_
        air_density (Float_0D): _description_
        T_Kelvin (Float_0D): _description_

    Returns:
        Tuple[Float_0D, Float_0D]: _description_
    """
    # this subroutine is uncessary for CanAlfalfa since we measure and input ustar
    # filter sensible heat flux to reduce run to run instability
    H_old_update = 0.85 * sensible_heat_flux + 0.15 * H_old
    # z/L
    zl = -(0.4 * 9.8 * H_old_update * 14.75) / (
        air_density * 1005.0 * T_Kelvin * jnp.power(ustar, 3.0)
    )
    # jax.debug.print(
    #     "jax - {a} {b} {c} {d}", a=H_old, b=air_density, c=T_Kelvin, d=ustar
    # )
    return H_old_update, zl
