"""
Partitioning solar radiation into direct and diffuse, visible and near-infrared components.
The implementation is based on:
(1) the Weiss-Norman ( 1985, Agric. forest Meteorol. 34: 205-213)
(2) the DIFFUSE_DIRECT_RADIATION() subroutine of CANOAK

Author: Peishi Jiang
Date: 2023.3.10
"""  # noqa: E501

import jax
import jax.numpy as jnp

from ....shared_utilities.types import Float_0D, Float_1D
from ....shared_utilities.constants import PO as p0
from ....shared_utilities.constants import RADIANS as RADD

# # Fraction of NIR and PAR (visible)
# fir, fv = 0.54, 0.46


def partition_solar_radiation(
    solar_rad: Float_0D, solar_elev_angle: Float_0D, pres: Float_0D
) -> Float_1D:
    """Partitioning solar radiation into direct and diffuse, visible and
       near-infrared components.

    Args:
        solar_rad (Float_0D): the incoming solar radiation [W m-2]
        solar_elev_angle (Float_0D): solar elevation angle [degree]
        pres (Float_0D): the atmospheric pressure [kPa]

    Returns:
        Float_1D: the partitioned radiation components [W m-2]
    """
    # Convert degree to radians
    solar_elev_rad = solar_elev_angle * RADD
    sine_solar_elev = jnp.sin(solar_elev_rad)

    # Pressure impact
    ru = pres / (p0 * sine_solar_elev)

    # Calculate potential visible direct PAR
    rdvis = 624.0 * jnp.exp(-0.185 * ru) * sine_solar_elev

    # Calculate potential diffuse PAR
    rsvis = 0.4 * (624.0 * sine_solar_elev - rdvis)

    # Calculate potential direct beam NIR
    wa = (
        1373.0 * 0.077 * jnp.power(2.0 * ru, 0.3)
    )  # water absorption in NIR for 10 mm precip water
    rdir = (748.0 * jnp.exp(-0.06 * ru) - wa) * sine_solar_elev
    rdir = jnp.max(jnp.array([0, rdir]))

    # Calculate potential diffuse NIR
    rsdir = 0.6 * (748.0 - rdvis / sine_solar_elev - wa) * sine_solar_elev
    rsdir = jnp.max(jnp.array([0, rsdir]))

    # Calculate potential total PAR and NIR
    # print(rdvis, rsvis, rsdir, rdir)
    rvt = rdvis + rsvis  # PAR
    rit = rdir + rsdir  # NIR
    cond1 = rvt <= 0
    cond2 = rit <= 0
    rvt = jax.lax.cond(cond1, set_pointone, keep_as_is, rvt)
    rit = jax.lax.cond(cond2, set_pointone, keep_as_is, rit)

    # Calculate PAR and NIR from observed solar radiation
    ratio_v = rvt / (rvt + rit)
    ratio_i = 1 - ratio_v
    svt = solar_rad * ratio_v
    sit = solar_rad * ratio_i
    # print("The ratio of PAR in the total solar radiation: ", ratio_v)
    # print(svt, sit, ratio_v, ratio_i)

    # Calculate the fraction of PAR in the direct beam
    ratio = solar_rad / (rvt + rit)
    ratio = jnp.min(jnp.array([0.88, ratio]))
    value = (0.9 - ratio) / 0.7
    # print(ratio, value)
    fdv = rdvis / rvt * (1 - jnp.power(value, 2.0 / 3.0))
    fdv = jnp.max(jnp.array([0, fdv]))
    fdv = jnp.min(jnp.array([1, fdv]))
    fsv = 1 - fdv

    # Calculate the fraction of NIR in the direct beam
    value = (0.88 - ratio) / 0.68
    fdir = rdir / rit * (1 - jnp.power(value, 2.0 / 3.0))
    fdir = jnp.max(jnp.array([0, fdir]))
    fdir = jnp.min(jnp.array([1, fdir]))
    fsir = 1 - fdir
    # print(rdvis/rvt, fdv, fdir)

    # Calculate direct and diffuse PAR and NIR
    sdv, ssv = fdv * svt, fsv * svt
    sdi, ssi = fdir * sit, fsir * sit

    # sdv: the direct beam PAR
    # ssv: the diffuse PAR
    # sdi: the direct beam NIR
    # ssi: the diffuse NIR
    return jnp.array([sdv, ssv, sdi, ssi])


def set_pointone(x):
    return 0.1


def keep_as_is(x):
    return x
