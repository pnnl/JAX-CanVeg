"""
Radiation transfer functions, including:
- diffuse_direct_radiation()
- sky_ir()
- rad_tran_canopy()
- ir_rad_tran_canopy()

Author: Peishi Jiang
Date: 2023.07.27.
"""

import jax
import jax.numpy as jnp

import equinox as eqx

from functools import partial
from typing import Tuple

from ...subjects import SunAng, LeafAng, ParNir, Para, Ir, SunShadedCan, Soil, Met
from ...subjects import Lai
from ...shared_utilities.types import Float_0D, Float_1D
from ...shared_utilities.utils import dot, add


def diffuse_direct_radiation(
    solar_sine_beta: Float_1D, rglobal: Float_1D, parin: Float_1D, press_kpa: Float_1D
) -> Tuple[Float_1D, Float_1D, Float_1D, Float_1D, Float_1D]:
    solar_constant = 1360.8  # [W m-2]

    # Distribution of extraterrestrial solar radiation
    fnir, fvis = 0.517, 0.383
    # fuv, fnir, fvis = 0.087, 0.517, 0.383

    # visible direct and potential diffuse PAR
    ru = press_kpa / (101.3 * solar_sine_beta)
    ru = jnp.clip(ru, a_min=0.0)

    rdvis = solar_constant * fvis * jnp.exp(-0.185 * ru) * solar_sine_beta
    rsvis = 0.4 * (solar_constant * fvis * solar_sine_beta - rdvis)

    # water absorption in NIR for 10 mm precip water
    # @jnp.vectorize
    # def compute_wa(ru_e):
    #     return jax.lax.cond(
    #         ru_e<0.,
    #         lambda: 0.,
    #         lambda: solar_constant * 0.077 * jnp.power((2.0 * ru_e), 0.3)
    #     )
    # wa = compute_wa(ru)
    # wa = compute_wa(ru, solar_sine_beta)
    wa = solar_constant * 0.077 * jnp.power((2.0 * ru), 0.3)

    # direct beam and potential diffuse NIR
    @jnp.vectorize
    def clean_night_to_zero(r_e, solar_sine_beta_e):
        return jax.lax.cond(solar_sine_beta_e < 0.0, lambda: 0.0, lambda: r_e)

    nircoef = solar_constant * fnir
    rdir = (nircoef * jnp.exp(-0.06 * ru) - wa) * solar_sine_beta
    # rdir = rdir.at[jnp.isnan(rdir)].set(0.0)
    rdir = clean_night_to_zero(rdir, solar_sine_beta)
    rsdir = 0.6 * (nircoef - rdvis / solar_sine_beta - wa) * solar_sine_beta
    # rsdir = rsdir.at[jnp.isnan(rsdir)].set(0.0)
    rsdir = clean_night_to_zero(rsdir, solar_sine_beta)

    rvt = rdvis + rsvis
    rit = rdir + rsdir
    rvt = clean_night_to_zero(rvt, solar_sine_beta)
    rit = clean_night_to_zero(rit, solar_sine_beta)
    # rvt = rvt.at[jnp.isnan(rvt)].set(0.0)
    # rit = rit.at[jnp.isnan(rit)].set(0.0)
    rvt = jnp.maximum(0.1, rvt)
    rit = jnp.maximum(0.1, rit)

    @jnp.vectorize
    def clean_night_to_one(r_e, solar_sine_beta_e):
        return jax.lax.cond(solar_sine_beta_e < 0.0, lambda: 1.0, lambda: r_e)

    @jnp.vectorize
    def clean_inf_to_one(r_e):
        return jax.lax.cond(jnp.isinf(r_e), lambda: 1.0, lambda: r_e)

    # jax.debug.print("{a}", a=jnp.isnan(rvt+rit).sum())
    ratrad = rglobal / (rvt + rit)
    # ratrad = ratrad.at[jnp.isnan(ratrad)].set(1.0)
    # ratrad = ratrad.at[jnp.isinf(ratrad)].set(1.0)
    ratrad = clean_night_to_one(ratrad, solar_sine_beta)
    ratrad = clean_inf_to_one(ratrad)
    ratrad = jnp.clip(ratrad, a_min=0.22, a_max=0.89)
    # ratrad = jnp.maximum(0.22, ratrad)
    # ratrad = jnp.minimum(0.89, ratrad)

    # ratio is the ratio between observed and potential radiation
    # NIR flux density as a function of PAR
    # since NIR is used in energy balance calculations
    # convert it to W m-2: divide PAR by 4.6
    nirx = rglobal - (parin / 4.6)

    # fraction PAR direct and diffuse
    xvalue = (0.9 - ratrad) / 0.70
    fvsb = rdvis / rvt * (1.0 - jnp.power(xvalue, 0.67))
    fvsb = jnp.clip(fvsb, a_min=0.0, a_max=1.0)
    # fvd = 1.0 - fvsb
    # fvsb = jnp.maximum(0.0, fvsb)
    # fvsb = jnp.minimum(1.0, fvsb)

    # note PAR has been entered in units of uE m-2 s-1
    par_beam = fvsb * parin
    # par_beam = par_beam.at[par_beam < 0].set(0.0)
    par_beam = jnp.clip(par_beam, a_min=0.0)
    par_diffuse = parin - par_beam

    @jnp.vectorize
    def clean_zeroin_to_zero(r_e, in_e):
        return jax.lax.cond(in_e == 0, lambda: 0.0, lambda: r_e)

    par_beam = clean_zeroin_to_zero(par_beam, parin)
    par_diffuse = clean_zeroin_to_zero(par_diffuse, parin)
    # par_beam = par_beam.at[parin == 0].set(0.001)
    # par_diffuse = par_diffuse.at[parin == 0].set(0.001)
    # par_beam = par_beam.at[parin == 0].set(0.0)
    # par_diffuse = par_diffuse.at[parin == 0].set(0.0)
    # jax.debug.print("par_beam {a}", a=par_beam[:3])

    # NIR beam and diffuse flux densities
    xvalue = (0.9 - ratrad) / 0.68
    fansb = rdir / rit * (1.0 - jnp.power(xvalue, 0.67))
    fansb = jnp.clip(fansb, a_min=0.0, a_max=1.0)
    # fand = 1. - fansb
    nir_beam = fansb * nirx
    # nir_beam = nir_beam.at[nir_beam < 0].set(0.0)
    nir_beam = jnp.clip(nir_beam, a_min=0.0)
    nir_diffuse = nirx - nir_beam
    nir_beam = clean_zeroin_to_zero(nir_beam, nirx)
    nir_diffuse = clean_zeroin_to_zero(nir_diffuse, nirx)
    # nir_beam = nir_beam.at[nirx == 0].set(0.1)
    # nir_diffuse = nir_diffuse.at[nirx == 0].set(0.1)
    # nir_beam = nir_beam.at[nirx == 0].set(0.0)
    # nir_diffuse = nir_diffuse.at[nirx == 0].set(0.0)

    # nir_diffuse = nirx - nir_beam
    # nir_beam = nirx - nir_diffuse
    # par_beam = parin - par_diffuse

    # Check nan again!
    par_beam = clean_night_to_zero(par_beam, solar_sine_beta)
    par_diffuse = clean_night_to_zero(par_diffuse, solar_sine_beta)
    nir_beam = clean_night_to_zero(nir_beam, solar_sine_beta)
    nir_diffuse = clean_night_to_zero(nir_diffuse, solar_sine_beta)
    # par_beam = par_beam.at[jnp.isnan(par_beam)].set(0.0)
    # par_diffuse = par_diffuse.at[jnp.isnan(par_diffuse)].set(0.0)
    # nir_beam = nir_beam.at[jnp.isnan(nir_beam)].set(0.0)
    # nir_diffuse = nir_diffuse.at[jnp.isnan(nir_diffuse)].set(0.0)

    return ratrad, par_beam, par_diffuse, nir_beam, nir_diffuse


# def diffuse_direct_radiation_old(
#     solar_sine_beta: Float_1D, rglobal: Float_1D, parin: Float_1D, press_kpa: Float_1D
# ) -> Tuple[Float_1D, Float_1D, Float_1D, Float_1D, Float_1D]:
#     solar_constant = 1360.8  # [W m-2]

#     # Distribution of extraterrestrial solar radiation
#     fnir, fvis = 0.517, 0.383
#     # fuv, fnir, fvis = 0.087, 0.517, 0.383

#     # visible direct and potential diffuse PAR
#     ru = press_kpa / (101.3 * solar_sine_beta)

#     @jnp.vectorize
#     def func_night(ru_e):
#         return jax.lax.cond(ru_e <= 0, lambda: jnp.nan, lambda: ru_e)

#     # ru = ru.at[ru <= 0].set(jnp.nan) # for negative sun angles getting complex numbers  # noqa: E501
#     ru = func_night(ru)
#     rdvis = solar_constant * fvis * jnp.exp(-0.185 * ru) * solar_sine_beta
#     rsvis = 0.4 * (solar_constant * fvis * solar_sine_beta - rdvis)

#     # water absorption in NIR for 10 mm precip water
#     wa = solar_constant * 0.077 * jnp.power((2.0 * ru), 0.3)

#     # direct beam and potential diffuse NIR
#     @jnp.vectorize
#     def clean_nan_to_zero(r_e):
#         return jax.lax.cond(jnp.isnan(r_e), lambda: 0.0, lambda: r_e)

#     nircoef = solar_constant * fnir
#     rdir = (nircoef * jnp.exp(-0.06 * ru) - wa) * solar_sine_beta
#     # rdir = rdir.at[jnp.isnan(rdir)].set(0.0)
#     rdir = clean_nan_to_zero(rdir)
#     rsdir = 0.6 * (nircoef - rdvis / solar_sine_beta - wa) * solar_sine_beta
#     # rsdir = rsdir.at[jnp.isnan(rsdir)].set(0.0)
#     rsdir = clean_nan_to_zero(rsdir)

#     rvt = rdvis + rsvis
#     rit = rdir + rsdir
#     rvt = clean_nan_to_zero(rvt)
#     rit = clean_nan_to_zero(rit)
#     # rvt = rvt.at[jnp.isnan(rvt)].set(0.0)
#     # rit = rit.at[jnp.isnan(rit)].set(0.0)
#     # rvt = jnp.maximum(0.1, rvt)
#     # rit = jnp.maximum(0.1, rit)

#     @jnp.vectorize
#     def clean_nan_to_one(r_e):
#         return jax.lax.cond(jnp.isnan(r_e), lambda: 1.0, lambda: r_e)

#     @jnp.vectorize
#     def clean_inf_to_one(r_e):
#         return jax.lax.cond(jnp.isinf(r_e), lambda: 1.0, lambda: r_e)

#     ratrad = rglobal / (rvt + rit)
#     # ratrad = ratrad.at[jnp.isnan(ratrad)].set(1.0)
#     # ratrad = ratrad.at[jnp.isinf(ratrad)].set(1.0)
#     ratrad = clean_nan_to_one(ratrad)
#     ratrad = clean_inf_to_one(ratrad)
#     ratrad = jnp.clip(ratrad, a_min=0.22, a_max=0.89)
#     # ratrad = jnp.maximum(0.22, ratrad)
#     # ratrad = jnp.minimum(0.89, ratrad)

#     # ratio is the ratio between observed and potential radiation
#     # NIR flux density as a function of PAR
#     # since NIR is used in energy balance calculations
#     # convert it to W m-2: divide PAR by 4.6
#     nirx = rglobal - (parin / 4.6)

#     # fraction PAR direct and diffuse
#     xvalue = (0.9 - ratrad) / 0.70
#     fvsb = rdvis / rvt * (1.0 - jnp.power(xvalue, 0.67))
#     fvsb = jnp.clip(fvsb, a_min=0.0, a_max=1.0)
#     # fvd = 1.0 - fvsb
#     # fvsb = jnp.maximum(0.0, fvsb)
#     # fvsb = jnp.minimum(1.0, fvsb)

#     # note PAR has been entered in units of uE m-2 s-1
#     par_beam = fvsb * parin
#     # par_beam = par_beam.at[par_beam < 0].set(0.0)
#     par_beam = jnp.clip(par_beam, a_min=0.0)
#     par_diffuse = parin - par_beam

#     @jnp.vectorize
#     def clean_zeroin_to_zero(r_e, in_e):
#         return jax.lax.cond(in_e == 0, lambda: 0.0, lambda: r_e)

#     par_beam = clean_zeroin_to_zero(par_beam, parin)
#     par_diffuse = clean_zeroin_to_zero(par_diffuse, parin)
#     # par_beam = par_beam.at[parin == 0].set(0.001)
#     # par_diffuse = par_diffuse.at[parin == 0].set(0.001)
#     # par_beam = par_beam.at[parin == 0].set(0.0)
#     # par_diffuse = par_diffuse.at[parin == 0].set(0.0)
#     # jax.debug.print("par_beam {a}", a=par_beam[:3])

#     # NIR beam and diffuse flux densities
#     xvalue = (0.9 - ratrad) / 0.68
#     fansb = rdir / rit * (1.0 - jnp.power(xvalue, 0.67))
#     fansb = jnp.clip(fansb, a_min=0.0, a_max=1.0)
#     # fand = 1. - fansb
#     nir_beam = fansb * nirx
#     # nir_beam = nir_beam.at[nir_beam < 0].set(0.0)
#     nir_beam = jnp.clip(nir_beam, a_min=0.0)
#     nir_diffuse = nirx - nir_beam
#     nir_beam = clean_zeroin_to_zero(nir_beam, nirx)
#     nir_diffuse = clean_zeroin_to_zero(nir_diffuse, nirx)
#     # nir_beam = nir_beam.at[nirx == 0].set(0.1)
#     # nir_diffuse = nir_diffuse.at[nirx == 0].set(0.1)
#     # nir_beam = nir_beam.at[nirx == 0].set(0.0)
#     # nir_diffuse = nir_diffuse.at[nirx == 0].set(0.0)

#     # nir_diffuse = nirx - nir_beam
#     # nir_beam = nirx - nir_diffuse
#     # par_beam = parin - par_diffuse

#     # Check nan again!
#     par_beam = clean_nan_to_zero(par_beam)
#     par_diffuse = clean_nan_to_zero(par_diffuse)
#     nir_beam = clean_nan_to_zero(nir_beam)
#     nir_diffuse = clean_nan_to_zero(nir_diffuse)
#     # par_beam = par_beam.at[jnp.isnan(par_beam)].set(0.0)
#     # par_diffuse = par_diffuse.at[jnp.isnan(par_diffuse)].set(0.0)
#     # nir_beam = nir_beam.at[jnp.isnan(nir_beam)].set(0.0)
#     # nir_diffuse = nir_diffuse.at[jnp.isnan(nir_diffuse)].set(0.0)

#     return ratrad, par_beam, par_diffuse, nir_beam, nir_diffuse


# # @jax.jit
# def diffuse_direct_radiation(
#     solar_sine_beta: Float_1D, rglobal: Float_1D, parin: Float_1D, press_kpa: Float_1D
# ) -> Tuple[Float_1D, Float_1D, Float_1D, Float_1D, Float_1D]:
#     solar_constant = 1360.8  # [W m-2]

#     # Distribution of extraterrestrial solar radiation
#     fnir, fvis = 0.517, 0.383
#     # fuv, fnir, fvis = 0.087, 0.517, 0.383

#     # visible direct and potential diffuse PAR
#     ru = press_kpa / (101.3 * solar_sine_beta)
#     ru = ru.at[ru <= 0].set(jnp.nan) # for negative sun angles getting complex numbers
#     rdvis = solar_constant * fvis * jnp.exp(-0.185 * ru) * solar_sine_beta
#     rsvis = 0.4 * (solar_constant * fvis * solar_sine_beta - rdvis)

#     # water absorption in NIR for 10 mm precip water
#     wa = solar_constant * 0.077 * jnp.power((2.0 * ru), 0.3)

#     # direct beam and potential diffuse NIR
#     nircoef = solar_constant * fnir
#     rdir = (nircoef * jnp.exp(-0.06 * ru) - wa) * solar_sine_beta
#     rdir = rdir.at[jnp.isnan(rdir)].set(0.0)
#     # rdir = jnp.maximum(0.0, rdir)
#     rsdir = 0.6 * (nircoef - rdvis / solar_sine_beta - wa) * solar_sine_beta
#     rsdir = rsdir.at[jnp.isnan(rsdir)].set(0.0)
#     # rsdir = jnp.maximum(0.0, rsdir)

#     rvt = rdvis + rsvis
#     rit = rdir + rsdir
#     rvt = rvt.at[jnp.isnan(rvt)].set(0.0)
#     rit = rit.at[jnp.isnan(rit)].set(0.0)
#     # rvt = jnp.maximum(0.1, rvt)
#     # rit = jnp.maximum(0.1, rit)

#     ratrad = rglobal / (rvt + rit)
#     ratrad = ratrad.at[jnp.isnan(ratrad)].set(1.0)
#     ratrad = ratrad.at[jnp.isinf(ratrad)].set(1.0)
#     ratrad = jnp.clip(ratrad, a_min=0.22, a_max=0.89)
#     # ratrad = jnp.maximum(0.22, ratrad)
#     # ratrad = jnp.minimum(0.89, ratrad)

#     # ratio is the ratio between observed and potential radiation
#     # NIR flux density as a function of PAR
#     # since NIR is used in energy balance calculations
#     # convert it to W m-2: divide PAR by 4.6
#     nirx = rglobal - (parin / 4.6)

#     # fraction PAR direct and diffuse
#     xvalue = (0.9 - ratrad) / 0.70
#     fvsb = rdvis / rvt * (1.0 - jnp.power(xvalue, 0.67))
#     fvsb = jnp.clip(fvsb, a_min=0.0, a_max=1.0)
#     # fvd = 1.0 - fvsb
#     # fvsb = jnp.maximum(0.0, fvsb)
#     # fvsb = jnp.minimum(1.0, fvsb)

#     # note PAR has been entered in units of uE m-2 s-1
#     par_beam = fvsb * parin
#     par_beam = par_beam.at[par_beam < 0].set(0.0)
#     par_diffuse = parin - par_beam
#     # par_beam = par_beam.at[parin == 0].set(0.001)
#     # par_diffuse = par_diffuse.at[parin == 0].set(0.001)
#     par_beam = par_beam.at[parin == 0].set(0.0)
#     par_diffuse = par_diffuse.at[parin == 0].set(0.0)
#     # jax.debug.print("par_beam {a}", a=par_beam[:3])

#     # NIR beam and diffuse flux densities
#     xvalue = (0.9 - ratrad) / 0.68
#     fansb = rdir / rit * (1.0 - jnp.power(xvalue, 0.67))
#     fansb = jnp.clip(fansb, a_min=0.0, a_max=1.0)
#     # fand = 1. - fansb
#     nir_beam = fansb * nirx
#     nir_beam = nir_beam.at[nir_beam < 0].set(0.0)
#     nir_diffuse = nirx - nir_beam
#     # nir_beam = nir_beam.at[nirx == 0].set(0.1)
#     # nir_diffuse = nir_diffuse.at[nirx == 0].set(0.1)
#     nir_beam = nir_beam.at[nirx == 0].set(0.0)
#     nir_diffuse = nir_diffuse.at[nirx == 0].set(0.0)

#     # nir_diffuse = nirx - nir_beam
#     # nir_beam = nirx - nir_diffuse
#     # par_beam = parin - par_diffuse

#     # Check nan
#     par_beam = par_beam.at[jnp.isnan(par_beam)].set(0.0)
#     par_diffuse = par_diffuse.at[jnp.isnan(par_diffuse)].set(0.0)
#     nir_beam = nir_beam.at[jnp.isnan(nir_beam)].set(0.0)
#     nir_diffuse = nir_diffuse.at[jnp.isnan(nir_diffuse)].set(0.0)

#     return ratrad, par_beam, par_diffuse, nir_beam, nir_diffuse


def sky_ir(T: Float_1D, ratrad: Float_1D, sigma: Float_0D) -> Float_1D:
    """Infrared radiation from sky, W m-2, using algorithm from Norman.

    Args:
        T (Float_0D): _description_
        ratrad (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    product = (
        (1.0 - 0.261 * jnp.exp(-0.000777 * jnp.power((273.16 - T), 2.0))) * ratrad
        + 1
        - ratrad
    )
    y = sigma * jnp.power(T, 4.0) * product
    # product = (
    #     1.0 - 0.261 * jnp.exp(-0.000777 * jnp.power((273.16 - T), 2.0) * ratrad
    #     + 1
    #     - ratrad)
    # )
    # y = sigma * jnp.power(T, 4.0)
    return y


def sky_ir_v2(met: Met, ratrad: Float_1D, sigma: Float_0D) -> Float_1D:
    """Choi, Minha, Jennifer M. Jacobs, and William P. Kustas. 2008.
        'Assessment of clear and cloudy sky parameterizations for daily downwelling
        longwave radiation over different land surfaces in Florida, USA'
        , Geophysical Research Letters, 35.

    Args:
        met (Met): _description_
        ratrad (Float_1D): _description_
        sigma (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    ea_mb = met.eair_Pa / 100
    Rldc = (0.605 + 0.048 * jnp.power(ea_mb, 0.5)) * sigma * jnp.power(met.T_air_K, 4.0)

    # c
    c = 1 - ratrad

    y = Rldc * (1 - c) + c * sigma * jnp.power(met.T_air_K, 4.0)
    # y = sigma * jnp.power(T, 4.0)
    return y


# @eqx.filter_jit
def rad_tran_canopy(
    sunang: SunAng,
    leafang: LeafAng,
    rad: ParNir,
    prm: Para,
    lai: Lai,
    reflect: Float_0D,
    trans: Float_0D,
    soil_refl: Float_0D,
    niter: int,
) -> ParNir:
    """This subroutine computes the flux density of direct and diffuse
       radiation in the near infrared waveband.  The Markov model is used
       to compute the probability of beam penetration through clumped foliage.

       The algorithms of Norman (1979) are used.

        Norman, J.M. 1979. Modeling the complete crop canopy.
        Modification of the Aerial Environment of Crops.
        B. Barfield and J. Gerber, Eds. American Society of Agricultural Engineers,
        249-280.

    Args:
        sunang (SunAng): _description_
        leafang (LeafAng): _description_
        rad (ParNir): _description_
        prm (Para): _description_
        niter (Int_0D): _description_
        mask_night (HashableArrayWrapper): _description_

    Returns:
        ParNir: _description_
    """
    absorbed = 1 - reflect - trans

    # ntime = rad.inbeam.shape[0]
    # jtot = prm.markov.shape[0]
    ntime, jtot = rad.sh_abs.shape
    jktot = jtot + 1
    # ntime = prm.ntime
    # jtot = prm.jtot
    # jktot = prm.jktot

    # Incident radiation above the canopy
    # Compute the amount of diffuse light that is transmitted through a layer.
    # Diffuse radiation through gpas and absorbed radiation transmitted
    # through leaves
    # @jnp.vectorize
    # def calculate_fraction_beam(inbeam_e, incoming_e):
    #     return jax.lax.cond(
    #         incoming_e == 0,
    #         lambda: 0.0,
    #         lambda: inbeam_e / incoming_e
    #     )
    # fraction_beam = calculate_fraction_beam(rad.inbeam, rad.incoming)
    rad_incoming_cp = rad.incoming
    rad_incoming_cp = jnp.clip(rad_incoming_cp, a_min=0.01)
    fraction_beam = rad.inbeam / rad_incoming_cp
    # fraction_beam = rad.inbeam / rad.incoming

    # # At night if rad.incoming is zero then set fraction_beam to zero
    @jnp.vectorize
    def set_night_to_zero(fraction_beam_e, rad_incoming_e):
        return jax.lax.cond(rad_incoming_e == 0, lambda: 0.0, lambda: fraction_beam_e)

    fraction_beam = set_night_to_zero(fraction_beam, rad.incoming)

    # fraction_beam = jnp.nan_to_num(fraction_beam, nan=0.0)
    # beam_top = fraction_beam # compute beam from initial ones

    # Compute beam radiation transfer and probability of beam
    # @jnp.vectorize
    @partial(jax.vmap, in_axes=(0, 0), out_axes=0)
    def convert_night_to_zero(v, sin_beta_e):
        return jax.lax.cond(sin_beta_e <= 0, lambda: jnp.zeros(jktot), lambda: v)

    # a = dot(
    #         leafang.Gfunc / sunang.sin_beta,  # (ntime,)
    #         # a,  # (ntime,)
    #         # -(prm.dff * prm.markov),  # (ntime, jtot)
    #         -(lai.dff * prm.markov),  # (ntime, jtot)
    #     )
    # jax.debug.print("# of infs: {x}", x=jnp.isinf(a).sum())
    # jax.debug.print("a stats: {x1}, {x2}", x1=a.min(), x2=a.max())
    exp_direct = jnp.exp(
        dot(
            leafang.Gfunc / sunang.sin_beta,  # (ntime,)
            # -(prm.dff * prm.markov),  # (ntime, jtot)
            -(lai.dff * prm.markov),  # (ntime, jtot)
        )
    )  # (ntime, jtot)
    exp_direct = jnp.concatenate(
        [exp_direct, jnp.ones([ntime, 1])], axis=1
    )  # (ntime, jktot)
    exp_direct = convert_night_to_zero(exp_direct, sunang.sin_beta)
    # exp_direct = exp_direct.at[mask_night.val, :].set(0.0)
    P0 = jax.lax.cumprod(exp_direct, axis=1, reverse=True)  # (ntime, jktot)
    Tbeam = P0
    # rad.prob_beam = jnp.concatenate(
    #     [rad.prob_beam[:,:prm.jtot-1],
    #      jnp.ones([prm.ntime,1]),
    #      rad.prob_beam[:,prm.jtot:]]
    # )

    # rad.prob_beam(:,prm.jtot)=ones;
    prob_beam = jnp.concatenate(
        [prm.markov * P0[:, :jtot], jnp.ones([ntime, 1])], axis=1
    )  # (ntime, jktot)
    prob_beam = jnp.clip(prob_beam, a_max=1.0)
    prob_beam = convert_night_to_zero(prob_beam, sunang.sin_beta)
    # prob_beam = prob_beam.at[mask_night.val, :].set(0.0)
    # rad.prob_shade = 1 - rad.prob_beam
    # jax.debug.print("prob_beam: {a}", a=prob_beam.mean(axis=0))

    # Calculate beam PAR that is transmitted downward
    # (ntime, jtot)
    incoming_beam = dot(rad.incoming * fraction_beam, Tbeam[:, 1:])
    incoming_beam = incoming_beam * (1 - exp_direct[:, :-1])
    sdn = incoming_beam * trans

    # Calculate beam PAR that is reflected upward
    # (ntime, jtot)
    # sup = incoming_beam * rad.reflect
    sup = incoming_beam * reflect

    # Calculate the transmission and reflection of each layer
    # reflectance_layer = (1 - leafang.integ_exp_diff) * rad.reflect  # (ntime, jtot)
    reflectance_layer = (1 - leafang.integ_exp_diff) * reflect  # (ntime, jtot)
    transmission_layer = (
        # leafang.integ_exp_diff + (1 - leafang.integ_exp_diff) * rad.trans
        leafang.integ_exp_diff
        + (1 - leafang.integ_exp_diff) * trans
    )  # (ntime, jtot)  # noqa: E501

    # Compute diffuse and complementary radiation
    dn_top = (1 - fraction_beam) * rad.incoming
    dn_init = jnp.concatenate(
        [jnp.zeros([ntime, jtot]), jnp.expand_dims(dn_top, axis=-1)], axis=1
    )
    up_init = jnp.zeros([ntime, jktot])

    def calculate_dnup(c2, j):
        up, dn = c2[0], c2[1]
        dn_top, up_bot = dn[:, -1], up[:, 0]

        def calculate_dnup_layer(c, i):
            dn_layer_t, up_layer_b = c, up[:, i]
            dn_layer_b = (
                dn_layer_t * transmission_layer[:, i]
                + up_layer_b * reflectance_layer[:, i]
                + sdn[:, i]
            )
            up_layer_t = (
                up_layer_b * transmission_layer[:, i]
                + dn_layer_t * reflectance_layer[:, i]
                + sup[:, i]
            )
            cnew = dn_layer_b
            out_layer = (dn_layer_b, up_layer_t)
            return cnew, out_layer

        _, out = jax.lax.scan(
            calculate_dnup_layer, dn_top, jnp.arange(jtot - 1, -1, -1)
        )
        outdn, outup = out[0], out[1]
        outdn = outdn.T
        dn = jnp.concatenate([outdn[:, ::-1], jnp.expand_dims(dn_top, axis=-1)], axis=1)

        # up_bot = (dn[:, 0] + rad.incoming * fraction_beam * Tbeam[:, 0])*rad.soil_refl
        up_bot = (dn[:, 0] + rad.incoming * fraction_beam * Tbeam[:, 0]) * soil_refl
        outup = outup.T
        up = jnp.concatenate([jnp.expand_dims(up_bot, axis=-1), outup[:, ::-1]], axis=1)

        c2new = [up, dn]
        return c2new, None

    carry, _ = jax.lax.scan(calculate_dnup, [up_init, dn_init], xs=None, length=niter)
    up, dn = carry[0], carry[1]

    # jax.debug.print("up: {a}", a=up.mean(axis=0))
    # jax.debug.print("dn: {a}", a=dn.mean(axis=0))

    # upward diffuse radiation flux density, on the horizontal
    up_flux = up

    # downward beam radiation flux density, incident on the horizontal
    beam_flux = dot(rad.incoming * fraction_beam, Tbeam)

    # downward diffuse radiation flux density on the horizontal
    dn_flux = dn

    # # total downward radiation, incident on the horizontal
    # rad.total = rad.beam_flux + rad.dn_flux

    # amount of radiation absorbed on the sun and shade leaves
    # sh_abs = (dn_flux[:, :jtot] + up_flux[:, :jtot]) * rad.absorbed
    sh_abs = (dn_flux[:, :jtot] + up_flux[:, :jtot]) * absorbed
    normal = beam_flux[:, jtot] * leafang.Gfunc / sunang.sin_beta  # (ntime,)
    normal = jnp.clip(normal, a_min=0.0)
    # sun_normal_abs = normal * rad.absorbed
    sun_normal_abs = normal * absorbed
    sun_abs = add(sun_normal_abs, sh_abs)

    rad = eqx.tree_at(
        lambda t: (t.prob_beam, t.beam_flux, t.up_flux, t.dn_flux, t.sh_abs, t.sun_abs),
        rad,
        (prob_beam, beam_flux, up_flux, dn_flux, sh_abs, sun_abs),
    )

    return rad


# @eqx.filter_jit
def ir_rad_tran_canopy(
    leafang: LeafAng,
    ir: Ir,
    rad: ParNir,
    soil: Soil,
    sun: SunShadedCan,
    shade: SunShadedCan,
    prm: Para,
) -> Ir:
    """This subroutine computes the flux density of diffuse
       radiation in the infrared waveband.  The Markov model is used
       to compute the probability of beam penetration through clumped foliage.

       The algorithms of Norman (1979) are used.

       Norman, J.M. 1979. Modeling the complete crop canopy.
       Modification of the Aerial Environment of Crops.
       B. Barfield and J. Gerber, Eds. American Society of Agricultural Engineers, 249-280.

       And adaption algorithms from Bonan, Climate Change and Terrestrial Ecosystem
       Modeling of Norman's model

    Args:
        leafang (LeafAng): _description_
        ir (Ir): _description_
        rad (ParNir): _description_
        sun (SunShadedCan): _description_
        shade (SunShadedCan): _description_
        prm (Para): _description_

    Returns:
        Ir: _description_
    """  # noqa: E501
    _, jktot = ir.ir_dn.shape
    jtot = jktot - 1

    # Set upper boundary condition
    ir_dn = jnp.concatenate(
        [ir.ir_dn[:, :jtot], jnp.expand_dims(ir.ir_in, axis=1)], axis=1
    )
    ir_up = jnp.concatenate(
        [ir.ir_up[:, :jtot], jnp.expand_dims(ir.ir_in, axis=1)], axis=1
    )

    # Compute IR radiative source flux as a function of leaf temperature weighted
    # according to sunlit and shaded fractions
    IR_source_sun = rad.prob_beam[:, :jtot] * jnp.power(sun.Tsfc, 4.0)
    IR_source_shade = rad.prob_shade[:, :jtot] * jnp.power(shade.Tsfc, 4.0)
    IR_source = prm.epsigma * (IR_source_sun + IR_source_shade)

    # Compute downward IR from the Top down
    # following Bonan and consider forward and backward scattering of IR, scat=(1-ep)/2
    # normally we assume relectance is 1-ep and transmission is zero, but that only
    # allows backward scattering and the IR fluxes are not so great leaving the canopy
    scat = (1 - prm.ep) / 2
    forward_scat = leafang.integ_exp_diff + (1 - leafang.integ_exp_diff) * scat
    backward_scat = (1 - leafang.integ_exp_diff) * scat

    sdn = IR_source * (1 - leafang.integ_exp_diff)
    sup = IR_source * (1 - leafang.integ_exp_diff)

    # def calculate_dnup(c2, j):
    #     ir_dn, ir_up = c2[0], c2[1]
    #     dn_top, up_bot = ir_dn[:, -1], ir_up[:, 0]

    #     def calculate_dn_layer(c, i):
    #         dn_layer_t, up_layer_b = c, ir_up[:, i]

    #         dn_layer_b = (
    #             dn_layer_t * forward_scat[:, i]
    #             + up_layer_b * backward_scat[:, i]
    #             + sdn[:, i]
    #         )

    #         up_layer_t = (
    #             up_layer_b * forward_scat[:, i]
    #             + dn_layer_t * backward_scat[:, i]
    #             + sup[:, i]
    #         )

    #         cnew = dn_layer_b
    #         out_layer = (dn_layer_b, up_layer_t)
    #         return cnew, out_layer

    #     _, out = jax.lax.scan(
    #         calculate_dn_layer,
    #         dn_top,
    #         jnp.arange(prm.jtot - 1, -1, -1),
    #     )
    #     outdn, outup = out[0], out[1]
    #     outdn = outdn.T
    #     ir_dn=jnp.concatenate([outdn[:,::-1],jnp.expand_dims(dn_top, axis=-1)],axis=1)

    #     up_bot = (1 - prm.epsoil) * ir_dn[:, 0] + prm.epsoil * prm.sigma * jnp.power(
    #         soil.sfc_temperature, 4
    #     )
    #     outup = outup.T
    #     ir_up=jnp.concatenate([jnp.expand_dims(up_bot,axis=-1),outup[:,::-1]],axis=1)

    #     cnew = [ir_dn, ir_up]

    #     return cnew, None

    # carry, _ = jax.lax.scan(calculate_dnup, [ir_dn, ir_up], xs=None, length=50)
    # ir_dn, ir_up = carry[0], carry[1]

    def calculate_dnup(c2, j):
        ir_dn, ir_up = c2[0], c2[1]

        def calculate_dn_layer(c, i):
            dn_layer_t, up_layer_b = c, ir_up[:, i]
            dn_layer_b = (
                dn_layer_t * forward_scat[:, i]
                + up_layer_b * backward_scat[:, i]
                + sdn[:, i]
            )
            cnew = dn_layer_b
            return cnew, cnew

        _, out = jax.lax.scan(
            # calculate_dn_layer, ir_dn[:, -1], jnp.arange(prm.jtot - 1, -1, -1)
            calculate_dn_layer,
            ir_dn[:, -1],
            jnp.arange(jtot - 1, -1, -1),
        )
        out = out.T
        ir_dn = jnp.concatenate([out[:, ::-1], ir_dn[:, -1:]], axis=1)

        # Compute upward IR from the bottom up with the lower boundary condition
        # based on soil temperature
        up_bot = (1 - prm.epsoil) * ir_dn[:, 0] + prm.epsoil * prm.sigma * jnp.power(
            soil.sfc_temperature, 4
        )

        def calculate_up_layer(c, i):
            un_layer_b, dn_layer_t = c, ir_dn[:, i + 1]
            up_layer_t = (
                un_layer_b * forward_scat[:, i]
                + dn_layer_t * backward_scat[:, i]
                # un_layer_b * leafang.integ_exp_diff[:,i]
                + sup[:, i]
            )
            cnew = up_layer_t
            return cnew, cnew

        _, out = jax.lax.scan(calculate_up_layer, up_bot, jnp.arange(jtot))
        out = out.T
        ir_up = jnp.concatenate([jnp.expand_dims(up_bot, axis=-1), out], axis=1)

        cnew = [ir_dn, ir_up]

        return cnew, cnew

    carry, _ = jax.lax.scan(calculate_dnup, [ir_dn, ir_up], xs=None, length=1)
    ir_dn, ir_up = carry[0], carry[1]

    # IR shade on top + bottom of leaves
    ir_shade = ir_up[:, :jtot] + ir_dn[:, :jtot]

    # Bonan shows IR Balance for ground area
    ir_balance = (1 - leafang.integ_exp_diff) * (
        (ir_up[:, :jtot] + ir_dn[:, 1:jktot]) * prm.ep - 2 * IR_source
    )

    ir = eqx.tree_at(
        lambda t: (
            t.ir_dn,
            t.ir_up,
            t.IR_source_sun,
            t.IR_source_shade,
            t.IR_source,
            t.shade,
            t.balance,
        ),
        ir,
        (ir_dn, ir_up, IR_source_sun, IR_source_shade, IR_source, ir_shade, ir_balance),
    )

    return ir
