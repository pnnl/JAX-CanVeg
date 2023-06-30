"""
Radiation transfer functions, including:
- rnet()
- par()
- par_day()
- par_night()
- diffuse_direct_radiation()
- diffuse_direct_radiation_day()
- diffuse_direct_radiation_night()
- nir()
- sky_ir()
- irflux()
- g_func_diffuse()
- gfunc()

Author: Peishi Jiang
Date: 2023.06.28.
"""

import jax
import jax.numpy as jnp

from typing import Tuple

from ...shared_utilities.types import Float_0D, Float_1D
from ...shared_utilities.constants import ep, markov


def rnet(
    ir_dn: Float_1D,
    ir_up: Float_1D,
    par_sun: Float_1D,
    nir_sun: Float_1D,
    par_shade: Float_1D,
    nir_shade: Float_1D,
) -> Tuple[Float_1D, Float_1D]:
    """Energy balance and photosynthesis are performed for vegetation
       between levels and based on the energy incident to that level

    Args:
        ir_dn (Float_1D): _description_
        ir_up (Float_1D): _description_
        par_sun (Float_1D): _description_
        nir_sun (Float_1D): _description_
        par_shade (Float_1D): _description_
        nir_shade (Float_1D): _description_

    Returns:
        Tuple[Float_1D, Float_1D]: sunlit net radiation, shaded net radiation
    """

    # Infrared radiation on leaves
    ir_shade = ir_dn + ir_up
    ir_shade *= ep

    # Available energy on leaves for evaporation
    rnet_sun = par_sun + nir_sun + ir_shade
    rnet_sh = par_shade + nir_shade + ir_shade

    return rnet_sun, rnet_sh


def par(
    solar_sine_beta: Float_0D,
    parin: Float_0D,
    par_beam: Float_0D,
    par_reflect: Float_0D,
    par_trans: Float_0D,
    par_soil_refl: Float_0D,
    par_absorbed: Float_0D,
    dLAIdz: Float_1D,
    exxpdir: Float_1D,
    Gfunc_solar: Float_1D,
) -> Tuple[
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
]:
    (
        sun_lai,
        shd_lai,
        prob_beam,
        prob_sh,
        par_up,
        par_down,
        beam_flux_par,
        quantum_sh,
        quantum_sun,
        par_shade,
        par_sun,
    ) = jax.lax.cond(
        solar_sine_beta > 0.1,
        par_day,
        par_night,
        solar_sine_beta,
        parin,
        par_beam,
        par_reflect,
        par_trans,
        par_soil_refl,
        par_absorbed,
        dLAIdz,
        exxpdir,
        Gfunc_solar,
    )
    return (
        sun_lai,
        shd_lai,
        prob_beam,
        prob_sh,
        par_up,
        par_down,
        beam_flux_par,
        quantum_sh,
        quantum_sun,
        par_shade,
        par_sun,
    )


def par_day(
    solar_sine_beta: Float_0D,
    parin: Float_0D,
    par_beam: Float_0D,
    par_reflect: Float_0D,
    par_trans: Float_0D,
    par_soil_refl: Float_0D,
    par_absorbed: Float_0D,
    dLAIdz: Float_1D,
    exxpdir: Float_1D,
    Gfunc_solar: Float_1D,
) -> Tuple[
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
]:
    # Level 1 is the soil surface and level jktot is the
    # top of the canopy.  layer 1 is the layer above
    # the soil and layer jtot is the top layer.
    sze = dLAIdz.size
    jtot, jktot = sze - 2, sze - 1

    sup, sdn, adum = jnp.zeros(sze), jnp.zeros(sze), jnp.zeros(sze)
    prob_sh, prob_beam = jnp.zeros(sze), jnp.zeros(sze)
    sun_lai, shd_lai = jnp.zeros(sze), jnp.zeros(sze)
    par_up, par_down = jnp.zeros(sze), jnp.zeros(sze)
    quantum_sh, quantum_sun = jnp.zeros(sze), jnp.zeros(sze)
    par_shade, par_sun = jnp.zeros(sze), jnp.zeros(sze)
    beam_flux_par = jnp.zeros(sze)

    fraction_beam = par_beam / parin
    beam = jnp.zeros(jktot) + fraction_beam
    # tbeam = jnp.zeros(jktot) + fraction_beam
    sumlai = jnp.zeros(sze)
    sumlai = sumlai.at[:jtot].set(jnp.cumsum(dLAIdz[::-1][2:])[::-1])

    # Compute probability of penetration for direct and
    # diffuse radiation for each layer in the canopy

    # Diffuse radiation reflected by layer
    reflectance_layer = (1.0 - exxpdir) * par_reflect

    # Diffuse radiation transmitted through layer
    transmission_layer = (1.0 - exxpdir) * par_trans + exxpdir

    # Compute the probability of beam penetration
    exp_direct = jnp.exp(-dLAIdz * markov * Gfunc_solar / solar_sine_beta)
    pen2 = jnp.exp(-sumlai * markov * Gfunc_solar / solar_sine_beta)
    pen2 = pen2.at[-2:].set(0)

    # sunlit and shaded LAI
    sun_lai = solar_sine_beta * (1 - pen2) / (markov * Gfunc_solar)
    shd_lai = sumlai - sun_lai
    sun_lai = sun_lai.at[-2:].set(0)
    shd_lai = shd_lai.at[-2:].set(0)

    # probability of beam
    prob_beam = markov * pen2
    # beam = beam.at[:-1].set(beam[1:] * exp_direct)
    def update_beam(carry, i):
        # carry_new = carry * jnp.power(exp_direct[jtot-1-i], i)
        carry_new = carry * exp_direct[jtot - 1 - i]
        return carry_new, carry_new

    _, beam_update = jax.lax.scan(
        f=update_beam, init=beam[jktot - 1], xs=jnp.arange(jtot)
    )
    beam = beam.at[:-1].set(beam_update[::-1])
    qu = 1.0 - prob_beam
    qu = jnp.clip(qu, a_min=0.0, a_max=1.0)
    qu = qu.at[-2:].set(0)

    # probability of umbra
    prob_sh = qu
    tbeam = beam

    # beam PAR that is reflected upward by a layer
    sup = sup.at[1:-1].set(tbeam[1:] - tbeam[:-1]) * par_reflect
    # print(qu)

    # beam PAR that is transmitted downward
    sdn = sdn.at[:-2].set(tbeam[1:] - tbeam[:-1]) * par_trans

    # Initiate scattering using the technique of NORMAN (1979).
    # scattering is computed using an iterative technique.
    # Here Adum is the ratio up/down diffuse radiation.
    sup = sup.at[0].set(tbeam[0] * par_soil_refl)
    par_down = par_down.at[jktot - 1].set(1.0 - fraction_beam)
    adum = adum.at[0].set(par_soil_refl)
    tlay2 = transmission_layer * transmission_layer
    # adum = adum.at[1:].set(adum)
    def update_adum(carry, i):
        carry_new = (
            carry * tlay2[i] / (1 - carry * reflectance_layer[i]) + reflectance_layer[i]
        )
        return carry_new, carry_new

    _, adum_update = jax.lax.scan(update_adum, adum[0], jnp.arange(jtot))
    adum = adum.at[1:-1].set(adum_update)

    def update_pard(carry, i):
        carry_new = (
            carry
            * transmission_layer[i - 1]
            / (1.0 - adum[i] * reflectance_layer[i - 1])
            + sdn[i - 1]
        )
        return carry_new, carry_new

    _, pard_update = jax.lax.scan(
        update_pard, par_down[jktot - 1], jnp.arange(1, jktot)[::-1]
    )
    par_down = par_down.at[:-2].set(pard_update[::-1])
    par_up = adum * par_down + sup
    par_up = par_up.at[0].set(par_soil_refl * par_down[0] + sup[0])

    # Iterative calculation of upward diffuse and downward beam +
    # diffuse PAR.
    def update_parupdown(carry, i):
        par_up, par_down = carry[0], carry[1]
        # downward --
        def calculate_down(c, j):
            # carry_new = carry * jnp.power(exp_direct[jtot-1-i], i)
            c_new = (
                transmission_layer[j] * c + par_up[j] * reflectance_layer[j] + sdn[j]
            )
            return c_new, c_new

        _, down = jax.lax.scan(
            f=calculate_down, init=par_down[jktot - 1], xs=jnp.arange(jtot)[::-1]
        )
        par_down = par_down.at[:jtot].set(down[::-1])
        # upward --
        par_up = par_up.at[0].set((par_down[0] + tbeam[0]) * par_soil_refl)

        def calculate_up(c, j):
            # carry_new = carry * jnp.power(exp_direct[jtot-1-i], i)
            c_new = (
                reflectance_layer[j] * par_down[j + 1]
                + c * transmission_layer[j]
                + sup[j + 1]
            )
            return c_new, c_new

        _, up = jax.lax.scan(f=calculate_up, init=par_up[0], xs=jnp.arange(jtot))
        par_up = par_up.at[1:jktot].set(up)
        # up = reflectance_layer[:jtot]*par_down[:jtot]+ \
        #     par_up[:jtot]*transmission_layer[:jtot] + sup[1:jktot]
        # par_up = par_up.at[1:jktot].set(up)
        carry_new = [par_up, par_down]
        return carry_new, carry_new

    carry_new, _ = jax.lax.scan(update_parupdown, [par_up, par_down], jnp.arange(5))
    par_up, par_down = carry_new[0], carry_new[1]
    # jax.debug.print("par_up: {x}", x=par_up)
    # jax.debug.print("par_down: {x}", x=par_down)

    # Compute flux density of PAR
    par_up = par_up * parin
    par_up = jnp.clip(par_up, a_min=0.001)
    par_up = par_up.at[jktot:].set(0)
    beam_flux_par = beam_flux_par.at[:-1].set(beam * parin)
    beam_flux_par = jnp.clip(beam_flux_par, a_min=0.001)
    beam_flux_par = beam_flux_par.at[jktot:].set(0)
    par_down = par_down * parin
    par_down = jnp.clip(par_down, a_min=0.001)
    par_down = par_down.at[jktot:].set(0)
    # par_down = par_down.at[par_down<=0].set(0.001)
    # par_total = beam_flux_par[jktot-1]+par_down[jktot-1]

    # PSUN is the radiation incident on the mean leaf normal
    par_beam = jnp.maximum(par_beam, 0.001)
    par_normal_quanta = par_beam * Gfunc_solar / solar_sine_beta
    par_normal_abs_energy = par_normal_quanta * par_absorbed / 4.6  # W m-2
    par_normal_abs_quanta = par_normal_quanta * par_absorbed  # umol m-2 s-1
    quantum_sh = (par_down + par_up) * par_absorbed
    quantum_sun = quantum_sh + par_normal_abs_quanta
    par_shade = quantum_sh / 4.6  # W m-2
    par_sun = par_normal_abs_energy + par_shade

    quantum_sh = quantum_sh.at[jtot:].set(0)
    quantum_sun = quantum_sun.at[jtot:].set(0)
    par_shade = par_shade.at[jtot:].set(0)
    par_sun = par_sun.at[jtot:].set(0)

    return (
        sun_lai,
        shd_lai,
        prob_beam,
        prob_sh,
        par_up,
        par_down,
        beam_flux_par,
        quantum_sh,
        quantum_sun,
        par_shade,
        par_sun,
    )


def par_night(
    solar_sine_beta: Float_0D,
    parin: Float_0D,
    par_beam: Float_0D,
    par_reflect: Float_0D,
    par_trans: Float_0D,
    par_soil_refl: Float_0D,
    par_absorbed: Float_0D,
    dLAIdz: Float_1D,
    exxpdir: Float_1D,
    Gfunc_solar: Float_1D,
) -> Tuple[
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
    Float_0D,
]:
    sze = dLAIdz.size
    jtot = sze - 2
    prob_sh, prob_beam = jnp.zeros(sze), jnp.zeros(sze)
    sun_lai, shd_lai = jnp.zeros(sze), jnp.zeros(sze)
    par_up, par_down = jnp.zeros(sze), jnp.zeros(sze)
    quantum_sh, quantum_sun = jnp.zeros(sze), jnp.zeros(sze)
    par_shade, par_sun = jnp.zeros(sze), jnp.zeros(sze)
    beam_flux_par = jnp.zeros(sze)

    prob_sh = prob_sh.at[:jtot].set(1.0)

    return (
        sun_lai,
        shd_lai,
        prob_beam,
        prob_sh,
        par_up,
        par_down,
        beam_flux_par,
        quantum_sh,
        quantum_sun,
        par_shade,
        par_sun,
    )


def diffuse_direct_radiation(
    solar_sine_beta: Float_0D, rglobal: Float_0D, parin: Float_0D, press_kpa: Float_0D
) -> Tuple[Float_0D, Float_0D, Float_0D, Float_0D, Float_0D]:
    """This subroutine uses the Weiss-Norman (1985, Agric. forest Meteorol. 34: 205-213)
       routine tocompute direct and diffuse PAR from total par

    Args:
        solar_sine_beta (Float_0D): _description_
        rglobal (Float_0D): _description_
        parin (Float_0D): _description_
        press_kpa (Float_0D): _description_

    Returns:
        Tuple[Float_0D, Float_0D, Float_0D, Float_0D, Float_0D]: _description_
    """
    ratrad, par_beam, par_diffuse, nir_beam, nir_diffuse = jax.lax.cond(
        parin != 0.0,
        diffuse_direct_radiation_day,
        diffuse_direct_radiation_night,
        solar_sine_beta,
        rglobal,
        parin,
        press_kpa,
    )
    return ratrad, par_beam, par_diffuse, nir_beam, nir_diffuse


def diffuse_direct_radiation_night(
    solar_sine_beta: Float_0D, rglobal: Float_0D, parin: Float_0D, press_kpa: Float_0D
) -> Tuple[Float_0D, Float_0D, Float_0D, Float_0D, Float_0D]:
    ratrad, par_beam, par_diffuse = 0.0, 0.0, 0.0
    nir_beam, nir_diffuse = 0.0, 0.0
    return ratrad, par_beam, par_diffuse, nir_beam, nir_diffuse


def diffuse_direct_radiation_day(
    solar_sine_beta: Float_0D, rglobal: Float_0D, parin: Float_0D, press_kpa: Float_0D
) -> Tuple[Float_0D, Float_0D, Float_0D, Float_0D, Float_0D]:
    # visible direct and potential diffuse PAR
    ru = press_kpa / (101.3 * solar_sine_beta)
    rdvis = 624.0 * jnp.exp(-0.185 * ru) * solar_sine_beta
    rsvis = 0.4 * (624.0 * solar_sine_beta - rdvis)

    # water absorption in NIR for 10 mm precip water
    wa = 1373.0 * 0.077 * jnp.power((2.0 * ru), 0.3)

    # direct beam and potential diffuse NIR
    rdir = (748.0 * jnp.exp(-0.06 * ru) - wa) * solar_sine_beta
    rdir = jnp.maximum(0.0, rdir)
    rsdir = 0.6 * (748.0 - rdvis / solar_sine_beta - wa) * solar_sine_beta
    rsdir = jnp.maximum(0.0, rsdir)

    rvt = rdvis + rsvis
    rit = rdir + rsdir
    rvt = jnp.maximum(0.1, rvt)
    rit = jnp.maximum(0.1, rit)

    ratrad = rglobal / (rvt + rit)
    ratrad = jnp.maximum(0.22, ratrad)
    ratrad = jnp.minimum(0.89, ratrad)

    # ratio is the ratio between observed and potential radiation
    # NIR flux density as a function of PAR
    # since NIR is used in energy balance calculations
    # convert it to W m-2: divide PAR by 4.6
    nirx = rglobal - (parin / 4.6)

    # fraction PAR direct and diffuse
    xvalue = (0.9 - ratrad) / 0.70
    fvsb = rdvis / rvt * (1.0 - jnp.power(xvalue, 0.67))
    fvsb = jnp.maximum(0.0, fvsb)
    fvsb = jnp.minimum(1.0, fvsb)
    # fvd = 1. - fvsb
    # note PAR has been entered in units of uE m-2 s-1
    def cond1_par():
        return fvsb * parin

    def cond2_par():
        return 0.001

    par_beam = jax.lax.cond(
        parin != 0,
        cond1_par,
        cond2_par,
    )
    par_beam = jnp.maximum(0.0, par_beam)
    par_diffuse = parin - par_beam

    # NIR beam and diffuse flux densities
    xvalue = (0.9 - ratrad) / 0.68
    fansb = rdir / rit * (1.0 - jnp.power(xvalue, 0.67))
    fansb = jnp.maximum(0.0, fansb)
    fansb = jnp.minimum(1.0, fansb)
    # fand = 1. - fansb
    def cond1_nir():
        return fansb * nirx

    def cond2_nir():
        return 0.1

    nir_beam = jax.lax.cond(
        nirx != 0,
        cond1_nir,
        cond2_nir,
    )
    nir_beam = jnp.maximum(0.0, nir_beam)
    nir_diffuse = nirx - nir_beam
    nir_beam = nirx - nir_diffuse

    return ratrad, par_beam, par_diffuse, nir_beam, nir_diffuse
