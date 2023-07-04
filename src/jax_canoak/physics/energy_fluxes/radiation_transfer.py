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
- nir_day()
- nir_night()
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

from ...physics.carbon_fluxes import freq

from ...shared_utilities.types import Float_0D, Float_1D
from ...shared_utilities.constants import ep, markov, sigma
from ...shared_utilities.constants import epsigma, epsoil, epm1
from ...shared_utilities.constants import PI180, PI9, PI2


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


def nir(
    solar_sine_beta: Float_0D,
    nir_beam: Float_0D,
    nir_diffuse: Float_0D,
    nir_reflect: Float_0D,
    nir_trans: Float_0D,
    nir_soil_refl: Float_0D,
    nir_absorbed: Float_0D,
    dLAIdz: Float_1D,
    exxpdir: Float_1D,
    Gfunc_solar: Float_1D,
) -> Tuple[Float_0D, Float_0D, Float_0D, Float_0D, Float_0D,]:
    nir_total = nir_beam + nir_diffuse
    (nir_dn, nir_up, beam_flux_nir, nir_sh, nir_sun,) = jax.lax.cond(
        (nir_total > 1.0) & (solar_sine_beta != 0.0),
        # solar_sine_beta != 0.0,
        nir_day,
        nir_night,
        solar_sine_beta,
        nir_beam,
        nir_diffuse,
        nir_reflect,
        nir_trans,
        nir_soil_refl,
        nir_absorbed,
        dLAIdz,
        exxpdir,
        Gfunc_solar,
    )
    return (
        nir_dn,
        nir_up,
        beam_flux_nir,
        nir_sh,
        nir_sun,
    )


def nir_day(
    solar_sine_beta: Float_0D,
    nir_beam: Float_0D,
    nir_diffuse: Float_0D,
    nir_reflect: Float_0D,
    nir_trans: Float_0D,
    nir_soil_refl: Float_0D,
    nir_absorbed: Float_0D,
    dLAIdz: Float_1D,
    exxpdir: Float_1D,
    Gfunc_solar: Float_1D,
) -> Tuple[Float_0D, Float_0D, Float_0D, Float_0D, Float_0D,]:
    # Level 1 is the soil surface and level jktot is the
    # top of the canopy.  layer 1 is the layer above
    # the soil and layer jtot is the top layer.
    sze = dLAIdz.size
    jtot = sze - 2
    jktot = jtot + 1

    sup, sdn, adum = jnp.zeros(sze), jnp.zeros(sze), jnp.zeros(sze)
    nir_dn, nir_up = jnp.zeros(sze), jnp.zeros(sze)
    nir_sh, nir_sun = jnp.zeros(sze), jnp.zeros(sze)
    beam_flux_nir = jnp.zeros(sze)

    fraction_beam = nir_beam / (nir_beam + nir_diffuse)
    beam = jnp.zeros(jktot) + fraction_beam
    # tbeam = jnp.zeros(jktot) + fraction_beam
    sumlai = jnp.zeros(sze)
    sumlai = sumlai.at[:jtot].set(jnp.cumsum(dLAIdz[::-1][2:])[::-1])

    # Compute probability of penetration for direct and
    # diffuse radiation for each layer in the canopy

    # Diffuse radiation reflected by layer
    reflectance_layer = (1.0 - exxpdir) * nir_reflect

    # Diffuse radiation transmitted through layer
    transmission_layer = (1.0 - exxpdir) * nir_trans + exxpdir

    # Compute the probability of beam penetration
    exp_direct = jnp.exp(-dLAIdz * markov * Gfunc_solar / solar_sine_beta)

    # probability of beam
    def update_beam(carry, i):
        # carry_new = carry * jnp.power(exp_direct[jtot-1-i], i)
        carry_new = carry * exp_direct[jtot - 1 - i]
        return carry_new, carry_new

    _, beam_update = jax.lax.scan(
        f=update_beam, init=beam[jktot - 1], xs=jnp.arange(jtot)
    )
    beam = beam.at[:-1].set(beam_update[::-1])
    tbeam = beam

    # beam PAR that is reflected upward by a layer
    sup = sup.at[1:-1].set(tbeam[1:] - tbeam[:-1]) * nir_reflect

    # beam PAR that is transmitted downward
    sdn = sdn.at[:-2].set(tbeam[1:] - tbeam[:-1]) * nir_trans

    # Initiate scattering using the technique of NORMAN (1979).
    # scattering is computed using an iterative technique.
    # Here Adum is the ratio up/down diffuse radiation.
    sup = sup.at[0].set(tbeam[0] * nir_soil_refl)
    nir_dn = nir_dn.at[jktot - 1].set(1.0 - fraction_beam)
    adum = adum.at[0].set(nir_soil_refl)
    tlay2 = transmission_layer * transmission_layer

    def update_adum(carry, i):
        carry_new = (
            carry * tlay2[i] / (1 - carry * reflectance_layer[i]) + reflectance_layer[i]
        )
        return carry_new, carry_new

    _, adum_update = jax.lax.scan(update_adum, adum[0], jnp.arange(jtot))
    adum = adum.at[1:-1].set(adum_update)

    def update_nird(carry, i):
        carry_new = (
            carry
            * transmission_layer[i - 1]
            / (1.0 - adum[i] * reflectance_layer[i - 1])
            + sdn[i - 1]
        )
        return carry_new, carry_new

    _, nird_update = jax.lax.scan(
        update_nird, nir_dn[jktot - 1], jnp.arange(1, jktot)[::-1]
    )
    nir_dn = nir_dn.at[:-2].set(nird_update[::-1])
    nir_up = adum * nir_dn + sup
    nir_up = nir_up.at[0].set(nir_soil_refl * nir_dn[0] + sup[0])

    # Iterative calculation of upward diffuse and downward beam +
    # diffuse PAR.
    def update_nirupdown(carry, i):
        nir_up, nir_dn = carry[0], carry[1]
        # downward --
        def calculate_down(c, j):
            c_new = (
                transmission_layer[j] * c + nir_up[j] * reflectance_layer[j] + sdn[j]
            )
            return c_new, c_new

        _, down = jax.lax.scan(
            f=calculate_down, init=nir_dn[jktot - 1], xs=jnp.arange(jtot)[::-1]
        )
        nir_dn = nir_dn.at[:jtot].set(down[::-1])
        # upward --
        nir_up = nir_up.at[0].set((nir_dn[0] + tbeam[0]) * nir_soil_refl)

        def calculate_up(c, j):
            c_new = (
                reflectance_layer[j] * nir_dn[j + 1]
                + c * transmission_layer[j]
                + sup[j + 1]
            )
            return c_new, c_new

        _, up = jax.lax.scan(f=calculate_up, init=nir_up[0], xs=jnp.arange(jtot))
        nir_up = nir_up.at[1:jktot].set(up)
        carry_new = [nir_up, nir_dn]
        return carry_new, carry_new

    carry_new, _ = jax.lax.scan(update_nirupdown, [nir_up, nir_dn], jnp.arange(5))
    nir_up, nir_dn = carry_new[0], carry_new[1]

    # Compute flux density of PAR
    nir_total = nir_beam + nir_diffuse
    nir_up = nir_up * nir_total
    nir_up = jnp.clip(nir_up, a_min=0.001)
    nir_up = nir_up.at[jktot:].set(0)
    beam_flux_nir = beam_flux_nir.at[:-1].set(beam * nir_total)
    beam_flux_nir = jnp.clip(beam_flux_nir, a_min=0.001)
    beam_flux_nir = beam_flux_nir.at[jktot:].set(0)
    nir_dn = nir_dn * nir_total
    nir_dn = jnp.clip(nir_dn, a_min=0.001)
    nir_dn = nir_dn.at[jktot:].set(0)

    # PSUN is the radiation incident on the mean leaf normal
    nir_normal = nir_beam * Gfunc_solar / solar_sine_beta
    nsunen = nir_normal * nir_absorbed
    nir_sh = nir_dn + nir_up
    nir_sh = nir_sh * nir_absorbed
    nir_sun = nsunen + nir_sh

    # jax.debug.print('nir_normal: {x}', x=nir_normal)
    # jax.debug.print('nir_beam: {x}', x=nir_beam)

    nir_sh = nir_sh.at[jtot:].set(0)
    nir_sun = nir_sun.at[jtot:].set(0)

    return (
        nir_dn,
        nir_up,
        beam_flux_nir,
        nir_sh,
        nir_sun,
    )


def nir_night(
    solar_sine_beta: Float_0D,
    nir_beam: Float_0D,
    nir_diffuse: Float_0D,
    nir_reflect: Float_0D,
    nir_trans: Float_0D,
    nir_soil_refl: Float_0D,
    nir_absorbed: Float_0D,
    dLAIdz: Float_1D,
    exxpdir: Float_1D,
    Gfunc_solar: Float_1D,
) -> Tuple[Float_0D, Float_0D, Float_0D, Float_0D, Float_0D,]:
    sze = dLAIdz.size
    nir_dn, nir_up = jnp.zeros(sze), jnp.zeros(sze)
    nir_sh, nir_sun = jnp.zeros(sze), jnp.zeros(sze)
    beam_flux_nir = jnp.zeros(sze)

    return (
        nir_dn,
        nir_up,
        beam_flux_nir,
        nir_sh,
        nir_sun,
    )


def sky_ir(T: Float_0D, ratrad: Float_0D) -> Float_0D:
    """Infrared radiation from sky, W m-2, using algorithm from Norman.

    Args:
        T (Float_0D): _description_
        ratrad (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    y = (
        sigma
        * jnp.power(T, 4.0)
        * (
            (1.0 - 0.261 * jnp.exp(-0.000777 * jnp.power((273.16 - T), 2.0))) * ratrad
            + 1
            - ratrad
        )
    )
    return y


def irflux(
    T_Kelvin: Float_0D,
    ratrad: Float_0D,
    sfc_temperature: Float_0D,
    exxpdir: Float_1D,
    sun_T_filter: Float_1D,
    shd_T_filter: Float_1D,
    prob_beam: Float_1D,
    prob_sh: Float_1D,
) -> Tuple[Float_1D, Float_1D]:
    """Compute probability of penetration for diffuse
       radiation for each layer in the canopy .
       IR radiation is isotropic.

    Args:
        T_Kelvin (Float_0D): _description_
        ratrad (Float_0D): _description_
        sfc_temperature (Float_0D): _description_
        exxpdir (Float_1D): _description_
        sun_T_filter (Float_1D): _description_
        shd_T_filter (Float_1D): _description_
        prob_beam (Float_1D): _description_
        prob_sh (Float_1D): _description_

    Returns:
        Tuple[Float_1D, Float_1D]: _description_
    """
    # Level 1 is the soil surface and level jktot is the
    # top of the canopy.  layer 1 is the layer above
    # the soil and layer jtot is the top layer.
    sze = exxpdir.size
    jtot, jktot = sze - 2, sze - 1
    SDN, SUP = jnp.zeros(sze), jnp.zeros(sze)
    ir_dn, ir_up = jnp.zeros(sze), jnp.zeros(sze)

    ir_in = sky_ir(T_Kelvin, ratrad)
    ir_dn = ir_dn.at[jktot - 1].set(ir_in)

    # Loop from layers jtot to 1
    #
    # Integrated probability of diffuse sky radiation penetration
    # EXPDIF[JJ] is computed in RAD
    # compute IR radiative source flux as a function of
    # leaf temperature weighted according to sunlit and shaded fractions
    # source=ep*sigma*(laisun*tksun^4 + laish*tksh^4)
    # remember energy balance is done on layers not levels.
    # so level jtot+1 must use tl from layer jtot
    Tk_sun_flit = sun_T_filter + 273.16
    Tk_shade_flit = shd_T_filter + 273.16
    IR_source_sun = prob_beam * jnp.power(Tk_sun_flit, 4.0)
    IR_source_shade = prob_sh * jnp.power(Tk_shade_flit, 4.0)
    IR_source = epsigma * (IR_source_sun + IR_source_shade)
    SDN = IR_source * (1 - exxpdir)
    SUP = SUP.at[1:jktot].set(IR_source[:jtot] * (1 - exxpdir[:jtot]))

    # Downward IR radiation, sum of that from upper layer that is transmitted
    # and the downward source generated in the upper layer.
    # REMEMBER LEVEL JJ IS AFFECTED BY temperature OF LAYER ABOVE WHICH IS JJ
    def update_ird(carry, i):
        carry_new = carry * exxpdir[i] + SDN[i]
        return carry_new, carry_new

    _, ird_update = jax.lax.scan(update_ird, ir_dn[jktot - 1], jnp.arange(jtot)[::-1])
    ir_dn = ir_dn.at[:-2].set(ird_update[::-1])
    emiss_IR_soil = epsigma * jnp.power((sfc_temperature + 273.16), 4.0)
    SUP = SUP.at[0].set(ir_dn[0] * (1.0 - epsoil))
    ir_up = ir_up.at[0].set(emiss_IR_soil + SUP[0])

    def update_iru(carry, i):
        carry_new = carry * exxpdir[i - 1] + SUP[i]
        return carry_new, carry_new

    _, iru_update = jax.lax.scan(update_iru, ir_up[0], jnp.arange(1, jktot))
    ir_up = ir_up.at[1:jktot].set(iru_update)

    # jax.debug.print("ir_up: {x}", x=ir_up)
    # jax.debug.print("ir_dn: {x}", x=ir_dn)

    # Iterative calculation of upward and downward IR
    def update_irupdown(carry, i):
        ir_up, ir_dn, SUP = carry[0], carry[1], carry[2]
        # downward --
        def calculate_down(c, j):
            # carry_new = carry * jnp.power(exp_direct[jtot-1-i], i)
            reflc_lay_ir = (1 - exxpdir[j]) * epm1
            c_new = exxpdir[j] * c + ir_up[j] * reflc_lay_ir + SDN[j]
            return c_new, c_new

        _, down = jax.lax.scan(
            f=calculate_down, init=ir_dn[jktot - 1], xs=jnp.arange(jtot)[::-1]
        )
        ir_dn = ir_dn.at[:jtot].set(down[::-1])

        # upward --
        SUP = SUP.at[0].set(ir_dn[0] * (1.0 - epsoil))
        ir_up = ir_up.at[0].set(emiss_IR_soil + SUP[0])

        def calculate_up(c, j):
            reflec_lay_ir = (1 - exxpdir[j - 1]) * epm1
            c_new = reflec_lay_ir * ir_dn[j] + c * exxpdir[j - 1] + SUP[j]
            return c_new, c_new

        _, up = jax.lax.scan(f=calculate_up, init=ir_up[0], xs=jnp.arange(1, jktot))
        ir_up = ir_up.at[1:jktot].set(up)
        carry_new = [ir_up, ir_dn, SUP]
        return carry_new, carry_new

    carry_new, _ = jax.lax.scan(update_irupdown, [ir_up, ir_dn, SUP], jnp.arange(2))
    ir_up, ir_dn = carry_new[0], carry_new[1]

    return ir_up, ir_dn


def g_func_diffuse(dLAIdz: Float_1D) -> Float_1D:
    """computes the G Function according to
       the algorithms of Lemeur (1973, Agric. Meteorol. 12: 229-247).

    Args:
        dLAIdz (Float_1D): _description_

    Returns:
        Float_1D: _description_
    """

    sze = dLAIdz.size
    jtot = sze - 2
    nsize = 18
    aden, TT = jnp.zeros(nsize), jnp.zeros(nsize)
    PGF, sin_TT = jnp.zeros(nsize), jnp.zeros(nsize)
    del_TT, del_sin = jnp.zeros(nsize), jnp.zeros(nsize)

    Gfunc_sky = jnp.zeros([sze, 19])

    llai = jnp.zeros(sze)
    llai = llai.at[:jtot].set(jnp.cumsum(dLAIdz[::-1][2:])[::-1])
    # llai = llai.at[:jtot].set(jnp.cumsum(dLAIdz))

    ang = 5.0 * PI180
    dang = 2.0 * ang

    # Midpoints for azimuth intervals
    aden = aden.at[:16].set(0.0625)

    K = 2 * jnp.arange(1, nsize) - 3
    TT = TT.at[:-1].set(0.1963 * K)
    sin_TT = sin_TT.at[:-1].set(jnp.sin(TT[:-1]))
    del_TT = del_TT.at[: nsize - 2].set(TT[1 : nsize - 1] - TT[: nsize - 2])
    del_sin = del_sin.at[: nsize - 2].set(sin_TT[1 : nsize - 1] - sin_TT[: nsize - 2])

    # jax.debug.print("TT: {x}.", x=TT)
    # jax.debug.print("sin_TT: {x}.", x=sin_TT)
    # jax.debug.print("del_TT: {x}.", x=del_TT)
    # jax.debug.print("del_sin: {x}.", x=del_sin)

    # Compute G function for each layer in the canopy
    def calculate_g_row(c, i):
        ang = c
        bang = ang
        # c_new = c + dLAIdz[ii]
        # bdens = freq(c_new)

        def calculate_g_each(c1, j):
            bang = c1
            ii = jtot - j - 1
            bdens = freq(llai[ii])
            PPP = 0.0

            def calculate_ppp(c2, k):
                PPP, PGF, bang = c2[0], c2[1], c2[2]
                aang = (k * 10.0 + 5.0) * PI180
                cos_B, sin_B = jnp.cos(bang), jnp.sin(bang)
                cos_A, sin_A = jnp.cos(aang), jnp.sin(aang)
                X, Y = cos_A * sin_B, sin_A * cos_B

                def calculate_pgf_1(PGF):
                    # jax.debug.print("-- {x} {y} {z}", z=aang-bang, y=aang, x=bang)
                    PGF = PGF.at[: nsize - 2].set(
                        X * del_TT[: nsize - 2] + Y * del_sin[: nsize - 2]
                    )
                    return PGF

                def calculate_pgf_2(PGF):
                    # jax.debug.print("aang-bang: {x}.", x=aang-bang)
                    T0, TII = 1.0 + X / Y, 1.0 - X / Y
                    square = jax.lax.cond(
                        T0 / TII > 0.0, lambda x: jnp.sqrt(x), lambda x: 0.0, T0 / TII
                    )
                    TT0 = 2.0 * jnp.arctan(square)
                    sin_TT0 = jnp.sin(TT0)
                    TT1 = PI2 - TT0
                    sin_TT1 = jnp.sin(TT1)
                    # jax.debug.print("-- {x} {y} {z}", z=aang-bang, y=aang, x=bang)

                    def calculate_pgf_2a(c, h):
                        h1 = h + 1
                        conds = jnp.array(
                            [
                                TT[h1] - TT0 <= 0,
                                (TT[h1] - TT0 > 0)
                                & (TT[h1] - TT1 <= 0)
                                & (TT0 - TT[h] <= 0),
                                (TT[h1] - TT0 > 0)
                                & (TT[h1] - TT1 <= 0)
                                & (TT0 - TT[h] > 0),
                                (TT[h1] - TT0 > 0)
                                & (TT[h1] - TT1 > 0)
                                & (TT1 - TT[h] <= 0),
                                (TT[h1] - TT0 > 0)
                                & (TT[h1] - TT1 > 0)
                                & (TT1 - TT[h] > 0),
                            ]
                        )
                        index = jnp.where(conds, size=1)[0][0]
                        PGF_each = jax.lax.switch(
                            index,
                            [
                                lambda: X * del_TT[h] + Y * del_sin[h],
                                lambda: -X * del_TT[h] - Y * del_sin[h],
                                lambda: (X * (TT0 - TT[h]) + Y * (sin_TT0 - sin_TT[h]))
                                - (X * (TT[h1] - TT0) + Y * (sin_TT[h1] - sin_TT0)),
                                lambda: X * del_TT[h] + Y * del_sin[h],
                                lambda: -(X * (TT1 - TT[h]) + Y * (sin_TT1 - sin_TT[h]))
                                + (X * (TT[h1] - TT1) + Y * (sin_TT[h1] - sin_TT1)),
                            ],
                        )
                        return c, PGF_each

                    _, PGF_update = jax.lax.scan(
                        calculate_pgf_2a, None, jnp.arange(nsize - 2)
                    )
                    PGF = PGF.at[: nsize - 2].set(PGF_update)
                    # jax.debug.print("PGF_update: {x}.", x=PGF_update)
                    return PGF

                PGF = jax.lax.cond(
                    # aang - bang<=0.0,
                    # jnp.round(aang,7) - jnp.round(bang,7)<=0.0,
                    (aang - bang < 0) | (jnp.isclose(aang, bang)),
                    calculate_pgf_1,
                    calculate_pgf_2,
                    PGF,
                )

                # Compute the integrated leaf orientation function, G
                PP = jnp.sum(PGF[: nsize - 2] * aden[: nsize - 2])  # type: ignore
                PPP += PP * bdens[k] * PI9  # type: ignore
                # jax.debug.print("{x} {y} {z}", x=aang-bang<=0.0, y=bang, z=aang)
                # jax.debug.print("aang-bang: {x}.", x=aang-bang)
                # jax.debug.print("PGF: {x}.", x=PGF)
                # jax.debug.print("PGF*aden: {x}.", x=PGF[:nsize-2] * aden[:nsize-2])
                # jax.debug.print("PP: {x}.", x=PP)

                return [PPP, PGF, bang], None

            c2_new, _ = jax.lax.scan(calculate_ppp, [PPP, PGF, bang], jnp.arange(9))
            PPP = c2_new[0]
            # jax.debug.print("PPP: {x}.", x=PPP)

            G = PPP
            c1_new = c2_new[2]
            return c1_new, G

        # _, G_row = jax.lax.scan(calculate_g_each, PPP, jnp.arange(9))
        _, G_row = jax.lax.scan(calculate_g_each, bang, jnp.arange(jtot))

        # jax.debug.print("G_row: {x}.", x=G_row)

        ang += dang
        c_new = ang
        return c_new, G_row

    _, Gfunc_sky_update = jax.lax.scan(calculate_g_row, ang, jnp.arange(9))

    Gfunc_sky = Gfunc_sky.at[:jtot, :9].set(Gfunc_sky_update.T)

    return Gfunc_sky
