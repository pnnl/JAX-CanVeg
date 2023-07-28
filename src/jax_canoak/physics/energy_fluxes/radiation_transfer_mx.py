"""
Radiation transfer functions, including:
- diffuse_direct_radiation()
- diffuse_direct_radiation_day()
- diffuse_direct_radiation_night()

Author: Peishi Jiang
Date: 2023.07.27.
"""

import jax
import jax.numpy as jnp

# from jax.scipy.special import gamma

from typing import Tuple

from ...shared_utilities.types import Float_0D, Float_1D
from ...shared_utilities.constants import markov, sigma
from ...shared_utilities.constants import epsigma, epsoil, epm1
from ...shared_utilities.constants import PI180, PI9, PI2


# @jax.jit
def diffuse_direct_radiation(
    solar_sine_beta: Float_1D, rglobal: Float_1D, parin: Float_1D, press_kpa: Float_1D
) -> Tuple[Float_1D, Float_1D, Float_1D, Float_1D, Float_1D]:
    solar_constant = 1360.8  # [W m-2]

    # Distribution of extraterrestrial solar radiation
    fnir, fvis = 0.517, 0.383
    # fuv, fnir, fvis = 0.087, 0.517, 0.383

    # visible direct and potential diffuse PAR
    ru = press_kpa / (101.3 * solar_sine_beta)
    ru = ru.at[ru <= 0].set(jnp.nan)  # for negative sun angles getting complex numbers
    rdvis = solar_constant * fvis * jnp.exp(-0.185 * ru) * solar_sine_beta
    rsvis = 0.4 * (solar_constant * fvis * solar_sine_beta - rdvis)

    # water absorption in NIR for 10 mm precip water
    wa = solar_constant * 0.077 * jnp.power((2.0 * ru), 0.3)

    # direct beam and potential diffuse NIR
    nircoef = solar_constant * fnir
    rdir = (nircoef * jnp.exp(-0.06 * ru) - wa) * solar_sine_beta
    rdir = rdir.at[jnp.isnan(rdir)].set(0.0)
    # rdir = jnp.maximum(0.0, rdir)
    rsdir = 0.6 * (nircoef - rdvis / solar_sine_beta - wa) * solar_sine_beta
    rsdir = rsdir.at[jnp.isnan(rsdir)].set(0.0)
    # rsdir = jnp.maximum(0.0, rsdir)

    rvt = rdvis + rsvis
    rit = rdir + rsdir
    rvt = rvt.at[jnp.isnan(rvt)].set(0.0)
    rit = rit.at[jnp.isnan(rit)].set(0.0)
    # rvt = jnp.maximum(0.1, rvt)
    # rit = jnp.maximum(0.1, rit)

    ratrad = rglobal / (rvt + rit)
    ratrad = ratrad.at[jnp.isnan(ratrad)].set(1.0)
    ratrad = ratrad.at[jnp.isinf(ratrad)].set(1.0)
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
    par_beam = par_beam.at[par_beam < 0].set(0.0)
    par_diffuse = parin - par_beam
    par_beam = par_beam.at[parin == 0].set(0.001)
    par_diffuse = par_diffuse.at[parin == 0].set(0.001)

    # NIR beam and diffuse flux densities
    xvalue = (0.9 - ratrad) / 0.68
    fansb = rdir / rit * (1.0 - jnp.power(xvalue, 0.67))
    fansb = jnp.clip(fansb, a_min=0.0, a_max=1.0)
    # fand = 1. - fansb
    nir_beam = fansb * nirx
    nir_beam = nir_beam.at[nir_beam < 0].set(0.0)
    nir_diffuse = nirx - nir_beam
    nir_beam = nir_beam.at[nirx == 0].set(0.1)
    nir_diffuse = nir_diffuse.at[nirx == 0].set(0.1)
    nir_diffuse = nirx - nir_beam
    nir_beam = nirx - nir_diffuse

    # Check nan
    par_beam = par_beam.at[jnp.isnan(par_beam)].set(0.0)
    par_diffuse = par_diffuse.at[jnp.isnan(par_diffuse)].set(0.0)
    nir_beam = nir_beam.at[jnp.isnan(nir_beam)].set(0.0)
    nir_diffuse = nir_diffuse.at[jnp.isnan(nir_diffuse)].set(0.0)

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
    # sumlai = jnp.zeros(sze)
    # sumlai = sumlai.at[:jtot].set(jnp.cumsum(dLAIdz[::-1][2:])[::-1])
    # sumlai = jnp.concatenate(
    #     [jax.lax.cumsum(dLAIdz[:jtot], reverse=True), jnp.zeros(sze - jtot)]
    # )

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
    # beam = beam.at[:-1].set(beam_update[::-1])
    beam = jnp.concatenate([beam_update[::-1], beam[-1:]])
    tbeam = beam

    # beam PAR that is reflected upward by a layer
    sup = jnp.concatenate(
        [
            jnp.array([tbeam[0] * nir_soil_refl]),
            (tbeam[1:] - tbeam[:-1]) * nir_reflect,
            sup[-1:] * nir_reflect,
        ]
    )
    # sup = sup.at[1:-1].set(tbeam[1:] - tbeam[:-1]) * nir_reflect

    # beam PAR that is transmitted downward
    sdn = jnp.concatenate([tbeam[1:] - tbeam[:-1], sdn[-2:]]) * nir_trans
    # sdn = sdn.at[:-2].set(tbeam[1:] - tbeam[:-1]) * nir_trans

    # Initiate scattering using the technique of NORMAN (1979).
    # scattering is computed using an iterative technique.
    # Here Adum is the ratio up/down diffuse radiation.
    # sup = sup.at[0].set(tbeam[0] * nir_soil_refl)
    # nir_dn = nir_dn.at[jktot - 1].set(1.0 - fraction_beam)
    # adum = adum.at[0].set(nir_soil_refl)
    nir_dn = jnp.concatenate(
        [nir_dn[:-2], jnp.array([1.0 - fraction_beam, nir_dn[-1]])]
    )
    adum = jnp.concatenate([jnp.array([nir_soil_refl]), adum[1:]])
    tlay2 = transmission_layer * transmission_layer

    def update_adum(carry, i):
        carry_new = (
            carry * tlay2[i] / (1 - carry * reflectance_layer[i]) + reflectance_layer[i]
        )
        return carry_new, carry_new

    _, adum_update = jax.lax.scan(update_adum, adum[0], jnp.arange(jtot))
    adum = jnp.concatenate([adum[:1], adum_update, adum[-1:]])
    # adum = adum.at[1:-1].set(adum_update)

    def update_nird(carry, i):
        carry_new = (
            carry
            * transmission_layer[i - 1]
            / (1.0 - adum[i] * reflectance_layer[i - 1])
            + sdn[i - 1]
        )
        return carry_new, carry_new

    _, nird_update = jax.lax.scan(
        # update_nird, nir_dn[jktot - 1], jnp.arange(1, jktot)[::-1]
        update_nird,
        nir_dn[jktot - 1],
        jnp.arange(jktot - 1, 0, -1),
    )
    # nir_dn = nir_dn.at[:-2].set(nird_update[::-1])
    nir_dn = jnp.concatenate([nird_update[::-1], nir_dn[-2:]])
    nir_up = adum * nir_dn + sup
    nir_up = jnp.concatenate(
        [jnp.array([nir_soil_refl * nir_dn[0] + sup[0]]), nir_up[1:]]
    )
    # nir_up = nir_up.at[0].set(nir_soil_refl * nir_dn[0] + sup[0])

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
            f=calculate_down,
            init=nir_dn[jktot - 1],
            xs=jnp.arange(jtot - 1, -1, -1)
            # f=calculate_down, init=nir_dn[jktot - 1], xs=jnp.arange(jtot)[::-1]
        )
        # nir_dn = nir_dn.at[:jtot].set(down[::-1])
        nir_dn = jnp.concatenate([down[::-1], nir_dn[-2:]])
        # upward --
        # nir_up = nir_up.at[0].set((nir_dn[0] + tbeam[0]) * nir_soil_refl)
        nir_up = jnp.concatenate(
            [jnp.array([(nir_dn[0] + tbeam[0]) * nir_soil_refl]), nir_up[1:]]
        )

        def calculate_up(c, j):
            c_new = (
                reflectance_layer[j] * nir_dn[j + 1]
                + c * transmission_layer[j]
                + sup[j + 1]
            )
            return c_new, c_new

        _, up = jax.lax.scan(f=calculate_up, init=nir_up[0], xs=jnp.arange(jtot))
        # nir_up = nir_up.at[1:jktot].set(up)
        nir_up = jnp.concatenate([nir_up[:1], up, nir_up[-1:]])
        carry_new = [nir_up, nir_dn]
        return carry_new, carry_new

    carry_new, _ = jax.lax.scan(update_nirupdown, [nir_up, nir_dn], jnp.arange(5))
    nir_up, nir_dn = carry_new[0], carry_new[1]

    # Compute flux density of PAR
    nir_total = nir_beam + nir_diffuse
    nir_up = nir_up * nir_total
    nir_up = jnp.clip(nir_up, a_min=0.001)
    # nir_up = nir_up.at[jktot:].set(0)
    # beam_flux_nir = beam_flux_nir.at[:-1].set(beam * nir_total)
    nir_up = jnp.concatenate([nir_up[:jktot], jnp.array([0])])
    beam_flux_nir = jnp.concatenate([beam * nir_total, beam_flux_nir[-1:]])
    beam_flux_nir = jnp.clip(beam_flux_nir, a_min=0.001)
    # beam_flux_nir = beam_flux_nir.at[jktot:].set(0)
    beam_flux_nir = jnp.concatenate([beam_flux_nir[:jktot], jnp.array([0])])
    nir_dn = nir_dn * nir_total
    nir_dn = jnp.clip(nir_dn, a_min=0.001)
    nir_dn = jnp.concatenate([nir_dn[:jktot], jnp.array([0])])
    # nir_dn = nir_dn.at[jktot:].set(0)

    # PSUN is the radiation incident on the mean leaf normal
    nir_normal = nir_beam * Gfunc_solar / solar_sine_beta
    nsunen = nir_normal * nir_absorbed
    nir_sh = nir_dn + nir_up
    nir_sh = nir_sh * nir_absorbed
    nir_sun = nsunen + nir_sh

    # jax.debug.print('nir_normal: {x}', x=nir_normal)
    # jax.debug.print('nir_beam: {x}', x=nir_beam)

    nir_sh = jnp.concatenate([nir_sh[:jtot], jnp.zeros(sze - jtot)])
    nir_sun = jnp.concatenate([nir_sun[:jtot], jnp.zeros(sze - jtot)])
    # nir_sh = nir_sh.at[jtot:].set(0)
    # nir_sun = nir_sun.at[jtot:].set(0)

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
    # ir_dn = ir_dn.at[jktot - 1].set(ir_in)
    ir_dn = jnp.concatenate([ir_dn[:jtot], jnp.array([ir_in, ir_dn[-1]])])

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
    SUP = jnp.concatenate([SUP[:1], IR_source[:jtot] * (1 - exxpdir[:jtot]), SUP[-1:]])
    # SUP = SUP.at[1:jktot].set(IR_source[:jtot] * (1 - exxpdir[:jtot]))

    # Downward IR radiation, sum of that from upper layer that is transmitted
    # and the downward source generated in the upper layer.
    # REMEMBER LEVEL JJ IS AFFECTED BY temperature OF LAYER ABOVE WHICH IS JJ
    def update_ird(carry, i):
        carry_new = carry * exxpdir[i] + SDN[i]
        return carry_new, carry_new

    _, ird_update = jax.lax.scan(
        update_ird, ir_dn[jktot - 1], jnp.arange(jtot - 1, -1, -1)
    )
    # _, ird_update = jax.lax.scan(update_ird, ir_dn[jktot - 1], jnp.arange(jtot)[::-1])
    # ir_dn = ir_dn.at[:-2].set(ird_update[::-1])
    ir_dn = jnp.concatenate([ird_update[::-1], ir_dn[-2:]])
    emiss_IR_soil = epsigma * jnp.power((sfc_temperature + 273.16), 4.0)
    SUP = jnp.concatenate([jnp.array([ir_dn[0] * (1.0 - epsoil)]), SUP[1:]])
    ir_up = jnp.concatenate([jnp.array([emiss_IR_soil + SUP[0]]), ir_up[1:]])
    # SUP = SUP.at[0].set(ir_dn[0] * (1.0 - epsoil))
    # ir_up = ir_up.at[0].set(emiss_IR_soil + SUP[0])

    def update_iru(carry, i):
        carry_new = carry * exxpdir[i - 1] + SUP[i]
        return carry_new, carry_new

    _, iru_update = jax.lax.scan(update_iru, ir_up[0], jnp.arange(1, jktot))
    ir_up = jnp.concatenate([ir_up[:1], iru_update, ir_up[-1:]])
    # ir_up = ir_up.at[1:jktot].set(iru_update)

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
            f=calculate_down,
            init=ir_dn[jktot - 1],
            xs=jnp.arange(jtot - 1, -1, -1)
            # f=calculate_down, init=ir_dn[jktot - 1], xs=jnp.arange(jtot)[::-1]
        )
        ir_dn = jnp.concatenate([down[::-1], ir_dn[-2:]])
        # ir_dn = ir_dn.at[:jtot].set(down[::-1])

        # upward --
        SUP = jnp.concatenate([jnp.array([ir_dn[0] * (1.0 - epsoil)]), SUP[1:]])
        ir_up = jnp.concatenate([jnp.array([emiss_IR_soil + SUP[0]]), ir_up[1:]])
        # SUP = SUP.at[0].set(ir_dn[0] * (1.0 - epsoil))
        # ir_up = ir_up.at[0].set(emiss_IR_soil + SUP[0])

        def calculate_up(c, j):
            reflec_lay_ir = (1 - exxpdir[j - 1]) * epm1
            c_new = reflec_lay_ir * ir_dn[j] + c * exxpdir[j - 1] + SUP[j]
            return c_new, c_new

        _, up = jax.lax.scan(f=calculate_up, init=ir_up[0], xs=jnp.arange(1, jktot))
        ir_up = jnp.concatenate([ir_up[:1], up, ir_up[-1:]])
        # ir_up = ir_up.at[1:jktot].set(up)
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

    # llai = jnp.zeros(sze)
    # llai = llai.at[:jtot].set(jnp.cumsum(dLAIdz[::-1][2:])[::-1])
    # llai = llai.at[:jtot].set(jnp.cumsum(dLAIdz))
    llai = jnp.concatenate(
        [jax.lax.cumsum(dLAIdz[:jtot], reverse=True), jnp.zeros(sze - jtot)]
    )

    ang = 5.0 * PI180
    dang = 2.0 * ang

    # Midpoints for azimuth intervals
    aden = aden.at[:16].set(0.0625)

    K = 2 * jnp.arange(1, nsize) - 3
    # TT = TT.at[:-1].set(0.1963 * K)
    # sin_TT = sin_TT.at[:-1].set(jnp.sin(TT[:-1]))
    # del_TT = del_TT.at[: nsize - 2].set(TT[1 : nsize - 1] - TT[: nsize - 2])
    # del_sin = del_sin.at[: nsize - 2].set(sin_TT[1 : nsize - 1] - sin_TT[: nsize - 2])
    TT = jnp.concatenate([0.1963 * K, TT[-1:]])
    sin_TT = jnp.concatenate([jnp.sin(TT[:-1]), sin_TT[-1:]])
    del_TT = jnp.concatenate([TT[1 : nsize - 1] - TT[: nsize - 2], del_TT[-1:]])
    del_sin = jnp.concatenate(
        [sin_TT[1 : nsize - 1] - sin_TT[: nsize - 2], del_sin[-1:]]
    )

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
                    # PGF = PGF.at[: nsize - 2].set(
                    #     X * del_TT[: nsize - 2] + Y * del_sin[: nsize - 2]
                    # )
                    PGF = jnp.concatenate(
                        [X * del_TT[: nsize - 2] + Y * del_sin[: nsize - 2], PGF[-2:]]
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
                    # PGF = PGF.at[: nsize - 2].set(PGF_update)
                    PGF = jnp.concatenate([PGF_update, PGF[-2:]])
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

    # Gfunc_sky = Gfunc_sky.at[:jtot, :9].set(Gfunc_sky_update.T)
    Gfunc_sky = jnp.concatenate([Gfunc_sky_update.T, jnp.zeros([sze - jtot, 9])])
    Gfunc_sky = jnp.concatenate([Gfunc_sky, jnp.zeros([sze, 10])], axis=1)

    return Gfunc_sky


def gfunc(solar_beta_rad: Float_0D, dLAIdz: Float_1D) -> Float_1D:
    """This subroutine computes the G function according to the algorithms of:

        Lemeur, R. 1973.  A method for simulating the direct solar
        radiaiton regime of sunflower, Jerusalem artichoke, corn and soybean
        canopies using actual stand structure data. Agricultural Meteorology. 12,229-247

        This progrom computes G for a given sun angle.  G changes with height due
        to change leaf angles.

    Args:
        solar_beta_rad (Float_0D): _description_
        dLAIdz (Float_1D): _description_

    Returns:
        Float_1D: _description_
    """
    sze = dLAIdz.size
    jtot = sze - 2
    nsize = 18
    aden, TT = jnp.zeros(nsize), jnp.zeros(nsize)
    pgg, sin_TT = jnp.zeros(nsize), jnp.zeros(nsize)
    del_TT, del_sin = jnp.zeros(nsize), jnp.zeros(nsize)

    Gfunc_solar = jnp.zeros(sze)

    # llai = jnp.zeros(sze)
    # llai = llai.at[:jtot].set(jnp.cumsum(dLAIdz[::-1][2:])[::-1])
    llai = jnp.concatenate(
        [jax.lax.cumsum(dLAIdz[:jtot], reverse=True), jnp.zeros(sze - jtot)]
    )

    # Midpoint of azimuthal intervals
    aden = aden.at[:16].set(0.0625)
    K = 2 * jnp.arange(1, nsize) - 3
    # TT = TT.at[:-1].set(3.14159265 / 16.0 * K)
    # sin_TT = sin_TT.at[:-1].set(jnp.sin(TT[:-1]))
    # del_TT = del_TT.at[: nsize - 2].set(TT[1 : nsize - 1] - TT[: nsize - 2])
    # del_sin = del_sin.at[: nsize - 2].set(sin_TT[1 : nsize - 1] - sin_TT[: nsize - 2])
    TT = jnp.concatenate([3.14159265 / 16.0 * K, TT[-1:]])
    sin_TT = jnp.concatenate([jnp.sin(TT[:-1]), sin_TT[-1:]])
    del_TT = jnp.concatenate([TT[1 : nsize - 1] - TT[: nsize - 2], del_TT[-1:]])
    del_sin = jnp.concatenate(
        [sin_TT[1 : nsize - 1] - sin_TT[: nsize - 2], del_sin[-1:]]
    )

    # Compute the G function for each layer
    def calculate_g_each(c1, j):
        ii = jtot - j - 1
        bdens = freq(llai[ii])
        PPP = 0.0

        def calculate_ppp(c2, k):
            PPP, pgg = c2[0], c2[1]
            aang = (k * 10.0 + 5.0) * PI180
            cos_B, sin_B = jnp.cos(solar_beta_rad), jnp.sin(solar_beta_rad)
            cos_A, sin_A = jnp.cos(aang), jnp.sin(aang)
            X, Y = cos_A * sin_B, sin_A * cos_B

            def calculate_pgf_1(pgg):
                # jax.debug.print("-- {x} {y} {z}", z=aang-bang, y=aang, x=bang)
                # pgg = pgg.at[: nsize - 2].set(
                #     X * del_TT[: nsize - 2] + Y * del_sin[: nsize - 2]
                # )
                pgg = jnp.concatenate(
                    [X * del_TT[: nsize - 2] + Y * del_sin[: nsize - 2], pgg[-2:]]
                )
                return pgg

            def calculate_pgf_2(pgg):
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
                            (TT[h1] - TT0 > 0) & (TT[h1] - TT1 > 0) & (TT1 - TT[h] > 0),
                        ]
                    )
                    index = jnp.where(conds, size=1)[0][0]
                    pgg_each = jax.lax.switch(
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
                    return c, pgg_each

                _, pgg_update = jax.lax.scan(
                    calculate_pgf_2a, None, jnp.arange(nsize - 2)
                )
                # pgg = pgg.at[: nsize - 2].set(pgg_update)
                pgg = jnp.concatenate([pgg_update, pgg[-2:]])
                return pgg

            pgg = jax.lax.cond(
                (aang - solar_beta_rad < 0) | (jnp.isclose(aang, solar_beta_rad)),
                calculate_pgf_1,
                calculate_pgf_2,
                pgg,
            )

            # Compute the integrated leaf orientation function, G
            PP = jnp.sum(pgg[: nsize - 2] * aden[: nsize - 2])  # type: ignore
            PPP += PP * bdens[k] * PI9  # type: ignore

            return [PPP, pgg], None

        c2_new, _ = jax.lax.scan(calculate_ppp, [PPP, pgg], jnp.arange(9))
        PPP = c2_new[0]

        G = jnp.maximum(PPP, 0.001)
        return c1, G

    # _, G_row = jax.lax.scan(calculate_g_each, PPP, jnp.arange(9))
    _, Gfunc_solar_update = jax.lax.scan(calculate_g_each, None, jnp.arange(jtot))

    # Gfunc_solar = Gfunc_solar.at[:jtot].set(Gfunc_solar_update)
    Gfunc_solar = jnp.concatenate([Gfunc_solar_update, Gfunc_solar[-2:]])

    return Gfunc_solar


def gammaf(x: Float_0D) -> Float_0D:
    """Gamma function.

    Args:
        x (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    gam = (
        (1.0 / (12.0 * x))
        + (1.0 / (288.0 * x * x))
        - (139.0 / (51840.0 * jnp.power(x, 3.0)))
    )
    gam = gam + 1.0

    # x has to be positive !
    y = jax.lax.cond(
        x > 0,
        lambda x: jnp.sqrt(2.0 * jnp.pi / x) * jnp.power(x, x) * jnp.exp(-x) * gam,
        lambda x: jnp.inf,
        x,
    )

    return y


def freq(lflai: Float_0D) -> Float_0D:
    """Use the beta distribution to compute the probability frequency distribution
       for a known mean leaf inclication angle starting from the top of the canopy,
       where llai=0 (after Goel and Strebel (1984)).

    Args:
        lflai (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    # spherical leaf angle
    MEAN = 57.4
    STD = 26
    VAR = STD * STD + MEAN * MEAN
    nuu = (1.0 - VAR / (90.0 * MEAN)) / (VAR / (MEAN * MEAN) - 1.0)
    MU = nuu * ((90.0 / MEAN) - 1.0)
    SUM = nuu + MU

    # FL1 = gamma(SUM) / (gamma(nuu) * gamma(MU))
    FL1 = gammaf(SUM) / (gammaf(nuu) * gammaf(MU))
    # jax.debug.print("FL1: {a}; FL1-jax: {b}", a=gammaf(SUM), b=gamma(SUM))
    MU1 = MU - 1.0
    nu1 = nuu - 1.0

    CONS = 1.0 / 9.0

    # COMPUTE PROBABILITY DISTRIBUTION FOR 9 ANGLE CLASSES
    # BETWEEN 5 AND 85 DEGREES, WITH INCREMENTS OF 10 DEGREES
    def calculate_bden(carry, i):
        ANG = 10.0 * (i + 1) - 5.0
        FL2 = jnp.power((1.0 - ANG / 90.0), MU1)
        FL3 = jnp.power((ANG / 90.0), nu1)
        y = CONS * FL1 * FL2 * FL3
        return None, y

    _, bdens = jax.lax.scan(calculate_bden, None, jnp.arange(9))

    return bdens
