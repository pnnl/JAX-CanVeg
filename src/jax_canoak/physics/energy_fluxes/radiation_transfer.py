"""
Radiation transfer functions, including:
- rnet()
- par()
- par_day()
- par_night()
- diffuse_direct_radiation()
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
):
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
):
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
):
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
