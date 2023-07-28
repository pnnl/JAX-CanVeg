"""
Radiation transfer functions, including:
- diffuse_direct_radiation()
- sky_ir()
- rad_tran_canopy()

Author: Peishi Jiang
Date: 2023.07.27.
"""

import jax
import jax.numpy as jnp

from functools import partial
from typing import Tuple

from ...subjects import SunAng, LeafAng, ParNir, Para
from ...shared_utilities.types import Float_0D, Float_1D
from ...shared_utilities.types import HashableArrayWrapper
from ...shared_utilities.utils import dot


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


def sky_ir(T: Float_1D, ratrad: Float_1D, sigma: Float_0D) -> Float_1D:
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


@partial(jax.jit, static_argnames=["mask_night", "niter"])
def rad_tran_canopy(
    sunang: SunAng,
    leafang: LeafAng,
    rad: ParNir,
    prm: Para,
    mask_night: HashableArrayWrapper,
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
    # Incident radiation above the canopy
    # Compute the amount of diffuse light that is transmitted through a layer.
    # Diffuse radiation through gpas and absorbed radiation transmitted
    # through leaves
    fraction_beam = rad.inbeam / rad.incoming

    # At night if rad.incoming is zero then set fraction_beam to zero
    fraction_beam = jnp.nan_to_num(fraction_beam, nan=0.0)
    # beam_top = fraction_beam # compute beam from initial ones

    # Compute beam radiation transfer and probability of beam
    # jax.debug.print("{a}, {b}", a=leafang.Gfunc.shape, b=sunang.sin_beta.shape)
    exp_direct = jnp.exp(
        dot(
            leafang.Gfunc / sunang.sin_beta,  # (ntime,)
            -(prm.dff * prm.markov),  # (ntime, jtot)
        )
    )  # (ntime, jtot)
    exp_direct = jnp.concatenate(
        [exp_direct, jnp.ones([prm.ntime, 1])], axis=1
    )  # (ntime, jktot)
    exp_direct = exp_direct.at[mask_night.val, :].set(0.0)
    P0 = jax.lax.cumprod(exp_direct, axis=1, reverse=True)  # (ntime, jktot)
    Tbeam = P0
    # rad.prob_beam = jnp.concatenate(
    #     [rad.prob_beam[:,:prm.jtot-1],
    #      jnp.ones([prm.ntime,1]),
    #      rad.prob_beam[:,prm.jtot:]]
    # )

    # rad.prob_beam(:,prm.jtot)=ones;
    rad.prob_beam = jnp.concatenate(
        [prm.markov * P0[:, : prm.jtot], jnp.ones([prm.ntime, 1])], axis=1
    )  # (ntime, jktot)
    rad.prob_beam = jnp.clip(rad.prob_beam, a_max=1.0)
    rad.prob_beam = rad.prob_beam.at[mask_night.val, :].set(0.0)
    rad.prob_shade = 1 - rad.prob_beam

    # Calculate beam PAR that is transmitted downward
    # (ntime, jtot)
    incoming_beam = dot(rad.incoming * fraction_beam, Tbeam[:, 1:])
    incoming_beam = incoming_beam * (1 - exp_direct[:, :-1])
    sdn = incoming_beam * rad.trans

    # Calculate beam PAR that is reflected upward
    # (ntime, jtot)
    sup = incoming_beam * rad.reflect

    # Calculate the transmission and reflection of each layer
    reflectance_layer = (1 - leafang.integ_exp_diff) * rad.reflect  # (ntime, jtot)
    transmission_layer = (
        leafang.integ_exp_diff + (1 - leafang.integ_exp_diff) * rad.trans
    )  # (ntime, jtot)  # noqa: E501

    # Compute diffuse and complementary radiation
    dn_top = (1 - fraction_beam) * rad.incoming
    dn_init = jnp.concatenate(
        [jnp.zeros([prm.ntime, prm.jtot]), jnp.expand_dims(dn_top, axis=-1)], axis=1
    )
    up_init = jnp.zeros([prm.ntime, prm.jktot])

    # Now, let's iterate...

    def calculate_dnup(c2, j):
        up, dn = c2[0], c2[1]
        dn_top, up_bot = dn[:, -1], up[:, 0]
        # dn
        def calculate_dn_layer(c, i):
            dn_layer_t, up_layer_b = c, up[:, i]
            dn_layer_b = (
                dn_layer_t * transmission_layer[:, i]
                + up_layer_b * reflectance_layer[:, i]
                + sdn[:, i]
            )
            cnew = dn_layer_b
            return cnew, cnew

        _, out = jax.lax.scan(
            calculate_dn_layer, dn_top, jnp.arange(prm.jtot - 1, -1, -1)
        )
        out = out.T
        dn = jnp.concatenate([out[:, ::-1], jnp.expand_dims(dn_top, axis=-1)], axis=1)

        # up
        def calculate_up_layer(c, i):
            un_layer_b, dn_layer_t = c, dn[:, i + 1]
            up_layer_t = (
                un_layer_b * transmission_layer[:, i]
                + dn_layer_t * reflectance_layer[:, i]
                + sup[:, i]
            )
            cnew = up_layer_t
            return cnew, cnew

        _, out = jax.lax.scan(calculate_up_layer, up_bot, jnp.arange(prm.jtot))
        out = out.T
        up_bot = (dn[:, 0] + rad.incoming * fraction_beam * Tbeam[:, 0]) * rad.soil_refl
        up = jnp.concatenate([jnp.expand_dims(up_bot, axis=-1), out], axis=1)

        c2new = [up, dn]
        return c2new, None

    carry, _ = jax.lax.scan(calculate_dnup, [up_init, dn_init], xs=None, length=niter)
    up, dn = carry[0], carry[1]

    # upward diffuse radiation flux density, on the horizontal
    rad.up_flux = up

    # downward beam radiation flux density, incident on the horizontal
    rad.beam_flux = dot(rad.incoming * fraction_beam, Tbeam)

    # downward diffuse radiation flux density on the horizontal
    rad.dn_flux = dn

    # total downward radiation, incident on the horizontal
    rad.total = rad.beam_flux + rad.dn_flux

    # amount of radiation absorbed on the sun and shade leaves
    rad.sh_abs = (
        rad.dn_flux[:, : prm.jtot] + rad.up_flux[:, : prm.jtot]
    ) * rad.absorbed
    normal = dot(
        leafang.Gfunc / sunang.sin_beta, rad.beam_flux[:, : prm.jtot]
    )  # (ntime, jtot)  # noqa: E501
    normal = jnp.clip(normal, a_min=0.0)
    sun_normal_abs = normal * rad.absorbed
    rad.sun_abs = sun_normal_abs + rad.sh_abs

    return rad
