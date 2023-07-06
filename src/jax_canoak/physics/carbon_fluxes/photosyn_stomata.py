"""
Photosynthesis/stomatal conductance/respiratin, including:
- stomata()
- photosynthesis_amphi()
- temp_func()
- tboltz()
- soil_respiration()

Author: Peishi Jiang
Date: 2023.07.06.
"""

import jax
import jax.numpy as jnp

from typing import Tuple
from ...shared_utilities.types import Float_0D, Float_1D
from ...shared_utilities.constants import brs, rsm, rugc, hkin


def stomata(
    lai: Float_0D,
    pai: Float_0D,
    rcuticle: Float_0D,
    par_sun: Float_1D,
    par_shade: Float_1D,
) -> Tuple[Float_1D, Float_1D]:
    """First guess of rstom to run the energy balance model.
       It is later updated with the Ball-Berry model.

    Args:
        lai (Float_0D): _description_
        pai (Float_0D): _description_
        rcuticle (Float_0D): _description_
        par_sun (Float_1D): _description_
        par_shade (Float_1D): _description_

    Returns:
        Tuple[Float_1D, Float_1D]: _description_
    """
    sze = par_sun.size
    jtot = sze - 2
    sun_rs, shd_rs = jnp.zeros(sze), jnp.zeros(sze)

    rsfact = brs * rsm

    def update_rs(c, i):
        sun_rs_each = jax.lax.cond(
            (lai == pai) | (par_sun[i] < 5.0),
            lambda: rcuticle,
            lambda: rsm + rsfact / par_sun[i],
        )
        shd_rs_each = jax.lax.cond(
            (lai == pai) | (par_shade[i] < 5.0),
            lambda: rcuticle,
            lambda: rsm + rsfact / par_shade[i],
        )
        return c, [sun_rs_each, shd_rs_each]

    _, rs_update = jax.lax.scan(update_rs, None, jnp.arange(jtot))

    # jax.debug.print("rs_update: {x}", x=rs_update)

    sun_rs = sun_rs.at[:jtot].set(rs_update[0])
    shd_rs = shd_rs.at[:jtot].set(rs_update[1])

    return sun_rs, shd_rs


def photosynthesis_amphi():
    pass


def temp_func(
    rate: Float_0D, eact: Float_0D, tprime: Float_0D, tref: Float_0D, t_lk: Float_0D
) -> Float_0D:
    """Arhennius temperature function.

    Args:
        rate (Float_0D): _description_
        eact (Float_0D): _description_
        tprime (Float_0D): _description_
        tref (Float_0D): _description_
        t_lk (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    return rate * jnp.exp(tprime * eact / (tref * rugc * t_lk))


def tboltz(rate: Float_0D, eakin: Float_0D, topt: Float_0D, tl: Float_0D) -> Float_0D:
    """Boltzmann temperature distribution for photosynthesis

    Args:
        rate (Float_0D): _description_
        eakin (Float_0D): _description_
        topt (Float_0D): _description_
        tl (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    dtlopt = tl - topt
    prodt = rugc * topt * tl
    numm = rate * hkin * jnp.exp(eakin * (dtlopt) / (prodt))
    denom = hkin - eakin * (1.0 - jnp.exp(hkin * (dtlopt) / (prodt)))
    return numm / denom


def soil_respiration(
    Ts: Float_0D, base_respiration: Float_0D = 8.0
) -> Tuple[Float_0D, Float_0D]:
    """Compute soil respiration

    Args:
        Ts (Float_0D): _description_
        base_respiration (Float_0D, optional): _description_. Defaults to 8..
    """
    # After Hanson et al. 1993. Tree Physiol. 13, 1-15
    # reference soil respiration at 20 C, with value of about 5 umol m-2 s-1
    # from field studies

    # assume Q10 of 1.4 based on Mahecha et al Science 2010, Ea = 25169
    respiration_mole = base_respiration * jnp.exp(
        (25169.0 / 8.314) * ((1.0 / 295.0) - 1.0 / (Ts + 273.16))
    )

    # soil wetness factor from the Hanson model, assuming constant and wet soils
    respiration_mole *= 0.86

    # convert soilresp to mg m-2 s-1 from umol m-2 s-1
    respiration_mg = respiration_mole * 0.044

    return respiration_mole, respiration_mg
