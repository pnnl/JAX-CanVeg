"""
Functions for combined fluxes, including:
- energy_carbon_fluxes()

Author: Peishi Jiang
Date: 2023.07.14.
"""

# import jax
# import jax.numpy as jnp

import equinox as eqx

# from functools import partial

from typing import Tuple

from ..subjects import ParNir, Met, Prof, Para, SunShadedCan, Qin
from ..shared_utilities.types import HashableArrayWrapper
from .energy_fluxes import boundary_resistance_mx, leaf_energy_mx
from .carbon_fluxes import leaf_ps_mx

# from .energy_fluxes import boundary_resistance, energy_balance_amphi, llambda
# from .carbon_fluxes import photosynthesis_amphi


# @partial(jax.jit, static_argnames=["mask_turbulence"])
@eqx.filter_jit
def energy_carbon_fluxes(
    sun: SunShadedCan,
    shade: SunShadedCan,
    qin: Qin,
    quantum: ParNir,
    met: Met,
    prof: Prof,
    prm: Para,
    mask_turbulence: HashableArrayWrapper,
) -> Tuple[SunShadedCan, SunShadedCan]:
    """The ENERGY_AND_CARBON_FLUXES routine to computes coupled fluxes
       of energy, water and CO2 exchange, as well as leaf temperature.  Computataions
       are performed for each layer in the canopy and on the sunlit and shaded fractions

       Analytical solution for leaf energy balance and leaf temperature is used.  The
       program is derived from work by Paw U (1986) and was corrected for errors with a
       re-derivation of the equations.  The quadratic form of the solution is used,
       rather than the quartic version that Paw U prefers.

       Estimates of leaf temperature are used to adjust kinetic coefficients for enzyme
       kinetics, respiration and photosynthesis, as well as the saturation vapor
       pressure at the leaf surface.

       The Analytical solution of the coupled set of equations for photosynthesis and
       stomatal conductance by Baldocchi (1994, Tree Physiology) is used.  This
       equation is a solution to a cubic form of the photosynthesis equation.
       The photosynthesis algorithms are from the model of Farquhar.  Stomatal
       conductance is based on the algorithms of Ball- Berry and Collatz et al.,
       which couple gs to A.

    Args:
        sun (SunShadedCan): _description_
        shade (SunShadedCan): _description_
        qin (Qin): _description_
        quantum (ParNir): _description_
        met (Met): _description_
        prof (Prof): _description_
        prm (Para): _description_

    Returns:
        Tuple[SunShadedCan, SunShadedCan]: _description_
    """
    # Based on sunlit leaf temperature and air temperature of the layer
    # compute boundary layer resistances for heat, vapor and CO2
    boundary_layer_res = boundary_resistance_mx(
        prof, met, sun.Tsfc, prm, mask_turbulence
    )

    # Compute leaf photosynthesis
    ps = leaf_ps_mx(
        quantum.sun_abs,
        prof.co2[:, : prm.jtot],
        sun.Tsfc,
        boundary_layer_res.co2,
        met.P_kPa,
        prof.eair_Pa[:, : prm.jtot],
        prm,
    )
    # sun.Ps, sun.gs, sun.Resp = ps.aphoto, ps.gs_m_s, ps.rd
    sun = eqx.tree_at(
        lambda t: (t.Ps, t.gs, t.Resp), sun, (ps.aphoto, ps.gs_m_s, ps.rd)
    )

    # Compute energy balance on the top of sunlit leaves
    # pass and use prm if leaf is amphistomatous or hypostomatous
    sun = leaf_energy_mx(boundary_layer_res, qin.sun_abs, met, prof, sun, prm)

    # Compute energy balance for the bottom of sunlit leaves
    # if assuming hypostomatous assign gs a low value, eg 0.01m/s

    # Redo for shade fraction
    boundary_layer_res = boundary_resistance_mx(
        prof, met, shade.Tsfc, prm, mask_turbulence
    )
    ps = leaf_ps_mx(
        quantum.sh_abs,
        prof.co2[:, : prm.jtot],
        shade.Tsfc,
        boundary_layer_res.co2,
        met.P_kPa,
        prof.eair_Pa[:, : prm.jtot],
        prm,
    )
    # shade.Ps, shade.gs, shade.Resp = ps.aphoto, ps.gs_m_s, ps.rd
    shade = eqx.tree_at(
        lambda t: (t.Ps, t.gs, t.Resp), shade, (ps.aphoto, ps.gs_m_s, ps.rd)
    )
    shade = leaf_energy_mx(boundary_layer_res, qin.shade_abs, met, prof, shade, prm)

    return sun, shade
