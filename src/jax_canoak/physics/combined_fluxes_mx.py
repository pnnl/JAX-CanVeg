"""
Functions for combined fluxes, including:
- energy_carbon_fluxes()

Author: Peishi Jiang
Date: 2023.07.14.
"""

# import jax
# import jax.numpy as jnp

from typing import Tuple

from ..subjects import ParNir, Met, Prof, Para, SunShadedCan, Qin

# from .energy_fluxes import boundary_resistance, energy_balance_amphi, llambda
# from .carbon_fluxes import photosynthesis_amphi


def energy_carbon_fluxes(
    sun: SunShadedCan,
    shade: SunShadedCan,
    qin: Qin,
    quantum: ParNir,
    met: Met,
    prof: Prof,
    prm: Para,
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
    # TODO!

    return sun, shade
