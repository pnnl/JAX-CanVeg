"""
This is a big jax function for running canveg, given the inputs.

Author: Peishi Jiang
Date: 2023.8.13.
"""

# import jax

# import jax.tree_util as jtu
import jax.numpy as jnp
import equinox as eqx

# from functools import partial

from typing import Tuple

from .canveg import canveg_initialize_states
from ..shared_utilities.solver import fixed_point

from ..shared_utilities.types import Float_0D, Float_2D
from ..subjects import Para, Met, Prof, SunAng, LeafAng, SunShadedCan
from ..subjects import Veg, Soil, Rnet, Qin, Ir, ParNir, Lai, Can

from ..subjects import update_profile, calculate_veg, calculate_can

# from ..physics import energy_carbon_fluxes

# from jax_canveg.physics.energy_fluxes import rad_tran_canopy, sky_ir_v2
from ..physics.energy_fluxes import compute_qin, ir_rad_tran_canopy
from ..physics.energy_fluxes import uz, soil_energy_balance
from ..physics.carbon_fluxes import leaf_ps_gsswc_hybrid, soil_respiration_alfalfa
from ..physics.energy_fluxes import boundary_resistance, leaf_energy


def energy_carbon_fluxes_gsswc_hybrid(
    sun: SunShadedCan,
    shade: SunShadedCan,
    qin: Qin,
    quantum: ParNir,
    met: Met,
    prof: Prof,
    prm: Para,
    stomata: int,
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
    ntime, jtot = sun.Tsfc.shape

    # Based on sunlit leaf temperature and air temperature of the layer
    # compute boundary layer resistances for heat, vapor and CO2
    boundary_layer_res = boundary_resistance(prof, met, sun.Tsfc, prm)

    # Compute leaf photosynthesis
    ps = leaf_ps_gsswc_hybrid(
        quantum.sun_abs,
        prof.co2[:, :jtot],
        sun.Tsfc,
        boundary_layer_res.co2,
        met.P_kPa,
        prof.eair_Pa[:, :jtot],
        met.soilmoisture,
        prm,
        stomata,
    )
    # sun.Ps, sun.gs, sun.Resp = ps.aphoto, ps.gs_m_s, ps.rd
    sun = eqx.tree_at(
        lambda t: (t.Ps, t.gs, t.Resp, t.Leaf_RH),
        sun,
        (ps.aphoto, ps.gs_m_s, ps.rd, ps.Leaf_RH),
    )

    # Compute energy balance on the top of sunlit leaves
    # pass and use prm if leaf is amphistomatous or hypostomatous
    sun = leaf_energy(boundary_layer_res, qin.sun_abs, met, prof, sun, prm, stomata)

    # Compute energy balance for the bottom of sunlit leaves
    # if assuming hypostomatous assign gs a low value, eg 0.01m/s

    # Redo for shade fraction
    boundary_layer_res = boundary_resistance(prof, met, shade.Tsfc, prm)
    ps = leaf_ps_gsswc_hybrid(
        quantum.sh_abs,
        prof.co2[:, :jtot],
        shade.Tsfc,
        boundary_layer_res.co2,
        met.P_kPa,
        prof.eair_Pa[:, :jtot],
        met.soilmoisture,
        prm,
        stomata,
    )
    # shade.Ps, shade.gs, shade.Resp = ps.aphoto, ps.gs_m_s, ps.rd
    shade = eqx.tree_at(
        # lambda t: (t.Ps, t.gs, t.Resp), shade, (ps.aphoto, ps.gs_m_s, ps.rd)
        lambda t: (t.Ps, t.gs, t.Resp, t.Leaf_RH),
        shade,
        (ps.aphoto, ps.gs_m_s, ps.rd, ps.Leaf_RH),
    )
    shade = leaf_energy(
        boundary_layer_res, qin.shade_abs, met, prof, shade, prm, stomata
    )  # noqa: E501

    return sun, shade


@eqx.filter_jit
def canveg_gsswc_hybrid_each_iteration(
    states: Tuple[Met, Prof, Ir, Qin, SunShadedCan, SunShadedCan, Soil, Veg, Can],
    para: Para,
    dij: Float_2D,
    # RsoilDL: eqx.Module,
    leaf_ang: LeafAng,
    quantum: ParNir,
    nir: ParNir,
    lai: Lai,
    jtot: int,
    stomata: int,
    soil_mtime: int,
):
    met, prof, ir, qin, sun, shade, soil, veg, can = states

    # Update canopy wind profile with iteration of z/l and use in boundary layer
    # resistance computations
    wind = uz(met, para)
    prof = eqx.tree_at(lambda t: t.wind, prof, wind)

    # Compute IR fluxes with Bonan's algorithms of Norman model
    ir = ir_rad_tran_canopy(leaf_ang, ir, quantum, soil, sun, shade, para)

    # Incoming short and longwave radiation
    qin = compute_qin(quantum, nir, ir, para, qin)

    # Compute energy fluxes for H, LE, gs, A on Sun and Shade leaves
    # Compute new boundary layer conductances based on new leaf energy balance
    # and delta T, in case convection occurs
    # Different coefficients will be assigned if amphistomatous or hypostomatous
    # sun, shade = energy_carbon_fluxes(
    sun, shade = energy_carbon_fluxes_gsswc_hybrid(
        sun, shade, qin, quantum, met, prof, para, stomata
    )

    # Compute soil fluxes
    soil = soil_energy_balance(quantum, nir, ir, met, prof, para, soil, soil_mtime)  # type: ignore  # noqa: E501

    # Compute soil respiration
    soil_resp = soil_respiration_alfalfa(
        veg.Ps, soil.T_soil[:, 9], met.soilmoisture, met.zcanopy, veg.Rd, para
    )
    soil = eqx.tree_at(lambda t: t.resp, soil, soil_resp)

    # Compute profiles of C's, zero layer jtot+1 as that is not a dF/dz or
    # source/sink level
    prof = update_profile(met, para, prof, quantum, sun, shade, soil, veg, lai, dij)

    # compute met.zL from HH and met.ustar
    HH = jnp.sum(
        (quantum.prob_beam[:, :jtot] * sun.H + quantum.prob_shade[:, :jtot] * shade.H)
        * lai.dff[:, :jtot],
        axis=1,
    )
    zL = -(0.4 * 9.8 * HH * para.meas_ht) / (
        met.air_density * 1005 * met.T_air_K * jnp.power(met.ustar, 3.0)
    )
    zL = jnp.clip(zL, a_min=-3, a_max=0.25)
    met = eqx.tree_at(lambda t: t.zL, met, zL)

    # Compute canopy integrated fluxes
    veg = calculate_veg(para, lai, quantum, sun, shade)

    # Compute the whole column fluxes
    can = calculate_can(quantum, nir, ir, veg, soil, jtot)

    states_new = [met, prof, ir, qin, sun, shade, soil, veg, can]

    return states_new


@eqx.filter_jit
def canveg_gsswc_hybrid(
    # def canveg_batch(
    para: Para,
    met: Met,
    dij: Float_2D,
    # RsoilDL: eqx.Module,
    # Location parameters
    lat_deg: Float_0D,
    long_deg: Float_0D,
    time_zone: int,
    # Static parameters
    leafangle: int,
    stomata: int,
    n_can_layers: int,
    n_total_layers: int,
    n_soil_layers: int,
    # ntime: int,
    time_batch_size: int,
    dt_soil: Float_0D,
    soil_mtime: int,
    niter: int,
) -> Tuple[
    Met,
    Prof,
    ParNir,
    ParNir,
    Ir,
    Rnet,
    Qin,
    SunAng,
    LeafAng,
    Lai,
    SunShadedCan,
    SunShadedCan,
    Soil,
    Veg,
    Can,
]:
    # Initialization
    quantum, nir, rnet, lai, sun_ang, leaf_ang, initials = canveg_initialize_states(
        para,
        met,
        lat_deg,
        long_deg,
        time_zone,
        leafangle,
        n_can_layers,
        n_total_layers,
        n_soil_layers,
        time_batch_size,
        dt_soil,
        soil_mtime,
    )

    # ---------------------------------------------------------------------------- #
    #                     Iterations                                               #
    # ---------------------------------------------------------------------------- #
    # compute Tsfc -> IR -> Rnet -> Energy balance -> Tsfc
    # loop again and apply updated Tsfc info until convergence
    # This is where things should be jitted as a whole
    args = [
        dij,
        # RsoilDL,
        leaf_ang,
        quantum,
        nir,
        lai,
        n_can_layers,
        stomata,
        soil_mtime,
    ]
    finals = fixed_point(
        canveg_gsswc_hybrid_each_iteration, initials, para, niter, *args
    )

    met, prof, ir, qin, sun, shade, soil, veg, can = finals

    return (
        met,
        prof,
        quantum,
        nir,
        ir,
        rnet,
        qin,
        sun_ang,
        leaf_ang,
        lai,
        sun,
        shade,
        soil,
        veg,
        can,
    )
