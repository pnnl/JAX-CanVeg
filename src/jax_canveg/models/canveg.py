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

from typing import Tuple, Callable

from ..shared_utilities.types import Float_0D, Float_2D
from ..shared_utilities.solver import fixed_point

from ..subjects import Para, Met, Prof, SunAng, LeafAng, SunShadedCan
from ..subjects import Veg, Soil, Rnet, Qin, Ir, ParNir, Lai, Can
from ..subjects import initialize_profile, initialize_model_states

from ..shared_utilities.utils import dot
from ..subjects import update_profile, calculate_veg, calculate_can
from ..physics import energy_carbon_fluxes

# from jax_canveg.physics.energy_fluxes import rad_tran_canopy, sky_ir_v2
from ..physics.energy_fluxes import rad_tran_canopy, sky_ir
from ..physics.energy_fluxes import compute_qin, ir_rad_tran_canopy
from ..physics.energy_fluxes import uz, soil_energy_balance
from ..physics.energy_fluxes import diffuse_direct_radiation
from ..physics.carbon_fluxes import angle, leaf_angle

# from ..physics.carbon_fluxes import soil_respiration


@eqx.filter_jit
def canveg_initialize_states(
    para: Para,
    met: Met,
    # Location parameters
    lat_deg: Float_0D,
    long_deg: Float_0D,
    time_zone: int,
    # Static parameters
    leafangle: int,
    n_can_layers: int,
    n_total_layers: int,
    n_soil_layers: int,
    time_batch_size: int,
    dt_soil: Float_0D,
    soil_mtime: int,
    par_niter: int = 5,
    nir_niter: int = 25,
):
    # jtot, jtot_total = n_can_layers, n_total_layers
    jtot = n_can_layers
    ntime = time_batch_size

    # ntime, jtot, jtot_total = met.zL.size, setup.n_can_layers, setup.n_total_layers
    # dt_soil, soil_mtime = setup.dt_soil, setup.soil_mtime
    # n_soil_layers = setup.n_soil_layers
    # z = jnp.zeros(jtot)

    # ---------------------------------------------------------------------------- #
    #                     Initialize profiles of scalars/sources/sinks             #
    # ---------------------------------------------------------------------------- #
    # prof = initialize_profile(met, para, ntime, jtot, jtot_total)
    prof = initialize_profile(met, para)

    # ---------------------------------------------------------------------------- #
    #                     Initialize model states                        #
    # ---------------------------------------------------------------------------- #
    (
        soil,
        quantum,
        nir,
        ir,
        qin,
        rnet,
        sun,
        shade,
        veg,
        lai,
        can,
    ) = initialize_model_states(  # noqa: E501
        met, para, ntime, jtot, dt_soil, soil_mtime, n_soil_layers
    )

    # ---------------------------------------------------------------------------- #
    #                     Compute sun angles                                       #
    # ---------------------------------------------------------------------------- #
    sun_ang = angle(lat_deg, long_deg, time_zone, met.day, met.hhour)

    # ---------------------------------------------------------------------------- #
    #                     Compute leaf angle                                       #
    # ---------------------------------------------------------------------------- #
    leaf_ang = leaf_angle(sun_ang, para, leafangle, lai)

    # ---------------------------------------------------------------------------- #
    #                     Compute direct and diffuse radiations                    #
    # ---------------------------------------------------------------------------- #
    ratrad, par_beam, par_diffuse, nir_beam, nir_diffuse = diffuse_direct_radiation(
        sun_ang.sin_beta, met.rglobal, met.parin, met.P_kPa
    )
    quantum = eqx.tree_at(
        lambda t: (t.inbeam, t.indiffuse), quantum, (par_beam, par_diffuse)
    )
    nir = eqx.tree_at(lambda t: (t.inbeam, t.indiffuse), nir, (nir_beam, nir_diffuse))

    # ---------------------------------------------------------------------------- #
    #                     Initialize IR fluxes with air temperature                #
    # ---------------------------------------------------------------------------- #
    ir_in = sky_ir(met.T_air_K, ratrad, para.sigma)
    # ir_in = sky_ir_v2(met, ratrad, para.sigma)
    ir_dn = dot(ir_in, ir.ir_dn)
    ir_up = dot(ir_in, ir.ir_up)
    ir = eqx.tree_at(lambda t: (t.ir_in, t.ir_dn, t.ir_up), ir, (ir_in, ir_dn, ir_up))

    # ---------------------------------------------------------------------------- #
    #                     Compute radiation fields             #
    # ---------------------------------------------------------------------------- #
    # PAR
    quantum = rad_tran_canopy(
        sun_ang,
        leaf_ang,
        quantum,
        para,
        lai,
        para.par_reflect,
        para.par_trans,
        para.par_soil_refl,
        niter=par_niter,
    )
    # NIR
    nir = rad_tran_canopy(
        sun_ang,
        leaf_ang,
        nir,
        para,
        lai,
        para.nir_reflect,
        para.nir_trans,
        para.nir_soil_refl,
        niter=nir_niter,
    )

    states_initial = [met, prof, quantum, nir, ir, qin, sun, shade, soil, veg, can]

    return rnet, lai, sun_ang, leaf_ang, states_initial


# @eqx.filter_jit
def canveg_each_iteration_v1(
    states: Tuple[Met, Prof, Ir, Qin, SunShadedCan, SunShadedCan, Soil, Veg, Can],
    para: Para,
    dij: Float_2D,
    leaf_ang: LeafAng,
    quantum: ParNir,
    nir: ParNir,
    lai: Lai,
    jtot: int,
    stomata_type: int,
    soil_mtime: int,
    leafrh_func: Callable,
    soilresp_func: Callable,
):
    met, prof, ir, qin, sun, shade, soil, veg, can = states
    # jax.debug.print("T soil: {a}", a=soil.T_soil[10,:])
    # jax.debug.print("T sfc: {a}", a=soil.sfc_temperature[10])
    # jax.debug.print("Tsfc: {a}", a=prof.Tair_K.mean())
    # jax.debug.print("T soil surface: {a}", a=soil.sfc_temperature.mean())

    # Update canopy wind profile with iteration of z/l and use in boundary layer
    # resistance computations
    # wind = uz(met, para, jtot)
    wind = uz(met, para)
    prof = eqx.tree_at(lambda t: t.wind, prof, wind)

    # Compute IR fluxes with Bonan's algorithms of Norman model
    ir = ir_rad_tran_canopy(leaf_ang, ir, quantum, soil, sun, shade, para)
    # jax.debug.print("ir_dn: {x}", x=ir.ir_dn)

    # Incoming short and longwave radiation
    qin = compute_qin(quantum, nir, ir, para, qin)

    # Compute energy fluxes for H, LE, gs, A on Sun and Shade leaves
    # Compute new boundary layer conductances based on new leaf energy balance
    # and delta T, in case convection occurs
    # Different coefficients will be assigned if amphistomatous or hypostomatous
    # sun, shade = energy_carbon_fluxes(
    sun, shade = energy_carbon_fluxes(
        sun, shade, qin, quantum, met, prof, para, stomata_type, leafrh_func
    )

    # Compute soil fluxes
    soil = soil_energy_balance(quantum, nir, ir, met, prof, para, soil, soil_mtime)  # type: ignore  # noqa: E501

    # Compute soil respiration
    # soil_resp = soil_respiration_alfalfa(
    #     veg.Ps, soil.T_soil[:, 9], met.soilmoisture, met.zcanopy, veg.Rd, para
    # )
    soil_resp = soilresp_func(
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


def canveg_each_iteration(
    states: Tuple[
        Met, Prof, ParNir, ParNir, Ir, Qin, SunShadedCan, SunShadedCan, Soil, Veg, Can
    ],
    para: Para,
    dij: Float_2D,
    sun_ang: SunAng,
    leaf_ang: LeafAng,
    lai: Lai,
    jtot: int,
    stomata_type: int,
    soil_mtime: int,
    leafrh_func: Callable,
    soilresp_func: Callable,
):
    met, prof, quantum, nir, ir, qin, sun, shade, soil, veg, can = states

    # Update PAR (quantum) and NIR (nir) again
    # PAR
    quantum = rad_tran_canopy(
        sun_ang,
        leaf_ang,
        quantum,
        para,
        lai,
        para.par_reflect,
        para.par_trans,
        para.par_soil_refl,
        niter=1,
    )
    # NIR
    nir = rad_tran_canopy(
        sun_ang,
        leaf_ang,
        nir,
        para,
        lai,
        para.nir_reflect,
        para.nir_trans,
        para.nir_soil_refl,
        niter=1,
    )

    # Update canopy wind profile with iteration of z/l and use in boundary layer
    # resistance computations
    # wind = uz(met, para, jtot)
    wind = uz(met, para)
    prof = eqx.tree_at(lambda t: t.wind, prof, wind)

    # Compute IR fluxes with Bonan's algorithms of Norman model
    ir = ir_rad_tran_canopy(leaf_ang, ir, quantum, soil, sun, shade, para)
    # jax.debug.print("ir_dn: {x}", x=ir.ir_dn)

    # Incoming short and longwave radiation
    qin = compute_qin(quantum, nir, ir, para, qin)

    # Compute energy fluxes for H, LE, gs, A on Sun and Shade leaves
    # Compute new boundary layer conductances based on new leaf energy balance
    # and delta T, in case convection occurs
    # Different coefficients will be assigned if amphistomatous or hypostomatous
    # sun, shade = energy_carbon_fluxes(
    sun, shade = energy_carbon_fluxes(
        sun, shade, qin, quantum, met, prof, para, stomata_type, leafrh_func
    )

    # Compute soil fluxes
    soil = soil_energy_balance(quantum, nir, ir, met, prof, para, soil, soil_mtime)  # type: ignore  # noqa: E501

    # Compute soil respiration
    # soil_resp = soil_respiration_alfalfa(
    #     veg.Ps, soil.T_soil[:, 9], met.soilmoisture, met.zcanopy, veg.Rd, para
    # )
    soil_resp = soilresp_func(
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

    states_new = [met, prof, quantum, nir, ir, qin, sun, shade, soil, veg, can]

    return states_new


# @eqx.filter_jit
def canveg(
    # def canveg_batch(
    para: Para,
    met: Met,
    dij: Float_2D,
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
    # Functions
    leafrh_func: Callable,
    soilresp_func: Callable,
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
    rnet, lai, sun_ang, leaf_ang, initials = canveg_initialize_states(
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
    # finals = canveg_fixed_point(
    #     initials, para, niter, dij, leaf_ang, quantum, nir, lai,
    #     n_can_layers, stomata, soil_mtime
    # )
    args = [
        dij,
        sun_ang,
        leaf_ang,
        lai,
        n_can_layers,
        stomata,
        soil_mtime,
        leafrh_func,
        soilresp_func,
    ]
    finals = fixed_point(canveg_each_iteration, initials, para, niter, *args)

    met, prof, quantum, nir, ir, qin, sun, shade, soil, veg, can = finals

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


################################################################
# The followings are functions for extracting or getting
# a particular variable from the output.
# The list of states follows --
# [Met, Prof, ParNir, ParNir, Ir, Rnet, Qin,
#  SunShadedCan, SunShadedCan, Soil, Veg, Can,]
################################################################
def get_all(states):
    return states


def update_all(states, new_states):
    return new_states


def get_canle(states):
    return states[-1].LE


def update_canle(states, can_le):
    can = states[-1]
    can_new = eqx.tree_at(lambda t: t.LE, can, can_le)
    states[-1] = can_new
    return states


def get_cannee(states):
    return states[-1].NEE


def update_cannee(states, can_nee):
    can = states[-1]
    can_new = eqx.tree_at(lambda t: t.NEE, can, can_nee)
    states[-1] = can_new
    return states


def get_canlenee(states):
    # return states[-1].LE, states[-1].NEE
    le, nee = states[-1].LE, states[-1].NEE
    return jnp.array([le, nee]).T  # shape (Nt, 2)


def update_canlenee(states, newstates):
    can_le, can_nee = newstates[:, 0], newstates[:, 1]
    can = states[-1]
    can_new = eqx.tree_at(lambda t: (t.NEE, t.LE), can, (can_nee, can_le))
    states[-1] = can_new
    return states


def get_soilresp(states):
    return states[-3].resp


def update_soilresp(states, soil_resp):
    soil = states[-3]
    soil_new = eqx.tree_at(lambda t: t.resp, soil, soil_resp)
    states[-3] = soil_new
    return states
