"""
This is a big jax function for running canoak, given the inputs.

Author: Peishi Jiang
Date: 2023.8.13.
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from typing import Tuple

from jax_canoak.shared_utilities.types import Float_1D, Float_2D
from jax_canoak.shared_utilities.types import HashableArrayWrapper

from jax_canoak.subjects import Para, Met, Prof, SunAng, LeafAng, SunShadedCan
from jax_canoak.subjects import Setup, Veg, Soil, Qin, Ir, ParNir, Lai

from jax_canoak.shared_utilities.utils import dot
from jax_canoak.subjects import update_profile, calculate_veg
from jax_canoak.physics import energy_carbon_fluxes

# from jax_canoak.physics.energy_fluxes import rad_tran_canopy, sky_ir_v2
from jax_canoak.physics.energy_fluxes import rad_tran_canopy, sky_ir
from jax_canoak.physics.energy_fluxes import compute_qin, ir_rad_tran_canopy
from jax_canoak.physics.energy_fluxes import uz, soil_energy_balance


def canoak(
    para: Para,
    setup: Setup,
    met: Met,
    prof: Prof,
    dij: Float_2D,
    lai: Lai,
    sun_ang: SunAng,
    leaf_ang: LeafAng,
    quantum: ParNir,
    nir: ParNir,
    ir: Ir,
    qin: Qin,
    ratrad: Float_1D,
    sun: SunShadedCan,
    shade: SunShadedCan,
    veg: Veg,
    soil: Soil,
    mask_night_hashable: HashableArrayWrapper,
    mask_turbulence_hashable: HashableArrayWrapper,
    soil_mtime: int,
    niter: int = 15,
) -> Tuple[Met, Prof, Ir, Qin, SunShadedCan, SunShadedCan, Soil, Veg]:
    # # ---------------------------------------------------------------------------- #
    # #                     Compute direct and diffuse radiations                    #
    # # ---------------------------------------------------------------------------- #
    # ratrad, par_beam, par_diffuse, nir_beam, nir_diffuse = diffuse_direct_radiation(
    #     sun_ang.sin_beta, met.rglobal, met.parin, met.P_kPa
    # )
    # quantum = eqx.tree_at(
    #     lambda t: (t.inbeam, t.indiffuse), quantum, (par_beam, par_diffuse)
    # )
    # nir = eqx.tree_at(lambda t: (t.inbeam, t.indiffuse), nir, (nir_beam, nir_diffuse))

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
        sun_ang, leaf_ang, quantum, para, lai, mask_night_hashable, niter=5
    )
    # NIR
    nir = rad_tran_canopy(
        sun_ang, leaf_ang, nir, para, lai, mask_night_hashable, niter=25
    )  # noqa: E501

    # ---------------------------------------------------------------------------- #
    #                     Iterations                                               #
    # ---------------------------------------------------------------------------- #
    # compute Tsfc -> IR -> Rnet -> Energy balance -> Tsfc
    # loop again and apply updated Tsfc info until convergence
    # This is where things should be jitted as a whole
    def iteration(c, i):
        met, prof, ir, qin, sun, shade, soil, veg = c
        # jax.debug.print("T soil: {a}", a=soil.T_soil[10,:])
        # jax.debug.print("T sfc: {a}", a=soil.sfc_temperature[10])

        # Update canopy wind profile with iteration of z/l and use in boundary layer
        # resistance computations
        wind = uz(met, para, setup)
        prof = eqx.tree_at(lambda t: t.wind, prof, wind)

        # Compute IR fluxes with Bonan's algorithms of Norman model
        ir = ir_rad_tran_canopy(leaf_ang, ir, quantum, soil, sun, shade, para)
        # jax.debug.print("ir: {a}", a=ir.ir_dn[10,:])

        # Incoming short and longwave radiation
        qin = compute_qin(quantum, nir, ir, para, qin)

        # Compute energy fluxes for H, LE, gs, A on Sun and Shade leaves
        # Compute new boundary layer conductances based on new leaf energy balance
        # and delta T, in case convection occurs
        # Different coefficients will be assigned if amphistomatous or hypostomatous
        sun, shade = energy_carbon_fluxes(
            sun, shade, qin, quantum, met, prof, para, setup, mask_turbulence_hashable
        )

        # Compute soil fluxes
        soil = soil_energy_balance(quantum, nir, ir, met, prof, para, soil, soil_mtime)  # type: ignore  # noqa: E501

        # Compute profiles of C's, zero layer jtot+1 as that is not a dF/dz or
        # source/sink level
        prof = update_profile(met, para, prof, quantum, sun, shade, soil, veg, lai, dij)

        # compute met.zL from HH and met.ustar
        HH = jnp.sum(
            (
                quantum.prob_beam[:, : para.jtot] * sun.H
                + quantum.prob_shade[:, : para.jtot] * shade.H
            )
            * lai.dff[:, : para.jtot],
            axis=1,
        )
        zL = -(0.4 * 9.8 * HH * para.meas_ht) / (
            met.air_density * 1005 * met.T_air_K * jnp.power(met.ustar, 3.0)
        )
        zL = jnp.clip(zL, a_min=-3, a_max=0.25)
        met = eqx.tree_at(lambda t: t.zL, met, zL)

        # Compute canopy integrated fluxes
        veg = calculate_veg(para, lai, quantum, sun, shade)

        cnew = [met, prof, ir, qin, sun, shade, soil, veg]
        return cnew, None

    initials = [met, prof, ir, qin, sun, shade, soil, veg]
    finals, _ = jax.lax.scan(iteration, initials, xs=None, length=niter)

    met, prof, ir, qin, sun, shade, soil, veg = finals

    return met, prof, ir, qin, sun, shade, soil, veg
