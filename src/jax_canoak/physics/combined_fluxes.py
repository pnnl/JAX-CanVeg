"""
Functions for combined fluxes, including:
- energy_and_carbon_fluxes()

Author: Peishi Jiang
Date: 2023.07.14.
"""

import jax
import jax.numpy as jnp

from typing import Tuple

from .energy_fluxes import boundary_resistance, energy_balance_amphi, llambda
from .carbon_fluxes import photosynthesis_amphi
from ..shared_utilities.types import Float_0D, Float_1D
from ..shared_utilities.constants import mass_CO2


def energy_and_carbon_fluxes(
    ht: Float_0D,
    grasshof: Float_0D,
    press_kPa: Float_0D,
    co2air: Float_0D,
    wnd: Float_0D,
    pr33: Float_0D,
    sc33: Float_0D,
    scc33: Float_0D,
    rhovva: Float_0D,
    air_density: Float_0D,
    lai: Float_0D,
    pai: Float_0D,
    pstat273: Float_0D,
    kballstr: Float_0D,
    tair_filter: Float_1D,
    zzz_ht: Float_1D,
    prob_beam: Float_1D,
    prob_sh: Float_1D,
    rnet_sun: Float_1D,
    rnet_sh: Float_1D,
    quantum_sun: Float_1D,
    quantum_sh: Float_1D,
    can_co2_air: Float_1D,
    rhov_air: Float_1D,
    rhov_filter: Float_1D,
    dLAIdz: Float_1D,
    sun_rs: Float_1D,
    shd_rs: Float_1D,
    sun_tleaf: Float_1D,
    shd_tleaf: Float_1D,
) -> Tuple:
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
        ht (Float_0D): _description_
        grasshof (Float_0D): _description_
        press_kPa (Float_0D): _description_
        co2air (Float_0D): _description_
        wnd (Float_0D): _description_
        pr33 (Float_0D): _description_
        sc33 (Float_0D): _description_
        scc33 (Float_0D): _description_
        rhovva (Float_0D): _description_
        air_density (Float_0D): _description_
        lai (Float_0D): _description_
        pai (Float_0D): _description_
        pstat273 (Float_0D): _description_
        kballstr (Float_0D): _description_
        tair_filter (Float_1D): _description_
        zzz_ht (Float_1D): _description_
        prob_beam (Float_1D): _description_
        prob_sh (Float_1D): _description_
        rnet_sun (Float_1D): _description_
        rnet_sh (Float_1D): _description_
        quantum_sun (Float_1D): _description_
        quantum_sh (Float_1D): _description_
        can_co2_air (Float_1D): _description_
        rhov_air (Float_1D): _description_
        rhov_filter (Float_1D): _description_
        dLAIdz (Float_1D): _description_
        sun_rs (Float_1D): _description_
        shd_rs (Float_1D): _description_
        sun_tleaf (Float_1D): _description_
        shd_tleaf (Float_1D): _description_

    Returns:
        Tuple: _description_
    """
    sze = rnet_sun.size
    jtot = sze - 2
    press_Pa = press_kPa * 1000.0
    # jax.debug.print("jax -- sun_rs: {x}; shd_rs: {y}", x=sun_rs, y=shd_rs)

    def calculate_energy_carbon(
        rnet_z,
        quantum_z,
        cca_z,
        t_sfc_K,
        tleaf_z,
        tair_filter_z,
        rhov_filter_z,
        rhov_air_z,
        rs,
        latent,
        zzz,
    ):
        # Compute the resistances for heat and vapor transfer, rh and rv,
        # for each layer, s/m
        boundres_heat, boundres_vapor, boundres_co2 = boundary_resistance(
            zzz,
            ht,
            tleaf_z,
            grasshof,
            press_kPa,
            wnd,
            pr33,
            sc33,
            scc33,
            tair_filter_z,
        )
        # jax.debug.print(
        #     "boundres (jax): {x} {a} {b} {c}",
        #     x=press_Pa, a=boundres_heat,
        #     b=boundres_vapor, c=boundres_co2
        # )
        # Compute energy balance of leaves
        t_sfc_K, LE_leaf, H_leaf, lout_leaf = energy_balance_amphi(
            rnet_z,
            t_sfc_K,
            rhovva,
            rhov_filter_z,
            rs,
            air_density,
            latent,
            press_Pa,
            boundres_heat,
        )
        # Compute photosynthesis
        rs, A_mg, resp, internal_CO2, wj_leaf, wc_leaf = photosynthesis_amphi(
            quantum_z,
            cca_z,
            t_sfc_K,
            LE_leaf,
            boundres_vapor,
            pstat273,
            kballstr,
            latent,
            co2air,
            boundres_co2,
            rhov_air_z,
        )
        t_sfc_C = t_sfc_K - 273.16
        rn = rnet_z - lout_leaf
        A = A_mg * 1000.0 / mass_CO2
        gs = 1.0 / rs
        return (
            t_sfc_C,
            H_leaf,
            LE_leaf,
            lout_leaf,
            rn,
            A,
            gs,
            rs,
            boundres_heat,
            boundres_vapor,
            boundres_co2,
            internal_CO2,
            resp,
            wj_leaf,
            wc_leaf,
        )

    def return_zeros(
        rnet_z,
        quantum_z,
        cca_z,
        t_sfc_K,
        tleaf_z,
        tair_filter_z,
        rhov_filter_z,
        rhov_air_z,
        rs,
        latent,
        zzz,
    ):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def update_energy_carbon_layer(carry, i):
        # First compute energy balance of sunlit leaf, then
        # repeat and compute algorithm for shaded leaves.
        # The stomatal resistances on the sunlit and shaded leaves are pre-estimated
        # as a function of PAR using STOMATA
        # Remember layer is two-sided so include upper and lower streams
        # are considered.
        # KC is the convective transfer coeff. (W M-2 K-1). A factor
        # of 2 is applied since convective heat transfer occurs
        # on both sides of leaves.
        # To calculate the total leaf resistance we must combine stomatal
        # and the leaf boundary layer resistance.  Many crops are amphistomatous
        # so KE must be multiplied by 2.  Deciduous forest, on the other hand
        # has stomata on one side of the leaf.
        tair_K_filtered = tair_filter[i] + 273.16  # absolute air temperature

        # Initialize surface temperature with air temperature
        t_sfc_K = tair_K_filtered

        latent = llambda(t_sfc_K)
        # latent18=latent*18.
        zzz = zzz_ht[i]
        cca, tair_filter_z = can_co2_air[i], tair_filter[i]
        rhov_filter_z, rhov_air_z = rhov_filter[i], rhov_air[i]

        # ---- Energy balance on sunlit leaves
        # update latent heat with new temperature during each call of this routine
        rnet_sun_z, tleaf_sun_z, rs_sun_z = rnet_sun[i], sun_tleaf[i], sun_rs[i]
        quantum_sun_z = quantum_sun[i]
        (
            t_sfc_C_sun,
            H_leaf_sun,
            LE_leaf_sun,
            lout_leaf_sun,
            rn_sun,
            A_sun,
            gs_sun,
            rs_sun,
            boundres_heat_sun,
            boundres_vapor_sun,
            boundres_co2_sun,
            internal_CO2_sun,
            resp_sun,
            wj_leaf_sun,
            wc_leaf_sun,
        ) = jax.lax.cond(
            prob_beam[i] > 0.0,
            calculate_energy_carbon,
            return_zeros,
            rnet_sun_z,
            quantum_sun_z,
            cca,
            t_sfc_K,
            tleaf_sun_z,
            tair_filter_z,
            rhov_filter_z,
            rhov_air_z,
            rs_sun_z,
            latent,
            zzz,
        )

        # ---- Energy balance on shaded leaves
        # update latent heat with new temperature during each call of this routine
        t_sfc_K = t_sfc_C_sun + 273.16
        rnet_shd_z, tleaf_shd_z, rs_shd_z = rnet_sh[i], shd_tleaf[i], shd_rs[i]
        quantum_shd_z = quantum_sh[i]
        (
            t_sfc_C_shd,
            H_leaf_shd,
            LE_leaf_shd,
            lout_leaf_shd,
            rn_shd,
            A_shd,
            gs_shd,
            rs_shd,
            boundres_heat_shd,
            boundres_vapor_shd,
            boundres_co2_shd,
            internal_CO2_shd,
            resp_shd,
            wj_leaf_shd,
            wc_leaf_shd,
        ) = calculate_energy_carbon(
            rnet_shd_z,
            quantum_shd_z,
            cca,
            t_sfc_K,
            tleaf_shd_z,
            tair_filter_z,
            rhov_filter_z,
            rhov_air_z,
            rs_shd_z,
            latent,
            zzz,
        )

        # jax.debug.print("t_sfc_K: {a}, shd_tleaf_each: {b}",
        #                 a=t_sfc_K, b=t_sfc_C_shd)
        output = [
            t_sfc_C_sun,
            H_leaf_sun,
            LE_leaf_sun,
            lout_leaf_sun,
            rn_sun,
            A_sun,
            gs_sun,
            rs_sun,
            boundres_heat_sun,
            boundres_vapor_sun,
            boundres_co2_sun,
            internal_CO2_sun,
            resp_sun,
            wj_leaf_sun,
            wc_leaf_sun,
            t_sfc_C_shd,
            H_leaf_shd,
            LE_leaf_shd,
            lout_leaf_shd,
            rn_shd,
            A_shd,
            gs_shd,
            rs_shd,
            boundres_heat_shd,
            boundres_vapor_shd,
            boundres_co2_shd,
            internal_CO2_shd,
            resp_shd,
            wj_leaf_shd,
            wc_leaf_shd,
        ]

        return carry, output

    _, outputs = jax.lax.scan(update_energy_carbon_layer, None, jnp.arange(jtot))
    (
        sun_tleaf,
        sun_H_leaf,
        sun_LE_leaf,
        sun_lout_leaf,
        sun_rn,
        sun_A,
        sun_gs,
        sun_rs,
        sun_rbh,
        sun_rbv,
        sun_rbco2,
        sun_ci,
        sun_resp,
        sun_wj,
        sun_wc,
        shd_tleaf,
        shd_H_leaf,
        shd_LE_leaf,
        shd_lout_leaf,
        shd_rn,
        shd_A,
        shd_gs,
        shd_rs,
        shd_rbh,
        shd_rbv,
        shd_rbco2,
        shd_ci,
        shd_resp,
        shd_wj,
        shd_wc,
    ) = outputs
    # jax.debug.print("sun_tleaf: {a}; shd_tleaf: {b}", a=sun_tleaf, b=shd_tleaf)

    # Compute layer energy fluxes, weighted by leaf area and sun and shaded fractions
    dLEdz = dLAIdz[:jtot] * (
        prob_beam[:jtot] * sun_LE_leaf + prob_sh[:jtot] * shd_LE_leaf
    )
    dHdz = dLAIdz[:jtot] * (prob_beam[:jtot] * sun_H_leaf + prob_sh[:jtot] * shd_H_leaf)
    dRNdz = dLAIdz[:jtot] * (prob_beam[:jtot] * sun_rn + prob_sh[:jtot] * shd_rn)

    # Photosynthesis of the layer,  prof.dPsdz has units mg m-3 s-1
    dPsdz = dLAIdz[:jtot] * (sun_A * prob_beam[:jtot] + shd_A * prob_sh[:jtot])
    Ci = sun_ci * prob_beam[:jtot] + shd_ci * prob_sh[:jtot]
    shd_cica = shd_ci / can_co2_air[:jtot]
    sun_cica = sun_ci / can_co2_air[:jtot]

    # Scaling boundary layer conductance for vapor, 1/rbv
    drbv = prob_beam[:jtot] / sun_rbv + prob_sh[:jtot] / shd_rbv

    # Photosynthesis of layer, prof.dPsdz has units of micromoles m-2 s-1
    dPsdz = dLAIdz[:jtot] * (sun_A * prob_beam[:jtot] + shd_A * prob_sh[:jtot])

    # Respiration of the layer, micromol m-2 s-1
    dRESPdz = dLAIdz[:jtot] * (sun_resp * prob_beam[:jtot] + shd_resp * prob_sh[:jtot])

    # Stomotal conductance m s-1
    dStomCondz = dLAIdz[:jtot] * (prob_beam[:jtot] * sun_gs + prob_sh[:jtot] * shd_gs)

    return (
        sun_rs,
        shd_rs,
        sun_gs,
        shd_gs,
        sun_resp,
        shd_resp,
        sun_wj,
        shd_wj,
        sun_wc,
        shd_wc,
        sun_A,
        shd_A,
        sun_rbh,
        shd_rbh,
        sun_rbv,
        shd_rbv,
        sun_rbco2,
        shd_rbco2,
        sun_ci,
        shd_ci,
        sun_cica,
        shd_cica,
        sun_tleaf,
        shd_tleaf,
        dLEdz,
        dHdz,
        dRNdz,
        dPsdz,
        Ci,
        drbv,
        dRESPdz,
        dStomCondz,
    )
