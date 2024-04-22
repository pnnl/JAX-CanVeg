"""
Classes for model states.
- BatchedProf()
- BatchedSunAng()
- BatchedLeafAng()
- BatchedParNir()
- BatchedIr()
- BatchedRnet()
- BatchedSunShadedCan()
- BatchedBoundLayerRes()
- BatchedQin()
- BatchedVeg()
- BatchedLai()
- BatchedPs()
- BatchedSoil()

Author: Peishi Jiang
Date: 2023.7.24.
"""

# import jax
# import jax.numpy as jnp

import jax.tree_util as jtu

import equinox as eqx

from typing import Tuple

# from .meterology import Met
# from .parameters import Para
# from .utils import soil_sfc_res, conc
# from .utils import llambda as flambda
from .states import Prof, SunAng, LeafAng, ParNir, Ir, Rnet
from .states import SunShadedCan, Qin, Obs
from .states import Veg, Lai, Soil, Can

from ..shared_utilities.types import Float_3D, Float_2D, Float_1D

# from ..shared_utilities.utils import dot, minus
from ..shared_utilities.constants import PI


class BatchedProf(eqx.Module):
    co2: Float_3D
    Tair_K: Float_3D
    Told_K: Float_3D
    eair_Pa: Float_3D
    eair_old_Pa: Float_3D
    wind: Float_3D
    Tsfc: Float_3D
    H: Float_3D
    LE: Float_3D
    Rnet: Float_3D
    Ps: Float_3D


class BatchedSunAng(eqx.Module):
    sin_beta: Float_2D
    beta_rad: Float_2D

    @property
    def beta_deg(self):
        return self.beta_rad * 180 / PI

    @property
    def theta_rad(self):
        return PI / 2.0 - self.beta_rad

    @property
    def theta_deg(self):
        return self.theta_rad * 180.0 / PI


class BatchedLeafAng(eqx.Module):
    pdf: Float_2D
    Gfunc: Float_2D
    thetaSky: Float_2D
    Gfunc_Sky: Float_2D
    integ_exp_diff: Float_3D


class BatchedParNir(eqx.Module):
    sh_abs: Float_3D
    sun_abs: Float_3D
    sun: Float_3D
    dn_flux: Float_3D
    up_flux: Float_3D
    sun_normal: Float_3D
    sh_flux: Float_3D
    beam_flux: Float_3D
    inbeam: Float_3D
    indiffuse: Float_3D
    prob_beam: Float_3D
    sun_lai: Float_3D
    shade_lai: Float_3D

    @property
    def total(self):
        return self.beam_flux + self.dn_flux

    @property
    def incoming(self):
        return self.inbeam + self.indiffuse

    @property
    def prob_shade(self):
        return 1 - self.prob_beam


class BatchedIr(eqx.Module):
    ir_in: Float_2D
    ir_dn: Float_3D
    ir_up: Float_3D
    IR_source_sun: Float_3D
    IR_source_shade: Float_3D
    IR_source: Float_3D
    shade: Float_3D
    shade_top: Float_3D
    shade_bottom: Float_3D
    balance: Float_3D


class BatchedRnet(eqx.Module):
    sun: Float_3D
    sh: Float_3D
    sun_top: Float_3D
    sh_top: Float_3D
    sh_bottom: Float_3D


class BatchedSunShadedCan(eqx.Module):
    Ps: Float_3D
    Resp: Float_3D
    gs: Float_3D
    vpd_Pa: Float_3D
    LE: Float_3D
    H: Float_3D
    Rnet: Float_3D
    Lout: Float_3D
    closure: Float_3D
    Tsfc: Float_3D
    Tsfc_new: Float_3D
    Tsfc_old: Float_3D


class BatchedBoundLayerRes(eqx.Module):
    heat: Float_3D
    vapor: Float_3D
    co2: Float_3D


class BatchedQin(eqx.Module):
    sun_abs: Float_3D
    shade_abs: Float_3D


class BatchedVeg(eqx.Module):
    Ps: Float_2D
    gs: Float_2D
    Rd: Float_2D
    H: Float_2D
    LE: Float_2D
    Rnet: Float_2D
    Tsfc: Float_2D
    vpd: Float_2D

    @property
    def GPP(self):
        return self.Ps + self.Rd


class BatchedLai(eqx.Module):
    lai: Float_2D
    dff: Float_3D
    sumlai: Float_3D
    dff_clmp: Float_3D
    adens: Float_3D


class BatchedPs(eqx.Module):
    aphoto: Float_3D
    ci: Float_3D
    gs_co2: Float_3D
    gs_m_s: Float_3D
    wj: Float_3D
    wc: Float_3D
    wp: Float_3D
    jsucrose: Float_3D
    Ag: Float_3D
    x1: Float_3D
    x2: Float_3D
    x3: Float_3D
    p: Float_3D
    q: Float_3D
    r: Float_3D
    rd: Float_3D
    rstom: Float_3D


class BatchedCan(eqx.Module):
    rnet_calc: Float_2D
    rnet: Float_2D
    LE: Float_2D
    H: Float_2D
    NEE: Float_2D
    avail: Float_2D
    gsoil: Float_2D
    # albedo_calc: Float_2D
    # nir_albedo_calc: Float_2D
    nir_refl: Float_2D


class BatchedObs(eqx.Module):
    LE: Float_2D
    H: Float_2D
    GPP: Float_2D
    rnet: Float_2D
    albedo: Float_2D
    Fco2: Float_2D
    gsoil: Float_2D


class BatchedSoil(eqx.Module):
    dt: Float_1D
    n_soil: Float_1D
    depth: Float_1D
    mtime: Float_1D
    water_content_15cm: Float_2D
    water_content_litter: Float_1D
    bulkdensity: Float_1D
    clay_fraction: Float_1D
    peat_fraction: Float_1D
    Cp_water: Float_1D
    Cp_air: Float_1D
    Cp_org: Float_1D
    Cp_mineral: Float_1D
    K_mineral: Float_1D
    K_org: Float_1D
    K_water: Float_1D
    dz: Float_2D
    vol: Float_2D
    T_soil: Float_3D
    T_soil_old: Float_3D
    T_soil_up_boundary: Float_2D
    T_soil_low_bound: Float_2D
    sfc_temperature: Float_2D
    sfc_temperature_old: Float_2D
    bulk_density: Float_3D
    K_air: Float_2D
    Cp_soil: Float_2D
    evap: Float_2D
    heat: Float_2D
    rnet: Float_2D
    gsoil: Float_2D
    lout: Float_2D
    llout: Float_2D
    resp: Float_2D

    # @property
    # def bulkdensity_kg_m3(self):
    #     return self.bulkdensity * 100 * 100 * 100 / 1000

    # @property
    # def pore_fraction(self):
    #     # from alfalfa, 1 minus ratio bulk density 1.00
    #     # g cm-3/2.65 g cm-3, density of solids
    #     return 1 - self.bulkdensity / 2.65

    # @property
    # def mineral_fraction(self):
    #     return 1 - self.pore_fraction - self.peat_fraction

    # @property
    # def air_fraction(self):
    #     return self.pore_fraction - self.water_content_15cm

    # @property
    # def n_soil_1(self):
    #     return self.n_soil + 1

    # @property
    # def n_soil_2(self):
    #     return self.n_soil + 2

    # @property
    # def z_soil(self):
    #     # (n_soil_1,)
    #     return jnp.concatenate([jnp.array([0]), jnp.cumsum(self.dz)])

    # @property
    # def d2z(self):
    #     #  (n_soil,)
    #     d2z = self.z_soil[2:] - self.z_soil[:-2]
    #     return jnp.concatenate([self.dz[:1], d2z])

    # @property
    # def k_fluid(self):
    #     # (ntime,)
    #     fw = 1.0 / (1 + jnp.power((self.water_content_15cm / 0.15), -4))
    #     return self.K_air + fw * (self.K_water - self.K_air)  # (ntime,)

    # @property
    # def wt_air(self):
    #     return 2.0 / (3.0 * (1.0 + 0.2 * (self.K_air / self.k_fluid - 1))) + 1.0 / (
    #         3.0 * (1.0 + (1.0 - 2.0 * 0.2) * (self.K_air / self.k_fluid - 1))
    #     )

    # @property
    # def wt_water(self):
    #     return 2.0 / (3.0 * (1.0 + 0.2 * (self.K_water / self.k_fluid - 1))) + 1.0 / (
    #         3.0 * (1.0 + (1.0 - 2.0 * 0.2) * (self.K_water / self.k_fluid - 1))
    #     )

    # @property
    # def wt_org(self):
    #     return 2.0 / (3.0 * (1 + 0.2 * (self.K_org / self.k_fluid - 1))) + 1.0 / (
    #         3.0 * (1.0 + (1.0 - 2.0 * 0.2) * (self.K_org / self.k_fluid - 1))
    #     )

    # @property
    # def wt_mineral(self):
    #     return 2.0 / (3.0 * (1.0 + 0.2 * (self.K_mineral / self.k_fluid - 1))) + 1.0/(
    #         3.0 * (1.0 + (1.0 - 2.0 * 0.2) * (self.K_mineral / self.k_fluid - 1))
    #     )

    # @property
    # def K_soil_num(self):
    #     # thermal conductivity, W m-1  K-1
    #     # (ntime,)
    #     return (
    #         self.mineral_fraction * self.wt_mineral * self.K_mineral
    #         + self.air_fraction * self.wt_air * self.K_air
    #         + self.water_content_15cm * self.wt_water * self.K_water
    #         + self.peat_fraction * self.wt_org * self.K_mineral
    #     )

    # @property
    # def K_soil(self):
    #     # thermal conductivity, W m-1  K-1
    #     # (ntime,)
    #     return self.K_soil_num / (
    #         self.mineral_fraction * self.wt_mineral
    #         + self.air_fraction * self.wt_air
    #         + self.water_content_15cm * self.wt_water
    #         + self.peat_fraction * self.wt_org
    #     )

    # @property
    # def cp_soil(self):
    #     # rho Cp Volume/ 2 dt
    #     # transpose dz to make 2d matrix, (ntime, n_soil)
    #     return jnp.outer(self.Cp_soil, self.d2z) / (2 * self.dt)

    # @property
    # def k_conductivity_soil(self):
    #     # K/dz
    #     # (ntime, n_soil+1)
    #     ntime = self.K_soil.size
    #     k_conductivity_soil = jnp.outer(self.K_soil, 1.0 / self.dz)  # (ntime, n_soil)
    #     k_conductivity_soil = jnp.concatenate(
    #         [k_conductivity_soil, jnp.zeros([ntime, 1])],
    #         axis=1,
    #     )
    #     return k_conductivity_soil

    # @property
    # def k_conductivity_soil_bound(self):
    #     return self.k_conductivity_soil[..., 0]

    # @property
    # def resistance_h2o(self):
    #     return soil_sfc_res(self.water_content_15cm)


def convert_obs_to_batched_obs(obs: Obs, n_batch: int, batch_size: int) -> BatchedObs:
    n_total = n_batch * batch_size
    batched_obs = jtu.tree_map(
        lambda x: x[:n_total].reshape([n_batch, batch_size]), obs
    )
    return batched_obs


def convert_batchedstates_to_states(
    # prof: BatchedProf,
    # can: BatchedCan,
    # veg: BatchedVeg,
    # shade: BatchedSunShadedCan,
    # sun: BatchedSunShadedCan,
    # qin: BatchedQin,
    # rnet: BatchedRnet,
    # sun_ang: BatchedSunAng,
    # ir: BatchedIr,
    # nir: BatchedParNir,
    # quantum: BatchedParNir,
    # lai: BatchedLai,
    # leaf_ang: BatchedLeafAng,
    # soil: BatchedSoil,
    # TODO: --
    prof: Prof,
    can: Can,
    veg: Veg,
    shade: SunShadedCan,
    sun: SunShadedCan,
    qin: Qin,
    rnet: Rnet,
    sun_ang: SunAng,
    ir: Ir,
    nir: ParNir,
    quantum: ParNir,
    lai: Lai,
    leaf_ang: LeafAng,
    soil: Soil,
) -> Tuple[
    Prof,
    Can,
    Veg,
    SunShadedCan,
    Qin,
    Rnet,
    SunAng,
    Ir,
    ParNir,
    ParNir,
    Lai,
    LeafAng,
    Soil,
]:
    n_batch, batch_size = prof.co2.shape[0], prof.co2.shape[1]
    n_total = n_batch * batch_size

    prof = jtu.tree_map(lambda x: x.reshape(n_total, -1), prof)
    can = jtu.tree_map(lambda x: x.reshape(n_total), can)
    veg = jtu.tree_map(lambda x: x.reshape(n_total), veg)
    shade = jtu.tree_map(lambda x: x.reshape(n_total, -1), shade)
    sun = jtu.tree_map(lambda x: x.reshape(n_total, -1), sun)
    qin = jtu.tree_map(lambda x: x.reshape(n_total, -1), qin)
    # rnet = rnet.reshape(n_total)
    rnet = jtu.tree_map(lambda x: x.reshape(n_total, -1), rnet)
    sun_ang = jtu.tree_map(lambda x: x.reshape(n_total), sun_ang)

    ir = jtu.tree_map(lambda x: x.reshape(n_total, -1), ir)
    ir = eqx.tree_at(lambda t: (t.ir_in), ir, replace=(ir.ir_in.reshape(-1)))

    nir = jtu.tree_map(lambda x: x.reshape(n_total, -1), nir)
    nir = eqx.tree_at(lambda t: (t.inbeam), nir, replace=(nir.inbeam.reshape(-1)))

    quantum = jtu.tree_map(lambda x: x.reshape(n_total, -1), quantum)
    quantum = eqx.tree_at(
        lambda t: (t.inbeam), quantum, replace=(quantum.inbeam.reshape(-1))
    )

    lai = jtu.tree_map(lambda x: x.reshape(n_total, -1), lai)
    lai = eqx.tree_at(lambda t: (t.lai), lai, replace=(lai.lai.reshape(-1)))

    leaf_ang = eqx.tree_at(
        lambda t: (t.pdf, t.Gfunc, t.thetaSky, t.Gfunc_Sky, t.integ_exp_diff),
        leaf_ang,
        replace=(
            leaf_ang.pdf[0, :],
            leaf_ang.Gfunc.reshape(n_total),
            leaf_ang.thetaSky[0, :],
            leaf_ang.Gfunc_Sky[0, :],
            leaf_ang.integ_exp_diff.reshape(n_total, -1),
        ),
    )

    soil = eqx.tree_at(
        lambda t: (
            t.dt,
            # t.n_soil,
            t.depth,
            # t.mtime,
            t.water_content_15cm,
            t.water_content_litter,
            t.bulkdensity,
            t.clay_fraction,
            t.peat_fraction,
            t.Cp_water,
            t.Cp_air,
            t.Cp_org,
            t.Cp_mineral,
            t.K_mineral,
            t.K_org,
            t.K_water,
            t.dz,
            t.vol,
            t.T_soil,
            t.T_soil_old,
            t.T_soil_up_boundary,
            t.T_soil_low_bound,
            t.sfc_temperature,
            t.sfc_temperature_old,
            t.bulk_density,
            t.K_air,
            t.Cp_soil,
            t.evap,
            t.heat,
            t.rnet,
            t.gsoil,
            t.lout,
            t.llout,
            t.resp,
        ),
        soil,
        replace=(
            soil.dt[0],  # pyright: ignore
            # soil.n_soil[0],  # pyright: ignore
            soil.depth[0],  # pyright: ignore
            # soil.mtime[0],  # pyright: ignore
            soil.water_content_15cm.reshape(n_total),
            soil.water_content_litter[0],  # pyright: ignore
            soil.bulkdensity[0],  # pyright: ignore
            soil.clay_fraction[0],  # pyright: ignore
            soil.peat_fraction[0],  # pyright: ignore
            soil.Cp_water[0],  # pyright: ignore
            soil.Cp_air[0],  # pyright: ignore
            soil.Cp_org[0],  # pyright: ignore
            soil.Cp_mineral[0],  # pyright: ignore
            soil.K_mineral[0],  # pyright: ignore
            soil.K_org[0],  # pyright: ignore
            soil.K_water[0],  # pyright: ignore
            soil.dz[0],
            soil.vol[0],
            soil.T_soil.reshape(n_total, -1),
            soil.T_soil_old.reshape(n_total, -1),
            soil.T_soil_up_boundary.reshape(n_total),
            soil.T_soil_low_bound.reshape(n_total),
            soil.sfc_temperature.reshape(n_total),
            soil.sfc_temperature_old.reshape(n_total),
            soil.bulk_density.reshape(n_total, -1),
            soil.K_air.reshape(n_total),
            soil.Cp_soil.reshape(n_total),
            soil.evap.reshape(n_total),
            soil.heat.reshape(n_total),
            soil.rnet.reshape(n_total),
            soil.gsoil.reshape(n_total),
            soil.lout_sfc.reshape(n_total),
            # soil.lout.reshape(n_total),
            soil.llout.reshape(n_total),
            soil.resp.reshape(n_total),
        ),
    )

    return (  # pyright: ignore
        prof,
        can,
        veg,
        shade,
        sun,
        qin,
        rnet,
        sun_ang,
        ir,
        nir,
        quantum,
        lai,
        leaf_ang,
        soil,
    )
