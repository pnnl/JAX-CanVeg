"""
Classes for model states.
- Prof()
- SunAng()
- LeafAng()
- ParNir()
- Ir()
- Rnet()
- SunShadedCan()
- BoundLayerRes()
- Qin()
- Veg()
- Lai()
- Ps()
- Can()
- Obs()
- Soil()

Author: Peishi Jiang
Date: 2023.7.24.
"""

# import jax
import jax.numpy as jnp

import equinox as eqx

# from .meterology import Met
# from .parameters import Para
from .utils import soil_sfc_res

# from .utils import llambda as flambda

from ..shared_utilities.types import Float_2D, Float_1D, Float_0D

# from ..shared_utilities.utils import dot, minus
from ..shared_utilities.constants import PI


class Prof(eqx.Module):
    # zht: Float_1D
    # delz: Float_1D
    co2: Float_2D
    Tair_K: Float_2D
    Told_K: Float_2D
    eair_Pa: Float_2D
    eair_old_Pa: Float_2D
    wind: Float_2D
    Tsfc: Float_2D
    H: Float_2D
    LE: Float_2D
    Rnet: Float_2D
    Ps: Float_2D


class SunAng(eqx.Module):
    sin_beta: Float_1D
    beta_rad: Float_1D

    @property
    def beta_deg(self):
        return self.beta_rad * 180 / PI

    @property
    def theta_rad(self):
        return PI / 2.0 - self.beta_rad

    @property
    def theta_deg(self):
        return self.theta_rad * 180.0 / PI


class LeafAng(eqx.Module):
    pdf: Float_1D
    Gfunc: Float_1D
    thetaSky: Float_1D
    Gfunc_Sky: Float_1D
    integ_exp_diff: Float_2D


class ParNir(eqx.Module):
    sh_abs: Float_2D
    sun_abs: Float_2D
    sun: Float_2D
    dn_flux: Float_2D
    up_flux: Float_2D
    sun_normal: Float_2D
    sh_flux: Float_2D
    beam_flux: Float_2D
    inbeam: Float_1D
    indiffuse: Float_1D
    prob_beam: Float_2D
    sun_lai: Float_2D
    shade_lai: Float_2D
    # reflect: Float_0D
    # trans: Float_0D
    # soil_refl: Float_0D

    @property
    def total(self):
        return self.beam_flux + self.dn_flux

    @property
    def incoming(self):
        return self.inbeam + self.indiffuse

    @property
    def prob_shade(self):
        return 1 - self.prob_beam

    # @property
    # def absorbed(self):
    #     return 1 - self.reflect - self.trans


class Ir(eqx.Module):
    ir_in: Float_1D
    ir_dn: Float_2D
    ir_up: Float_2D
    IR_source_sun: Float_2D
    IR_source_shade: Float_2D
    IR_source: Float_2D
    shade: Float_2D
    shade_top: Float_2D
    shade_bottom: Float_2D
    balance: Float_2D


class Rnet(eqx.Module):
    sun: Float_2D
    sh: Float_2D
    sun_top: Float_2D
    sh_top: Float_2D
    sh_bottom: Float_2D


class SunShadedCan(eqx.Module):
    Ps: Float_2D
    Resp: Float_2D
    gs: Float_2D
    vpd_Pa: Float_2D
    LE: Float_2D
    H: Float_2D
    Rnet: Float_2D
    Lout: Float_2D
    closure: Float_2D
    Tsfc: Float_2D
    Tsfc_new: Float_2D
    Tsfc_old: Float_2D
    Leaf_RH: Float_2D


class BoundLayerRes(eqx.Module):
    heat: Float_2D
    vapor: Float_2D
    co2: Float_2D


class Qin(eqx.Module):
    sun_abs: Float_2D
    shade_abs: Float_2D


class Veg(eqx.Module):
    Ps: Float_1D
    gs: Float_1D
    Rd: Float_1D
    H: Float_1D
    LE: Float_1D
    Rnet: Float_1D
    Tsfc: Float_1D
    vpd: Float_1D

    @property
    def GPP(self):
        return self.Ps + self.Rd


class Lai(eqx.Module):
    lai: Float_1D
    dff: Float_2D
    sumlai: Float_2D
    dff_clmp: Float_2D
    adens: Float_2D


class Ps(eqx.Module):
    aphoto: Float_2D
    ci: Float_2D
    gs_co2: Float_2D
    gs_m_s: Float_2D
    wj: Float_2D
    wc: Float_2D
    wp: Float_2D
    jsucrose: Float_2D
    Ag: Float_2D
    x1: Float_2D
    x2: Float_2D
    x3: Float_2D
    p: Float_2D
    q: Float_2D
    r: Float_2D
    rd: Float_2D
    rstom: Float_2D
    Leaf_RH: Float_2D


class Can(eqx.Module):
    rnet_calc: Float_1D
    rnet: Float_1D
    LE: Float_1D
    H: Float_1D
    NEE: Float_1D
    avail: Float_1D
    gsoil: Float_1D
    # albedo_calc: Float_1D
    # nir_albedo_calc: Float_1D
    nir_refl: Float_1D


class Obs(eqx.Module):
    P: Float_1D
    LE: Float_1D
    H: Float_1D
    GPP: Float_1D
    rnet: Float_1D
    albedo: Float_1D
    Fco2: Float_1D
    gsoil: Float_1D
    Rsoil: Float_1D


class Soil(eqx.Module):
    dt: Float_0D
    # n_soil: int
    depth: Float_0D
    # mtime: int
    water_content_15cm: Float_1D
    water_content_litter: Float_0D
    bulkdensity: Float_0D
    clay_fraction: Float_0D
    peat_fraction: Float_0D
    Cp_water: Float_0D
    Cp_air: Float_0D
    Cp_org: Float_0D
    Cp_mineral: Float_0D
    K_mineral: Float_0D
    K_org: Float_0D
    K_water: Float_0D
    dz: Float_1D
    vol: Float_1D
    T_soil: Float_2D
    T_soil_old: Float_2D
    T_soil_up_boundary: Float_1D
    T_soil_low_bound: Float_1D
    sfc_temperature: Float_1D
    sfc_temperature_old: Float_1D
    bulk_density: Float_2D
    K_air: Float_1D
    Cp_soil: Float_1D
    evap: Float_1D
    heat: Float_1D
    rnet: Float_1D
    gsoil: Float_1D
    lout: Float_1D
    llout: Float_1D
    resp: Float_1D

    @property
    def bulkdensity_kg_m3(self):
        return self.bulkdensity * 100 * 100 * 100 / 1000

    @property
    def pore_fraction(self):
        # from alfalfa, 1 minus ratio bulk density 1.00
        # g cm-3/2.65 g cm-3, density of solids
        return 1 - self.bulkdensity / 2.65

    @property
    def mineral_fraction(self):
        return 1 - self.pore_fraction - self.peat_fraction

    @property
    def air_fraction(self):
        return self.pore_fraction - self.water_content_15cm

    # @property
    # def n_soil_1(self):
    #     return self.n_soil + 1

    # @property
    # def n_soil_2(self):
    #     return self.n_soil + 2

    @property
    def z_soil(self):
        # (n_soil_1,)
        return jnp.concatenate([jnp.array([0]), jnp.cumsum(self.dz)])

    @property
    def d2z(self):
        #  (n_soil,)
        d2z = self.z_soil[2:] - self.z_soil[:-2]
        return jnp.concatenate([self.dz[:1], d2z])

    @property
    def k_fluid(self):
        # (ntime,)
        fw = 1.0 / (1 + jnp.power((self.water_content_15cm / 0.15), -4))
        return self.K_air + fw * (self.K_water - self.K_air)  # (ntime,)

    @property
    def wt_air(self):
        return 2.0 / (3.0 * (1.0 + 0.2 * (self.K_air / self.k_fluid - 1))) + 1.0 / (
            3.0 * (1.0 + (1.0 - 2.0 * 0.2) * (self.K_air / self.k_fluid - 1))
        )

    @property
    def wt_water(self):
        return 2.0 / (3.0 * (1.0 + 0.2 * (self.K_water / self.k_fluid - 1))) + 1.0 / (
            3.0 * (1.0 + (1.0 - 2.0 * 0.2) * (self.K_water / self.k_fluid - 1))
        )

    @property
    def wt_org(self):
        return 2.0 / (3.0 * (1 + 0.2 * (self.K_org / self.k_fluid - 1))) + 1.0 / (
            3.0 * (1.0 + (1.0 - 2.0 * 0.2) * (self.K_org / self.k_fluid - 1))
        )

    @property
    def wt_mineral(self):
        return 2.0 / (3.0 * (1.0 + 0.2 * (self.K_mineral / self.k_fluid - 1))) + 1.0 / (
            3.0 * (1.0 + (1.0 - 2.0 * 0.2) * (self.K_mineral / self.k_fluid - 1))
        )

    @property
    def K_soil_num(self):
        # thermal conductivity, W m-1  K-1
        # (ntime,)
        return (
            self.mineral_fraction * self.wt_mineral * self.K_mineral
            + self.air_fraction * self.wt_air * self.K_air
            + self.water_content_15cm * self.wt_water * self.K_water
            + self.peat_fraction * self.wt_org * self.K_mineral
        )

    @property
    def K_soil(self):
        # thermal conductivity, W m-1  K-1
        # (ntime,)
        return self.K_soil_num / (
            self.mineral_fraction * self.wt_mineral
            + self.air_fraction * self.wt_air
            + self.water_content_15cm * self.wt_water
            + self.peat_fraction * self.wt_org
        )

    @property
    def cp_soil(self):
        # rho Cp Volume/ 2 dt
        # transpose dz to make 2d matrix, (ntime, n_soil)
        return jnp.outer(self.Cp_soil, self.d2z) / (2 * self.dt)

    @property
    def k_conductivity_soil(self):
        # K/dz
        # (ntime, n_soil+1)
        ntime = self.K_soil.size
        k_conductivity_soil = jnp.outer(self.K_soil, 1.0 / self.dz)  # (ntime, n_soil)
        k_conductivity_soil = jnp.concatenate(
            [k_conductivity_soil, jnp.zeros([ntime, 1])],
            axis=1,
        )
        return k_conductivity_soil

    @property
    def k_conductivity_soil_bound(self):
        return self.k_conductivity_soil[:, 0]

    @property
    def resistance_h2o(self):
        return soil_sfc_res(self.water_content_15cm)
