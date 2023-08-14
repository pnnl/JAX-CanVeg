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
- Soil()

Author: Peishi Jiang
Date: 2023.7.24.
"""

import jax
import jax.numpy as jnp
from math import floor

import equinox as eqx

# from ..physics.energy_fluxes import conc_mx

from .meterology import Met
from .parameters import Para
from .utils import soil_sfc_res, conc_mx
from .utils import llambda as flambda

# from ..physics.energy_fluxes.soil_energy_balance_mx import soil_sfc_res
from ..shared_utilities.types import Float_2D, Float_1D, Float_0D
from ..shared_utilities.utils import dot, minus
from ..shared_utilities.constants import PI


# dot = jax.vmap(lambda x, y: x * y, in_axes=(None, 1), out_axes=1)


class Prof(eqx.Module):
    zht: Float_1D
    delz: Float_1D
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
    inbeam: Float_2D
    indiffuse: Float_2D
    prob_beam: Float_2D
    sun_lai: Float_2D
    shade_lai: Float_2D
    reflect: Float_0D
    trans: Float_0D
    soil_refl: Float_0D

    @property
    def total(self):
        return self.up_flux + self.dn_flux

    @property
    def incoming(self):
        return self.inbeam + self.indiffuse

    @property
    def prob_shade(self):
        return 1 - self.prob_beam

    @property
    def absorbed(self):
        return 1 - self.reflect - self.trans


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


class Soil(object):
    def __init__(
        self,
        dt: Float_0D,
        n_soil: int,
        depth: Float_0D,
        mtime: int,
        water_content_15cm: Float_1D,
        water_content_litter: Float_0D,
        bulkdensity: Float_0D,
        bulkdensity_kg_m3: Float_0D,
        pore_fraction: Float_0D,
        clay_fraction: Float_0D,
        peat_fraction: Float_0D,
        mineral_fraction: Float_0D,
        air_fraction: Float_1D,
        Cp_water: Float_0D,
        Cp_air: Float_0D,
        Cp_org: Float_0D,
        Cp_mineral: Float_0D,
        K_mineral: Float_0D,
        K_org: Float_0D,
        K_water: Float_0D,
        n_soil_1: int,
        n_soil_2: int,
        dz: Float_1D,
        z_soil: Float_1D,
        d2z: Float_1D,
        vol: Float_1D,
        T_soil: Float_2D,
        T_soil_old: Float_2D,
        T_soil_up_boundary: Float_1D,
        sfc_temperature: Float_1D,
        sfc_temperature_old: Float_1D,
        bulk_density: Float_2D,
        K_air: Float_1D,
        Cp_soil: Float_1D,
        K_soil: Float_1D,
        cp_soil: Float_2D,
        k_conductivity_soil: Float_2D,
        k_conductivity_soil_bound: Float_1D,
        evap: Float_1D,
        heat: Float_1D,
        rnet: Float_1D,
        gsoil: Float_1D,
        lout: Float_1D,
        llout: Float_1D,
        resistance_h2o: Float_1D,
        T_soil_low_bound: Float_1D,
    ) -> None:
        self.dt = dt
        self.n_soil = n_soil
        self.depth = depth
        self.mtime = mtime
        self.water_content_15cm = water_content_15cm
        self.water_content_litter = water_content_litter
        self.bulkdensity = bulkdensity
        self.bulkdensity_kg_m3 = bulkdensity_kg_m3
        self.pore_fraction = pore_fraction
        self.clay_fraction = clay_fraction
        self.peat_fraction = peat_fraction
        self.mineral_fraction = mineral_fraction
        self.air_fraction = air_fraction
        self.Cp_water = Cp_water
        self.Cp_air = Cp_air
        self.Cp_org = Cp_org
        self.Cp_mineral = Cp_mineral
        self.K_mineral = K_mineral
        self.K_org = K_org
        self.K_water = K_water
        self.n_soil_1 = n_soil_1
        self.n_soil_2 = n_soil_2
        self.dz = dz
        self.z_soil = z_soil
        self.d2z = d2z
        self.vol = vol
        self.T_soil = T_soil
        self.T_soil_old = T_soil_old
        self.T_soil_up_boundary = T_soil_up_boundary
        self.sfc_temperature = sfc_temperature
        self.sfc_temperature_old = sfc_temperature_old
        self.bulk_density = bulk_density
        self.K_air = K_air
        self.Cp_soil = Cp_soil
        self.K_soil = K_soil
        self.cp_soil = cp_soil
        self.k_conductivity_soil = k_conductivity_soil
        self.k_conductivity_soil_bound = k_conductivity_soil_bound
        self.evap = evap
        self.heat = heat
        self.rnet = rnet
        self.gsoil = gsoil
        self.lout = lout
        self.llout = llout
        self.resistance_h2o = resistance_h2o
        self.T_soil_low_bound = T_soil_low_bound

    def _tree_flatten(self):
        children = (
            # self.dt,
            # self.n_soil,
            # self.depth,
            # self.mtime,
            self.water_content_15cm,
            self.water_content_litter,
            self.bulkdensity,
            self.bulkdensity_kg_m3,
            self.pore_fraction,
            self.clay_fraction,
            self.peat_fraction,
            self.mineral_fraction,
            self.air_fraction,
            self.Cp_water,
            self.Cp_air,
            self.Cp_org,
            self.Cp_mineral,
            self.K_mineral,
            self.K_org,
            self.K_water,
            self.n_soil_1,
            self.n_soil_2,
            self.dz,
            self.z_soil,
            self.d2z,
            self.vol,
            self.T_soil,
            self.T_soil_old,
            self.T_soil_up_boundary,
            self.sfc_temperature,
            self.sfc_temperature_old,
            self.bulk_density,
            self.K_air,
            self.Cp_soil,
            self.K_soil,
            self.cp_soil,
            self.k_conductivity_soil,
            self.k_conductivity_soil_bound,
            self.evap,
            self.heat,
            self.rnet,
            self.gsoil,
            self.lout,
            self.llout,
            self.resistance_h2o,
            self.T_soil_low_bound,
        )
        aux_data = {
            "dt": self.dt,
            "n_soil": self.n_soil,
            "depth": self.depth,
            "mtime": self.mtime,
        }
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        # return cls(*children, **aux_data)
        return cls(
            aux_data["dt"],
            aux_data["n_soil"],
            aux_data["depth"],
            aux_data["mtime"],
            *children
        )


def initialize_profile_mx(met: Met, para: Para):
    ntime, jtot = para.ntime, para.jtot
    nlayers = para.nlayers_atmos
    zht = para.zht
    delz = para.delz
    co2 = jnp.ones([ntime, nlayers])
    Tair_K = jnp.ones([ntime, nlayers])
    Told_K = jnp.ones([ntime, nlayers])
    eair_Pa = jnp.ones([ntime, nlayers])
    eair_old_Pa = jnp.ones([ntime, nlayers])
    wind = jnp.zeros([ntime, jtot])
    Tsfc = jnp.zeros([ntime, jtot])
    H = jnp.zeros([ntime, jtot])
    LE = jnp.zeros([ntime, jtot])
    Rnet = jnp.zeros([ntime, jtot])
    Ps = jnp.zeros([ntime, jtot])

    co2 = dot(met.CO2, co2)
    Tair_K = dot(met.T_air_K, Tair_K)
    Told_K = dot(met.T_air_K, Told_K)
    eair_Pa = dot(met.eair_Pa, eair_Pa)
    eair_old_Pa = dot(met.eair_Pa, eair_old_Pa)

    prof = Prof(
        zht,
        delz,
        co2,
        Tair_K,
        Told_K,
        eair_Pa,
        eair_old_Pa,
        wind,
        Tsfc,
        H,
        LE,
        Rnet,
        Ps,
    )
    return prof


def update_profile_mx(
    met: Met,
    para: Para,
    prof: Prof,
    quantum: ParNir,
    sun: SunShadedCan,
    shade: SunShadedCan,
    soil: Soil,
    lai: Lai,
    dij: Float_2D,
) -> Prof:
    Ps = (
        quantum.prob_beam[:, : para.nlayers] * sun.Ps
        + quantum.prob_shade[:, : para.nlayers] * shade.Ps
    ) * lai.adens
    LE = (
        quantum.prob_beam[:, : para.nlayers] * sun.LE
        + quantum.prob_shade[:, : para.nlayers] * shade.LE
    ) * lai.adens
    H = (
        quantum.prob_beam[:, : para.nlayers] * sun.H
        + quantum.prob_shade[:, : para.nlayers] * shade.H
    ) * lai.adens
    Rnet = (
        quantum.prob_beam[:, : para.nlayers] * sun.Rnet
        + quantum.prob_shade[:, : para.nlayers] * shade.Rnet
    ) * lai.adens
    Tsfc = (
        quantum.prob_beam[:, : para.nlayers] * sun.Tsfc
        + quantum.prob_shade[:, : para.nlayers] * shade.Tsfc
    )

    # Compute scalar profiles
    # it needs information on source/sink, Dij, soil boundary flux and factor for units
    fact_heatcoef = met.air_density * para.Cp
    soilflux = soil.heat  # assume soil heat flux is 20 W m-2 until soil sub is working
    Tair_K = conc_mx(
        H,
        soilflux,
        prof.delz,
        dij,
        met.ustar,
        met.zL,
        met.T_air_K,
        para.jtot,
        para.nlayers_atmos,
        fact_heatcoef,
    )
    Told_K = prof.Told_K

    # with larger Dij value I need to filter new T profiles
    Tair_K = 0.25 * Tair_K + 0.75 * Told_K
    Told_K = Tair_K

    # Compute vapor pressure profiles
    soilflux = soil.evap  # W m-2
    # in fConcMatrix fact.lecoef is in the denominator insteat of multiplier
    # if we divide W m -2 = J m-2 s-1 by Lambda we have g m-2 s-1
    # need to convert g to Pa
    # eair =rhovair R Tk/mv  Jones
    fact_lecoef = (
        flambda(Tair_K[:, para.jktot - 1])
        * 18.01
        / (1000 * 8.314 * Tair_K[:, para.jktot - 1])
    )  # noqa: E501
    eair_Pa = conc_mx(
        LE,
        soil.evap,
        prof.delz,
        dij,
        met.ustar,
        met.zL,
        met.eair_Pa,
        para.jtot,
        para.nlayers_atmos,
        fact_lecoef,
    )
    eair_old_Pa = prof.eair_old_Pa
    eair_Pa = 0.25 * eair_Pa + 0.75 * eair_old_Pa
    eair_old_Pa = eair_Pa

    # # TODO: Compute CO2 profiles
    # fact_co2=(28.97/44)*met.air_density_mole
    # Rsoil = SoilRespiration(
    #    Veg.Ps,soil.T_soil(:,10),met.soilmoisture,met.zcanopy,Veg.Rd,prm)
    # soilflux=Rsoil.Respiration;
    # [prof.co2]=fConcMatrix(-prof.Ps,soilflux, prof.delz, Dij,met,met.CO2,prm,fact.co2)

    # Update
    prof = eqx.tree_at(
        lambda t: (
            t.Ps,
            t.LE,
            t.H,
            t.Rnet,
            t.Tsfc,
            t.Tair_K,
            t.Told_K,
            t.eair_Pa,
            t.eair_old_Pa,
        ),
        prof,
        (Ps, LE, H, Rnet, Tsfc, Tair_K, Told_K, eair_Pa, eair_old_Pa),
    )

    return prof


def calculate_veg_mx(
    para: Para, lai: Lai, quantum: ParNir, sun: SunShadedCan, shade: SunShadedCan
) -> Veg:
    veg_Ps = jnp.sum(
        (
            quantum.prob_beam[:, : para.nlayers] * sun.Ps
            + quantum.prob_shade[:, : para.nlayers] * shade.Ps
        )
        * lai.dff,
        axis=1,
    )
    veg_Rd = jnp.sum(
        (
            quantum.prob_beam[:, : para.nlayers] * sun.Resp
            + quantum.prob_shade[:, : para.nlayers] * shade.Resp
        )
        * lai.dff,
        axis=1,
    )
    veg_LE = jnp.sum(
        (
            quantum.prob_beam[:, : para.nlayers] * sun.LE
            + quantum.prob_shade[:, : para.nlayers] * shade.LE
        )
        * lai.dff,
        axis=1,
    )
    veg_H = jnp.sum(
        (
            quantum.prob_beam[:, : para.nlayers] * sun.H
            + quantum.prob_shade[:, : para.nlayers] * shade.H
        )
        * lai.dff,
        axis=1,
    )
    veg_gs = jnp.sum(
        (
            quantum.prob_beam[:, : para.nlayers] * sun.gs
            + quantum.prob_shade[:, : para.nlayers] * shade.gs
        )
        * lai.dff,
        axis=1,
    )
    veg_Rnet = jnp.sum(
        (
            quantum.prob_beam[:, : para.nlayers] * sun.Rnet
            + quantum.prob_shade[:, : para.nlayers] * shade.Rnet
        )
        * lai.dff,
        axis=1,
    )
    veg_Tsfc = jnp.sum(
        (
            quantum.prob_beam[:, : para.nlayers] * sun.Tsfc
            + quantum.prob_shade[:, : para.nlayers] * shade.Tsfc
        )
        * lai.dff,
        axis=1,
    )
    veg_Tsfc = veg_Tsfc / lai.lai
    veg_vpd = jnp.sum(
        (
            quantum.prob_beam[:, : para.nlayers] * sun.vpd_Pa
            + quantum.prob_shade[:, : para.nlayers] * shade.vpd_Pa
        )
        * lai.dff,
        axis=1,
    )
    veg_vpd = veg_vpd / lai.lai

    veg = Veg(veg_Ps, veg_gs, veg_Rd, veg_H, veg_LE, veg_Rnet, veg_Tsfc, veg_vpd)

    return veg


def initialize_model_states(met: Met, para: Para):
    # Soil
    soil = initialize_soil(para, met)

    # Quantum and NIR
    quantum = initialize_parnir(para, "par")
    nir = initialize_parnir(para, "nir")

    # IR
    ir = initialize_ir(para)

    # # Veg
    # veg = initialize_veg(para)

    # Qin
    qin = initialize_qin(para)

    # TODO: remove RNet since it is not used
    # RNet
    rnet = initialize_rnet(para)

    # Sun
    sun = initialize_sunshade(para, met)

    # Shade
    shade = initialize_sunshade(para, met)

    # Lai
    lai = initialize_lai(para, met)

    return soil, quantum, nir, ir, qin, rnet, sun, shade, lai


def initialize_parnir(para: Para, wavebnd: str) -> ParNir:
    ntime, jtot = para.ntime, para.jtot
    jktot = jtot + 1
    sh_abs = jnp.zeros([ntime, jtot])
    sun_abs = jnp.zeros([ntime, jtot])
    sun = jnp.zeros([ntime, jktot])
    dn_flux = jnp.zeros([ntime, jktot])
    up_flux = jnp.zeros([ntime, jktot])
    sun_normal = jnp.zeros([ntime, jktot])
    sh_flux = jnp.zeros([ntime, jktot])
    # incoming = jnp.zeros(ntime)
    beam_flux = jnp.zeros([ntime, jktot])
    # total = jnp.zeros([ntime, jktot])
    inbeam = jnp.zeros(ntime)
    indiffuse = jnp.zeros(ntime)
    prob_beam = jnp.zeros([ntime, jktot])
    # prob_shade = jnp.ones([ntime, jktot])
    sun_lai = jnp.zeros([ntime, jktot])
    shade_lai = jnp.zeros([ntime, jktot])
    if wavebnd == "par":
        reflect = jnp.array(para.par_reflect)
        trans = jnp.array(para.par_trans)
        soil_refl = jnp.array(para.par_soil_refl)
        # absorbed = jnp.array(para.par_absorbed)
    elif wavebnd == "nir":
        reflect = jnp.array(para.nir_reflect)
        trans = jnp.array(para.nir_trans)
        soil_refl = jnp.array(para.nir_soil_refl)
        # absorbed = jnp.array(para.nir_absorbed)
    rad = ParNir(
        sh_abs,
        sun_abs,
        sun,
        dn_flux,
        up_flux,
        sun_normal,
        sh_flux,
        # incoming,
        beam_flux,
        # total,
        inbeam,
        indiffuse,
        prob_beam,
        # prob_shade,
        sun_lai,
        shade_lai,
        reflect,  # pyright: ignore
        trans,  # pyright: ignore
        soil_refl,  # pyright: ignore
        # absorbed,  # pyright: ignore
    )
    return rad


def initialize_ir(para: Para) -> Ir:
    jktot, jtot, ntime = para.jktot, para.jtot, para.ntime
    ir_in = jnp.zeros(ntime)
    ir_dn = jnp.ones([ntime, jktot])
    ir_up = jnp.ones([ntime, jktot])
    # IR_source_sun = jnp.zeros([ntime, jktot])
    # IR_source_shade = jnp.zeros([ntime, jktot])
    # IR_source = jnp.zeros([ntime, jktot])
    IR_source_sun = jnp.zeros([ntime, jtot])
    IR_source_shade = jnp.zeros([ntime, jtot])
    IR_source = jnp.zeros([ntime, jtot])
    shade = jnp.zeros([ntime, jtot])
    shade_top = jnp.zeros([ntime, jktot])
    shade_bottom = jnp.zeros([ntime, jktot])
    balance = jnp.zeros([ntime, jtot])
    ir = Ir(
        ir_in,
        ir_dn,
        ir_up,
        IR_source_sun,
        IR_source_shade,
        IR_source,
        shade,
        shade_top,
        shade_bottom,
        balance,
    )
    return ir


def initialize_veg(para: Para) -> Veg:
    ntime = para.ntime
    Ps = jnp.zeros(ntime)
    gs = jnp.zeros(ntime)
    Rd = jnp.zeros(ntime)
    H = jnp.zeros(ntime)
    LE = jnp.zeros(ntime)
    Rnet = jnp.zeros(ntime)
    Tsfc = jnp.zeros(ntime)
    vpd = jnp.zeros(ntime)
    return Veg(Ps, gs, Rd, H, LE, Rnet, Tsfc, vpd)


def initialize_qin(para: Para) -> Qin:
    ntime, jtot = para.ntime, para.jtot
    sun_abs = jnp.zeros([ntime, jtot])
    shade_abs = jnp.zeros([ntime, jtot])
    return Qin(sun_abs, shade_abs)


def initialize_rnet(para: Para) -> Rnet:
    ntime, jktot = para.ntime, para.jktot
    sun = jnp.zeros([ntime, jktot])
    sh = jnp.zeros([ntime, jktot])
    sun_top = jnp.zeros([ntime, jktot])
    sh_top = jnp.zeros([ntime, jktot])
    sh_bottom = jnp.zeros([ntime, jktot])
    return Rnet(sun, sh, sun_top, sh_top, sh_bottom)


def initialize_sunshade(para: Para, met: Met) -> SunShadedCan:
    ntime, jtot = para.ntime, para.jtot
    Ps = jnp.zeros([ntime, jtot])
    Resp = jnp.zeros([ntime, jtot])
    gs = jnp.zeros([ntime, jtot])
    vpd_Pa = jnp.zeros([ntime, jtot])
    LE = jnp.zeros([ntime, jtot])
    H = jnp.zeros([ntime, jtot])
    Rnet = jnp.zeros([ntime, jtot])
    Lout = jnp.zeros([ntime, jtot])
    closure = jnp.zeros([ntime, jtot])
    Tsfc = jnp.ones([ntime, jtot])
    Tsfc_old = jnp.ones([ntime, jtot])
    Tsfc_new = jnp.ones([ntime, jtot])

    Tsfc = dot(met.T_air_K, Tsfc)

    return SunShadedCan(
        Ps, Resp, gs, vpd_Pa, LE, H, Rnet, Lout, closure, Tsfc, Tsfc_new, Tsfc_old
    )


def initialize_lai(para: Para, met: Met) -> Lai:
    lai = met.lai
    dff = jnp.ones([para.ntime, para.nlayers]) / para.nlayers  # (ntime,nlayers)
    dff = dot(lai, dff)  # (ntime, nlayers)
    # TODO: double check!
    # self.sumlai = jax.lax.cumsum(self.dff, axis=1, reverse=True) #(ntime,nlayers)
    sumlai = minus(lai, jax.lax.cumsum(dff, axis=1))  # (ntime, nlayers)
    sumlai = jnp.clip(sumlai, a_min=0.0)  # (ntime, nlayers)
    dff_clmp = dff / para.markov  # (ntime, nlayers)

    # divide by height of the layers in the canopy
    adens = dff[:, : para.nlayers] / para.dht_canopy  # (ntime, nlayers)

    return Lai(lai, dff, sumlai, dff_clmp, adens)


def initialize_soil(
    para: Para, met: Met, dt: Float_0D = 20.0, n_soil: int = 10, depth: Float_0D = 0.15
) -> Soil:
    # Soil water content
    water_content_15cm = (
        met.soilmoisture
    )  # measured at 10 cmwater content of soil m3 m-3    # noqa: E501

    # Water content of litter. Values ranged between 0.02 and 0.126
    water_content_litter = 0.0  # assumed constant but needs to vary  # noqa: E501

    # fraction porosity + mineral + organic = 1
    # airborne fraction = porosity - volumetric water content
    bulkdensity = 1.06  # g cm-3   Data from Tyler Anthony
    bulkdensity_kg_m3 = bulkdensity * 100 * 100 * 100 / 1000
    pore_fraction = (
        1 - bulkdensity / 2.65
    )  # from alfalfa, 1 minus ratio bulk density 1.00 g cm-3/2.65 g cm-3, density of solids  # noqa: E501
    clay_fraction = 0.3  #  Clay fraction
    peat_fraction = 0.08  #  SOM = a C; C = 4.7%, a = 1.72  Kuno Kasak, 2019 Bouldin Alfalfa  # noqa: E501
    mineral_fraction = (
        1 - pore_fraction - peat_fraction
    )  # from bulk density asssuming density of solids is 2.65  # noqa: E501
    air_fraction = pore_fraction - met.soilmoisture

    # J kg-1 K-1, heat capacity
    Cp_water = 4180.0
    Cp_air = 1065.0
    Cp_org = 1920.0
    Cp_mineral = 870.0

    # W m-1 K-1, thermal conductivity
    K_mineral = 2.5
    K_org = 0.8
    K_water = 0.25

    # Time step in seconds
    dt = dt
    mtime = floor(1800.0 / dt)  # time steps per half hour

    n_soil = n_soil  # number of soil layers
    n_soil_1 = n_soil + 1  # number of soil levels
    n_soil_2 = n_soil + 2  # number of soil levels

    # Compute soils depths, from 0 to base
    # we define n layers and n+1 levels, including top and bottom boundaries
    # define Geometric Depths in Soil
    depth = depth

    # exponential change in soil depth
    nfact = jnp.power(2, jnp.arange(1, n_soil_1))
    dz = depth / jnp.sum(nfact) * nfact  # n_soil
    z_soil = jnp.concatenate([jnp.array([0]), jnp.cumsum(dz)])  # n_soil_1
    d2z = z_soil[2:] - z_soil[:-2]
    d2z = jnp.concatenate([dz[:1], d2z])  # n_soil

    # Soil volume
    aream2 = 1  # meter squared area
    vol = aream2 * dz

    # initialize soil temperature to the deep base temperature at 15 cm
    # self.T_soil = jnp.ones([para.ntime, self.n_soil_2]) * met.Tsoil + 273.15
    # jax.debug.print("{a}", a=met.Tsoil)
    T_soil = (
        dot(met.Tsoil, jnp.ones([para.ntime, n_soil_2])) + 273.15
    )  # (ntime, n_soil_2)  # noqa: E501

    # initialize upper boundary temperature as air temperature
    # in later iterations this is reset to the radiative surface
    # temperature
    T_soil_up_boundary = met.T_air_K
    sfc_temperature = T_soil_up_boundary
    sfc_temperature_old = sfc_temperature
    bulk_density = (
        jnp.ones([para.ntime, n_soil]) * 0.83
    )  # soil bulk density for the alfalfa, g cm-3, (ntime, n_soil)  # noqa: E501

    # thermal conductivity code from Campbell and Norman
    fw = 1.0 / (
        1 + jnp.power((met.soilmoisture / 0.15), -4)
    )  # terms for Stefan flow as water evaporates in the pores  # noqa: E501
    K_air = (
        0.024 + 44100 * 2.42e-5 * fw * met.air_density_mole * met.dest / met.P_Pa
    )  # (ntime,)  # noqa: E501
    k_fluid = K_air + fw * (K_water - K_air)  # (ntime,)
    wt_air = 2.0 / (3.0 * (1.0 + 0.2 * (K_air / k_fluid - 1))) + 1.0 / (
        3.0 * (1.0 + (1.0 - 2.0 * 0.2) * (K_air / k_fluid - 1))
    )  # noqa: E501
    wt_water = 2.0 / (3.0 * (1.0 + 0.2 * (K_water / k_fluid - 1))) + 1.0 / (
        3.0 * (1.0 + (1.0 - 2.0 * 0.2) * (K_water / k_fluid - 1))
    )  # noqa: E501
    wt_org = 2.0 / (3.0 * (1 + 0.2 * (K_org / k_fluid - 1))) + 1.0 / (
        3.0 * (1.0 + (1.0 - 2.0 * 0.2) * (K_org / k_fluid - 1))
    )  # noqa: E501
    wt_mineral = 2.0 / (3.0 * (1.0 + 0.2 * (K_mineral / k_fluid - 1))) + 1.0 / (
        3.0 * (1.0 + (1.0 - 2.0 * 0.2) * (K_mineral / k_fluid - 1))
    )  # noqa: E501

    # compute heat capacity and thermal conductivty weighting by mineral, water, air
    # and organic fractions
    # rho_s Cs dT/dt = d(k dT/dz)/dz
    # soil density rho (kg m-3) *  Heat Capacity, J kg-1 K-1 -> J m-3 K-1
    # factors of 1000 is kg H2O m-3, or density of mineral soil (2650) and
    # peat (1300) in terms of kg m-3
    Cp_soil = (
        met.air_density * air_fraction * Cp_air
        + 1000.000 * Cp_water * met.soilmoisture
        + 1300.000 * Cp_org * peat_fraction
        + 2650.000 * Cp_mineral * mineral_fraction
    )  # (ntime,)

    # thermal conductivity, W m-1  K-1
    K_soil_num = (
        mineral_fraction * wt_mineral * K_mineral
        + air_fraction * wt_air * K_air
        + water_content_15cm * wt_water * K_water
        + peat_fraction * wt_org * K_mineral
    )  # (ntime,)
    K_soil = K_soil_num / (
        mineral_fraction * wt_mineral
        + air_fraction * wt_air
        + water_content_15cm * wt_water
        + peat_fraction * wt_org
    )  # (ntime,)

    # rho Cp Volume/ 2 dt
    cp_soil = jnp.outer(Cp_soil, d2z) / (
        2 * dt
    )  # transpose dz to make 2d matrix, (ntime, n_soil)  # noqa: E501

    # K/dz
    # self.k_conductivity_soil = self.K_soil / self.dz  # (ntime, n_soil)
    k_conductivity_soil = jnp.outer(K_soil, 1.0 / dz)  # (ntime, n_soil)
    k_conductivity_soil_bound = k_conductivity_soil[:, 0]
    k_conductivity_soil = jnp.concatenate(
        # [k_conductivity_soil[:, :-1], jnp.zeros([para.ntime, 1])], axis=1
        [k_conductivity_soil, jnp.zeros([para.ntime, 1])],
        axis=1,
    )  # (ntime, n_soil+1)
    # self.k_conductivity_soil[:,-1]=0

    # Energy initialization
    evap = jnp.zeros(para.ntime)  # initialization
    heat = jnp.zeros(para.ntime)
    # self.lout=jnp.zeros(para.ntime)
    rnet = jnp.zeros(para.ntime)
    gsoil = jnp.zeros(para.ntime)
    lout = para.epsigma * jnp.power(met.T_air_K, 4)  # initialization
    llout = lout

    resistance_h2o = soil_sfc_res(water_content_15cm)

    # lower boundary
    T_soil_low_bound = met.Tsoil + 273.15
    T_soil = jnp.concatenate(
        [T_soil[:, :-1], jnp.expand_dims(T_soil_low_bound, axis=1)],
        axis=1,
    )
    T_soil_old = T_soil

    return Soil(
        dt,
        n_soil,
        depth,
        mtime,
        water_content_15cm,
        water_content_litter,
        bulkdensity,
        bulkdensity_kg_m3,
        pore_fraction,
        clay_fraction,
        peat_fraction,
        mineral_fraction,
        air_fraction,
        Cp_water,
        Cp_air,
        Cp_org,
        Cp_mineral,
        K_mineral,
        K_org,
        K_water,
        n_soil_1,
        n_soil_2,
        dz,
        z_soil,
        d2z,
        vol,
        T_soil,
        T_soil_old,
        T_soil_up_boundary,
        sfc_temperature,
        sfc_temperature_old,
        bulk_density,
        K_air,
        Cp_soil,
        K_soil,
        cp_soil,
        k_conductivity_soil,
        k_conductivity_soil_bound,
        evap,
        heat,
        rnet,
        gsoil,
        lout,
        llout,
        resistance_h2o,
        T_soil_low_bound,
    )


# def initialize_model_states(met: Met, para: Para):
#     soil = Soil(met, para)
#     quantum, nir = ParNir(para.ntime, para.jtot), ParNir(para.ntime, para.jtot)
#     quantum.reflect = para.par_reflect  # reflectance of leaf
#     quantum.trans = para.par_trans  # transmittances of leaf
#     quantum.soil_refl = para.par_soil_refl  # soil reflectances
#     quantum.absorbed = para.par_absorbed  # fraction absorbed
#     nir.reflect = para.nir_reflect  # reflectance of leaf
#     nir.trans = para.nir_trans  # transmittances of leaf
#     nir.soil_refl = para.nir_soil_refl  # soil reflectances
#     nir.absorbed = para.nir_absorbed  # fraction absorbed

#     ir, veg = Ir(para.ntime, para.jtot), Veg(para.ntime)
#     qin, rnet = Qin(para.ntime, para.jtot), Rnet(para.ntime, para.jktot)
#     sun, shade = SunShadedCan(para.ntime, para.jtot), SunShadedCan(
#         para.ntime, para.jtot
#     )  # noqa: E501
#     # dot = jax.vmap(lambda x, y: x * y, in_axes=(None, 1), out_axes=1)
#     sun.Tsfc = dot(met.T_air_K, sun.Tsfc)
#     shade.Tsfc = dot(met.T_air_K, shade.Tsfc)

#     return soil, quantum, nir, ir, veg, qin, rnet, sun, shade
