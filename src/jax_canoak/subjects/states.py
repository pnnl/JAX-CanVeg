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

from .meterology import Met
from .parameters import Para

from .utils import soil_sfc_res

# from ..physics.energy_fluxes.soil_energy_balance_mx import soil_sfc_res
from ..shared_utilities.types import Float_2D, Float_1D, Float_0D
from ..shared_utilities.utils import dot, minus


# dot = jax.vmap(lambda x, y: x * y, in_axes=(None, 1), out_axes=1)


class Prof(object):
    def __init__(
        self,
        zht: Float_1D,
        delz: Float_1D,
        co2: Float_2D,
        Tair_K: Float_2D,
        Told_K: Float_2D,
        eair_Pa: Float_2D,
        eair_old_Pa: Float_2D,
        wind: Float_2D,
        Tsfc: Float_2D,
        H: Float_2D,
        LE: Float_2D,
        Rnet: Float_2D,
        Ps: Float_2D,
    ) -> None:
        self.zht = zht
        self.delz = delz
        self.co2 = co2
        self.Tair_K = Tair_K
        self.Told_K = Told_K
        self.eair_Pa = eair_Pa
        self.eair_old_Pa = eair_old_Pa
        self.wind = wind
        self.Tsfc = Tsfc
        self.H = H
        self.LE = LE
        self.Rnet = Rnet
        self.Ps = Ps

    def _tree_flatten(self):
        children = (
            self.zht,
            self.delz,
            self.co2,
            self.Tair_K,
            self.Told_K,
            self.eair_Pa,
            self.eair_old_Pa,
            self.wind,
            self.Tsfc,
            self.H,
            self.LE,
            self.Rnet,
            self.Ps,
        )
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class SunAng(object):
    def __init__(
        self,
        sin_beta: Float_1D,
        beta_rad: Float_1D,
        beta_deg: Float_1D,
        theta_rad: Float_1D,
        theta_deg: Float_1D,
    ) -> None:
        self.sin_beta = sin_beta
        self.beta_rad = beta_rad
        self.beta_deg = beta_deg
        self.theta_rad = theta_rad
        self.theta_deg = theta_deg

    def _tree_flatten(self):
        children = (
            self.sin_beta,
            self.beta_rad,
            self.beta_deg,
            self.theta_rad,
            self.theta_deg,
        )
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class LeafAng(object):
    def __init__(
        self,
        pdf: Float_1D,
        Gfunc: Float_1D,
        thetaSky: Float_1D,
        Gfunc_Sky: Float_1D,
        integ_exp_diff: Float_2D,
    ) -> None:
        self.pdf = pdf
        self.Gfunc = Gfunc
        self.thetaSky = thetaSky
        self.Gfunc_Sky = Gfunc_Sky
        self.integ_exp_diff = integ_exp_diff

    def _tree_flatten(self):
        children = (
            self.pdf,
            self.Gfunc,
            self.thetaSky,
            self.Gfunc_Sky,
            self.integ_exp_diff,
        )
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class ParNir(object):
    def __init__(
        self,
        sh_abs: Float_2D,
        sun_abs: Float_2D,
        sun: Float_2D,
        dn_flux: Float_2D,
        up_flux: Float_2D,
        sun_normal: Float_2D,
        sh_flux: Float_2D,
        incoming: Float_2D,
        beam_flux: Float_2D,
        total: Float_2D,
        inbeam: Float_2D,
        indiffuse: Float_2D,
        prob_beam: Float_2D,
        prob_shade: Float_2D,
        sun_lai: Float_2D,
        shade_lai: Float_2D,
        reflect: Float_0D,
        trans: Float_0D,
        soil_refl: Float_0D,
        absorbed: Float_0D,
    ) -> None:
        self.sh_abs = sh_abs
        self.sun_abs = sun_abs
        self.sun = sun
        self.dn_flux = dn_flux
        self.up_flux = up_flux
        self.sun_normal = sun_normal
        self.sh_flux = sh_flux
        self.incoming = incoming
        self.beam_flux = beam_flux
        self.total = total
        # self.inbeam = jnp.zeros(ntime) + 999.
        self.inbeam = inbeam
        self.indiffuse = indiffuse
        self.prob_beam = prob_beam
        self.prob_shade = prob_shade
        self.sun_lai = sun_lai
        self.shade_lai = shade_lai
        self.reflect = reflect
        self.trans = trans
        self.soil_refl = soil_refl
        self.absorbed = absorbed

    def _tree_flatten(self):
        children = (
            self.sh_abs,
            self.sun_abs,
            self.sun,
            self.dn_flux,
            self.up_flux,
            self.sun_normal,
            self.sh_flux,
            self.incoming,
            self.beam_flux,
            self.total,
            self.inbeam,
            self.indiffuse,
            self.prob_beam,
            self.prob_shade,
            self.sun_lai,
            self.shade_lai,
            self.reflect,
            self.trans,
            self.soil_refl,
            self.absorbed,
        )
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class Ir(object):
    def __init__(
        self,
        ir_in: Float_1D,
        ir_dn: Float_2D,
        ir_up: Float_2D,
        IR_source_sun: Float_2D,
        IR_source_shade: Float_2D,
        IR_source: Float_2D,
        shade: Float_2D,
        shade_top: Float_2D,
        shade_bottom: Float_2D,
        balance: Float_2D,
    ) -> None:
        self.ir_in = ir_in
        self.ir_dn = ir_dn
        self.ir_up = ir_up
        self.IR_source_sun = IR_source_sun
        self.IR_source_shade = IR_source_shade
        self.IR_source = IR_source
        self.shade = shade
        self.shade_top = shade_top
        self.shade_bottom = shade_bottom
        self.balance = balance

    def _tree_flatten(self):
        children = (
            self.ir_in,
            self.ir_dn,
            self.ir_up,
            self.IR_source_sun,
            self.IR_source_shade,
            self.IR_source,
            self.shade,
            self.shade_top,
            self.shade_bottom,
            self.balance,
        )
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class Rnet(object):
    def __init__(
        self,
        sun: Float_2D,
        sh: Float_2D,
        sun_top: Float_2D,
        sh_top: Float_2D,
        sh_bottom: Float_2D,
    ) -> None:
        self.sun = sun
        self.sh = sh
        self.sun_top = sun_top
        self.sh_top = sh_top
        self.sh_bottom = sh_bottom

    def _tree_flatten(self):
        children = (self.sun, self.sh, self.sun_top, self.sh_top, self.sh_bottom)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class SunShadedCan(object):
    def __init__(
        self,
        Ps: Float_2D,
        Resp: Float_2D,
        gs: Float_2D,
        LE: Float_2D,
        H: Float_2D,
        Rnet: Float_2D,
        Lout: Float_2D,
        closure: Float_2D,
        Tsfc: Float_2D,
        Tsfc_new: Float_2D,
        Tsfc_old: Float_2D,
    ) -> None:
        self.Ps = Ps
        self.Resp = Resp
        self.gs = gs
        self.LE = LE
        self.H = H
        self.Rnet = Rnet
        self.Lout = Lout
        self.closure = closure
        self.Tsfc = Tsfc
        self.Tsfc_old = Tsfc_old
        self.Tsfc_new = Tsfc_new

    def _tree_flatten(self):
        children = (
            self.Ps,
            self.Resp,
            self.gs,
            self.LE,
            self.H,
            self.Rnet,
            self.Lout,
            self.closure,
            self.Tsfc,
            self.Tsfc_new,
            self.Tsfc_old,
        )
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class BoundLayerRes(object):
    def __init__(self, heat: Float_2D, vapor: Float_2D, co2: Float_2D) -> None:
        self.heat = heat
        self.vapor = vapor
        self.co2 = co2
        # self.heat = jnp.zeros([ntime, jtot])
        # self.vapor = jnp.zeros([ntime, jtot])
        # self.co2 = jnp.zeros([ntime, jtot])

    def _tree_flatten(self):
        children = (self.heat, self.vapor, self.co2)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class Qin(object):
    def __init__(self, sun_abs: Float_2D, shade_abs: Float_2D) -> None:
        self.sun_abs = sun_abs
        self.shade_abs = shade_abs

    def _tree_flatten(self):
        children = (self.sun_abs, self.shade_abs)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class Veg(object):
    def __init__(
        self, Ps: Float_1D, Rd: Float_1D, H: Float_1D, LE: Float_1D, Tsfc: Float_1D
    ) -> None:
        self.Ps = Ps
        self.Rd = Rd
        self.H = H
        self.LE = LE
        self.Tsfc = Tsfc

    def _tree_flatten(self):
        children = (self.Ps, self.Rd, self.H, self.LE, self.Tsfc)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class Lai(object):
    def __init__(
        self,
        lai: Float_1D,
        dff: Float_2D,
        sumlai: Float_2D,
        dff_clmp: Float_2D,
        adens: Float_2D,
    ) -> None:
        self.lai = lai
        self.dff = dff
        self.sumlai = sumlai
        self.dff_clmp = dff_clmp
        self.adens = adens

    def _tree_flatten(self):
        children = (self.lai, self.dff, self.sumlai, self.dff_clmp, self.adens)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class Ps(object):
    def __init__(
        self,
        aphoto: Float_2D,
        ci: Float_2D,
        gs_co2: Float_2D,
        gs_m_s: Float_2D,
        wj: Float_2D,
        wc: Float_2D,
        wp: Float_2D,
        jsucrose: Float_2D,
        Ag: Float_2D,
        x1: Float_2D,
        x2: Float_2D,
        x3: Float_2D,
        p: Float_2D,
        q: Float_2D,
        r: Float_2D,
        rd: Float_2D,
        rstom: Float_2D,
    ) -> None:
        self.aphoto = aphoto
        self.ci = ci
        self.gs_co2 = gs_co2
        self.gs_m_s = gs_m_s
        self.wj = wj
        self.wc = wc
        self.wp = wp
        self.jsucrose = jsucrose
        self.Ag = Ag
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.p = p
        self.q = q
        self.r = r
        self.rd = rd
        self.rstom = rstom

    def _tree_flatten(self):
        children = (
            self.aphoto,
            self.ci,
            self.gs_co2,
            self.wj,
            self.wc,
            self.wp,
            self.jsucrose,
            self.Ag,
            self.x1,
            self.x2,
            self.x3,
            self.p,
            self.q,
            self.r,
            self.rd,
            self.rstom,
        )
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


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


def initialize_model_states(met: Met, para: Para):
    # Soil
    soil = initialize_soil(para, met)

    # Quantum and NIR
    quantum = initialize_parnir(para, "par")
    nir = initialize_parnir(para, "nir")

    # IR
    ir = initialize_ir(para)

    # Veg
    veg = initialize_veg(para)

    # Qin
    qin = initialize_qin(para)

    # RNet
    rnet = initialize_rnet(para)

    # Sun
    sun = initialize_sunshade(para, met)

    # Shade
    shade = initialize_sunshade(para, met)

    # Lai
    lai = initialize_lai(para, met)

    return soil, quantum, nir, ir, veg, qin, rnet, sun, shade, lai


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
    incoming = jnp.zeros(ntime)
    beam_flux = jnp.zeros([ntime, jktot])
    total = jnp.zeros([ntime, jktot])
    inbeam = jnp.zeros(ntime)
    indiffuse = jnp.zeros(ntime)
    prob_beam = jnp.zeros([ntime, jktot])
    prob_shade = jnp.ones([ntime, jktot])
    sun_lai = jnp.zeros([ntime, jktot])
    shade_lai = jnp.zeros([ntime, jktot])
    if wavebnd == "par":
        reflect = para.par_reflect
        trans = para.par_trans
        soil_refl = para.par_soil_refl
        absorbed = para.par_absorbed
    elif wavebnd == "nir":
        reflect = para.nir_reflect
        trans = para.nir_trans
        soil_refl = para.nir_soil_refl
        absorbed = para.nir_absorbed
    rad = ParNir(
        sh_abs,
        sun_abs,
        sun,
        dn_flux,
        up_flux,
        sun_normal,
        sh_flux,
        incoming,
        beam_flux,
        total,
        inbeam,
        indiffuse,
        prob_beam,
        prob_shade,
        sun_lai,
        shade_lai,
        reflect,  # pyright: ignore
        trans,  # pyright: ignore
        soil_refl,  # pyright: ignore
        absorbed,  # pyright: ignore
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
    Rd = jnp.zeros(ntime)
    H = jnp.zeros(ntime)
    LE = jnp.zeros(ntime)
    Tsfc = jnp.zeros(ntime)
    return Veg(Ps, Rd, H, LE, Tsfc)


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
        Ps, Resp, gs, LE, H, Rnet, Lout, closure, Tsfc, Tsfc_new, Tsfc_old
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
