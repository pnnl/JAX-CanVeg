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

import equinox as eqx

from .meterology import Met
from .parameters import Para, Setup
from .utils import soil_sfc_res, conc
from .utils import llambda as flambda

from ..shared_utilities.types import Float_2D, Float_1D, Float_0D
from ..shared_utilities.utils import dot, minus
from ..shared_utilities.constants import PI


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
    inbeam: Float_1D
    indiffuse: Float_1D
    prob_beam: Float_2D
    sun_lai: Float_2D
    shade_lai: Float_2D
    reflect: Float_0D
    trans: Float_0D
    soil_refl: Float_0D

    @property
    def total(self):
        return self.beam_flux + self.dn_flux

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


class Can(eqx.Module):
    rnet_calc: Float_1D
    rnet: Float_1D
    LE: Float_1D
    H: Float_1D
    NEE: Float_1D
    avail: Float_1D
    gsoil: Float_1D
    albedo_calc: Float_1D
    nir_albedo_calc: Float_1D
    nir_refl: Float_1D


class Obs(eqx.Module):
    LE: Float_1D
    H: Float_1D
    GPP: Float_1D
    rnet: Float_1D
    albedo: Float_1D
    Fco2: Float_1D
    gsoil: Float_1D


class Soil(eqx.Module):
    dt: Float_0D
    n_soil: int
    depth: Float_0D
    mtime: int
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

    @property
    def n_soil_1(self):
        return self.n_soil + 1

    @property
    def n_soil_2(self):
        return self.n_soil + 2

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


def initialize_profile(met: Met, para: Para, setup: Setup):
    ntime, jtot = setup.ntime, setup.n_can_layers
    # nlayers = setup.nlayers_atmos
    nlayers = setup.n_total_layers
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


def update_profile(
    met: Met,
    para: Para,
    prof: Prof,
    quantum: ParNir,
    sun: SunShadedCan,
    shade: SunShadedCan,
    soil: Soil,
    veg: Veg,
    lai: Lai,
    dij: Float_2D,
) -> Prof:
    # from ..physics.carbon_fluxes import soil_respiration_alfalfa

    nlayers = sun.Tsfc.shape[1]
    nlayers_atmos = prof.Tair_K.shape[1]

    Ps = (
        quantum.prob_beam[:, :nlayers] * sun.Ps
        + quantum.prob_shade[:, :nlayers] * shade.Ps
    ) * lai.adens
    LE = (
        quantum.prob_beam[:, :nlayers] * sun.LE
        + quantum.prob_shade[:, :nlayers] * shade.LE
    ) * lai.adens
    H = (
        quantum.prob_beam[:, :nlayers] * sun.H
        + quantum.prob_shade[:, :nlayers] * shade.H
    ) * lai.adens
    Rnet = (
        quantum.prob_beam[:, :nlayers] * sun.Rnet
        + quantum.prob_shade[:, :nlayers] * shade.Rnet
    ) * lai.adens
    Tsfc = (
        quantum.prob_beam[:, :nlayers] * sun.Tsfc
        + quantum.prob_shade[:, :nlayers] * shade.Tsfc
    )

    # Compute scalar profiles
    # it needs information on source/sink, Dij, soil boundary flux and factor for units
    fact_heatcoef = met.air_density * para.Cp
    # soilflux = soil.heat # assume soil heat flux is 20 W m-2 until soil sub is working
    Tair_K = conc(
        H,
        soil.heat,
        prof.delz,
        dij,
        met.ustar,
        met.zL,
        met.T_air_K,
        nlayers,
        nlayers_atmos,
        fact_heatcoef,
    )
    Told_K = prof.Told_K

    # with larger Dij value I need to filter new T profiles
    Tair_K = 0.25 * Tair_K + 0.75 * Told_K
    Told_K = Tair_K

    # Compute vapor pressure profiles
    # soilflux = soil.evap  # W m-2
    # in fConcMatrix fact.lecoef is in the denominator insteat of multiplier
    # if we divide W m -2 = J m-2 s-1 by Lambda we have g m-2 s-1
    # need to convert g to Pa
    # eair =rhovair R Tk/mv  Jones
    fact_lecoef = (
        flambda(Tair_K[:, nlayers]) * 18.01 / (1000 * 8.314 * Tair_K[:, nlayers])
    )  # noqa: E501
    eair_Pa = conc(
        LE,
        soil.evap,
        prof.delz,
        dij,
        met.ustar,
        met.zL,
        met.eair_Pa,
        nlayers,
        nlayers_atmos,
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
    fact_co2 = (28.97 / 44.0) * met.air_density_mole
    # soil_resp = soil_respiration_alfalfa(
    #     veg.Ps, soil.T_soil[:, 9], met.soilmoisture, met.zcanopy, veg.Rd, para
    # )
    # soilflux = soil_resp
    co2 = conc(
        -Ps,
        soil.resp,
        prof.delz,
        dij,
        met.ustar,
        met.zL,
        met.CO2,
        nlayers,
        nlayers_atmos,
        fact_co2,
    )
    # jax.debug.print("soilflux: {a}", a=soilflux[:10])
    # jax.debug.print("Ps: {a}", a=Ps[:10])
    # jax.debug.print("met.zL: {a}", a=met.zL[:10])
    # jax.debug.print("met co2: {a}", a=met.CO2[:10])
    # jax.debug.print("co2: {a}", a=co2[:10,:5])

    # Update
    prof = eqx.tree_at(
        lambda t: (
            t.Ps,
            t.LE,
            t.H,
            t.co2,
            t.Rnet,
            t.Tsfc,
            t.Tair_K,
            t.Told_K,
            t.eair_Pa,
            t.eair_old_Pa,
        ),
        prof,
        (Ps, LE, H, co2, Rnet, Tsfc, Tair_K, Told_K, eair_Pa, eair_old_Pa),
    )

    return prof


def calculate_veg(
    para: Para, lai: Lai, quantum: ParNir, sun: SunShadedCan, shade: SunShadedCan
) -> Veg:
    # nlayers = para.jtot
    nlayers = sun.Tsfc.shape[1]
    veg_Ps = jnp.sum(
        (
            quantum.prob_beam[:, :nlayers] * sun.Ps
            + quantum.prob_shade[:, :nlayers] * shade.Ps
        )
        * lai.dff,
        axis=1,
    )
    veg_Rd = jnp.sum(
        (
            quantum.prob_beam[:, :nlayers] * sun.Resp
            + quantum.prob_shade[:, :nlayers] * shade.Resp
        )
        * lai.dff,
        axis=1,
    )
    veg_LE = jnp.sum(
        (
            quantum.prob_beam[:, :nlayers] * sun.LE
            + quantum.prob_shade[:, :nlayers] * shade.LE
        )
        * lai.dff,
        axis=1,
    )
    veg_H = jnp.sum(
        (
            quantum.prob_beam[:, :nlayers] * sun.H
            + quantum.prob_shade[:, :nlayers] * shade.H
        )
        * lai.dff,
        axis=1,
    )
    veg_gs = jnp.sum(
        (
            quantum.prob_beam[:, :nlayers] * sun.gs
            + quantum.prob_shade[:, :nlayers] * shade.gs
        )
        * lai.dff,
        axis=1,
    )
    veg_Rnet = jnp.sum(
        (
            quantum.prob_beam[:, :nlayers] * sun.Rnet
            + quantum.prob_shade[:, :nlayers] * shade.Rnet
        )
        * lai.dff,
        axis=1,
    )
    veg_Tsfc = jnp.sum(
        (
            quantum.prob_beam[:, :nlayers] * sun.Tsfc
            + quantum.prob_shade[:, :nlayers] * shade.Tsfc
        )
        * lai.dff,
        axis=1,
    )
    veg_Tsfc = veg_Tsfc / lai.lai
    veg_vpd = jnp.sum(
        (
            quantum.prob_beam[:, :nlayers] * sun.vpd_Pa
            + quantum.prob_shade[:, :nlayers] * shade.vpd_Pa
        )
        * lai.dff,
        axis=1,
    )
    veg_vpd = veg_vpd / lai.lai

    veg = Veg(veg_Ps, veg_gs, veg_Rd, veg_H, veg_LE, veg_Rnet, veg_Tsfc, veg_vpd)

    return veg


def initialize_model_states(met: Met, para: Para, setup: Setup):
    # Soil
    soil = initialize_soil(setup, para, met)

    # Quantum and NIR
    quantum = initialize_parnir(para, setup, "par")
    nir = initialize_parnir(para, setup, "nir")

    # IR
    ir = initialize_ir(setup)

    # Veg
    veg = initialize_veg(setup)

    # Qin
    qin = initialize_qin(setup)

    # TODO: remove RNet since it is not used
    # RNet
    rnet = initialize_rnet(setup)

    # Sun
    sun = initialize_sunshade(setup, met)

    # Shade
    shade = initialize_sunshade(setup, met)

    # Lai
    lai = initialize_lai(setup, para, met)

    return soil, quantum, nir, ir, qin, rnet, sun, shade, veg, lai


def initialize_parnir(para: Para, setup: Setup, wavebnd: str) -> ParNir:
    ntime, jtot = setup.ntime, setup.n_can_layers
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
    # TODO: how to link this with model parameters?
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


def initialize_ir(setup: Setup) -> Ir:
    jtot, ntime = setup.n_can_layers, setup.ntime
    jktot = jtot + 1
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


def initialize_veg(setup: Setup) -> Veg:
    ntime = setup.ntime
    Ps = jnp.zeros(ntime)
    gs = jnp.zeros(ntime)
    Rd = jnp.zeros(ntime)
    H = jnp.zeros(ntime)
    LE = jnp.zeros(ntime)
    Rnet = jnp.zeros(ntime)
    Tsfc = jnp.zeros(ntime)
    vpd = jnp.zeros(ntime)
    return Veg(Ps, gs, Rd, H, LE, Rnet, Tsfc, vpd)


def initialize_qin(
    setup: Setup,
) -> Qin:
    ntime, jtot = setup.ntime, setup.n_can_layers
    sun_abs = jnp.zeros([ntime, jtot])
    shade_abs = jnp.zeros([ntime, jtot])
    return Qin(sun_abs, shade_abs)


def initialize_rnet(setup: Setup) -> Rnet:
    ntime, jtot = setup.ntime, setup.n_can_layers
    jktot = jtot + 1
    sun = jnp.zeros([ntime, jktot])
    sh = jnp.zeros([ntime, jktot])
    sun_top = jnp.zeros([ntime, jktot])
    sh_top = jnp.zeros([ntime, jktot])
    sh_bottom = jnp.zeros([ntime, jktot])
    return Rnet(sun, sh, sun_top, sh_top, sh_bottom)


def initialize_sunshade(setup: Setup, met: Met) -> SunShadedCan:
    ntime, jtot = setup.ntime, setup.n_can_layers
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


def initialize_lai(setup: Setup, para: Para, met: Met) -> Lai:
    nlayers = setup.n_can_layers
    lai = met.lai
    dff = jnp.ones([setup.ntime, nlayers]) / nlayers  # (ntime,nlayers)
    dff = dot(lai, dff)  # (ntime, nlayers)
    # TODO: double check!
    # self.sumlai = jax.lax.cumsum(self.dff, axis=1, reverse=True) #(ntime,nlayers)
    sumlai = minus(lai, jax.lax.cumsum(dff, axis=1))  # (ntime, nlayers)
    sumlai = jnp.clip(sumlai, a_min=0.0)  # (ntime, nlayers)
    dff_clmp = dff / para.markov  # (ntime, nlayers)

    # divide by height of the layers in the canopy
    adens = dff[:, :nlayers] / para.dht_canopy  # (ntime, nlayers)

    return Lai(lai, dff, sumlai, dff_clmp, adens)


def initialize_soil(
    # para: Para, met: Met, n_soil: int = 10, depth: Float_0D = 0.15
    setup: Setup,
    para: Para,
    met: Met,
    depth: Float_0D = 0.15,
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
    dt = setup.dt_soil
    mtime = setup.soil_mtime
    # mtime = floor(1800.0 / dt)  # time steps per half hour

    n_soil = setup.n_soil_layers  # number of soil layers
    n_soil_1 = n_soil + 1
    n_soil_2 = n_soil + 2

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
    # self.T_soil = jnp.ones([setup.ntime, self.n_soil_2]) * met.Tsoil + 273.15
    # jax.debug.print("{a}", a=met.Tsoil)
    T_soil = (
        dot(met.Tsoil, jnp.ones([setup.ntime, n_soil_2])) + 273.15
    )  # (ntime, n_soil_2)  # noqa: E501

    # initialize upper boundary temperature as air temperature
    # in later iterations this is reset to the radiative surface
    # temperature
    T_soil_up_boundary = met.T_air_K
    sfc_temperature = T_soil_up_boundary
    sfc_temperature_old = sfc_temperature
    bulk_density = (
        jnp.ones([setup.ntime, n_soil]) * 0.83
    )  # soil bulk density for the alfalfa, g cm-3, (ntime, n_soil)  # noqa: E501

    # thermal conductivity code from Campbell and Norman
    fw = 1.0 / (
        1 + jnp.power((met.soilmoisture / 0.15), -4)
    )  # terms for Stefan flow as water evaporates in the pores  # noqa: E501
    K_air = (
        0.024 + 44100 * 2.42e-5 * fw * met.air_density_mole * met.dest / met.P_Pa
    )  # (ntime,)  # noqa: E501

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

    # Energy initialization
    evap = jnp.zeros(setup.ntime)  # initialization
    heat = jnp.zeros(setup.ntime)
    # self.lout=jnp.zeros(setup.ntime)
    rnet = jnp.zeros(setup.ntime)
    gsoil = jnp.zeros(setup.ntime)
    resp = jnp.zeros(setup.ntime)
    lout = para.epsigma * jnp.power(met.T_air_K, 4)  # initialization
    llout = lout

    # lower boundary
    T_soil_low_bound = met.Tsoil + 273.15
    T_soil = jnp.concatenate(
        [T_soil[:, :-1], jnp.expand_dims(T_soil_low_bound, axis=1)],
        axis=1,
    )
    T_soil_old = T_soil

    # Convert int/float to jnp.ndarray
    dt = jnp.array(dt)
    # n_soil = jnp.array(n_soil)
    depth = jnp.array(depth)
    # mtime = jnp.array(mtime)
    water_content_litter = jnp.array(water_content_litter)
    bulkdensity = jnp.array(bulkdensity)
    clay_fraction = jnp.array(clay_fraction)
    peat_fraction = jnp.array(peat_fraction)
    Cp_water = jnp.array(Cp_water)
    Cp_air = jnp.array(Cp_air)
    Cp_org = jnp.array(Cp_org)
    Cp_mineral = jnp.array(Cp_mineral)
    K_mineral = jnp.array(K_mineral)
    K_org = jnp.array(K_org)
    K_water = jnp.array(K_water)

    return Soil(
        dt,
        n_soil,
        depth,
        mtime,
        water_content_15cm,
        water_content_litter,
        bulkdensity,
        clay_fraction,
        peat_fraction,
        Cp_water,
        Cp_air,
        Cp_org,
        Cp_mineral,
        K_mineral,
        K_org,
        K_water,
        dz,
        vol,
        T_soil,
        T_soil_old,
        T_soil_up_boundary,
        T_soil_low_bound,
        sfc_temperature,
        sfc_temperature_old,
        bulk_density,
        K_air,
        Cp_soil,
        evap,
        heat,
        rnet,
        gsoil,
        lout,
        llout,
        resp,
    )
