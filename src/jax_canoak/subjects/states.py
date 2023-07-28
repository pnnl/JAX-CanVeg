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
- Soil()

Author: Peishi Jiang
Date: 2023.7.24.
"""

import jax
import jax.numpy as jnp

from .meterology import Met
from .parameters import Para

from .utils import soil_sfc_res

# from ..physics.energy_fluxes.soil_energy_balance_mx import soil_sfc_res
from ..shared_utilities.types import Float_0D, Int_0D


dot = jax.vmap(lambda x, y: x * y, in_axes=(None, 1), out_axes=1)


class Prof(object):
    def __init__(self, ntime: int, jtot: int, nlayers: int) -> None:
        self.ntime = ntime
        self.jtot = jtot
        self.nlayers = nlayers
        self.zht = jnp.zeros(nlayers)
        self.delz = jnp.zeros(nlayers)
        self.co2 = jnp.ones([ntime, nlayers])
        self.Tair_K = jnp.ones([ntime, nlayers])
        self.Told_K = jnp.ones([ntime, nlayers])
        self.eair_Pa = jnp.ones([ntime, nlayers])
        self.wind = jnp.zeros([ntime, jtot])
        self.Tsfc = jnp.zeros([ntime, jtot])
        self.H = jnp.zeros([ntime, jtot])
        self.LE = jnp.zeros([ntime, jtot])
        self.Ps = jnp.zeros([ntime, jtot])

    def _tree_flatten(self):
        children = (
            self.zht,
            self.delz,
            self.co2,
            self.Tair_K,
            self.Told_K,
            self.eair_Pa,
            self.wind,
            self.Tsfc,
            self.H,
            self.LE,
            self.Ps,
        )
        aux_data = {"ntime": self.ntime, "jtot": self.jtot, "nlayers": self.nlayers}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


class SunAng(object):
    def __init__(self, ntime: Int_0D) -> None:
        self.sin_beta = jnp.zeros(ntime)
        self.beta_rad = jnp.zeros(ntime)
        self.beta_deg = jnp.zeros(ntime)
        self.theta_rad = jnp.zeros(ntime)
        self.theta_deg = jnp.zeros(ntime)
        self.ntime = ntime

    def _tree_flatten(self):
        children = (
            self.beta_rad,
            self.sin_beta,
            self.beta_deg,
            self.theta_rad,
            self.theta_deg,
        )
        aux_data = {"ntime": self.ntime}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)
        # return cls(*children, **aux_data)
        # return cls(children[0])


class LeafAng(object):
    def __init__(self, ntime: int, jtot: int, nclass: int) -> None:
        self.ntime, self.jtot, self.nclass = ntime, jtot, nclass
        self.pdf = jnp.zeros(nclass)
        self.Gfunc = jnp.zeros(ntime)
        self.thetaSky = jnp.zeros(nclass)
        self.Gfunc_Sky = jnp.zeros(nclass)
        self.integ_exp_diff = jnp.zeros([ntime, jtot])

    def _tree_flatten(self):
        children = (
            self.pdf,
            self.Gfunc,
            self.thetaSky,
            self.Gfunc_Sky,
            self.integ_exp_diff,
        )
        aux_data = {"ntime": self.ntime, "jtot": self.jtot, "nclass": self.nclass}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


class ParNir(object):
    def __init__(self, ntime: int, jtot: int) -> None:
        self.ntime, self.jtot = ntime, jtot
        jktot = jtot + 1
        self.sh_abs = jnp.zeros([ntime, jtot])
        self.sun_abs = jnp.zeros([ntime, jtot])
        self.sun = jnp.zeros([ntime, jktot])
        self.dn_flux = jnp.zeros([ntime, jktot])
        self.up_flux = jnp.zeros([ntime, jktot])
        self.sun_normal = jnp.zeros([ntime, jktot])
        self.sh_flux = jnp.zeros([ntime, jktot])
        self.incoming = jnp.zeros(ntime)
        self.beam_flux = jnp.zeros([ntime, jktot])
        self.total = jnp.zeros([ntime, jktot])
        self.inbeam = jnp.zeros(ntime)
        self.indiffuse = jnp.zeros(ntime)
        self.prob_beam = jnp.zeros([ntime, jktot])
        self.prob_shade = jnp.ones([ntime, jktot])
        self.sun_lai = jnp.zeros([ntime, jktot])
        self.shade_lai = jnp.zeros([ntime, jktot])
        self.reflect = 0.0
        self.trans = 0.0
        self.soil_refl = 0.0
        self.absorbed = 0.0

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
            self.prob_beam,
            self.prob_shade,
            self.sun_lai,
            self.shade_lai,
            self.reflect,
            self.trans,
            self.soil_refl,
            self.absorbed,
        )
        aux_data = {"ntime": self.ntime, "jtot": self.jtot}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


class Ir(object):
    def __init__(self, ntime: int, jtot: int) -> None:
        self.ntime, self.jtot = ntime, jtot
        jktot = jtot + 1
        self.ir_in = jnp.zeros(ntime)
        self.ir_dn = jnp.ones([ntime, jktot])
        self.ir_up = jnp.ones([ntime, jktot])
        self.IR_source_sun = jnp.zeros([ntime, jktot])
        self.IR_source_shade = jnp.zeros([ntime, jktot])
        self.IR_source = jnp.zeros([ntime, jktot])
        self.shade = jnp.zeros([ntime, jtot])
        self.shade_top = jnp.zeros([ntime, jktot])
        self.shade_bottom = jnp.zeros([ntime, jktot])

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
        )
        aux_data = {"ntime": self.ntime, "jtot": self.jtot}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


class Rnet(object):
    def __init__(self, ntime: int, jktot: int) -> None:
        self.ntime, self.jktot = ntime, jktot
        self.sun = jnp.zeros([ntime, jktot])
        self.sh = jnp.zeros([ntime, jktot])
        self.sun_top = jnp.zeros([ntime, jktot])
        self.sh_top = jnp.zeros([ntime, jktot])
        self.sh_bottom = jnp.zeros([ntime, jktot])

    def _tree_flatten(self):
        children = (self.sun, self.sh, self.sun_top, self.sh_top, self.sh_bottom)
        aux_data = {"ntime": self.ntime, "jktot": self.jktot}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


class SunShadedCan(object):
    def __init__(self, ntime: int, jktot: int) -> None:
        self.ntime, self.jktot = ntime, jktot
        self.Ps = jnp.zeros([ntime, jktot])
        self.Resp = jnp.zeros([ntime, jktot])
        self.gs = jnp.zeros([ntime, jktot])
        self.LE = jnp.zeros([ntime, jktot])
        self.H = jnp.zeros([ntime, jktot])
        self.Tsfc = jnp.ones([ntime, jktot])
        self.Tsfc_old = jnp.ones([ntime, jktot])
        self.Tsfc_new = jnp.ones([ntime, jktot])

    def _tree_flatten(self):
        children = (
            self.Ps,
            self.Resp,
            self.gs,
            self.LE,
            self.H,
            self.Tsfc,
            self.Tsfc_new,
            self.Tsfc_old,
        )
        aux_data = {"ntime": self.ntime, "jktot": self.jktot}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


class BoundLayerRes(object):
    def __init__(self, ntime: int, jktot: int) -> None:
        self.ntime, self.jktot = ntime, jktot
        self.heat = jnp.zeros([ntime, jktot])
        self.vapor = jnp.zeros([ntime, jktot])
        self.co2 = jnp.zeros([ntime, jktot])

    def _tree_flatten(self):
        children = (self.heat, self.vapor, self.co2)
        aux_data = {"ntime": self.ntime, "jktot": self.jktot}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


class Qin(object):
    def __init__(self, ntime: int, jktot: int) -> None:
        self.ntime, self.jktot = ntime, jktot
        self.sun_abs = jnp.zeros([ntime, jktot])
        self.shade_abs = jnp.zeros([ntime, jktot])

    def _tree_flatten(self):
        children = (self.sun_abs, self.shade_abs)
        aux_data = {"ntime": self.ntime, "jktot": self.jktot}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


class Veg(object):
    def __init__(self, ntime: int) -> None:
        self.ntime = ntime
        self.Ps = jnp.zeros(ntime)
        self.Rd = jnp.zeros(ntime)
        self.H = jnp.zeros(ntime)
        self.LE = jnp.zeros(ntime)
        self.Tsfc = jnp.zeros(ntime)

    def _tree_flatten(self):
        children = (self.Ps, self.Rd, self.H, self.LE, self.Tsfc)
        aux_data = {"ntime": self.ntime}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


class Soil(object):
    def __init__(
        self,
        met: Met,
        para: Para,
        # water_content_litter: Float_0D = 0.0,
        # bulkdensity: Float_0D = 1.06,
        # clay_fraction: Float_0D = 0.3,
        # peat_fraction: Float_0D = 0.08,
        dt: Float_0D = 20.0,
        n_soil: int = 10,
        depth: Float_0D = 0.15,
    ) -> None:
        self.met, self.para = met, para
        # Soil water content
        self.water_content_15cm = (
            met.soilmoisture
        )  # measured at 10 cmwater content of soil m3 m-3    # noqa: E501

        # Water content of litter. Values ranged between 0.02 and 0.126
        self.water_content_litter = (
            0.0  # assumed constant but needs to vary  # noqa: E501
        )

        # fraction porosity + mineral + organic = 1
        # airborne fraction = porosity - volumetric water content
        self.bulkdensity = 1.06  # g cm-3   Data from Tyler Anthony
        self.bulkdensity_kg_m3 = self.bulkdensity * 100 * 100 * 100 / 1000
        self.pore_fraction = (
            1 - self.bulkdensity / 2.65
        )  # from alfalfa, 1 minus ratio bulk density 1.00 g cm-3/2.65 g cm-3, density of solids  # noqa: E501
        self.clay_fraction = 0.3  #  Clay fraction
        self.peat_fraction = 0.08  #  SOM = a C; C = 4.7%, a = 1.72  Kuno Kasak, 2019 Bouldin Alfalfa  # noqa: E501
        self.mineral_fraction = (
            1 - self.pore_fraction - self.peat_fraction
        )  # from bulk density asssuming density of solids is 2.65  # noqa: E501
        self.air_fraction = self.pore_fraction - met.soilmoisture

        # J kg-1 K-1, heat capacity
        self.Cp_water = 4180
        self.Cp_air = 1065
        self.Cp_org = 1920
        self.Cp_mineral = 870

        # W m-1 K-1, thermal conductivity
        self.K_mineral = 2.5
        self.K_org = 0.8
        self.K_water = 0.25

        # Time step in seconds
        self.dt = dt
        self.mtime = jnp.floor(1800.0 / self.dt)  # time steps per half hour

        self.n_soil = n_soil  # number of soil layers
        self.n_soil_1 = self.n_soil + 1  # number of soil levels
        self.n_soil_2 = self.n_soil + 2  # number of soil levels

        # Compute soils depths, from 0 to base
        # we define n layers and n+1 levels, including top and bottom boundaries
        # define Geometric Depths in Soil
        self.depth = depth

        # exponential change in soil depth
        nfact = jnp.power(2, jnp.arange(1, self.n_soil_1))
        self.dz = self.depth / jnp.sum(nfact) * nfact  # n_soil
        self.z_soil = jnp.concatenate([jnp.array([0]), jnp.cumsum(self.dz)])  # n_soil_1
        self.d2z = self.z_soil[2:] - self.z_soil[:-2]
        self.d2z = jnp.concatenate([self.dz[:1], self.d2z])  # n_soil

        # Soil volume
        aream2 = 1  # meter squared area
        self.vol = aream2 * self.dz

        # initialize soil temperature to the deep base temperature at 15 cm
        # self.T_soil = jnp.ones([para.ntime, self.n_soil_2]) * met.Tsoil + 273.15
        # jax.debug.print("{a}", a=met.Tsoil)
        self.T_soil = (
            dot(met.Tsoil, jnp.ones([para.ntime, self.n_soil_2])) + 273.15
        )  # (ntime, n_soil_2)  # noqa: E501

        # initialize upper boundary temperature as air temperature
        # in later iterations this is reset to the radiative surface
        # temperature
        self.T_soil_up_boundary = met.T_air_K
        self.sfc_temperature = self.T_soil_up_boundary
        self.sfc_temperature_old = self.sfc_temperature
        self.bulk_density = (
            jnp.ones([para.ntime, self.n_soil]) * 0.83
        )  # soil bulk density for the alfalfa, g cm-3, (ntime, n_soil)  # noqa: E501

        # thermal conductivity code from Campbell and Norman
        fw = 1.0 / (
            1 + jnp.power((met.soilmoisture / 0.15), -4)
        )  # terms for Stefan flow as water evaporates in the pores  # noqa: E501
        self.K_air = (
            0.024 + 44100 * 2.42e-5 * fw * met.air_density_mole * met.dest / met.P_Pa
        )  # (ntime,)  # noqa: E501
        k_fluid = self.K_air + fw * (self.K_water - self.K_air)  # (ntime,)
        wt_air = 2.0 / (3.0 * (1.0 + 0.2 * (self.K_air / k_fluid - 1))) + 1.0 / (
            3.0 * (1.0 + (1.0 - 2.0 * 0.2) * (self.K_air / k_fluid - 1))
        )  # noqa: E501
        wt_water = 2.0 / (3.0 * (1.0 + 0.2 * (self.K_water / k_fluid - 1))) + 1.0 / (
            3.0 * (1.0 + (1.0 - 2.0 * 0.2) * (self.K_water / k_fluid - 1))
        )  # noqa: E501
        wt_org = 2.0 / (3.0 * (1 + 0.2 * (self.K_org / k_fluid - 1))) + 1.0 / (
            3.0 * (1.0 + (1.0 - 2.0 * 0.2) * (self.K_org / k_fluid - 1))
        )  # noqa: E501
        wt_mineral = 2.0 / (
            3.0 * (1.0 + 0.2 * (self.K_mineral / k_fluid - 1))
        ) + 1.0 / (
            3.0 * (1.0 + (1.0 - 2.0 * 0.2) * (self.K_mineral / k_fluid - 1))
        )  # noqa: E501

        # compute heat capacity and thermal conductivty weighting by mineral, water, air
        # and organic fractions
        # rho_s Cs dT/dt = d(k dT/dz)/dz
        # soil density rho (kg m-3) *  Heat Capacity, J kg-1 K-1 -> J m-3 K-1
        # factors of 1000 is kg H2O m-3, or density of mineral soil (2650) and
        # peat (1300) in terms of kg m-3
        self.Cp_soil = (
            met.air_density * self.air_fraction * self.Cp_air
            + 1000.000 * self.Cp_water * met.soilmoisture
            + 1300.000 * self.Cp_org * self.peat_fraction
            + 2650.000 * self.Cp_mineral * self.mineral_fraction
        )  # (ntime,)

        # thermal conductivity, W m-1  K-1
        K_soil_num = (
            self.mineral_fraction * wt_mineral * self.K_mineral
            + self.air_fraction * wt_air * self.K_air
            + self.water_content_15cm * wt_water * self.K_water
            + self.peat_fraction * wt_org * self.K_mineral
        )  # (ntime,)
        self.K_soil = K_soil_num / (
            self.mineral_fraction * wt_mineral
            + self.air_fraction * wt_air
            + self.water_content_15cm * wt_water
            + self.peat_fraction * wt_org
        )  # (ntime,)

        # rho Cp Volume/ 2 dt
        self.cp_soil = jnp.outer(self.Cp_soil, self.d2z) / (
            2 * self.dt
        )  # transpose dz to make 2d matrix, (ntime, n_soil)  # noqa: E501

        # K/dz
        # self.k_conductivity_soil = self.K_soil / self.dz  # (ntime, n_soil)
        self.k_conductivity_soil = jnp.outer(
            self.K_soil, 1.0 / self.dz
        )  # (ntime, n_soil)
        self.k_conductivity_soil_bound = self.k_conductivity_soil[:, 0]
        self.k_conductivity_soil = jnp.concatenate(
            [self.k_conductivity_soil[:, :-1], jnp.zeros([para.ntime, 1])], axis=1
        )
        # self.k_conductivity_soil[:,-1]=0

        # Energy initialization
        self.evap = jnp.zeros(para.ntime)  # initialization
        self.heat = jnp.zeros(para.ntime)
        # self.lout=jnp.zeros(para.ntime)
        self.rnet = jnp.zeros(para.ntime)
        self.gsoil = jnp.zeros(para.ntime)
        self.lout = para.epsigma * jnp.power(met.T_air_K, 4)  # initialization
        self.llout = self.lout

        self.resistance_h2o = soil_sfc_res(self.water_content_15cm)

        # lower boundary
        self.T_soil_low_bound = met.Tsoil + 273.15
        self.T_soil = jnp.concatenate(
            [self.T_soil[:, :-1], jnp.expand_dims(self.T_soil_low_bound, axis=1)],
            axis=1,
        )
        # self.T_soil[:,-1]= self.T_soil_low_bound

    def _tree_flatten(self):
        children = (
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
            self.mtime,
        )
        aux_data = {
            "met": self.met,
            "para": self.para,
            "n_soil": self.n_soil,
            "depth": self.depth,
            "dt": self.dt,
        }
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


def initialize_profile_mx(met: Met, para: Para):
    prof = Prof(para.ntime, para.jtot, para.nlayers_atmos)

    prof.co2 = dot(met.CO2, prof.co2)
    prof.Tair_K = dot(met.T_air_K, prof.Tair_K)
    prof.Told_K = dot(met.T_air_K, prof.Told_K)
    prof.eair_Pa = dot(met.eair_Pa, prof.eair_Pa)

    # height of canopy layers
    prof.zht = para.zht
    prof.delz = para.delz

    return prof


def initialize_model_states(met: Met, para: Para):
    soil = Soil(met, para)
    quantum, nir = ParNir(para.ntime, para.jtot), ParNir(para.ntime, para.jtot)
    quantum.reflect = para.par_reflect  # reflectance of leaf
    quantum.trans = para.par_trans  # transmittances of leaf
    quantum.soil_refl = para.par_soil_refl  # soil reflectances
    quantum.absorbed = para.par_absorbed  # fraction absorbed
    nir.reflect = para.nir_reflect  # reflectance of leaf
    nir.trans = para.nir_trans  # transmittances of leaf
    nir.soil_refl = para.nir_soil_refl  # soil reflectances
    nir.absorbed = para.nir_absorbed  # fraction absorbed

    ir, veg = Ir(para.ntime, para.jtot), Veg(para.ntime)
    qin, rnet = Qin(para.ntime, para.jktot), Rnet(para.ntime, para.jktot)
    sun, shade = SunShadedCan(para.ntime, para.jktot), SunShadedCan(
        para.ntime, para.jktot
    )  # noqa: E501
    # dot = jax.vmap(lambda x, y: x * y, in_axes=(None, 1), out_axes=1)
    sun.Tsfc = dot(met.T_air_K, sun.Tsfc)
    shade.Tsfc = dot(met.T_air_K, shade.Tsfc)

    return soil, quantum, nir, ir, veg, qin, rnet, sun, shade
