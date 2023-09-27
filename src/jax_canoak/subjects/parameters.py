"""
Classes for model general parameters.

- Setup
- Para
- Rsoil_DL

Author: Peishi Jiang
Date: 2023.7.24.
"""

# import jax
import jax.numpy as jnp

# from math import floor

import equinox as eqx

# from equinox.nn import MLP

from typing import Optional
from ..shared_utilities.types import Float_0D, Float_1D

# from ..shared_utilities.utils import dot, minus
# dot = jax.vmap(lambda x, y: x * y, in_axes=(None, 1), out_axes=1)

# TODO: I need to separate the fixed setup and the parameters that can be estimated.
# TODO: Soil parameters should also go here.


class Setup(eqx.Module):
    # Location
    time_zone: int
    lat_deg: Float_0D
    long_deg: Float_0D
    # Leaf
    stomata: int  # amphistomatous = 1; hypostomatous = 0
    # hypo_amphi: int  # hypostomatous, 1, amphistomatous, 2
    # Leaf angle distributions
    # options include spherical - 1, planophile - 2, erectophile - 3, uniform - 4,
    # plagiophile - 5, extremophile - 6
    leafangle: int
    # Timesteps
    n_hr_per_day: int
    ntime: int
    dt_soil: int
    soil_mtime: int
    # time_batch_size: int
    # Number of layers
    n_can_layers: int
    n_total_layers: int
    n_soil_layers: int
    # dispersion
    npart: int
    # Number of iteration
    niter: int

    # @property
    # def soil_mtime(self):
    #     return floor(3600 * 24 / self.n_hr_per_day / self.dt_soil)

    @property
    def ndays(self):
        return self.ntime / self.n_hr_per_day

    @property
    def n_atmos_layers(self):
        return self.n_total_layers - self.n_can_layers

    # @property
    # def jtot(self):
    #     return self.n_can_layers

    # @property
    # def jktot(self):
    #     return self.jtot + 1

    # @property
    # def jtot3(self):
    #     return self.jtot * 3

    # @property
    # def jktot3(self):
    #     return self.jtot3 + 1


class VarStats(eqx.Module):
    T_air: Float_0D
    rglobal: Float_0D
    eair: Float_0D
    wind: Float_0D
    CO2: Float_0D
    P_kPa: Float_0D
    ustar: Float_0D
    Tsoil: Float_0D
    soilmoisture: Float_0D
    zcanopy: Float_0D
    lai: Float_0D
    rsoil: Float_0D
    LE: Float_0D
    H: Float_0D


class Para(eqx.Module):
    # Vertical profiles
    zht1: Float_1D
    zht2: Float_1D
    delz1: Float_1D
    delz2: Float_1D
    soil_depth: Float_0D
    # Leaf
    leaf_clumping_factor: Float_0D
    # Par and Nir
    par_reflect: Float_0D
    par_trans: Float_0D
    par_soil_refl: Float_0D
    nir_reflect: Float_0D
    nir_trans: Float_0D
    nir_soil_refl: Float_0D
    # Constants for IR fluxes
    sigma: Float_0D
    ep: Float_0D
    epsoil: Float_0D
    # Universal gas constant
    rugc: Float_0D
    rgc1000: Float_0D
    Cp: Float_0D
    # parameters for the Farquhar et al photosynthesis model
    # Consts for Photosynthesis model and kinetic equations.
    # for Vcmax and Jmax.  Taken from Harley and Baldocchi (1995, PCE)
    # carboxylation rate at 25 C temperature, umol m-2 s-1
    # these variables were computed from A-Ci measurements and the
    # Sharkey Excel spreadsheet tool
    vcopt: Float_0D
    jmopt: Float_0D
    rd25: Float_0D
    hkin: Float_0D
    skin: Float_0D
    ejm: Float_0D
    evc: Float_0D
    # Enzyme constants & partial pressure of O2 and CO2
    # Michaelis-Menten K values. From survey of literature.
    kc25: Float_0D
    ko25: Float_0D
    o2: Float_0D
    ekc: Float_0D
    eko: Float_0D
    erd: Float_0D
    ektau: Float_0D
    tk_25: Float_0D
    # Ps code was blowing up at high leaf temperatures, so reduced opt
    # and working perfectly
    toptvc: Float_0D
    toptjm: Float_0D
    kball: Float_0D
    bprime: Float_0D
    # simple values for priors and first iteration
    rsm: Float_0D
    brs: Float_0D
    qalpha: Float_0D
    lleaf: Float_0D
    # Water content thresholds
    theta_min: Float_0D  # field capacity
    theta_max: Float_0D  # wilting point
    # Diffusivity values for 273 K and 1013 mb (STP) using values from Massman (1998) Atmos Environment  # noqa: E501
    # These values are for diffusion in air.  When used these values must be adjusted for  # noqa: E501
    # temperature and pressure
    # nu, Molecular viscosity
    nuvisc: Float_0D
    # Diffusivity of CO2
    dc: Float_0D
    # Diffusivity of heat
    dh: Float_0D
    # Diffusivity of water
    dv: Float_0D
    # Diffusivity of ozone
    do3: Float_0D
    betfac: Float_0D
    # Other constants
    Mair: Float_0D
    dLdT: Float_0D
    extinct: Float_0D
    # # Deep learning models
    # RsoilDL: MLP
    # Meterological stats
    var_mean: Optional[VarStats]
    var_std: Optional[VarStats]
    var_max: Optional[VarStats]
    var_min: Optional[VarStats]

    def __init__(
        self,
        # Vertical profiles
        zht1: Float_1D,
        zht2: Float_1D,
        delz1: Float_1D,
        delz2: Float_1D,
        soil_depth: Float_0D,
        # Leaf
        leaf_clumping_factor: Float_0D,
        # Par and Nir
        par_reflect: Float_0D = 0.05,
        par_trans: Float_0D = 0.05,
        par_soil_refl: Float_0D = 0.05,
        nir_reflect: Float_0D = 0.60,
        nir_trans: Float_0D = 0.20,
        nir_soil_refl: Float_0D = 0.10,
        # parameters for the Farquhar et al photosynthesis model
        vcopt: Float_0D = 171.0,
        jmopt: Float_0D = 259.0,
        rd25: Float_0D = 2.68,
        hkin: Float_0D = 200000.0,
        skin: Float_0D = 710.0,
        ejm: Float_0D = 55000.0,
        evc: Float_0D = 55000.0,
        # Enzyme constants & partial pressure of O2 and CO2
        # Michaelis-Menten K values. From survey of literature.
        kc25: Float_0D = 274.6,
        ko25: Float_0D = 419.8,
        o2: Float_0D = 210.0,
        ekc: Float_0D = 80500.0,
        eko: Float_0D = 14500.0,
        erd: Float_0D = 38000.0,
        ektau: Float_0D = -29000.0,
        # Ps code was blowing up at high leaf temperatures, so reduced opt
        # and working perfectly
        toptvc: Float_0D = 303.0,
        toptjm: Float_0D = 303.0,
        kball: Float_0D = 8.17,
        bprime: Float_0D = 0.05,
        # simple values for priors and first iteration
        rsm: Float_0D = 145.0,
        brs: Float_0D = 60.0,
        qalpha: Float_0D = 0.22,
        lleaf: Float_0D = 0.04,
        # Water content thresholds
        theta_min: Float_0D = 0.03,  # wilting point
        theta_max: Float_0D = 0.2,  # field capacity
        # Meterological stats
        var_mean: Optional[VarStats] = None,
        var_std: Optional[VarStats] = None,
        var_max: Optional[VarStats] = None,
        var_min: Optional[VarStats] = None,
    ) -> None:
        # Vertical profiles
        self.zht1 = zht1
        self.zht2 = zht2
        self.delz1 = delz1
        self.delz2 = delz2
        self.soil_depth = jnp.array(soil_depth)

        # Leaf clumping
        self.leaf_clumping_factor = jnp.array(leaf_clumping_factor)

        # Reflectance, transmiss, soil reflectance
        self.par_reflect = jnp.array(par_reflect)
        self.par_trans = jnp.array(par_trans)
        self.par_soil_refl = jnp.array(par_soil_refl)
        self.nir_reflect = jnp.array(nir_reflect)
        self.nir_trans = jnp.array(nir_trans)
        self.nir_soil_refl = jnp.array(nir_soil_refl)

        # IR fluxes
        self.sigma = jnp.array(5.670367e-08)  # Stefan Boltzmann constant W m-2 K-4
        # self.ep = jnp.array(0.985)  # emissivity of leaves
        self.ep = jnp.array(0.98)  # emissivity of leaves
        self.epsoil = jnp.array(0.98)  # Emissivity of soil

        # Universal gas constant
        self.rugc = jnp.array(8.314)  # J mole-1 K-1
        self.rgc1000 = jnp.array(8314.0)  # gas constant times 1000.
        self.Cp = jnp.array(1005.0)  # specific heat of air, J kg-1 K-1

        # parameters for the Farquhar et al photosynthesis model
        # Consts for Photosynthesis model and kinetic equations.
        # for Vcmax and Jmax.  Taken from Harley and Baldocchi (1995, PCE)
        # carboxylation rate at 25 C temperature, umol m-2 s-1
        # these variables were computed from A-Ci measurements and the
        # Sharkey Excel spreadsheet tool
        self.vcopt = jnp.array(
            vcopt
        )  # carboxylation rate at 25 C temperature, umol m-2 s-1; from field measurements Rey Sanchez alfalfa  # noqa: E501
        self.jmopt = jnp.array(
            jmopt
        )  # electron transport rate at 25 C temperature, umol m-2 s-1, field measuremetns Jmax = 1.64 Vcmax  # noqa: E501
        self.rd25 = jnp.array(
            rd25
        )  # dark respiration at 25 C, rd25= 0.34 umol m-2 s-1, field measurements   # noqa: E501
        self.hkin = jnp.array(hkin)  # enthalpy term, J mol-1
        self.skin = jnp.array(skin)  # entropy term, J K-1 mol-1
        self.ejm = jnp.array(
            ejm
        )  # activation energy for electron transport, J mol-1  # noqa: E501
        self.evc = jnp.array(evc)  # activation energy for carboxylation, J mol-1

        # Enzyme constants & partial pressure of O2 and CO2
        # Michaelis-Menten K values. From survey of literature.
        self.kc25 = jnp.array(kc25)  # kinetic coef for CO2 at 25 C, microbars
        self.ko25 = jnp.array(ko25)  # kinetic coef for O2 at 25C,  millibars
        self.o2 = jnp.array(o2)  # oxygen concentration  mmol mol-1
        self.ekc = jnp.array(ekc)  # Activation energy for K of CO2; J mol-1
        self.eko = jnp.array(eko)  # Activation energy for K of O2, J mol-1
        self.erd = jnp.array(
            erd
        )  # activation energy for dark respiration, eg Q10=2  # noqa: E501
        self.ektau = jnp.array(ektau)  # J mol-1 (Jordan and Ogren, 1984)
        self.tk_25 = jnp.array(298.16)  # absolute temperature at 25 C

        # Ps code was blowing up at high leaf temperatures, so reduced opt
        # and working perfectly
        self.toptvc = jnp.array(
            toptvc
        )  # optimum temperature for maximum carboxylation, 311 K  # noqa: E501
        self.toptjm = jnp.array(
            toptjm
        )  # optimum temperature for maximum electron transport,311 K  # noqa: E501
        self.kball = jnp.array(
            kball
        )  # Ball-Berry stomatal coefficient for stomatal conductance, data from Camilo Rey Sanchez bouldin Alfalfa  # noqa: E501
        self.bprime = jnp.array(
            bprime
        )  # intercept leads to too big LE0.14;    % mol m-2 s-1 h2o..Camilo Rey Sanchez..seems high  # noqa: E501
        # self.bprime16 = self.bprime / 1.6  # intercept for CO2, bprime16 = bprime /1.6;  # noqa: E501

        # simple values for priors and first iteration
        self.rsm = jnp.array(rsm)  # Minimum stomatal resistance, s m-1.
        self.brs = jnp.array(brs)  # curvature coeffient for light response
        self.qalpha = jnp.array(qalpha)  #  leaf quantum yield, electrons
        # self.qalpha2 = 0.0484  # qalpha squared, qalpha2 = pow(qalpha, 2.0);
        self.lleaf = jnp.array(lleaf)  # leaf length, m, alfalfa, across the trifoliate

        # Water content thresholds
        self.theta_min = theta_min  # wilting point
        self.theta_max = theta_max  # field capacity

        # Diffusivity values for 273 K and 1013 mb (STP) using values from Massman (1998) Atmos Environment  # noqa: E501
        # These values are for diffusion in air.  When used these values must be adjusted for  # noqa: E501
        # temperature and pressure
        # nu, Molecular viscosity
        self.nuvisc = jnp.array(13.27)  # mm2 s-1
        # self.nnu = 0.00001327  # m2 s-1

        # Diffusivity of CO2
        self.dc = jnp.array(13.81)  # / mm2 s-1
        # self.ddc = 0.00001381  # / m2 s-1

        # Diffusivity of heat
        self.dh = jnp.array(18.69)  # mm2 s-1
        # self.ddh = 0.00001869  # m2 s-1

        # Diffusivity of water vapor
        self.dv = jnp.array(21.78)  # / mm2 s-1
        # self.ddv = 0.00002178  # m2 s-1

        # Diffusivity of ozone
        self.do3 = jnp.array(14.44)  # m2 s-1
        # self.ddo3 = 0.00001444  # m2 s-1
        self.betfac = jnp.array(1.5)  # the boundary layer sheltering factor from Grace

        self.Mair = jnp.array(28.97)
        self.dLdT = jnp.array(-2370.0)
        self.extinct = jnp.array(2.0)  # extinction coefficient wind in canopy

        # Initialize the Rsoil DL module
        # TODO: this is a hack to make sure that the RsoilDL module is initialized
        # self.RsoilDL = MLP(in_size=2, out_size=1, width_size=6, depth=2, key=jax.random.PRNGKey(0))  # noqa: E501

        # Get the variable stats
        self.var_mean = var_mean
        self.var_std = var_std
        self.var_max = var_max
        self.var_min = var_min

    @property
    def dht(self):
        # zero plane displacement height, m
        return 0.6 * self.veg_ht

    @property
    def z0(self):
        # aerodynamic roughness height, m
        return 0.1 * self.veg_ht

    @property
    def zht(self):
        # aerodynamic roughness height, m
        return jnp.concatenate([self.zht1, self.zht2])

    # @property
    # def n_can_layer(self):
    #     return self.zht1.size

    # @property
    # def jtot(self):
    #     return self.zht1.size

    # @property
    # def n_atmos_layers(self):
    #     return self.zht2.size

    # @property
    # def jktot(self):
    #     return self.jtot + 1

    # @property
    # def jtot3(self):
    #     return self.jtot * 3

    # @property
    # def jktot3(self):
    #     return self.jtot3 + 1

    @property
    def delz(self):
        # height increment of each layer
        return jnp.concatenate([self.delz1, self.delz2])

    @property
    def veg_ht(self):
        # return self.zht[self.jtot - 1]
        return self.zht1[-1]

    @property
    def meas_ht(self):
        return self.zht[-1]

    @property
    def dht_canopy(self):
        # height increment of each layer
        # return self.veg_ht / self.n_can_layer
        return self.veg_ht / self.zht1.size

    @property
    def ht_atmos(self):
        return self.meas_ht - self.veg_ht

    @property
    def dht_atmos(self):
        # return self.ht_atmos / self.n_atmos_layers
        return self.ht_atmos / self.zht2.size

    # @property
    # def nlayers_atmos(self):
    #     # return self.jtot + jnp.floor(self.ht_atmos / self.dht_atmos).astype(int)
    #     return self.n_can_layer + self.n_atmos_layers

    @property
    def markov(self):
        # zero plane displacement height, m
        # return jnp.ones(self.n_can_layer) * self.leaf_clumping_factor
        return jnp.ones(self.zht1.size) * self.leaf_clumping_factor

    @property
    def par_absorbed(self):
        return 1.0 - self.par_reflect - self.par_trans

    @property
    def nir_absorbed(self):
        return 1.0 - self.nir_reflect - self.nir_trans

    @property
    def epm1(self):
        return 1 - self.ep  # 1-ep

    @property
    def epsigma(self):
        return self.ep * self.sigma

    @property
    def epsigma2(self):
        return 2 * self.ep * self.sigma

    @property
    def epsigma4(self):
        return 4 * self.ep * self.sigma

    @property
    def epsigma6(self):
        return 6 * self.ep * self.sigma

    @property
    def epsigma8(self):
        return 8 * self.ep * self.sigma

    @property
    def epsigma12(self):
        return 12 * self.ep * self.sigma

    @property
    def ir_reflect(self):
        return 1.0 - self.ep  # tranmittance is zero, 1 minus emissivity

    @property
    def ir_trans(self):
        return 0.0

    @property
    def ir_absorbed(self):
        return self.ep

    @property
    def ir_soil_refl(self):
        return 1 - self.epsoil

    @property
    def bprime16(self):
        return self.bprime / 1.6

    @property
    def qalpha2(self):
        return jnp.power(self.qalpha, 2.0)

    @property
    def nnu(self):
        return self.nuvisc * 1e-6

    @property
    def ddc(self):
        return self.dc * 1e-6

    @property
    def ddh(self):
        return self.dh * 1e-6

    @property
    def ddv(self):
        return self.dv * 1e-6

    @property
    def ddo3(self):
        return self.do3 * 1e-6

    @property
    def lfddh(self):
        # Constants for leaf boundary layers
        return self.lleaf / self.ddh

    @property
    def pr(self):
        # Prandtl Number
        return self.nuvisc / self.dh

    @property
    def pr33(self):
        # Prandtl Number
        return self.pr**0.33

    @property
    def lfddv(self):
        return self.lleaf / self.ddv

    @property
    def sc(self):
        # SCHMIDT NUMBER FOR VAPOR
        return self.nuvisc / self.dv

    @property
    def sc33(self):
        # SCHMIDT NUMBER FOR VAPOR
        return self.sc**0.33

    @property
    def scc(self):
        # SCHMIDT NUMBER FOR CO2
        return self.nuvisc / self.dc

    @property
    def scc33(self):
        # SCHMIDT NUMBER FOR CO2
        return self.scc**0.33

    @property
    def grasshof(self):
        # Grasshof Number
        return 9.8 * self.lleaf**3 / (self.nnu**2)


# class RsoilDL(eqx.Module):
#     layers: List[Linear]

#     def __init__(
#         self,
#         n_input: int,
#         n_hidden: int,
#         n_output: int,
#         depth: int=1,
#         random_seed: int=0,
#     ):
#         super().__init__()
#         key = jax.random.PRNGKey(random_seed)

#         layers = []
#         # First layer
#         layer1 = Linear(n_input, n_hidden, key=key)
#         layers.append(layer1)
#         # Hidden layers
#         for i in range(depth):
#             key_used, key = jax.random.split(key)
#             if i == depth - 1:
#                 layer = Linear(n_hidden, n_output, key=key)
#             else:
#                 layer = Linear(n_hidden, n_hidden, key=key)
#             layers.append(layer)
#         self.layers = layers

#     def __call__(self, x):
#         layers = self.layers

#         def calculate_layer(x, layer):
#             x = layer(x)
#             x = jax.nn.relu(x)
#             return x, x
#         _, x = jax.lax.scan(calculate_layer, x, layers)
#         return x


# def initialize_parameters(
#     time_zone: int = -8,
#     latitude: Float_0D = 38.0991538,
#     longitude: Float_0D = -121.49933,
#     stomata: int = 1,
#     leafangle: int = 1,
#     leaf_clumping_factor: Float_0D = 0.95,
#     veg_ht: Float_0D = 0.8,
#     meas_ht: Float_0D = 5.0,
#     soil_depth: Float_0D = 0.15,
#     n_can_layers: int = 30,
#     n_atmos_layers: int = 50,
#     n_soil_layers: int = 10,
#     n_hr_per_day: int = 48,
#     n_time: int = 200,
#     # time_batch_size: int = 1,
#     dt_soil: int = 20,
#     par_reflect: Float_0D = 0.05,
#     par_trans: Float_0D = 0.05,
#     par_soil_refl: Float_0D = 0.05,
#     nir_reflect: Float_0D = 0.60,
#     nir_trans: Float_0D = 0.20,
#     nir_soil_refl: Float_0D = 0.10,
#     theta_min: Float_0D = 0.05,  # wilting point
#     theta_max: Float_0D = 0.2,  # field capacity
#     npart: int = 1000000,
#     niter: int = 15,
#     met: Optional[Met] = None,
#     obs: Optional[Obs] = None,
# ) -> Tuple[Setup, Para]:

#     dht_canopy = veg_ht / n_can_layers
#     ht_atmos = meas_ht - veg_ht
#     dht_atmos = ht_atmos / n_atmos_layers

#     n_total_layers = n_can_layers + n_atmos_layers

#     # Layer depths
#     zht1 = jnp.arange(1, n_can_layers + 1)
#     zht1 = zht1 * dht_canopy
#     delz1 = jnp.ones(n_can_layers) * dht_canopy
#     zht2 = jnp.arange(1, n_total_layers - n_can_layers + 1) * dht_atmos + veg_ht
#     delz2 = jnp.ones(n_total_layers - n_can_layers) * dht_atmos

#     # Calculate meterological mean and standard deviation
#     if obs is None:
#         var_mean, var_std = None, None
#     else:
#         var_mean, var_std = calculate_var_stats(met, obs)

#     # Number of time steps for solving soil energy balancd
#     soil_mtime = floor(3600 * 24 / n_hr_per_day / dt_soil)
#     setup = Setup(
#         time_zone=time_zone,
#         lat_deg=latitude,
#         long_deg=longitude,
#         stomata=stomata,
#         leafangle=leafangle,
#         n_hr_per_day=n_hr_per_day,
#         ntime=n_time,
#         # time_batch_size=time_batch_size,
#         n_can_layers=n_can_layers,
#         n_total_layers=n_total_layers,
#         n_soil_layers=n_soil_layers,
#         dt_soil=dt_soil,
#         soil_mtime=soil_mtime,
#         npart=npart,
#         niter=niter,
#     )

#     para = Para(
#         leaf_clumping_factor=leaf_clumping_factor,
#         zht1=zht1,
#         zht2=zht2,
#         delz1=delz1,
#         delz2=delz2,
#         soil_depth=soil_depth,
#         par_reflect=par_reflect,
#         par_trans=par_trans,
#         par_soil_refl=par_soil_refl,
#         nir_reflect=nir_reflect,
#         nir_trans=nir_trans,
#         nir_soil_refl=nir_soil_refl,
#         theta_min=theta_min,
#         theta_max=theta_max,
#         var_mean=var_mean,
#         var_std=var_std,
#     )

#     return setup, para
