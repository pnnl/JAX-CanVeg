"""
Functions for initializing and updating parameters, meterological variabbles
and model states.

Author: Peishi Jiang
Date: 2023.9.25.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import pandas as pd
import numpy as np

import equinox as eqx

from typing import Optional, Tuple
from math import floor

from .meterology import Met
from .parameters import Para, Setup, VarStats
from .states import Obs, Prof, Qin, Veg, Lai, Soil
from .states import ParNir, Ir, Rnet, SunShadedCan, Can
from ..shared_utilities.utils import dot, minus
from ..shared_utilities.types import Float_2D, Int_0D, Float_0D, Float_1D
from .utils import conc
from .utils import llambda as flambda

# near_zero = 1e-20
near_zero = 5e-5

############################################################################
# Parameters and setup
############################################################################
def initialize_parameters(
    time_zone: int = -8,
    latitude: Float_0D = 38.0991538,
    longitude: Float_0D = -121.49933,
    stomata: int = 1,
    leafangle: int = 1,
    leaf_clumping_factor: Float_0D = 0.95,
    veg_ht: Float_0D = 0.8,
    meas_ht: Float_0D = 5.0,
    soil_depth: Float_0D = 0.15,
    n_can_layers: int = 30,
    n_atmos_layers: int = 50,
    n_soil_layers: int = 10,
    n_hr_per_day: int = 48,
    n_time: int = 200,
    # time_batch_size: int = 1,
    dt_soil: Float_0D = 20.0,
    par_reflect: Float_0D = 0.05,
    par_trans: Float_0D = 0.05,
    par_soil_refl: Float_0D = 0.05,
    nir_reflect: Float_0D = 0.60,
    nir_trans: Float_0D = 0.20,
    nir_soil_refl: Float_0D = 0.10,
    theta_min: Float_0D = 0.05,  # wilting point
    theta_max: Float_0D = 0.2,  # field capacity
    npart: int = 1000000,
    niter: int = 15,
    met: Optional[Met] = None,
    obs: Optional[Obs] = None,
    RsoilDL: Optional[eqx.Module] = None,
    LeafRHDL: Optional[eqx.Module] = None,
    bprimeDL: Optional[eqx.Module] = None,
    gscoefDL: Optional[eqx.Module] = None,
    get_para_bounds: bool = False,
    # ) -> Tuple[Setup, Para]:
):

    dht_canopy = veg_ht / n_can_layers
    ht_atmos = meas_ht - veg_ht
    dht_atmos = ht_atmos / n_atmos_layers

    n_total_layers = n_can_layers + n_atmos_layers

    # Layer depths
    zht1 = jnp.arange(1, n_can_layers + 1)
    zht1 = zht1 * dht_canopy
    delz1 = jnp.ones(n_can_layers) * dht_canopy
    zht2 = jnp.arange(1, n_total_layers - n_can_layers + 1) * dht_atmos + veg_ht
    delz2 = jnp.ones(n_total_layers - n_can_layers) * dht_atmos

    # Calculate meterological mean and standard deviation
    if (met is not None) and (obs is not None):
        var_mean, var_std, var_max, var_min = calculate_var_stats(met, obs)
    else:
        var_mean, var_std, var_max, var_min = None, None, None, None

    # Number of time steps for solving soil energy balancd
    soil_mtime = floor(3600 * 24 / n_hr_per_day / dt_soil)
    setup = Setup(
        time_zone=time_zone,
        lat_deg=latitude,
        long_deg=longitude,
        stomata=stomata,
        leafangle=leafangle,
        n_hr_per_day=n_hr_per_day,
        ntime=n_time,
        # time_batch_size=time_batch_size,
        n_can_layers=n_can_layers,
        n_total_layers=n_total_layers,
        n_soil_layers=n_soil_layers,
        dt_soil=dt_soil,
        soil_mtime=soil_mtime,
        npart=npart,
        niter=niter,
    )

    para = Para(
        leaf_clumping_factor=leaf_clumping_factor,
        zht1=zht1,
        zht2=zht2,
        delz1=delz1,
        delz2=delz2,
        soil_depth=soil_depth,
        par_reflect=par_reflect,
        par_trans=par_trans,
        par_soil_refl=par_soil_refl,
        nir_reflect=nir_reflect,
        nir_trans=nir_trans,
        nir_soil_refl=nir_soil_refl,
        theta_min=theta_min,
        theta_max=theta_max,
        var_mean=var_mean,
        var_std=var_std,
        var_max=var_max,
        var_min=var_min,
        RsoilDL=RsoilDL,
        LeafRHDL=LeafRHDL,
        bprimeDL=bprimeDL,
        gscoefDL=gscoefDL,
    )

    if not get_para_bounds:
        return setup, para
    else:
        para_min = jtu.tree_map(lambda _: -99999.0, para)
        para_min = eqx.tree_at(
            lambda t: [
                t.leaf_clumping_factor,
                t.par_reflect,
                t.par_trans,
                t.par_soil_refl,
                t.nir_reflect,
                t.nir_trans,
                t.nir_soil_refl,
                t.bprime,
                t.lleaf,
                t.theta_min,
                t.theta_max,
            ],
            para_min,
            replace=[
                near_zero,
                near_zero,
                near_zero,
                near_zero,
                near_zero,
                near_zero,
                near_zero,
                near_zero,
                near_zero,
                near_zero,
                near_zero,
            ],
        )
        para_max = jtu.tree_map(lambda _: 99999.0, para)
        para_max = eqx.tree_at(
            lambda t: [
                t.leaf_clumping_factor,
                t.par_reflect,
                t.par_trans,
                t.par_soil_refl,
                t.nir_reflect,
                t.nir_trans,
                t.nir_soil_refl,
                t.theta_min,
                t.theta_max,
                t.lleaf,
            ],
            para_max,
            replace=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1],
        )
        return setup, para, para_min, para_max


############################################################################
# Meterology
############################################################################
def initialize_met(data: Float_2D, ntime: Int_0D, zL0: Float_1D) -> Met:
    assert ntime == data.shape[0]
    year = jnp.array(data[:, 0])  # day of year
    day = jnp.array(data[:, 1])  # day of year
    hhour = jnp.array(data[:, 2])  # hour
    # self.T_air_K = jnp.array(data[:, 2]) + 273.15  # air temperature, K
    T_air = jnp.array(data[:, 3])  # air temperature, degC
    rglobal = jnp.array(data[:, 4])  # global shortwave radiation, W m-2
    eair = jnp.array(data[:, 5])  # vapor pressure, kPa
    wind = jnp.array(data[:, 6])  # wind velocity, m/s
    CO2 = jnp.array(data[:, 7])  # CO2, ppm
    P_kPa = jnp.array(data[:, 8])  # atmospheric pressure, kPa
    ustar = jnp.array(data[:, 9])  # friction velocity, m/s
    Tsoil = jnp.array(data[:, 10])  # soil temperature, C...16 cm
    soilmoisture = jnp.array(data[:, 11])  # soil moisture, fraction
    zcanopy = jnp.array(data[:, 12])  # aerodynamic canopy height
    lai = jnp.array(data[:, 13])  # leaf area index [-]

    # Some operations to ensure stability
    wind = jnp.clip(wind, a_min=0.75)
    # ustar = jnp.clip(ustar, a_min=0.75)
    ustar = jnp.clip(ustar, a_min=0.1)
    rglobal = jnp.clip(rglobal, a_min=0.0)

    # Convert the following int and float to jax.ndarray
    # ntime = jnp.array(ntime)
    # Mair = jnp.array(Mair)
    # rugc = jnp.array(rugc)

    met = Met(
        # ntime,
        # Mair,
        # rugc,
        zL0,
        year,
        day,
        hhour,
        T_air,
        rglobal,
        eair,
        wind,
        CO2,
        P_kPa,
        ustar,
        Tsoil,
        soilmoisture,
        zcanopy,
        lai,
    )

    return met


def get_met_forcings(f_forcing: str, lai: Optional[Float_0D] = None) -> Tuple[Met, int]:
    # Load the modeling forcing text file
    # This should be a matrix of forcing data with each column representing
    # a time series of observations
    forcing_data = np.loadtxt(f_forcing, delimiter=",")
    forcing_data = jnp.array(forcing_data)
    n_time = forcing_data.shape[0]
    # Initialize the zl length with zeros
    zl0 = jnp.zeros(n_time)
    # Set up the lai if not None
    if lai is not None:
        forcing_data = jnp.concatenate(
            # [forcing_data[:, :12], jnp.ones([n_time, 1]) * lai], axis=1
            [forcing_data[:, :13], jnp.ones([n_time, 1]) * lai],
            axis=1,
        )
    # Initialize the met instance
    met = initialize_met(forcing_data, n_time, zl0)
    return met, n_time


def get_obs(f_obs: str) -> Obs:
    # Load the observations forcing text file
    obs = pd.read_csv(f_obs)
    obs.interpolate(method="linear", limit_direction="both", inplace=True)
    ntime = obs.shape[0]
    nan = jnp.nan * jnp.ones(ntime)
    # nan = jnp.ones(ntime)

    # Precipitation
    if "P_mm" in obs:
        P_obs = jnp.array(obs["P_mm"])
    else:
        P_obs = nan

    # Latent heat flux
    if "LE" in obs:
        LE_obs = jnp.array(obs["LE"])
    elif "LE_F_MDS" in obs:
        LE_obs = jnp.array(obs["LE_F_MDS"])
    else:
        LE_obs = nan

    # Sensible heat flux
    if "H" in obs:
        H_obs = jnp.array(obs["H"])
    elif "H_F_MDS" in obs:
        H_obs = jnp.array(obs["H_F_MDS"])
    else:
        H_obs = nan

    # Ground heat flux
    if "G" in obs:
        Gsoil_obs = jnp.array(obs["G"])
    elif "G_F_MDS" in obs:
        Gsoil_obs = jnp.array(obs["G_F_MDS"])
    elif "G_5cm" in obs:
        Gsoil_obs = jnp.array(obs["G_5cm"])
    else:
        Gsoil_obs = nan

    # Net radiation
    if "NETRAD" in obs:
        Rnet_obs = jnp.array(obs["NETRAD"])
    else:
        Rnet_obs = nan

    # Gross primary productivity
    if "GPP" in obs:
        GPP_obs = jnp.array(obs["GPP"])
    else:
        GPP_obs = nan

    # Albedo
    if "ALBEDO" in obs:
        albedo_obs = jnp.array(obs["ALBEDO"])
    else:
        albedo_obs = nan

    # FCO2
    if "FCO2" in obs:
        Fco2_obs = jnp.array(obs["FCO2"])
    elif "Fco2" in obs:
        Fco2_obs = jnp.array(obs["Fco2"])
    elif "FC" in obs:
        Fco2_obs = jnp.array(obs["FC"])
    elif "NEE_CUT_REF" in obs:
        Fco2_obs = jnp.array(obs["NEE_CUT_REF"])
    elif "NEE_VUT_REF" in obs:
        Fco2_obs = jnp.array(obs["NEE_VUT_REF"])
    else:
        Fco2_obs = nan

    # Soil respiration
    if "Rsoil" in obs:
        Rsoil_obs = jnp.array(obs["Rsoil"])
    else:
        Rsoil_obs = nan

    # P_obs = jnp.array(obs["P_mm"]) if "P_mm" in obs else nan
    # LE_obs = jnp.array(obs["LE"]) if "LE" in obs else nan
    # H_obs = jnp.array(obs["H"]) if "H" in obs else nan
    # Gsoil_obs = jnp.array(obs["G_5cm"]) if "G_5cm" in obs else nan
    # Rnet_obs = jnp.array(obs["NETRAD"]) if "NETRAD" in obs else nan
    # GPP_obs = jnp.array(obs["GPP"]) if "GPP" in obs else nan
    # albedo_obs = jnp.array(obs["ALBEDO"]) if "ALBEDO" in obs else nan
    # Fco2_obs = jnp.array(obs["FCO2"]) if "FCO2" in obs else nan
    # Rsoil_obs = jnp.array(obs["Rsoil"]) if "Rsoil" in obs else nan
    # LE_obs, H_obs = jnp.array(obs["LE"]), jnp.array(obs["H"])
    # Gsoil_obs, Rnet_obs = jnp.array(obs["G_5cm"]), jnp.array(obs["NETRAD"])
    # nan = jnp.nan * jnp.ones(LE_obs.size)

    # # TODO: need to define the key words for the following variables
    # GPP_obs, albedo_obs, Fco2_obs, Rsoil_obs = nan, nan, nan, nan

    obs = Obs(
        P_obs,
        LE_obs,
        H_obs,
        GPP_obs,
        Rnet_obs,
        albedo_obs,
        Fco2_obs,
        Gsoil_obs,
        Rsoil_obs,
    )

    return obs


############################################################################
# States
############################################################################
# def initialize_profile(met: Met, para: Para, ntime: int, jtot: int, nlayers: int):
def initialize_profile(met: Met, para: Para):
    # ntime, jtot = setup.ntime, setup.n_can_layers
    # nlayers = setup.n_total_layers
    # ntime, jtot = met.zL.size, para.zht1.size
    # nlayers = para.zht.size
    # zht = para.zht
    # delz = para.delz
    ntime = met.zL.size
    nlayers = para.zht.size
    jtot = para.zht1.size
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
        # zht,
        # delz,
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
        para.delz,
        dij,
        met.ustar,
        met.zL,
        met.T_air_K,
        nlayers,
        nlayers_atmos,
        fact_heatcoef,
    )
    Told_K = prof.Told_K
    # jax.debug.print('Tair_K: {a}', a=Tair_K[18950, 28:36])
    # jax.debug.print('Told_K: {a}', a=Told_K[18950, 28:36])
    # jax.debug.print('soil.heat: {a}', a=soil.heat)

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
        para.delz,
        dij,
        met.ustar,
        met.zL,
        met.eair_Pa,
        nlayers,
        nlayers_atmos,
        fact_lecoef,
    )
    # jax.debug.print('fact_lecoef: {a}', a=fact_lecoef[11254:11257])
    # jax.debug.print('LE: {a}', a=LE[11254:11257,:10])
    # jax.debug.print('soil.evap: {a}', a=soil.evap[11254:11257])
    # jax.debug.print('met.zL: {a}', a=met.zL[11254:11257])
    # jax.debug.print('eair_Pa: {a}', a=eair_Pa[11254:11257,:10])
    # jax.debug.print('fact_lecoef: {a}', a=fact_lecoef[:1])
    # jax.debug.print('LE: {a}', a=LE[:1,:1])
    # jax.debug.print('soil.evap: {a}', a=soil.evap[:1])
    # jax.debug.print('met.zL: {a}', a=met.zL[:1])
    # jax.debug.print('eair_Pa: {a}', a=eair_Pa[:1,:1])
    eair_old_Pa = prof.eair_old_Pa
    eair_Pa = 0.25 * eair_Pa + 0.75 * eair_old_Pa
    eair_old_Pa = eair_Pa

    # jax.debug.print('eair_Pa: {a}', a=eair_Pa[18950, 28:36])
    # jax.debug.print('met.zl: {a}', a=met.zL)
    # jax.debug.print('co2: {a}', a=eair_Pa[18950, 28:36])

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
        para.delz,
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


def initialize_model_states(
    met: Met,
    para: Para,
    ntime: int,
    jtot: int,
    dt_soil: Float_0D,
    soil_mtime: int,
    n_soil_layers: int,
):
    ntime, jtot = met.zL.size, para.zht1.size
    depth = para.soil_depth
    # Soil
    soil = initialize_soil(para, met, ntime, dt_soil, soil_mtime, n_soil_layers, depth)

    # Quantum and NIR
    # quantum = initialize_parnir(para, setup, "par")
    # nir = initialize_parnir(para, setup, "nir")
    quantum = initialize_parnir(para, ntime, jtot, 0)
    nir = initialize_parnir(para, ntime, jtot, 1)

    # IR
    # ir = initialize_ir(setup)
    ir = initialize_ir(ntime, jtot)

    # Veg
    veg = initialize_veg(ntime)

    # Qin
    qin = initialize_qin(ntime, jtot)

    # TODO: remove RNet since it is not used
    # RNet
    rnet = initialize_rnet(ntime, jtot)

    # Sun
    sun = initialize_sunshade(ntime, jtot, met)

    # Shade
    shade = initialize_sunshade(ntime, jtot, met)

    # Lai
    lai = initialize_lai(ntime, jtot, para, met)

    # Can
    can = calculate_can(quantum, nir, ir, veg, soil, jtot)

    return soil, quantum, nir, ir, qin, rnet, sun, shade, veg, lai, can


# def initialize_parnir(para: Para, setup: Setup, wavebnd: str) -> ParNir:
def initialize_parnir(para: Para, ntime: int, jtot: int, wavebnd: int) -> ParNir:
    # ntime, jtot = setup.ntime, setup.n_can_layers
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
    # # TODO: how to link this with model parameters?
    # ratios = jax.lax.switch(
    #     wavebnd,
    #     [
    #         lambda: (
    #             jnp.array(para.par_reflect),
    #             jnp.array(para.par_trans),
    #             jnp.array(para.par_soil_refl),
    #         ),
    #         lambda: (
    #             jnp.array(para.nir_reflect),
    #             jnp.array(para.nir_trans),
    #             jnp.array(para.nir_soil_refl),
    #         ),
    #     ],
    # )
    # reflect, trans, soil_refl = ratios[0], ratios[1], ratios[2]
    # if wavebnd == "par":
    #     reflect = jnp.array(para.par_reflect)
    #     trans = jnp.array(para.par_trans)
    #     soil_refl = jnp.array(para.par_soil_refl)
    #     # absorbed = jnp.array(para.par_absorbed)
    # elif wavebnd == "nir":
    #     reflect = jnp.array(para.nir_reflect)
    #     trans = jnp.array(para.nir_trans)
    #     soil_refl = jnp.array(para.nir_soil_refl)
    #     # absorbed = jnp.array(para.nir_absorbed)
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
        # reflect,  # pyright: ignore
        # trans,  # pyright: ignore
        # soil_refl,  # pyright: ignore
        # absorbed,  # pyright: ignore
    )
    return rad


def initialize_ir(ntime: int, jtot: int) -> Ir:
    # jtot, ntime = setup.n_can_layers, setup.ntime
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


def initialize_veg(ntime: int) -> Veg:
    # ntime = setup.ntime
    Ps = jnp.zeros(ntime)
    gs = jnp.zeros(ntime)
    Rd = jnp.zeros(ntime)
    H = jnp.zeros(ntime)
    LE = jnp.zeros(ntime)
    Rnet = jnp.zeros(ntime)
    Tsfc = jnp.zeros(ntime)
    vpd = jnp.zeros(ntime)
    return Veg(Ps, gs, Rd, H, LE, Rnet, Tsfc, vpd)


def initialize_qin(ntime: int, jtot: int) -> Qin:
    # ntime, jtot = setup.ntime, setup.n_can_layers
    sun_abs = jnp.zeros([ntime, jtot])
    shade_abs = jnp.zeros([ntime, jtot])
    return Qin(sun_abs, shade_abs)


def initialize_rnet(ntime: int, jtot: int) -> Rnet:
    # ntime, jtot = setup.ntime, setup.n_can_layers
    jktot = jtot + 1
    sun = jnp.zeros([ntime, jktot])
    sh = jnp.zeros([ntime, jktot])
    sun_top = jnp.zeros([ntime, jktot])
    sh_top = jnp.zeros([ntime, jktot])
    sh_bottom = jnp.zeros([ntime, jktot])
    return Rnet(sun, sh, sun_top, sh_top, sh_bottom)


def initialize_sunshade(ntime: int, jtot: int, met: Met) -> SunShadedCan:
    # ntime, jtot = setup.ntime, setup.n_can_layers
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

    Leaf_RH = jnp.zeros([ntime, jtot])

    Tsfc = dot(met.T_air_K, Tsfc)

    return SunShadedCan(
        Ps,
        Resp,
        gs,
        vpd_Pa,
        LE,
        H,
        Rnet,
        Lout,
        closure,
        Tsfc,
        Tsfc_new,
        Tsfc_old,
        Leaf_RH,
    )


def initialize_lai(ntime: int, jtot: int, para: Para, met: Met) -> Lai:
    # nlayers = setup.n_can_layers
    lai = met.lai
    dff = jnp.ones([ntime, jtot]) / jtot  # (ntime,nlayers)
    dff = dot(lai, dff)  # (ntime, nlayers)
    # TODO: double check!
    # self.sumlai = jax.lax.cumsum(self.dff, axis=1, reverse=True) #(ntime,nlayers)
    sumlai = minus(lai, jax.lax.cumsum(dff, axis=1))  # (ntime, nlayers)
    sumlai = jnp.clip(sumlai, a_min=0.0)  # (ntime, nlayers)
    dff_clmp = dff / para.markov  # (ntime, nlayers)

    # divide by height of the layers in the canopy
    adens = dff[:, :jtot] / para.dht_canopy  # (ntime, nlayers)

    return Lai(lai, dff, sumlai, dff_clmp, adens)


def initialize_soil(
    # para: Para, met: Met, n_soil: int = 10, depth: Float_0D = 0.15
    # setup: Setup,
    para: Para,
    met: Met,
    ntime: int,
    # dt_soil: int,
    dt_soil: Float_0D,
    soil_mtime: int,
    n_soil_layers: int,
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
    dt = dt_soil
    # mtime = soil_mtime
    # mtime = floor(1800.0 / dt)  # time steps per half hour

    n_soil = n_soil_layers  # number of soil layers
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
        dot(met.Tsoil, jnp.ones([ntime, n_soil_2])) + 273.15
    )  # (ntime, n_soil_2)  # noqa: E501

    # initialize upper boundary temperature as air temperature
    # in later iterations this is reset to the radiative surface
    # temperature
    T_soil_up_boundary = met.T_air_K
    sfc_temperature = T_soil_up_boundary
    sfc_temperature_old = sfc_temperature
    bulk_density = (
        jnp.ones([ntime, n_soil]) * 0.83
    )  # soil bulk density for the alfalfa, g cm-3, (ntime, n_soil)  # noqa: E501

    # thermal conductivity code from Campbell and Norman
    fw = 1.0 / (
        1 + jnp.power((met.soilmoisture / 0.15), -4)
    )  # terms for Stefan flow as water evaporates in the pores  # noqa: E501
    K_air = (
        0.024 + 44100.0 * 2.42e-5 * fw * met.air_density_mole * met.dest / met.P_Pa
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
    evap = jnp.zeros(ntime)  # initialization
    heat = jnp.zeros(ntime)
    # self.lout=jnp.zeros(setup.ntime)
    rnet = jnp.zeros(ntime)
    gsoil = jnp.zeros(ntime)
    resp = jnp.zeros(ntime)
    lout_sfc = para.epsigma * jnp.power(met.T_air_K, 4)  # initialization
    llout = lout_sfc

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
        # n_soil,
        depth,
        # mtime,
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
        lout_sfc,
        llout,
        resp,
    )


def calculate_can(
    quantum: ParNir, nir: ParNir, ir: Ir, veg: Veg, soil: Soil, jtot: int
) -> Can:
    # Calculate the states/fluxes across the whole canopy
    rnet_calc = (
        quantum.beam_flux[:, jtot] / 4.6
        + quantum.dn_flux[:, jtot] / 4.6
        - quantum.up_flux[:, jtot] / 4.6
        + nir.beam_flux[:, jtot]
        + nir.dn_flux[:, jtot]
        - nir.up_flux[:, jtot]
        + ir.ir_dn[:, jtot]
        + -ir.ir_up[:, jtot]
    )
    LE = veg.LE + soil.evap
    H = veg.H + soil.heat
    rnet = veg.Rnet + soil.rnet
    NEE = soil.resp - veg.GPP
    avail = rnet_calc - soil.gsoil
    gsoil = soil.gsoil
    # albedo_calc = (quantum.up_flux[:, jtot] / 4.6 + nir.up_flux[:, jtot]) / (
    #     quantum.incoming / 4.6 + nir.incoming
    # )
    # nir_albedo_calc = nir.up_flux[:, jtot] / nir.incoming
    nir_refl = nir.up_flux[:, jtot] - nir.up_flux[:, 0]

    return Can(
        rnet_calc,
        rnet,
        LE,
        H,
        NEE,
        avail,
        gsoil,
        # albedo_calc,
        # nir_albedo_calc,
        nir_refl,
    )


def calculate_var_stats(
    met: Met, obs: Obs
) -> Tuple[VarStats, VarStats, VarStats, VarStats]:  # noqa: E501
    # Calculate the meteorological statistics (i.e., mean and standard deviation)
    T_air_mean = met.T_air.mean()
    rglobal_mean = met.rglobal.mean()
    eair_mean = met.eair.mean()
    wind_mean = met.wind.mean()
    CO2_mean = met.CO2.mean()
    P_kPa_mean = met.P_kPa.mean()
    ustar_mean = met.ustar.mean()
    Tsoil_mean = met.Tsoil.mean()
    soilmoisture_mean = met.soilmoisture.mean()
    zcanopy_mean = met.zcanopy.mean()
    lai_mean = met.lai.mean()
    rsoil_mean = obs.Rsoil.mean()
    LE_mean = obs.LE.mean()
    H_mean = obs.H.mean()
    vpd_mean = met.vpd.mean()

    # Calculate the meteorological statistics (i.e., mean and standard deviation)
    T_air_std = met.T_air.std()
    rglobal_std = met.rglobal.std()
    eair_std = met.eair.std()
    wind_std = met.wind.std()
    CO2_std = met.CO2.std()
    P_kPa_std = met.P_kPa.std()
    ustar_std = met.ustar.std()
    Tsoil_std = met.Tsoil.std()
    soilmoisture_std = met.soilmoisture.std()
    zcanopy_std = met.zcanopy.std()
    lai_std = met.lai.std()
    rsoil_std = obs.Rsoil.std()
    LE_std = obs.LE.std()
    H_std = obs.H.std()
    vpd_std = met.vpd.std()

    # Calculate the meteorological statistics (i.e., mean and standard deviation)
    T_air_max = met.T_air.max()
    rglobal_max = met.rglobal.max()
    eair_max = met.eair.max()
    wind_max = met.wind.max()
    CO2_max = met.CO2.max()
    P_kPa_max = met.P_kPa.max()
    ustar_max = met.ustar.max()
    Tsoil_max = met.Tsoil.max()
    soilmoisture_max = met.soilmoisture.max()
    zcanopy_max = met.zcanopy.max()
    lai_max = met.lai.max()
    rsoil_max = obs.Rsoil.max()
    LE_max = obs.LE.max()
    H_max = obs.H.max()
    vpd_max = met.vpd.max()

    # Calculate the meteorological statistics (i.e., mean and standard deviation)
    T_air_min = met.T_air.min()
    rglobal_min = met.rglobal.min()
    eair_min = met.eair.min()
    wind_min = met.wind.min()
    CO2_min = met.CO2.min()
    P_kPa_min = met.P_kPa.min()
    ustar_min = met.ustar.min()
    Tsoil_min = met.Tsoil.min()
    soilmoisture_min = met.soilmoisture.min()
    zcanopy_min = met.zcanopy.min()
    lai_min = met.lai.min()
    rsoil_min = obs.Rsoil.min()
    LE_min = obs.LE.min()
    H_min = obs.H.min()
    vpd_min = met.vpd.min()

    var_mean = VarStats(
        T_air_mean,
        rglobal_mean,
        eair_mean,
        wind_mean,
        CO2_mean,
        P_kPa_mean,
        ustar_mean,
        Tsoil_mean,
        soilmoisture_mean,
        zcanopy_mean,
        lai_mean,
        rsoil_mean,
        LE_mean,
        H_mean,
        vpd_mean,
    )

    var_std = VarStats(
        T_air_std,
        rglobal_std,
        eair_std,
        wind_std,
        CO2_std,
        P_kPa_std,
        ustar_std,
        Tsoil_std,
        soilmoisture_std,
        zcanopy_std,
        lai_std,
        rsoil_std,
        LE_std,
        H_std,
        vpd_std,
    )

    var_max = VarStats(
        T_air_max,
        rglobal_max,
        eair_max,
        wind_max,
        CO2_max,
        P_kPa_max,
        ustar_max,
        Tsoil_max,
        soilmoisture_max,
        zcanopy_max,
        lai_max,
        rsoil_max,
        LE_max,
        H_max,
        vpd_max,
    )

    var_min = VarStats(
        T_air_min,
        rglobal_min,
        eair_min,
        wind_min,
        CO2_min,
        P_kPa_min,
        ustar_min,
        Tsoil_min,
        soilmoisture_min,
        zcanopy_min,
        lai_min,
        rsoil_min,
        LE_min,
        H_min,
        vpd_min,
    )

    return var_mean, var_std, var_max, var_min
