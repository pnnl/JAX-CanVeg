"""
A class for meterology variables.

Author: Peishi Jiang
Date: 2023.7.24.
"""

import jax
import jax.numpy as jnp

from math import floor

# from typing import Array
from ..shared_utilities.types import Float_0D, Float_1D

dot = jax.vmap(lambda x, y: x * y, in_axes=(None, 1), out_axes=1)


class Para(object):

    # TODO: the parameter inputs should be more generic!
    def __init__(
        self,
        time_zone: int = -8,
        latitude: Float_0D = 38.0991538,
        longitude: Float_0D = -121.49933,
        stomata: int = 2,
        hypo_amphi: int = 1,
        veg_ht: Float_0D = 0.8,
        leafangle: int = 1,
        n_can_layers: int = 30,
        meas_ht: Float_0D = 5.0,
        n_hr_per_day: int = 48,
        n_time: int = 200,
    ) -> None:
        self.time_zone = time_zone  # time zone
        self.lat_deg = latitude  # latitude
        self.long_deg = longitude  # longitude
        # self.lat_deg= 38.1074;   % latitude
        # self.long_deg= -121.6469; % longitude

        self.stomata = stomata  # amphistomatous = 2; hypostomatous = 1
        self.hypo_amphi = hypo_amphi  # hypostomatous, 1, amphistomatous, 2
        self.veg_ht = (
            veg_ht  # vegetation canopy height, aerodynamic height, Bouldin, 2018
        )

        self.nlayers = n_can_layers

        self.meas_ht = meas_ht  # measurement height
        self.dht = 0.6 * self.veg_ht  # zero plane displacement height, m
        self.z0 = 0.1 * self.veg_ht  # aerodynamic roughness height, m
        self.markov = 0.95  # markov clumping factor, 0 to 1, from LI 2200
        self.markov = jnp.ones(self.nlayers) * self.markov
        # self.markov=ones(self.nlayers,1)*prm.markov;

        # TODO: The leaf area of each layer should be dynamic saved in staes
        # prm.LAI = 3.6;  % leaf area index  May day 141 to 157, 2019
        # prm.dff= 0.1; % leaf area of each layer
        # prm.nlayers=prm.LAI/prm.dff;  % number of canopy layers
        # prm.markov=ones(prm.nlayers,1)*prm.markov;
        # prm.dff=ones(prm.nlayers,1)*prm.dff;
        # prm.sumlai=prm.LAI-cumsum(prm.dff);
        # prm.sumlai(prm.sumlai <0)=0;
        # prm.dff_clmp=prm.dff./prm.markov;

        self.jtot = self.nlayers  # number of layers
        self.jktot = self.jtot + 1  # number of layers plus 1
        self.jtot3 = self.nlayers * 3  # 3 times layers for Dij
        self.jktot3 = self.jtot3 + 1  # 3 times layers for Dij
        self.dht_canopy = self.veg_ht / self.jtot  # height increment of each layer
        self.ht_atmos = self.meas_ht - self.veg_ht
        n_atmos_layers = 50
        self.dht_atmos = self.ht_atmos / n_atmos_layers
        self.nlayers_atmos = self.jtot + floor(self.ht_atmos / self.dht_atmos)
        # jnp.floor(self.ht_atmos / self.dht_atmos).astype(int)
        self.nlayers_atmos = int(self.nlayers_atmos)

        # Set up time
        self.hrs = n_hr_per_day  # half hour periods per day
        self.ntime = n_time
        self.ndays = n_time / n_hr_per_day

        # Initialize the lai states
        self.dff = jnp.zeros([self.ntime, self.nlayers])
        self.sumlai = jnp.zeros([self.ntime, self.nlayers])
        self.dff_clmp = jnp.zeros([self.ntime, self.nlayers])

        # calculate height of layers in the canopy and atmosphere
        zht1 = jnp.arange(1, self.jktot) * self.dht_canopy
        delz1 = jnp.ones(self.jtot) * self.dht_canopy
        zht2 = (
            jnp.arange(1, self.nlayers_atmos - self.jtot + 1) * self.dht_atmos
            + self.veg_ht
        )
        delz2 = jnp.ones(self.nlayers_atmos - self.jtot) * self.dht_atmos
        self.zht = jnp.concatenate([zht1, zht2])
        self.delz = jnp.concatenate([delz1, delz2])

        self.par_reflect = 0.05
        self.par_trans = 0.05
        self.par_absorbed = 1.0 - self.par_reflect - self.par_trans
        self.par_soil_refl = 0.05
        self.nir_reflect = 0.60  # 0.4 .based on field and Ocean Optics...wt with Planck Law. High leaf N and high reflected NIR  # noqa: E501
        self.nir_trans = 0.20  # 0.4 ...UCD presentation shows NIR transmission is about the same as reflectance; 80% NIR reflectance 700 to 1100 nm  # noqa: E501
        self.nir_soil_refl = 0.10  # Ocean Optics spectra May 2, 2019 after cutting
        self.nir_absorbed = 1.0 - self.nir_reflect - self.nir_trans

        # Leaf angle distributions
        # options include spherical - 1, planophile - 2, erectophile - 3, uniform - 4,
        # plagiophile - 5, extremophile - 6
        self.leafangle = leafangle

        # IR fluxes
        self.sigma = 5.670367e-08  # Stefan Boltzmann constant W m-2 K-4
        self.ep = 0.985  # emissivity of leaves
        self.epm1 = 0.015  # 1- ep
        self.epsoil = 0.98  # Emissivity of soil
        self.epsigma = 5.5566e-8  # ep*sigma
        # self.epm1=0.02                   # 1- ep
        self.epsigma2 = 11.1132e-8  # 2*ep*sigma
        self.epsigma4 = 22.2264e-8  # 4.0 * ep * sigma
        self.epsigma6 = 33.3396e-8  # 6.0 * ep * sigma
        self.epsigma8 = 44.448e-8  # 8.0 * ep * sigma
        self.epsigma12 = 66.6792e-8  # 12.0 * ep * sigma

        self.ir_reflect = 1 - self.ep  # tranmittance is zero, 1 minus emissivity
        self.ir_trans = 0
        self.ir_soil_refl = 1 - self.epsoil
        self.ir_absorbed = self.ep

        # Universal gas constant
        self.rugc = 8.314  # J mole-1 K-1
        self.rgc1000 = 8314  # gas constant times 1000.
        self.Cp = 1005  # specific heat of air, J kg-1 K-1

        # parameters for the Farquhar et al photosynthesis model
        # Consts for Photosynthesis model and kinetic equations.
        # for Vcmax and Jmax.  Taken from Harley and Baldocchi (1995, PCE)
        # carboxylation rate at 25 C temperature, umol m-2 s-1
        # these variables were computed from A-Ci measurements and the
        # Sharkey Excel spreadsheet tool
        self.vcopt = 171  # carboxylation rate at 25 C temperature, umol m-2 s-1; from field measurements Rey Sanchez alfalfa  # noqa: E501
        self.jmopt = 259  # electron transport rate at 25 C temperature, umol m-2 s-1, field measuremetns Jmax = 1.64 Vcmax  # noqa: E501
        self.rd25 = 2.68  # dark respiration at 25 C, rd25= 0.34 umol m-2 s-1, field measurements   # noqa: E501
        self.hkin = 200000.0  # enthalpy term, J mol-1
        self.skin = 710.0  # entropy term, J K-1 mol-1
        self.ejm = 55000.0  # activation energy for electron transport, J mol-1
        self.evc = 55000.0  # activation energy for carboxylation, J mol-1

        # Enzyme constants & partial pressure of O2 and CO2
        # Michaelis-Menten K values. From survey of literature.
        self.kc25 = 274.6  # kinetic coef for CO2 at 25 C, microbars
        self.ko25 = 419.8  # kinetic coef for O2 at 25C,  millibars
        self.o2 = 210.0  # oxygen concentration  mmol mol-1
        self.ekc = 80500.0  # Activation energy for K of CO2; J mol-1
        self.eko = 14500.0  # Activation energy for K of O2, J mol-1
        self.erd = 38000.0  # activation energy for dark respiration, eg Q10=2
        self.ektau = -29000.0  # J mol-1 (Jordan and Ogren, 1984)
        self.tk_25 = 298.16  # absolute temperature at 25 C

        # Ps code was blowing up at high leaf temperatures, so reduced opt
        # and working perfectly
        self.toptvc = 303.0  # optimum temperature for maximum carboxylation, 311 K
        self.toptjm = 303.0  # optimum temperature for maximum electron transport,311 K  # noqa: E501
        self.kball = 8.17  # Ball-Berry stomatal coefficient for stomatal conductance, data from Camilo Rey Sanchez bouldin Alfalfa  # noqa: E501
        self.bprime = 0.05  # intercept leads to too big LE0.14;    % mol m-2 s-1 h2o..Camilo Rey Sanchez..seems high  # noqa: E501
        self.bprime16 = self.bprime / 1.6  # intercept for CO2, bprime16 = bprime / 1.6;

        # simple values for priors and first iteration
        self.rsm = 145.0  # Minimum stomatal resistance, s m-1.
        self.brs = 60.0  # curvature coeffient for light response
        self.qalpha = 0.22  #  leaf quantum yield, electrons
        self.qalpha2 = 0.0484  # qalpha squared, qalpha2 = pow(qalpha, 2.0);
        self.lleaf = 0.04  # leaf length, m, alfalfa, across the trifoliate

        # Diffusivity values for 273 K and 1013 mb (STP) using values from Massman (1998) Atmos Environment  # noqa: E501
        # These values are for diffusion in air.  When used these values must be adjusted for  # noqa: E501
        # temperature and pressure
        # nu, Molecular viscosity
        self.nuvisc = 13.27  # mm2 s-1
        self.nnu = 0.00001327  # m2 s-1

        # Diffusivity of CO2
        self.dc = 13.81  # / mm2 s-1
        self.ddc = 0.00001381  # / m2 s-1

        # Diffusivity of heat
        self.dh = 18.69  # mm2 s-1
        self.ddh = 0.00001869  # m2 s-1

        # Diffusivity of water vapor
        self.dv = 21.78  # / mm2 s-1
        self.ddv = 0.00002178  # m2 s-1

        # Diffusivity of ozone
        self.do3 = 14.44  # m2 s-1
        self.ddo3 = 0.00001444  # m2 s-1
        self.betfac = 1.5  # the boundary layer sheltering factor from Grace

        # Constants for leaf boundary layers
        self.lfddh = self.lleaf / self.ddh

        # Prandtl Number
        self.pr = self.nuvisc / self.dh
        self.pr33 = self.pr**0.33

        self.lfddv = self.lleaf / self.ddv

        # SCHMIDT NUMBER FOR VAPOR
        self.sc = self.nuvisc / self.dv
        self.sc33 = self.sc**0.33

        # SCHMIDT NUMBER FOR CO2
        self.scc = self.nuvisc / self.dc
        self.scc33 = self.scc**0.33

        # Grasshof Number
        self.grasshof = 9.8 * self.lleaf**3 / (self.nnu**2)

        self.Mair = 28.97
        self.dLdT = -2370.0
        self.extinct = 2  # extinction coefficient wind in canopy

        # Dispersion Matrix Lagrangian model
        self.npart = 500000  # number of random walk particles, use about 10,000 for testing, up to 1M for smoother profiles  # noqa: E501, E501

    # def set_time(self, days: Float_1D) -> None:
    #     self.ntime=len(days)   # number of 30 minute runs
    #     self.ndays=self.ntime/self.hrs      # number of days

    def set_lai(self, lai: Float_1D) -> None:
        assert self.ntime == lai.size
        # self.dff = 1./ self.nlayers
        self.dff = (
            jnp.ones([self.ntime, self.nlayers]) / self.nlayers
        )  # (ntime,nlayers)
        self.dff = dot(lai, self.dff)  # (ntime, nlayers)
        self.sumlai = jax.lax.cumsum(self.dff, axis=1, reverse=True)  # (ntime, nlayers)
        self.sumlai = jnp.clip(self.sumlai, a_min=0.0)  # (ntime, nlayers)
        self.dff_clmp = self.dff / self.markov  # (ntime, nlayers)

    def _tree_flatten(self):
        children = (
            self.nlayers,
            self.dht,
            self.z0,
            self.markov,
            self.jktot,
            self.jtot3,
            self.jktot3,
            self.dht_canopy,
            self.ht_atmos,
            self.dht_atmos,
            self.nlayers_atmos,  # noqa: E501
            self.ndays,
            self.dff,
            self.sumlai,
            self.dff_clmp,
            self.zht,
            self.delz,
            self.par_reflect,
            self.par_trans,
            self.par_absorbed,
            self.par_soil_refl,
            self.nir_reflect,
            self.nir_trans,
            self.nir_soil_refl,
            self.nir_absorbed,
            self.sigma,
            self.ep,
            self.epm1,
            self.epsoil,
            self.epsigma,
            self.epsigma2,
            self.epsigma4,
            self.epsigma6,
            self.epsigma8,
            self.epsigma12,
            self.ir_reflect,
            self.ir_trans,
            self.ir_soil_refl,
            self.ir_absorbed,
            self.rugc,
            self.rgc1000,
            self.Cp,
            self.vcopt,
            self.jmopt,
            self.rd25,
            self.hkin,
            self.skin,
            self.ejm,
            self.evc,
            self.kc25,
            self.ko25,
            self.o2,
            self.ekc,
            self.eko,
            self.erd,
            self.ektau,
            self.tk_25,
            self.toptvc,
            self.toptjm,
            self.kball,
            self.bprime,
            self.bprime16,
            self.rsm,
            self.brs,
            self.qalpha,
            self.qalpha2,
            self.lleaf,
            self.nuvisc,
            self.nnu,
            self.dc,
            self.ddc,
            self.dh,
            self.ddh,
            self.dv,
            self.ddv,
            self.do3,
            self.ddo3,
            self.betfac,
            self.lfddh,
            self.pr,
            self.pr33,
            self.lfddv,
            self.sc,
            self.sc33,
            self.scc,
            self.scc33,
            self.grasshof,
            self.Mair,
            self.dLdT,
            self.extinct,
            self.npart,
        )
        aux_data = {
            "n_can_layers": self.jtot,
            "n_time": self.ntime,
            "n_hr_per_day": self.hrs,
            "stomata": self.stomata,
            "hypo_amphi": self.hypo_amphi,
            "veg_ht": self.veg_ht,
            "time_zone": self.time_zone,
            "latitude": self.lat_deg,
            "longitude": self.long_deg,  # noqa: E501
            "meas_ht": self.meas_ht,
            "leafangle": self.leafangle,
        }
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)

        # time_zone: int = -8,
        # latitude: Float_0D = 38.0991538,
        # longitude: Float_0D = -121.49933,
        # stomata: int = 2,
        # hypo_amphi: int = 1,
        # veg_ht: Float_0D = 0.8,
        # leafangle: int = 1,
        # n_can_layers: int = 30,
        # meas_ht: Float_0D = 5.0,
        # n_hr_per_day: int = 48,
        # n_time: int = 200,
