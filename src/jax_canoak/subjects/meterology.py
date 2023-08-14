"""
A class for meterology variables.

Author: Peishi Jiang
Date: 2023.7.24.
"""

import jax.numpy as jnp

import equinox as eqx

# from .parameters import Para
from ..shared_utilities.types import Float_2D, Int_0D, Float_0D, Float_1D
from .utils import es as fes
from .utils import llambda, desdt, des2dt

Mair_default = 28.97  # the molecular weight of air
rugc_default = 8.314  # J mole-1 K-1


class Met(eqx.Module):
    ntime: Int_0D
    Mair: Float_0D
    rugc: Float_0D
    zL: Float_1D
    day: Float_1D
    hhour: Float_1D
    T_air: Float_1D
    rglobal: Float_1D
    eair: Float_1D
    wind: Float_1D
    CO2: Float_1D
    P_kPa: Float_1D
    ustar: Float_1D
    Tsoil: Float_1D
    soilmoisture: Float_1D
    zcanopy: Float_1D
    lai: Float_1D

    @property
    def T_air_K(self):
        return self.T_air + 273.15

    @property
    def parin(self):
        return 4.6 * self.rglobal / 2.0

    @property
    def eair_Pa(self):
        return self.eair * 1000  # vapor pressure, Pa

    @property
    def P_Pa(self):
        return self.P_kPa * 1000  # pressure, Pa

    @property
    def es(self):
        return fes(self.T_air_K)  # saturation vapor pressure, Pa

    @property
    def vpd_Pa(self):
        return self.es - self.eair_Pa  # atmospheric vapor pressure deficit, Pa

    @property
    def air_density(self):
        # air density, kg m-3
        return self.P_kPa * self.Mair / (self.rugc * self.T_air_K)

    @property
    def air_density_mole(self):
        return 1000.0 * self.air_density / self.Mair

    @property
    def dest(self):
        return desdt(self.T_air_K)

    @property
    def d2est(self):
        return des2dt(self.T_air_K)  # second derivative es(T)

    @property
    def llambda(self):
        # latent heat of vaporization, J kg-1
        return jnp.vectorize(llambda)(self.T_air_K)


def initialize_met(
    data: Float_2D,
    ntime: Int_0D,
    zL0: Float_1D,
    Mair: Float_0D = Mair_default,
    rugc: Float_0D = rugc_default,
) -> Met:
    day = jnp.array(data[:, 0])  # day of year
    hhour = jnp.array(data[:, 1])  # hour
    # self.T_air_K = jnp.array(data[:, 2]) + 273.15  # air temperature, K
    T_air = jnp.array(data[:, 2])  # air temperature, degC
    rglobal = jnp.array(data[:, 3])  # global shortwave radiation, W m-2
    eair = jnp.array(data[:, 4])  # vapor pressure, kPa
    wind = jnp.array(data[:, 5])  # wind velocity, m/s
    CO2 = jnp.array(data[:, 6])  # CO2, ppm
    P_kPa = jnp.array(data[:, 7])  # atmospheric pressure, kPa
    ustar = jnp.array(data[:, 8])  # friction velocity, m/s
    Tsoil = jnp.array(data[:, 9])  # soil temperature, C...16 cm
    soilmoisture = jnp.array(data[:, 10])  # soil moisture, fraction
    zcanopy = jnp.array(data[:, 11])  # aerodynamic canopy height
    lai = jnp.array(data[:, 12])  # leaf area index [-]

    # Some operations to ensure stability
    wind = jnp.clip(wind, a_min=0.75)
    ustar = jnp.clip(ustar, a_min=0.1)
    rglobal = jnp.clip(rglobal, a_min=0.0)

    # Convert the following int and float to jax.ndarray
    ntime = jnp.array(ntime)
    Mair = jnp.array(Mair)
    rugc = jnp.array(rugc)

    met = Met(
        ntime,
        Mair,
        rugc,
        zL0,
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
