"""
A class for meterology variables.

Author: Peishi Jiang
Date: 2023.7.24.
"""

# import jax

# import numpy as np
import jax.numpy as jnp

import equinox as eqx

# from typing import Optional, Tuple

# from .parameters import Para
from ..shared_utilities.types import Float_1D
from .utils import es as fes
from .utils import llambda, desdt, des2dt

Mair = 28.97  # the molecular weight of air
rugc = 8.314  # J mole-1 K-1


class Met(eqx.Module):
    # ntime: Int_0D
    # Mair: Float_0D
    # rugc: Float_0D
    zL: Float_1D
    year: Float_1D
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
        return self.P_kPa * Mair / (rugc * self.T_air_K)

    @property
    def air_density_mole(self):
        return 1000.0 * self.air_density / Mair

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


# def initialize_met(data: Float_2D, ntime: Int_0D, zL0: Float_1D) -> Met:
#     assert ntime == data.shape[0]
#     year = jnp.array(data[:, 0])  # day of year
#     day = jnp.array(data[:, 1])  # day of year
#     hhour = jnp.array(data[:, 2])  # hour
#     # self.T_air_K = jnp.array(data[:, 2]) + 273.15  # air temperature, K
#     T_air = jnp.array(data[:, 3])  # air temperature, degC
#     rglobal = jnp.array(data[:, 4])  # global shortwave radiation, W m-2
#     eair = jnp.array(data[:, 5])  # vapor pressure, kPa
#     wind = jnp.array(data[:, 6])  # wind velocity, m/s
#     CO2 = jnp.array(data[:, 7])  # CO2, ppm
#     P_kPa = jnp.array(data[:, 8])  # atmospheric pressure, kPa
#     ustar = jnp.array(data[:, 9])  # friction velocity, m/s
#     Tsoil = jnp.array(data[:, 10])  # soil temperature, C...16 cm
#     soilmoisture = jnp.array(data[:, 11])  # soil moisture, fraction
#     zcanopy = jnp.array(data[:, 12])  # aerodynamic canopy height
#     lai = jnp.array(data[:, 13])  # leaf area index [-]

#     # Some operations to ensure stability
#     wind = jnp.clip(wind, a_min=0.75)
#     # ustar = jnp.clip(ustar, a_min=0.75)
#     ustar = jnp.clip(ustar, a_min=0.1)
#     rglobal = jnp.clip(rglobal, a_min=0.0)

#     # Convert the following int and float to jax.ndarray
#     # ntime = jnp.array(ntime)
#     # Mair = jnp.array(Mair)
#     # rugc = jnp.array(rugc)

#     met = Met(
#         # ntime,
#         # Mair,
#         # rugc,
#         zL0,
#         year,
#         day,
#         hhour,
#         T_air,
#         rglobal,
#         eair,
#         wind,
#         CO2,
#         P_kPa,
#         ustar,
#         Tsoil,
#         soilmoisture,
#         zcanopy,
#         lai,
#     )

#     return met


# def get_met_forcings(f_forcing: str, lai: Optional[Float_0D] = None)->Tuple[Met,int]:
#     # Load the modeling forcing text file
#     # This should be a matrix of forcing data with each column representing
#     # a time series of observations
#     forcing_data = np.loadtxt(f_forcing, delimiter=",")
#     forcing_data = jnp.array(forcing_data)
#     n_time = forcing_data.shape[0]
#     # Initialize the zl length with zeros
#     zl0 = jnp.zeros(n_time)
#     # Set up the lai if not None
#     if lai is not None:
#         forcing_data = jnp.concatenate(
#             # [forcing_data[:, :12], jnp.ones([n_time, 1]) * lai], axis=1
#             [forcing_data[:, :13], jnp.ones([n_time, 1]) * lai], axis=1
#         )
#     # Initialize the met instance
#     met = initialize_met(forcing_data, n_time, zl0)
#     return met, n_time
