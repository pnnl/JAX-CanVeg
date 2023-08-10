"""
A class for meterology variables.

Author: Peishi Jiang
Date: 2023.7.24.
"""

import jax.numpy as jnp

# from .parameters import Para
from ..shared_utilities.types import Float_2D, Int_0D, Float_0D, Float_1D
from .utils import es as fes
from .utils import llambda, desdt, des2dt

Mair = 28.97
rugc = 8.314  # J mole-1 K-1


class Met(object):
    def __init__(
        self,
        data: Float_2D,
        ntime: Int_0D,
        zL: Float_1D,
        Mair: Float_0D = Mair,
        rugc: Float_0D = rugc,
    ) -> None:
        assert data.shape[0] == ntime
        self.ntime = ntime
        self.Mair, self.rugc = Mair, rugc
        self.day = jnp.array(data[:, 0])  # day of year
        self.hhour = jnp.array(data[:, 1])  # hour
        # self.T_air_K = jnp.array(data[:, 2]) + 273.15  # air temperature, K
        self.T_air = jnp.array(data[:, 2])  # air temperature, degC
        self.rglobal = jnp.array(data[:, 3])  # global shortwave radiation, W m-2
        self.eair = jnp.array(data[:, 4])  # vapor pressure, kPa
        self.wind = jnp.array(data[:, 5])  # wind velocity, m/s
        self.CO2 = jnp.array(data[:, 6])  # CO2, ppm
        self.P_kPa = jnp.array(data[:, 7])  # atmospheric pressure, kPa
        self.ustar = jnp.array(data[:, 8])  # friction velocity, m/s
        self.Tsoil = jnp.array(data[:, 9])  # soil temperature, C...16 cm
        self.soilmoisture = jnp.array(data[:, 10])  # soil moisture, fraction
        self.zcanopy = jnp.array(data[:, 11])  # aerodynamic canopy height
        self.lai = jnp.array(data[:, 12])  # leaf area index [-]

        # Some operations to ensure stability
        self.wind = jnp.clip(self.wind, a_min=0.75)
        self.ustar = jnp.clip(self.ustar, a_min=0.75)
        self.rglobal = jnp.clip(self.rglobal, a_min=0.0)

        # self.zL = jnp.zeros(self.day.size)  # z/L initial value at 0
        self.zL = zL

        # # Calculate other meterological forcings/states
        # self.parin = (
        #     4.6 * self.rglobal / 2.0
        # )  # visible, or photosynthetic photon flux density, umol m-2 s-1  # noqa: E501
        # self.eair_Pa = self.eair * 1000  # vapor pressure, Pa
        # self.P_Pa = self.P_kPa * 1000  # pressure, Pa

        # # Compute gas and model coefficients that are used repeatedly
        # self.es = fes(self.T_air_K)  # saturation vapor pressure, Pa
        # self.vpd_Pa = self.es - self.eair_Pa  # atmospheric vapor pressure deficit, Pa
        # self.air_density = (
        #     self.P_kPa * Mair / (rugc * self.T_air_K)
        # )  # air density, kg m-3  # noqa: E501
        # self.air_density_mole = (
        #     1000.0 * self.air_density / Mair
        # )  # air density, moles m-3
        # self.dest = desdt(
        #     self.T_air_K
        # )  # slope saturation vapor pressure Temperature Pa K-1  # noqa: E501
        # self.d2est = des2dt(self.T_air_K)  # second derivative es(T)
        # self.llambda = jnp.vectorize(llambda)(
        #     self.T_air_K
        # )  # latent heat of vaporization, J kg-1  # noqa: E501

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
        return self.P_kPa * self.Mair / (rugc * self.T_air_K)

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

    def _tree_flatten(self):
        children = (
            self.day,
            self.hhour,
            self.T_air,
            self.rglobal,
            self.eair,
            self.wind,
            self.CO2,
            self.P_kPa,
            self.ustar,
            self.Tsoil,
            self.soilmoisture,
            self.soilmoisture,
            self.zcanopy,
            self.lai,
            # self.parin,
            # self.P_Pa,
            # self.es,
            # self.vpd_Pa,
            # self.air_density,
            # self.air_density_mole,
            # self.dest,
            # self.d2est,
            # self.llambda,
            self.zL,
        )
        aux_data = {"ntime": self.ntime, "Mair": self.Mair, "rugc": self.rugc}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        data = jnp.stack(children[:13]).T
        # aux_data['zL'] = children[-1]
        return cls(data, zL=children[-1], **aux_data)
