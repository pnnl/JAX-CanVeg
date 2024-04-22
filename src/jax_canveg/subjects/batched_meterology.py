"""
A class for meterology variables.

Author: Peishi Jiang
Date: 2023.7.24.
"""

# import jax
import jax.tree_util as jtu

# import numpy as np
import jax.numpy as jnp

import equinox as eqx

# from typing import Optional, Tuple

# from .parameters import Para
from ..shared_utilities.types import Float_2D
from .utils import es as fes
from .utils import llambda, desdt, des2dt

from .meterology import Met

Mair = 28.97  # the molecular weight of air
rugc = 8.314  # J mole-1 K-1


class BatchedMet(eqx.Module):
    zL: Float_2D
    year: Float_2D
    day: Float_2D
    hhour: Float_2D
    T_air: Float_2D
    rglobal: Float_2D
    eair: Float_2D
    wind: Float_2D
    CO2: Float_2D
    P_kPa: Float_2D
    ustar: Float_2D
    Tsoil: Float_2D
    soilmoisture: Float_2D
    zcanopy: Float_2D
    lai: Float_2D

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


def convert_met_to_batched_met(met: Met, n_batch: int, batch_size: int) -> BatchedMet:
    n_total = n_batch * batch_size
    batched_met = jtu.tree_map(
        lambda x: x[:n_total].reshape([n_batch, batch_size]), met
    )
    return batched_met


def convert_batchedmet_to_met(batched_met: BatchedMet) -> Met:
    met = jtu.tree_map(lambda x: x.reshape(-1), batched_met)
    return met
