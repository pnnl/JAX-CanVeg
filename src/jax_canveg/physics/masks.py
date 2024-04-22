"""
Functions used for generating masks.

- generate_night_mask()
- generate_turbulence_mask()

Author: Peishi Jiang
Date: 2023.8.16.
"""

import jax.numpy as jnp
from ..subjects import SunAng, Para, Prof, Met
from ..shared_utilities.utils import dot
from ..shared_utilities.types import HashableArrayWrapper


def generate_night_mask(sun_ang: SunAng) -> HashableArrayWrapper:
    mask_night = sun_ang.sin_beta <= 0.0
    return HashableArrayWrapper(mask_night)


def generate_turbulence_mask(para: Para, met: Met, prof: Prof) -> HashableArrayWrapper:
    nnu_T_P = dot(
        para.nnu * (101.3 / met.P_kPa),
        # jnp.power(prof.Tair_K[:, : para.jtot] / 273.16, 1.81),
        jnp.power(prof.Tair_K[:, : para.zht1.size] / 273.16, 1.81),
    )
    # Re = para.lleaf * prof.wind[:, : para.jtot] / nnu_T_P
    Re = para.lleaf * prof.wind[:, : para.zht1.size] / nnu_T_P
    mask_turbulence = Re > 14000.0
    return HashableArrayWrapper(mask_turbulence)
