"""
Run the batched mode of canveg.

Author: Peishi Jiang
Date: 2023.09.11.
"""

import jax

from typing import Tuple

from .canveg_eqx import CanvegBase
from ..subjects import BatchedMet
from ..subjects import Met, Prof, SunAng
from ..subjects import LeafAng, SunShadedCan, Can
from ..subjects import Veg, Soil, Rnet
from ..subjects import Qin, Ir, ParNir, Lai
from ..subjects import convert_batchedstates_to_states
from ..subjects import convert_batchedmet_to_met


def run_canveg_in_batch_any(batched_met: BatchedMet, canveg_eqx: CanvegBase):
    # Run batch simulation
    result = jax.vmap(canveg_eqx)(batched_met)  # pyright: ignore

    return result


def run_canveg_in_batch(
    batched_met: BatchedMet, canveg_eqx: CanvegBase
) -> Tuple[
    Met,
    Prof,
    Can,
    Veg,
    SunShadedCan,
    SunShadedCan,
    Qin,
    Rnet,
    SunAng,
    Ir,
    ParNir,
    ParNir,
    Lai,
    LeafAng,
    Soil,
]:
    # Run batch simulation
    (
        batched_met,
        batched_prof,
        batched_quantum,
        batched_nir,
        batched_ir,
        batched_rnet,
        batched_qin,
        batched_sun_ang,
        batched_leaf_ang,
        batched_lai,
        batched_sun,
        batched_shade,
        batched_soil,
        batched_veg,
        batched_can,
    ) = jax.vmap(canveg_eqx)(
        batched_met  # pyright: ignore
    )

    # Reshape the results
    (  # pyright: ignore
        prof,
        can,
        veg,
        shade,
        sun,
        qin,
        rnet,
        sun_ang,
        ir,
        nir,
        quantum,
        lai,
        leaf_ang,
        soil,
    ) = convert_batchedstates_to_states(
        batched_prof,
        batched_can,
        batched_veg,
        batched_shade,
        batched_sun,
        batched_qin,
        batched_rnet,
        batched_sun_ang,
        batched_ir,
        batched_nir,
        batched_quantum,
        batched_lai,
        batched_leaf_ang,
        batched_soil,
    )

    # Reshape batched_met to met
    met = convert_batchedmet_to_met(batched_met)

    return (  # pyright: ignore
        met,
        prof,
        can,
        veg,
        shade,
        sun,
        qin,
        rnet,
        sun_ang,
        ir,
        nir,
        quantum,
        lai,
        leaf_ang,
        soil,
    )
