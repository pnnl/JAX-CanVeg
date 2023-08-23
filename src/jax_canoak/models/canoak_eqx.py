"""
An equinox module class for canoak.

Author: Peishi Jiang
Date: 2023.08.22.
"""

import equinox as eqx

from typing import Tuple

from .canoak import canoak
from ..subjects import Para, Met, Prof, SunAng, LeafAng, SunShadedCan
from ..subjects import Setup, Veg, Soil, Rnet, Qin, Ir, ParNir, Lai, Can
from ..shared_utilities.types import Float_2D


class CanoakBase(eqx.Module):
    para: Para
    setup: Setup
    dij: Float_2D

    @eqx.filter_jit
    def __call__(
        self, met: Met
    ) -> Tuple[
        Met,
        Prof,
        ParNir,
        ParNir,
        Ir,
        Rnet,
        Qin,
        SunAng,
        LeafAng,
        Lai,
        SunShadedCan,
        SunShadedCan,
        Soil,
        Veg,
        Can,
    ]:
        para, setup = self.para, self.setup
        dij = self.dij
        soil_mtime, niter = setup.soil_mtime, setup.niter
        results = canoak(para, setup, met, dij, soil_mtime, niter)
        return results

    def get_can_rnet(self, met: Met):
        results = self(met)
        return results[-1].rnet
