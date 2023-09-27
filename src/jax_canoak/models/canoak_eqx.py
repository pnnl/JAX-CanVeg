"""
An equinox module class for canoak.

Author: Peishi Jiang
Date: 2023.08.22.
"""

import jax
import equinox as eqx
from equinox.nn import MLP

from typing import Tuple, Optional

# from math import ceil

from .canoak import canoak
from .canoak_rsoil_hybrid import canoak_rsoil_hybrid

# from ..subjects import Para, Met, Prof, SunAng, LeafAng, SunShadedCan
# from ..subjects import Setup, Veg, Soil, Rnet, Qin, Ir, ParNir, Lai, Can
from ..subjects import Para, Met, Prof, SunAng
from ..subjects import LeafAng, SunShadedCan, Can
from ..subjects import Setup, Veg, Soil, Rnet
from ..subjects import Qin, Ir, ParNir, Lai
from ..shared_utilities.types import Float_2D, Float_0D


class CanoakBase(eqx.Module):
    para: Para
    dij: Float_2D
    # Setup
    lat_deg: Float_0D
    long_deg: Float_0D
    time_zone: int
    leafangle: int
    stomata: int
    n_can_layers: int
    n_total_layers: int
    n_soil_layers: int
    # ntime: int
    dt_soil: int
    soil_mtime: int
    # batch_size: int
    # n_batch: int
    niter: int

    def __init__(self, para: Para, setup: Setup, dij: Float_2D):
        self.para = para
        self.dij = dij
        # Location parameters
        self.lat_deg = setup.lat_deg
        self.long_deg = setup.long_deg
        self.time_zone = setup.time_zone
        # Static parameters
        self.leafangle = setup.leafangle
        self.stomata = setup.stomata
        self.n_can_layers = setup.n_can_layers
        self.n_total_layers = setup.n_total_layers
        self.n_soil_layers = setup.n_soil_layers
        self.dt_soil = setup.dt_soil
        self.soil_mtime = setup.soil_mtime
        # self.ntime = setup.time_batch_size
        # self.ntime = setup.ntime
        # self.batch_size = setup.time_batch_size
        # self.n_batch = ceil(setup.ntime / setup.time_batch_size)
        # self.ntime = self.batch_size * self.n_batch
        self.niter = setup.niter

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
        para, dij = self.para, self.dij
        # Location parameters
        lat_deg = self.lat_deg
        long_deg = self.long_deg
        time_zone = self.time_zone
        # Static parameters
        leafangle = self.leafangle
        stomata = self.stomata
        n_can_layers = self.n_can_layers
        n_total_layers = self.n_total_layers
        n_soil_layers = self.n_soil_layers
        # ntime = self.ntime
        dt_soil = self.dt_soil
        soil_mtime = self.soil_mtime
        niter = self.niter

        # Number of time steps from met
        # batch_size = self.batch_size
        # n_batch = self.n_batch
        ntime = met.zL.size

        results = canoak(
            para,
            met,
            dij,
            lat_deg,
            long_deg,
            time_zone,
            leafangle,
            stomata,
            n_can_layers,
            n_total_layers,
            n_soil_layers,
            ntime,
            dt_soil,
            soil_mtime,
            niter,
        )
        return results

    def get_can_rnet(self, met: Met):
        results = self(met)
        return results[-1].rnet

    def get_can_le(self, met: Met):
        results = self(met)
        return results[-1].LE

    def get_soil_resp(self, met: Met):
        results = self(met)
        soil = results[-3]
        return soil.resp


class CanoakRsoilHybrid(CanoakBase):
    RsoilDL: eqx.Module

    def __init__(
        self,
        para: Para,
        setup: Setup,
        dij: Float_2D,
        RsoilDL: Optional[eqx.Module] = None,
    ):
        super(CanoakRsoilHybrid, self).__init__(para, setup, dij)
        if RsoilDL is None:
            RsoilDL = MLP(
                in_size=2, out_size=1, width_size=6, depth=2, key=jax.random.PRNGKey(0)
            )
        self.RsoilDL = RsoilDL

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
        para, dij = self.para, self.dij
        RsoilDL = self.RsoilDL
        # Location parameters
        lat_deg = self.lat_deg
        long_deg = self.long_deg
        time_zone = self.time_zone
        # Static parameters
        leafangle = self.leafangle
        stomata = self.stomata
        n_can_layers = self.n_can_layers
        n_total_layers = self.n_total_layers
        n_soil_layers = self.n_soil_layers
        # ntime = self.ntime
        dt_soil = self.dt_soil
        soil_mtime = self.soil_mtime
        niter = self.niter

        # Number of time steps from met
        # batch_size = self.batch_size
        # n_batch = self.n_batch
        ntime = met.zL.size

        results = canoak_rsoil_hybrid(
            para,
            met,
            dij,
            RsoilDL,
            lat_deg,
            long_deg,
            time_zone,
            leafangle,
            stomata,
            n_can_layers,
            n_total_layers,
            n_soil_layers,
            ntime,
            dt_soil,
            soil_mtime,
            niter,
        )
        return results
