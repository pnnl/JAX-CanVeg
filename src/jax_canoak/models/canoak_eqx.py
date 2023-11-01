"""
An equinox module class for canoak.

Author: Peishi Jiang
Date: 2023.08.22.
"""

# import jax
import equinox as eqx

# from equinox.nn import MLP

from typing import Tuple, Callable

# from math import ceil

from .canoak import canoak
from .canoak_rsoil_hybrid import canoak_rsoil_hybrid

from ..shared_utilities.solver import implicit_func_fixed_point
from .canoak import canoak_initialize_states, canoak_each_iteration
from .canoak import get_all, update_all
from .canoak_rsoil_hybrid import canoak_rsoil_hybrid_each_iteration
from .canoak_leafrh_hybrid import canoak_leafrh_hybrid_each_iteration

from ..subjects import Para, Met, Prof, SunAng
from ..subjects import LeafAng, SunShadedCan, Can
from ..subjects import Setup, Veg, Soil, Rnet
from ..subjects import Qin, Ir, ParNir, Lai
from ..shared_utilities.types import Float_2D, Float_0D


########################################################################
# The base class for Canoak model
########################################################################
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
    dt_soil: Float_0D
    soil_mtime: int
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
        self.niter = setup.niter

    def __call__(self, met: Met, *args):
        raise NotImplementedError("This is the base model; not working!")


########################################################################
# The classes for performing canoak without IFT
########################################################################
class Canoak(CanoakBase):
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


class CanoakRsoilHybrid(Canoak):
    # RsoilDL: eqx.Module

    # def __init__(
    #     self,
    #     para: Para,
    #     setup: Setup,
    #     dij: Float_2D,
    #     RsoilDL: Optional[eqx.Module] = None,
    # ):
    #     super(CanoakRsoilHybrid, self).__init__(para, setup, dij)
    #     if RsoilDL is None:
    #         RsoilDL = MLP(
    #             in_size=2, out_size=1, width_size=6, depth=2,key=jax.random.PRNGKey(0)
    #         )
    #     self.RsoilDL = RsoilDL

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

        results = canoak_rsoil_hybrid(
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


########################################################################
# The classes for performing canoak with IFT
########################################################################
class CanoakIFT(CanoakBase):

    # def __call__(
    #     self,
    #     initial_states: Tuple[Met,Prof,Ir,Qin,SunShadedCan,SunShadedCan,Soil,Veg,Can],
    #     drivers: Tuple[LeafAng, ParNir, ParNir, Lai],
    #     update_substates_func: Callable = update_all,
    #     get_substates_func: Callable = get_all,
    # ):
    #     para, dij = self.para, self.dij
    #     # Some configurations
    #     stomata = self.stomata
    #     n_can_layers = self.n_can_layers
    #     soil_mtime = self.soil_mtime
    #     niter = self.niter

    #     # Get the drivers
    #     leaf_ang, quantum = drivers[0], drivers[1]
    #     nir, lai = drivers[2], drivers[3]

    #     # Forward runs
    #     args = [dij, leaf_ang, quantum, nir, lai, n_can_layers, stomata, soil_mtime]
    #     states_final = implicit_func_fixed_point(
    #         canoak_each_iteration,
    #         update_substates_func,
    #         get_substates_func,
    #         initial_states,
    #         para,
    #         niter,
    #         *args
    #     )

    #     return states_final

    def __call__(
        self,
        met: Met,
        update_substates_func: Callable = update_all,
        get_substates_func: Callable = get_all,
    ):
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
        dt_soil = self.dt_soil
        soil_mtime = self.soil_mtime
        niter = self.niter

        # Number of time steps from met
        ntime = met.zL.size

        # Initialization
        quantum, nir, rnet, lai, sun_ang, leaf_ang, initials = canoak_initialize_states(
            para,
            met,
            lat_deg,
            long_deg,
            time_zone,
            leafangle,
            n_can_layers,
            n_total_layers,
            n_soil_layers,
            ntime,
            dt_soil,
            soil_mtime,
        )
        states_guess = initials

        # Forward runs
        args = [dij, leaf_ang, quantum, nir, lai, n_can_layers, stomata, soil_mtime]
        states_final = implicit_func_fixed_point(
            canoak_each_iteration,
            update_substates_func,
            get_substates_func,
            states_guess,
            para,
            niter,
            *args
        )

        return states_final, [quantum, nir, rnet, sun_ang, leaf_ang, lai]

    def get_fixed_point_states(
        self,
        met: Met,
        update_substates_func: Callable = update_all,
        get_substates_func: Callable = get_all,
    ):
        results = self(met, update_substates_func, get_substates_func)
        # Not including quantum, nir, rnet, sun_ang, leaf_ang, and lai
        return results[0]


class CanoakRsoilHybridIFT(CanoakIFT):
    def __call__(
        self,
        met: Met,
        update_substates_func: Callable = update_all,
        get_substates_func: Callable = get_all,
    ):
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
        dt_soil = self.dt_soil
        soil_mtime = self.soil_mtime
        niter = self.niter

        # Number of time steps from met
        ntime = met.zL.size

        # Initialization
        quantum, nir, rnet, lai, sun_ang, leaf_ang, initials = canoak_initialize_states(
            para,
            met,
            lat_deg,
            long_deg,
            time_zone,
            leafangle,
            n_can_layers,
            n_total_layers,
            n_soil_layers,
            ntime,
            dt_soil,
            soil_mtime,
        )
        states_guess = initials

        # Forward runs
        args = [dij, leaf_ang, quantum, nir, lai, n_can_layers, stomata, soil_mtime]
        states_final = implicit_func_fixed_point(
            canoak_rsoil_hybrid_each_iteration,
            update_substates_func,
            get_substates_func,
            states_guess,
            para,
            niter,
            *args
        )

        return states_final, [quantum, nir, rnet, sun_ang, leaf_ang, lai]


class CanoakLeafRHHybridIFT(CanoakIFT):
    def __call__(
        self,
        met: Met,
        update_substates_func: Callable = update_all,
        get_substates_func: Callable = get_all,
    ):
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
        dt_soil = self.dt_soil
        soil_mtime = self.soil_mtime
        niter = self.niter

        # Number of time steps from met
        ntime = met.zL.size

        # Initialization
        quantum, nir, rnet, lai, sun_ang, leaf_ang, initials = canoak_initialize_states(
            para,
            met,
            lat_deg,
            long_deg,
            time_zone,
            leafangle,
            n_can_layers,
            n_total_layers,
            n_soil_layers,
            ntime,
            dt_soil,
            soil_mtime,
        )
        states_guess = initials

        # Forward runs
        args = [dij, leaf_ang, quantum, nir, lai, n_can_layers, stomata, soil_mtime]
        states_final = implicit_func_fixed_point(
            canoak_leafrh_hybrid_each_iteration,
            update_substates_func,
            get_substates_func,
            states_guess,
            para,
            niter,
            *args
        )

        return states_final, [quantum, nir, rnet, sun_ang, leaf_ang, lai]
