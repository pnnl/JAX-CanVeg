"""
An equinox module class for canveg.

Author: Peishi Jiang
Date: 2023.08.22.
"""

# import jax
import equinox as eqx

# from equinox.nn import MLP

from typing import Tuple, Callable

# from math import ceil

from .canveg import canveg

from ..shared_utilities.solver import implicit_func_fixed_point

# from ..shared_utilities.solver import implicit_func_fixed_point_canveg_main

from .canveg import canveg_initialize_states, canveg_each_iteration
from .canveg import get_all, update_all

# from .canveg_rsoil_hybrid import canveg_rsoil_hybrid
# from .canveg_rsoil_hybrid import canveg_rsoil_hybrid_each_iteration
# from .canveg_leafrh_hybrid import canveg_leafrh_hybrid_each_iteration
# from .canveg_gs_hybrid import canveg_gs_hybrid_each_iteration
# from .canveg_gsswc_hybrid import canveg_gsswc_hybrid_each_iteration

# Soil respiration function
from ..physics.carbon_fluxes import soil_respiration_alfalfa
from ..physics.carbon_fluxes import soil_respiration_q10_power
from ..physics.carbon_fluxes import soil_respiration_dnn

# Leaf relative humidity function
from ..physics.carbon_fluxes import calculate_leaf_rh_physics
from ..physics.carbon_fluxes import calculate_leaf_rh_nn

from ..subjects import Para, Met, Prof, SunAng
from ..subjects import LeafAng, SunShadedCan, Can
from ..subjects import Setup, Veg, Soil, Rnet
from ..subjects import Qin, Ir, ParNir, Lai
from ..shared_utilities.types import Float_2D, Float_0D


########################################################################
# The base class for CanVeg model
########################################################################
class CanvegBase(eqx.Module):
    para: Para
    dij: Float_2D
    # Setup
    lat_deg: Float_0D
    long_deg: Float_0D
    time_zone: int
    leafangle: int
    stomata: int
    leafrh: int
    soilresp: int
    n_can_layers: int
    n_total_layers: int
    n_soil_layers: int
    dt_soil: Float_0D
    soil_mtime: int
    niter: int

    leafrh_func: Callable
    soilresp_func: Callable

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
        self.leafrh = setup.leafrh
        self.soilresp = setup.soilresp
        self.n_can_layers = setup.n_can_layers
        self.n_total_layers = setup.n_total_layers
        self.n_soil_layers = setup.n_soil_layers
        self.dt_soil = setup.dt_soil
        self.soil_mtime = setup.soil_mtime
        self.niter = setup.niter

        # Get leaf relative humidity calculation function
        self.leafrh_func = self._get_leafrh_func()

        # Get soil respiration calculation function
        self.soilresp_func = self._get_soilresp_func()

    def __call__(self, met: Met, *args):
        raise NotImplementedError("This is the base model; not working!")

    def _get_leafrh_func(self):
        if self.leafrh == 0:
            return calculate_leaf_rh_physics
        elif self.leafrh == 1:
            return calculate_leaf_rh_nn
        else:
            raise Exception(f"Unknown leaf relative humidity module type {self.leafrh}")

    def _get_soilresp_func(self):
        if self.soilresp == 0:
            return soil_respiration_alfalfa
        elif self.soilresp == 1:
            return soil_respiration_q10_power
        elif self.soilresp == 2:
            return soil_respiration_dnn
        else:
            raise Exception(f"Unknown soil respiration module type {self.soilresp}")


########################################################################
# The classes for performing Canveg without IFT
########################################################################
class Canveg(CanvegBase):
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
        # Functions
        leafrh_func = self.leafrh_func
        soilresp_func = self.soilresp_func

        # Number of time steps from met
        ntime = met.zL.size

        results = canveg(
            para,
            met,
            dij,
            lat_deg,
            long_deg,
            time_zone,
            leafangle=leafangle,
            stomata=stomata,
            leafrh_func=leafrh_func,
            soilresp_func=soilresp_func,
            n_can_layers=n_can_layers,
            n_total_layers=n_total_layers,
            n_soil_layers=n_soil_layers,
            time_batch_size=ntime,
            dt_soil=dt_soil,
            soil_mtime=soil_mtime,
            niter=niter,
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


########################################################################
# The classes for performing canveg with IFT
########################################################################
class CanvegIFT(CanvegBase):

    # @eqx.filter_jit
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
        # Functions
        leafrh_func = self.leafrh_func
        soilresp_func = self.soilresp_func

        # Number of time steps from met
        ntime = met.zL.size

        # Initialization
        rnet, lai, sun_ang, leaf_ang, initials = canveg_initialize_states(
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
        args = [
            dij,
            sun_ang,
            leaf_ang,
            lai,
            n_can_layers,
            stomata,
            soil_mtime,
            leafrh_func,
            soilresp_func,
        ]
        states_final = implicit_func_fixed_point(
            states_guess,
            para,
            args,
            iter_func=canveg_each_iteration,
            update_substates_func=update_substates_func,
            get_substates_func=get_substates_func,
            niter=niter,
        )

        return states_final, [rnet, sun_ang, leaf_ang, lai]
        # return states_final, [quantum, nir, rnet, sun_ang, leaf_ang, lai]

    def get_fixed_point_states(
        self,
        met: Met,
        update_substates_func: Callable = update_all,
        get_substates_func: Callable = get_all,
        scaler: Callable = lambda x: x,
    ):
        results = self(met, update_substates_func, get_substates_func)
        # Not including rnet, sun_ang, leaf_ang, and lai
        result = results[0]

        # Scale the result if needed
        scaled_result = scaler(result)
        return scaled_result


# class CanvegRsoilHybridIFT(CanvegIFT):
#     def __call__(
#         self,
#         met: Met,
#         update_substates_func: Callable = update_all,
#         get_substates_func: Callable = get_all,
#     ):
#         para, dij = self.para, self.dij
#         # Location parameters
#         lat_deg = self.lat_deg
#         long_deg = self.long_deg
#         time_zone = self.time_zone
#         # Static parameters
#         leafangle = self.leafangle
#         stomata = self.stomata
#         n_can_layers = self.n_can_layers
#         n_total_layers = self.n_total_layers
#         n_soil_layers = self.n_soil_layers
#         dt_soil = self.dt_soil
#         soil_mtime = self.soil_mtime
#         niter = self.niter

#         # Number of time steps from met
#         ntime = met.zL.size

#         # Initialization
#         quantum,nir,rnet,lai, sun_ang, leaf_ang, initials = canveg_initialize_states(
#             para,
#             met,
#             lat_deg,
#             long_deg,
#             time_zone,
#             leafangle,
#             n_can_layers,
#             n_total_layers,
#             n_soil_layers,
#             ntime,
#             dt_soil,
#             soil_mtime,
#         )
#         states_guess = initials

#         # Forward runs
#         args = [dij, leaf_ang, quantum, nir, lai, n_can_layers, stomata, soil_mtime]
#         # states_final = implicit_func_fixed_point(
#         #     canveg_rsoil_hybrid_each_iteration,
#         #     update_substates_func,
#         #     get_substates_func,
#         #     states_guess,
#         #     para,
#         #     niter,
#         #     *args
#         # )
#         states_final = implicit_func_fixed_point(
#             states_guess,
#             para,
#             args,
#             iter_func=canveg_rsoil_hybrid_each_iteration,
#             update_substates_func=update_substates_func,
#             get_substates_func=get_substates_func,
#             niter=niter,
#         )

#         return states_final, [quantum, nir, rnet, sun_ang, leaf_ang, lai]


# class CanvegLeafRHHybridIFT(CanvegIFT):
#     def __call__(
#         self,
#         met: Met,
#         update_substates_func: Callable = update_all,
#         get_substates_func: Callable = get_all,
#     ):
#         para, dij = self.para, self.dij
#         # Location parameters
#         lat_deg = self.lat_deg
#         long_deg = self.long_deg
#         time_zone = self.time_zone
#         # Static parameters
#         leafangle = self.leafangle
#         stomata = self.stomata
#         n_can_layers = self.n_can_layers
#         n_total_layers = self.n_total_layers
#         n_soil_layers = self.n_soil_layers
#         dt_soil = self.dt_soil
#         soil_mtime = self.soil_mtime
#         niter = self.niter

#         # Number of time steps from met
#         ntime = met.zL.size

#         # Initialization
#         quantum,nir,rnet,lai, sun_ang, leaf_ang, initials = canveg_initialize_states(
#             para,
#             met,
#             lat_deg,
#             long_deg,
#             time_zone,
#             leafangle,
#             n_can_layers,
#             n_total_layers,
#             n_soil_layers,
#             ntime,
#             dt_soil,
#             soil_mtime,
#         )
#         states_guess = initials

#         # Forward runs
#         args = [dij, leaf_ang, quantum, nir, lai, n_can_layers, stomata, soil_mtime]
#         # states_final = implicit_func_fixed_point(
#         #     canveg_leafrh_hybrid_each_iteration,
#         #     update_substates_func,
#         #     get_substates_func,
#         #     states_guess,
#         #     para,
#         #     niter,
#         #     *args
#         # )
#         states_final = implicit_func_fixed_point(
#             states_guess,
#             para,
#             args,
#             iter_func=canveg_leafrh_hybrid_each_iteration,
#             update_substates_func=update_substates_func,
#             get_substates_func=get_substates_func,
#             niter=niter,
#         )

#         return states_final, [quantum, nir, rnet, sun_ang, leaf_ang, lai]


# class CanvegGSHybridIFT(CanvegIFT):
#     def __call__(
#         self,
#         met: Met,
#         update_substates_func: Callable = update_all,
#         get_substates_func: Callable = get_all,
#     ):
#         para, dij = self.para, self.dij
#         # Location parameters
#         lat_deg = self.lat_deg
#         long_deg = self.long_deg
#         time_zone = self.time_zone
#         # Static parameters
#         leafangle = self.leafangle
#         stomata = self.stomata
#         n_can_layers = self.n_can_layers
#         n_total_layers = self.n_total_layers
#         n_soil_layers = self.n_soil_layers
#         dt_soil = self.dt_soil
#         soil_mtime = self.soil_mtime
#         niter = self.niter

#         # Number of time steps from met
#         ntime = met.zL.size

#         # Initialization
#         quantum,nir,rnet,lai, sun_ang, leaf_ang, initials = canveg_initialize_states(
#             para,
#             met,
#             lat_deg,
#             long_deg,
#             time_zone,
#             leafangle,
#             n_can_layers,
#             n_total_layers,
#             n_soil_layers,
#             ntime,
#             dt_soil,
#             soil_mtime,
#         )
#         states_guess = initials

#         # Forward runs
#         args = [dij, leaf_ang, quantum, nir, lai, n_can_layers, stomata, soil_mtime]
#         # states_final = implicit_func_fixed_point(
#         #     canveg_gs_hybrid_each_iteration,
#         #     update_substates_func,
#         #     get_substates_func,
#         #     states_guess,
#         #     para,
#         #     niter,
#         #     *args
#         # )
#         states_final = implicit_func_fixed_point(
#             states_guess,
#             para,
#             args,
#             iter_func=canveg_gs_hybrid_each_iteration,
#             update_substates_func=update_substates_func,
#             get_substates_func=get_substates_func,
#             niter=niter,
#         )

#         return states_final, [quantum, nir, rnet, sun_ang, leaf_ang, lai]


# class CanvegGSSWCHybridIFT(CanvegIFT):
#     def __call__(
#         self,
#         met: Met,
#         update_substates_func: Callable = update_all,
#         get_substates_func: Callable = get_all,
#     ):
#         para, dij = self.para, self.dij
#         # Location parameters
#         lat_deg = self.lat_deg
#         long_deg = self.long_deg
#         time_zone = self.time_zone
#         # Static parameters
#         leafangle = self.leafangle
#         stomata = self.stomata
#         n_can_layers = self.n_can_layers
#         n_total_layers = self.n_total_layers
#         n_soil_layers = self.n_soil_layers
#         dt_soil = self.dt_soil
#         soil_mtime = self.soil_mtime
#         niter = self.niter

#         # Number of time steps from met
#         ntime = met.zL.size

#         # Initialization
#         quantum,nir,rnet,lai, sun_ang, leaf_ang, initials = canveg_initialize_states(
#             para,
#             met,
#             lat_deg,
#             long_deg,
#             time_zone,
#             leafangle,
#             n_can_layers,
#             n_total_layers,
#             n_soil_layers,
#             ntime,
#             dt_soil,
#             soil_mtime,
#         )
#         states_guess = initials

#         # Forward runs
#         args = [dij, leaf_ang, quantum, nir, lai, n_can_layers, stomata, soil_mtime]
#         states_final = implicit_func_fixed_point(
#             canveg_gsswc_hybrid_each_iteration,
#             update_substates_func,
#             get_substates_func,
#             states_guess,
#             para,
#             niter,
#             *args
#         )

#         return states_final, [quantum, nir, rnet, sun_ang, leaf_ang, lai]
