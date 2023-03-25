"""
This class contains the following information of the soil domain:
- the physical properties (e.g., conductivity);
- the dynamical states (e.g., soil moisture);
- the spatial discretization;
- the temporal domain where the model is run.

Author: Peishi Jiang
Date: 2023. 3. 8.
"""
# TODO:
# (1) Add a unit test
# (2) Change the units to W m-2 and degK

from typing import Optional

from ..shared_utilities.types import Float_general
from ..shared_utilities.domain import Time
from ..shared_utilities.domain import BaseSpace

from .utils import parameter_initialize, state_initialize
from .base import BaseSubject

parameters_default = {
    "theta_sat": 0.4,
    "theta_r"  : 0.0001,
    "theta_wp" : 0.01,
    "theta_lim": 0.2,
    "ksat"     : 3e-7*86400,
    "alpha"    : 1.32,
    "n"        : 2.,
    "cs"       : 2e-3,
    "kthermal" : 2.*86400*1e-6,
    "rho"      : 1300.
}
states_default = {
    "temp": 20.,
    "vwc" : 0.07
}


class Soil(BaseSubject):

    def __init__(
        self, 
        ts       : Time, 
        space    : BaseSpace,
        theta_sat: Optional[Float_general]=None,
        theta_r  : Optional[Float_general]=None,
        theta_wp : Optional[Float_general]=None,
        theta_lim: Optional[Float_general]=None,
        ksat     : Optional[Float_general]=None,
        alpha    : Optional[Float_general]=None,
        n        : Optional[Float_general]=None,
        cs       : Optional[Float_general]=None,
        kthermal : Optional[Float_general]=None,
        rho      : Optional[Float_general]=None,
        temp     : Optional[Float_general]=None,
        vwc      : Optional[Float_general]=None,
    ) -> None:
        """The soil subject class.

        Args:
            ts (Time): See BaseSubject.
            space (BaseSpace): See BaseSubject.
            theta_sat (Float_general): Saturated volumetric soil water content [-]
            theta_r (Float_general): Residual water content [-]
            theta_wp (Float_general): Wilting point [-]
            theta_lim (Float_general): Limiting soil moisture for vegetation [-]
            ksat (Float_general): Saturated hydraulic conductivity [m d-1]
            alpha (Float_general): Van Genuchten parameter alpha
            n (Float_general): Van Genuchten parameter n
            cs (Float_general): Soil heat capacity [MJ kg-1 degC-1 or MJ kg-1 degK-1]
            kthermal (Float_general): Soil thermal conductivity [MJ m-1 d-1 degC-1 or MJ m-1 d-1 degK-1]
            rho (Float_general): Soil density [kg m-3]
            temp (Float_general): Soil temperature [degK]
            vwc (Float_general): Volumetric soil water content [-]
        """
        super().__init__(ts, space)
        # Parameters
        self.parameters = {
            "theta_sat": theta_sat,
            "theta_r"  : theta_r,
            "theta_wp" : theta_wp,
            "theta_lim": theta_lim,
            "ksat"     : ksat,
            "alpha"    : alpha,
            "n"        : n,
            "cs"       : cs,
            "kthermal" : kthermal,
            "rho"      : rho
        }
        # self.theta_sat = theta_sat
        # self.theta_r   = theta_r
        # self.theta_wp  = theta_wp
        # self.theta_lim = theta_lim
        # self.ksat      = ksat
        # self.alpha     = alpha
        # self.n         = n
        # self.cs        = cs
        # self.kthermal  = kthermal
        # self.rho       = rho
    
        # States
        self.states = {
            "temp": temp,
            "vwc" : vwc
        }
        # self.temp = temp
        # self.vwc  = vwc

        self._parameter_initialize()
        self._state_initialize()

    def _parameter_initialize(self) -> None:
        for paraname, para in self.parameters.items():
            self.parameters[paraname] = parameter_initialize(
                para, self.spatial_domain, paraname, parameters_default[paraname] 
            )

    def _state_initialize(self) -> None:
        for statename, state in self.states.items():
            self.states[statename] = state_initialize(
                state, self.temporal_domain, self.spatial_domain, statename, states_default[statename] 
            )