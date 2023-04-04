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

from typing import Optional

from ..shared_utilities.types import Float_general
from ..shared_utilities.domain import Time
from ..shared_utilities.domain import BaseSpace

from .utils import parameter_initialize, state_initialize
from .base import BaseSubject

parameters_default = {
    "θ_sat": 0.4,
    "θ_r": 0.0001,
    "θ_wp": 0.01,
    "θ_lim": 0.2,
    # "ksat": 3e-7 * 86400,
    "k_sat": 3e-7,
    "alpha": 1.32,
    "n": 2.0,
    # "cv": 2e-3,
    # "κ": 2.0 * 86400 * 1e-6,
    "cv": 2.5e6,
    "κ": 2.0,
    "ρ": 1300.0,
}
states_default = {"Tsoil": 20.0, "θ": 0.07}


class Soil(BaseSubject):
    def __init__(
        self,
        ts: Time,
        space: BaseSpace,
        θ_sat: Optional[Float_general] = None,
        θ_r: Optional[Float_general] = None,
        θ_wp: Optional[Float_general] = None,
        θ_lim: Optional[Float_general] = None,
        k_sat: Optional[Float_general] = None,
        alpha: Optional[Float_general] = None,
        n: Optional[Float_general] = None,
        cv: Optional[Float_general] = None,
        κ: Optional[Float_general] = None,
        ρ: Optional[Float_general] = None,
        Tsoil: Optional[Float_general] = None,
        θ: Optional[Float_general] = None,
    ) -> None:
        """The soil subject class.

        Args:
            ts (Time): See BaseSubject.
            space (BaseSpace): See BaseSubject.
            θ_sat (Float_general): Saturated volumetric soil water content [-]
            θ_r (Float_general): Residual water content [-]
            θ_wp (Float_general): Wilting point [-]
            θ_lim (Float_general): Limiting soil moisture for vegetation [-]
            k_sat (Float_general): Saturated hydraulic conductivity [m s-1]
            alpha (Float_general): Van Genuchten parameter alpha
            n (Float_general): Van Genuchten parameter n
            cv (Float_general): Soil heat capacity [J m-3 K-1]
            κ (Float_general): Soil thermal conductivity [W m-1 K-1]
            ρ (Float_general): Soil density [kg m-3]
            Tsoil (Float_general): Soil temperature [degK]
            θ (Float_general): Volumetric soil water content [-]
        """  # noqa: E501
        super().__init__(ts, space)
        # Parameters
        self.parameters = {
            "θ_sat": θ_sat,
            "θ_r": θ_r,
            "θ_wp": θ_wp,
            "θ_lim": θ_lim,
            "k_sat": k_sat,
            "alpha": alpha,
            "n": n,
            "cv": cv,
            "κ": κ,
            "ρ": ρ,
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
        self.states = {"Tsoil": Tsoil, "θ": θ}
        # self.temp = temp
        # self.vwc  = vwc

        self._parameter_initialize()
        self._state_initialize()

    def _parameter_initialize(self) -> None:
        for paraname, para in self.parameters.items():
            self.parameters[paraname] = parameter_initialize(
                paraname, self.spatial_domain, para, parameters_default[paraname]
            )

    def _state_initialize(self) -> None:
        for statename, state in self.states.items():
            self.states[statename] = state_initialize(
                statename,
                self.temporal_domain,
                self.spatial_domain,
                state,
                states_default[statename],
            )
