"""
This class contains the following information of the canopy domain :
- the physical properties;
- the dynamical states (e.g., canopy temperature);
- the spatial discretization;
- the temporal domain where the model is run.

Author: Peishi Jiang
Date: 2023. 3. 8.
"""

from typing import Optional

from ..shared_utilities.types import Float_general
from ..shared_utilities.domain import Time
from ..shared_utilities.domain import BaseSpace

from .utils import parameter_initialize, state_initialize
from .base import BaseSubject

parameters_default = {
}
states_default = {
    "temp": 20.,
    "sh":    0.,
    "rn":    0.,
    "le":    0.,
}

class Canopy(BaseSubject):

    def __init__(
        self, 
        ts       : Time, 
        space    : BaseSpace,
        temp     : Optional[Float_general]=None,
        rn       : Optional[Float_general]=None,
        sh       : Optional[Float_general]=None,
        le       : Optional[Float_general]=None,
    ) -> None:
        super().__init__(ts, space)
        # Parameters
        self.parameters = {
        }
    
        # States
        self.states = {
            "temp": temp,
            "rn"  : rn,
            "sh"  : sh,
            "le"  : le,
        }

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