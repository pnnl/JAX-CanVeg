"""
This class contains the following information of the soil surface :
- the physical properties (e.g., surface roughness length);
- the dynamical states (e.g., surface temperature);
- the spatial discretization;
- the temporal domain where the model is run.

Author: Peishi Jiang
Date: 2023. 3. 8.
"""

from typing import Optional

from ..shared_utilities.types import Float_0D, Float_general
from ..shared_utilities.domain import Time
from ..shared_utilities.domain import BaseSpace

from .utils import parameter_initialize, state_initialize
from .base import BaseSubject

# TODO: Add the documentation.

parameters_default = {}
states_default = {
    "l": -1.0,
    "u_a": 0.0,
    "q_a": 0.0,
    "ρ_a": 0.0,
    "T_a": 280.15,
    # "T_v" : 290.15,
    # "T_g" : 300.15,
    "T_v": 273.15,
    "T_g": 273.15,
    "S_v": 0.0,
    "S_g": 0.0,
    "L_v": 0.0,
    "L_g": 0.0,
    "H_v": 0.0,
    "H_g": 0.0,
    "E_v": 0.0,
    "E_g": 0.0,
    "G": 0.0,
}


class Surface(BaseSubject):
    def __init__(
        self,
        ts: Time,
        space: BaseSpace,
        l: Optional[Float_general] = None,  # noqa: E741
        u_a: Optional[Float_general] = None,
        q_a: Optional[Float_general] = None,
        ρ_a: Optional[Float_general] = None,
        T_a: Optional[Float_general] = None,
        T_v: Optional[Float_general] = None,
        T_g: Optional[Float_general] = None,
        S_v: Optional[Float_general] = None,
        S_g: Optional[Float_general] = None,
        L_v: Optional[Float_general] = None,
        L_g: Optional[Float_general] = None,
        H_v: Optional[Float_general] = None,
        H_g: Optional[Float_general] = None,
        E_v: Optional[Float_general] = None,
        E_g: Optional[Float_general] = None,
        G: Optional[Float_general] = None,
    ) -> None:
        super().__init__(ts, space)
        # Parameters
        self.parameters = {}

        # States
        self.states = {
            "l": l,
            "u_a": u_a,
            "q_a": q_a,
            "ρ_a": ρ_a,
            "T_a": T_a,
            "T_v": T_v,
            "T_g": T_g,
            "S_v": S_v,
            "S_g": S_g,
            "L_v": L_v,
            "L_g": L_g,
            "H_v": H_v,
            "H_g": H_g,
            "E_v": E_v,
            "E_g": E_g,
            "G": G,
        }

        self._parameter_initialize()
        self._state_initialize()

    def set_state_value(
        self,
        state_name: str,
        value: Float_0D,
        time_ind: int,
        space_ind: Optional[int] = None,
    ):
        if space_ind is not None:
            self.states[state_name] = (
                self.states[state_name]
                .at[(time_ind, space_ind)]  # pyright: ignore
                .set(value)
            )
        else:
            self.states[state_name] = (
                self.states[state_name].at[time_ind].set(value)  # pyright: ignore
            )

    def set_para_value(
        self, para_name: str, value: Float_0D, space_ind: Optional[int] = None
    ):
        if space_ind is not None:
            self.parameters[para_name] = (
                self.parameters[para_name].at[space_ind].set(value)
            )  # noqa: E501
        else:
            self.parameters[para_name] = self.parameters[para_name].at[:].set(value)

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
