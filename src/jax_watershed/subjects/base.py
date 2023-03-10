"""
This is the base subject class for a subject (i.e., soil, surface, and canopy). 
The base subject class should include the following three types of information:
- subject parameters (not changing with time)
- subject states (change with time)
- domain discretization

Author: Peishi Jiang
Date: 2023. 3. 7.
"""

from typing import List, AnyStr

from ..shared_utilities.domain import Time
from ..shared_utilities.domain import BaseSpace

class BaseSubject:

    def __init__(self, ts: Time, space: BaseSpace) -> None:
        """The base class for a subject

        Args:
            ts (Time): The temporal domain.
            space (BaseSpace): The spatial domain.
        """
        self._time  = ts
        self._space = space

        # The followings should be provided by each subject
        self.parameters, self.states = {}, {}

    @property
    def spatial_domain(self) -> BaseSpace:
        """The spatial domain of the subject."""
        return self._space

    @property
    def temporal_domain(self) -> Time:
        """The temporal domain of the subject."""
        return self._time

    @property
    def parameter_list(self) -> List[AnyStr]:
        # raise Exception("Not implemented")
        return list(self.parameters.keys())

    @property
    def state_list(self) -> List[AnyStr]:
        return list(self.states.keys())