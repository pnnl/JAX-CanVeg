"""
This is a class for time.

Author: Peishi Jiang
Date: 2023. 3. 7.
"""
# TODO: Implement and check time units and it has to be day for convenience!
# TODO: Clean up the time assignment!

from datetime import timedelta, datetime

import jax.numpy as jnp

# import numpy as np
from typing import Optional

# from typing import Optional, List

from ..types import Float_0D, Float_1D


class Time:
    def __init__(
        self,
        t0: Float_0D,
        tn: Float_0D,
        dt: Optional[Float_0D] = None,
        # t_list: Optional[List[float]] = None,
        t_list: Optional[Float_1D] = None,
        time_unit: str = "day",
        start_time: str = "1980-01-01 00:00:00",
    ) -> None:
        """A base class for time steps

        Args:
            t0 (Float_0D): the starting time.
            tn (Float_0D): the end time.
            nt (Optional[int], optional): Number of time steps. Defaults to None.
            t_list (Optional[List[Float_0D]], optional): List of time steps. Defaults to None.
            time_unit (str, optional): time units. Defaults to 'day'.
            start_time (str, oprtional): The starting time equaling to t0, following the
                                         format '%Y-%m-%d %H:%M:%S'. Defaults to '1980-01-01 00:00:00'.
        """  # noqa: E501
        self._t0, self._tn = t0, tn
        self._time_unit = time_unit.lower()
        self._start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

        if dt is None and t_list is not None:
            # t_list = [t0] + t_list + [tn]
            # t_list = list(set(t_list))
            # t_list.sort()
            # t_list = jnp.array(t_list)
            t_list = jnp.concatenate([jnp.array([t0]), t_list, jnp.array([tn])]).sort()
            self._t_list = t_list
        elif dt is not None and t_list is None:
            # self._t_list = list(range(t0, tn, dt)) + [tn]
            # t_list = list(np.arange(t0, tn, dt)) + [tn]
            # t_list = jnp.array(t_list)
            # t_list = list(np.arange(t0, tn, dt)) + [tn]
            # t_list = jnp.array(t_list)
            t_list = jnp.concatenate([jnp.arange(t0, tn, dt), jnp.array([tn])])
            self._t_list = t_list
        else:
            raise Exception("dt and t_list can not be both not None or both None!")

        self._nt = self._t_list.size

    @property
    def t0(self):
        return self._t0

    @property
    def tn(self):
        return self._tn

    @property
    def nt(self):
        return self._nt

    @property
    def time_unit(self):
        return self._time_unit

    @property
    def start_time(self):
        return self._start_time

    @property
    def t_list(self):
        return self._t_list

    def return_formatted_time(self, t: Float_0D) -> datetime:
        """Given the relative time step, return the exact time.

        Args:
            t (Float_0D): The relative time step.
        """
        # if self._time_unit in ['day', 'd']:
        #     dt = timedelta(days=t)
        # elif self._time_unit in ['sec', 's']:
        #     dt = timedelta(seconds=t)
        # elif self._time_unit in ['hour', 'hr']:
        #     dt = timedelta(hours=t)
        # else:
        #     raise Exception('Unknown time unit: {}'.format(self._time_unit))

        dt = timedelta(days=float(t))
        return self._start_time + dt
