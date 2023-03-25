"""
This is a class for time.

Author: Peishi Jiang
Date: 2023. 3. 7.
"""
# TODO: Implement and check time units and it has to be day for convenience!
# TODO: Clean up the time assignment!

from datetime import timedelta, datetime

import jax.numpy as jnp
import numpy as np
from typing import Optional, List


class Time:

    def __init__(
        self, 
        t0: float, 
        tn: float, 
        dt: Optional[float]=None,
        t_list: Optional[List[float]]=None,
        time_unit: str='day',
        start_time: str='1980-01-01',
    ) -> None:
        """A base class for time steps

        Args:
            t0 (float): the starting time.
            tn (float): the end time.
            nt (Optional[int], optional): Number of time steps. Defaults to None.
            t_list (Optional[List[float]], optional): List of time steps. Defaults to None.
            time_unit (str, optional): time units. Defaults to 'day'.
            start_time (str, oprtional): The starting time equaling to t0, following the format %Y-%m-%d. Defaults to '1980-01-01'.
        """
        self._t0, self._tn = t0, tn
        self._time_unit    = time_unit.lower()
        self._start_time   = datetime.strptime(start_time, '%Y-%m-%d')

        if dt is None:
            t_list = [t0] + t_list + [tn]
            t_list = list(set(t_list)); t_list.sort()
            t_list = jnp.array(t_list)
            self._t_list = t_list
        elif t_list is None:
            # self._t_list = list(range(t0, tn, dt)) + [tn]
            t_list = list(np.arange(t0, tn, dt)) + [tn]
            t_list = jnp.array(t_list)
            self._t_list = t_list
        else:
            raise Exception('dt and t_list can not be both not None!')

        self._nt = len(self._t_list)

    @property
    def t0(self):
        return self._t0

    @property
    def t1(self):
        return self._t1

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
    
    def return_formatted_time(self, t: float) -> datetime:
        """Given the relative time step, return the exact time.

        Args:
            t (float): The relative time step.
        """
        # if self._time_unit in ['day', 'd']:
        #     dt = timedelta(days=t)
        # elif self._time_unit in ['sec', 's']:
        #     dt = timedelta(seconds=t)
        # elif self._time_unit in ['hour', 'hr']:
        #     dt = timedelta(hours=t)
        # else:
        #     raise Exception('Unknown time unit: {}'.format(self._time_unit))

        dt = timedelta(days=t)
        return self._start_time + dt
