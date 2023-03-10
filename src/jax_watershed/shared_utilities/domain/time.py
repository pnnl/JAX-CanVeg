"""
This is a class for time.

Author: Peishi Jiang
Date: 2023. 3. 7.
"""
# TODO: Implement and check time units

from typing import Optional, List


class Time:

    def __init__(
        self, 
        t0: float, 
        tn: float, 
        dt: Optional[float]=None,
        t_list: Optional[List[float]]=None,
        time_unit: str='day',
    ) -> None:
        """A base class for time steps

        Args:
            t0 (float): the starting time.
            tn (float): the end time.
            nt (Optional[int], optional): Number of time steps. Defaults to None.
            t_list (Optional[List[float]], optional): List of time steps. Defaults to None.
            time_unit (str, optional): time units. Defaults to 'day'.
        """
        self._t0, self._tn = t0, tn
        self._time_unit    = time_unit

        if dt is None:
            t_list = [t0] + t_list + [tn]
            t_list = list(set(t_list)); t_list.sort()
            self._t_list = t_list
        elif t_list is None:
            self._t_list = list(range(t0, tn, dt)) + [tn]
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
    def t_list(self):
        return self._t_list