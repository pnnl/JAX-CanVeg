"""
Multiple classes for spatial discretization.

Author: Peishi Jiang
Date: 2023. 3. 7.
"""
# TODO: 
# (1) Currently, we only focus on the gridded discretization.
#     The unstructured mesh will be the future work.
# (2) Add a unit test

import jax.numpy as jnp

from typing import Tuple
from ..types import Float_1D, Float_2D, Float_3D

class BaseSpace:

    def __init__(self, ndim:int) -> None:
        """The base class for the spatial discretization.

        Args:
            ndim (int): The number of spatial dimension.
        """
        self._ndim = ndim
    
    @property
    def ndim(self) -> int:
        return self._ndim
    
    @property
    def shape(self) -> Tuple:
        raise Exception("Not implemented")

    @property
    def xs(self) -> Float_1D:
        raise Exception("Not implemented")

    @property
    def ys(self) -> Float_1D:
        raise Exception("Not implemented")

    @property
    def zs(self) -> Float_1D:
        raise Exception("Not implemented")

    @property
    def mesh(self) -> Float_1D:
        raise Exception("Not implemented")

class Column(BaseSpace):

    def __init__(self, xs: Float_1D) -> None:
        """A class for 1D or column-based domain.

        Args:
            xs (Float_1D): List of spatially discretized cells (i.e., the center of each cell).
        """
        super().__init__(ndim=1)
        self._xs = xs.sort()
    
    @property
    def xs(self) -> Float_1D:
        return self._xs

    @property
    def shape(self) -> Tuple:
        return (self._xs.size,)
    
    @property
    def mesh(self) -> Float_1D:
        return self._xs

class TwoDimSpace(BaseSpace):

    def __init__(self, xs: Float_1D, ys: Float_1D) -> None:
        """A class for two dimensional space using rectangular grids.

        Args:
            xs (Float_1D): The list of x-direction discretization.
            ys (Float_1D): The list of y-direction discretization.
        """
        super().__init__(ndim=2)
        self._xs, self._ys = xs.sort(), ys.sort()
    
    @property
    def xs(self) -> Float_1D:
        return self._xs
    
    @property
    def ys(self) -> Float_1D:
        return self._ys

    @property
    def shape(self) -> Tuple:
        return (self._xs.size, self._ys.size)

    @property
    def mesh(self) -> Float_2D:
        return jnp.meshgrid(self._xs, self._ys, indexing='xy')


class ThreeDimSpace(BaseSpace):

    def __init__(self, xs: Float_1D, ys: Float_1D, zs: Float_1D) -> None:
        """A class for three dimensional space using rectangular grids.

        Args:
            xs (Float_1D): The list of x-direction discretization. 
            ys (Float_1D): The list of y-direction discretization. 
            zs (Float_1D): The list of z-direction discretization. 
        """
        super().__init__(ndim=3)
        self._xs, self._ys, self._zs = xs.sort(), ys.sort(), zs.sort()

    @property
    def xs(self) -> Float_1D:
        return self._xs
    
    @property
    def ys(self) -> Float_1D:
        return self._ys

    @property
    def zs(self) -> Float_1D:
        return self._zs

    @property
    def shape(self) -> Tuple:
        return (self._xs.size, self._ys.size, self._zs.size)

    @property
    def mesh(self) -> Float_3D:
        return jnp.meshgrid(self._xs, self._ys, self._zs, indexing='xy')