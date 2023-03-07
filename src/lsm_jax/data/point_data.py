"""This implements the class of zero-dimensional data."""

from functools import partial

import jax
import jax.numpy as jnp

from .base import Data

from typing import List
from ..types import Float_0D, Float_1D, Float_2D, Float_3D

class PointData(Data):

    def __init__(self, varn_list: List[str], data: Float_2D, ts: Float_1D) -> None:
        super().__init__(varn_list, data)
        self.ts = ts
        self.nt = ts.size
        assert self.shape[1] == self.nt

    # @partial(jax.jit, static_argnums=[0])
    def interpolate_time(self, t: Float_0D) -> Float_1D:
        def interpolate(x):
            return jnp.interp(t, self.ts, x)
        x_interp = jax.vmap(interpolate, in_axes=(0,))(self.data)
        return x_interp

    def interpolate_time_by_varn(self, varn: str, ts_interp: Float_1D) -> Float_1D:
        idx = self.varn_list.index(varn)
        def interpolate(t):
            return jnp.interp(t, self.ts, self.data[idx,:])
        x_interp = jax.vmap(interpolate, in_axes=(0,))(ts_interp)
        return x_interp

    # @partial(jax.jit, static_argnums=[0])
    def interpolate_time_normalize(self, t: Float_0D) -> Float_1D:
        x_interp = self.interpolate_time(t)
        x_interp_normalize = self.normalize(x_interp)
        return x_interp_normalize
    
    # @partial(jax.jit, static_argnums=[0])
    def normalize(self, data: Float_2D) -> Float_2D:
        return super().normalize(data)

    # @partial(jax.jit, static_argnums=[0])
    def inverse_normalize(self, data: Float_2D) -> Float_2D:
        return super().inverse_normalize(data)


# TODO: The multiple point data class is useful to define river network catchment
class MultiplePointData(Data):

    def __init__(self, varn_list: List[str], data: Float_3D, ts: Float_1D, locs: Float_2D) -> None:
        super().__init__(varn_list, data)
        self.ts = ts
        self.locs = locs
        self.nt, self.nloc = ts.size, locs.shape[0]
        assert self.shape[1] == self.nt
        assert self.shape[2] == self.nloc
        assert locs.shape[0] == 3  # make sure we have x/y/z information