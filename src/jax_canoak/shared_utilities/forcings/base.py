"""This implement a base class for data."""
# TODO: Convert the dictionary-based forcing class to an array-based forcing class

import jax
import jax.numpy as jnp

# from sklearn.preprocessing import MinMaxScaler

from diffrax import AbstractGlobalInterpolation


# from typing import Dict, List, Tuple
# from ..types import Float_0D, Float_1D, Array, Numeric_ND

from typing import Dict, List
from ..types import Float_0D, Float_1D, Numeric_ND


class Data(object):
    def __init__(self, varn_list: List[str], data: Numeric_ND) -> None:
        self.varn_list = varn_list
        self.data = data
        # self.shape = data.shape
        self.shape = jnp.shape(data)
        self.n_var = self.shape[0]
        # self.n_var, self.nt= self.shapes[0], self.shapes[1]
        assert len(varn_list) == self.n_var

        self._generate_minmax_scaler()

    # def shape(self) -> Tuple:
    #     return self.shape

    def _generate_minmax_scaler(self) -> None:
        self.min = jnp.nanmin(
            self.data, axis=tuple(i for i in range(1, len(self.shape)))
        )
        self.max = jnp.nanmax(
            self.data, axis=tuple(i for i in range(1, len(self.shape)))
        )
        # self.scaler = lambda x: (x-self.min) / (self.max-self.min)
        # self.inverse_scaler = lambda x: x*(self.max-self.min) + self.min

    def normalize(self, data: Numeric_ND) -> Numeric_ND:
        def normalize_each(x, xmin, xmax):
            return (x - xmin) / (xmax - xmin)

        return jax.vmap(normalize_each, in_axes=(0, 0, 0))(data, self.min, self.max)

    def inverse_normalize(self, data: Numeric_ND) -> Numeric_ND:
        def normalize_each(x, xmin, xmax):
            return x * (xmax - xmin) + xmin

        return jax.vmap(normalize_each, in_axes=(0, 0, 0))(data, self.min, self.max)

    def interpolate_time(self, t: Float_0D):
        raise Exception("Not implmented.")

    def interpolate_time_normalize(self, t: Float_0D):
        raise Exception("Not implmented.")

    def interpolate_time_by_varn(self, varn: str, ts_interp: Float_1D) -> Float_1D:
        raise Exception("Not implmented.")

    def interpolate_time_normalize_by_varn(
        self, varn: str, ts_interp: Float_1D
    ) -> Float_1D:
        raise Exception("Not implmented.")

    def interpolate_space(self, x: Float_0D, y: Float_0D, z: Float_0D):
        raise Exception("Not implmented.")

    def interpolate_space_normalize(self, x: Float_0D, y: Float_0D, z: Float_0D):
        raise Exception("Not implmented.")


# class DataStatic(Data):

#     def __init__(self, varn_list: List[str], data: Numeric_ND) -> None:
#         super().__init__(varn_list, data)


# class DataDynamic(Data):

#     pass


class DataBase(object):
    def __init__(
        self, forcings: Dict[str, AbstractGlobalInterpolation], dt: Float_0D
    ) -> None:
        self.forcings = forcings
        self.varn_list = list(forcings.keys())
        self.dt = dt

        # Create the normalizing scaler
        self._generate_minmax_scaler()

    def _generate_minmax_scaler(self) -> None:
        self.scaler = dict()
        for varn in self.varn_list:
            ys = self.forcings[varn].ys
            minv, maxv = ys.min(), ys.max()
            self.scaler[varn] = {
                "min": minv,
                "maxv": maxv,
                "scaler": lambda y: (y - minv) / (maxv - minv),
                "inverse_scaler": lambda yinv: yinv * (maxv - minv) + minv,
            }

    # def inverse_normalize(self, varn: str, data: Array) -> Array:
    #     return self.scaler[varn]['inverse_scaler'](data)

    # @partial(jax.jit, static_argnums=[0])
    def evaluate(self, varn: str, t: Float_0D) -> Float_0D:
        return self.forcings[varn].evaluate(t)

    def evaluate_normalize(self, varn: str, t: Float_0D) -> Float_0D:
        out = self.forcings[varn].evaluate(t)
        return self.scaler[varn]["scaler"](out)

    # @partial(jax.jit, static_argnums=[0])
    def evaluate_list(self, varn_list: List[str], t: Float_0D) -> Float_1D:
        #     return jax.vmap(lambda varn: self.evaluate(varn, t))(varn_list)
        # return jax.lax.map(lambda varn: self.evaluate(varn, t), varn_list)
        return jnp.stack([self.evaluate(varn, t) for varn in varn_list])

    def evaluate_list_normalize(self, varn_list: List[str], t: Float_0D) -> Float_1D:
        return jnp.stack([self.evaluate_normalize(varn, t) for varn in varn_list])
