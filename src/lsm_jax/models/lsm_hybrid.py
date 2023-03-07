"""Implementation of the column-based land surface model."""

from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx
from equinox.module import static_field

from typing import List
from ..types import Float_1D, Float_0D

from .lsm import LSMVectorMassOnly
from ..data import DataBase

class LSMVectorMassOnlyHybrid(LSMVectorMassOnly):
    # dl_list: List[eqx.nn.MLP]
    dl: eqx.nn.MLP
    forcing_list: List[str] = static_field
    out_list: List[str] = static_field
    n_forcing: int = static_field
    n_out: int = static_field
    out_corrected_indices: List[int] = static_field

    def __init__(
        self,
        forcing_list: List[str],
        out_list: List[str],
        out_corrected_indices: List[int],
        width_size: int=4,
        depth: int=5,
        key: int=1234,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.forcing_list = forcing_list
        self.out_list = out_list
        self.n_forcing = len(forcing_list)
        self.n_out = len(out_list)
        self.out_corrected_indices = out_corrected_indices
        in_size = len(forcing_list)+len(out_corrected_indices)
        out_size = len(out_corrected_indices)

        # self.dl_list = [eqx.nn.MLP(in_size=len(forcing_list), out_size=1, width_size=4, depth=5) for i in out_corrected_indices]
        self.dl = eqx.nn.MLP(
            in_size=in_size, out_size=out_size, 
            width_size=width_size, depth=depth, key=jax.random.PRNGKey(key)
        )
    
    def __call__(self, t: Float_0D, y: Float_1D, forcings: DataBase) -> Float_1D:
        # The contributions from the physics process
        out = super().__call__(t, y, forcings)

        in_size = self.n_forcing + self.n_out

        # Get the inputs of the DL model
        inputs      = jnp.zeros(in_size)
        forcing_t   = forcings.evaluate_list_normalize(self.forcing_list, t)
        inputs      = inputs.at[jnp.arange(self.n_forcing)].set(forcing_t)
        inputs      = inputs.at[jnp.arange(self.n_forcing, in_size)].set(y[self.out_corrected_indices])

        # The contributions from DL corrections
        corrections = self.dl(inputs)

        # TODO: perform the inverse transformation on the corrections
        # print(corrections)
        # jax.lax.map(perform_correction, list(enumerate(self.out_corrected_indices)))
        # jax.lax.map(out.at[idx].add(corrections[i]), list(enumerate(self.out_corrected_indices)))
        out = out.at[jnp.array(self.out_corrected_indices)].add(corrections)

        return out