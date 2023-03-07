"""This implements the basic properties of ground surface using equinox."""

import jax.numpy as jnp

from jaxtyping import Array, Float, Int
from typing import Union, Optional
from ..types import Float_0D, Float_1D
from typeguard import typechecked

from .canopy import Canopy

import equinox as eqx
from equinox.module import static_field


# @typechecked
class Surface(eqx.Module):

    albedo      : Float_0D
    emissivity  : Float_0D
    canopy      : Canopy
    # Below should be static parameters!
    fbs         : Float_0D
    fv          : Float_0D
    xit         : Float_0D

    def __init__(
        self, 
        albedo      : Float_0D,
        emissivity  : Float_0D,
        canopy      : Canopy,
        fbs         : Float_0D,
        fv          : Float_0D,
        xit         : Float_0D,
        **kwargs
    ) -> None:
        """A class for surface properties including both bare ground and canopy.

        Args:
            albedo (Float_0D): Albeda [-]
            emissivity (Float_0D): Emissivity [-]
            fbs (Float_0D): Fraction of the bare soil [-]
            fv (Float_0D): Fraction of the vegetation [-]
            xit (Float_0D): Percentage of vegetation ET from the surface layer [-]
        """
        super().__init__(**kwargs)
        self.albedo     = albedo
        self.emissivity = emissivity
        self.canopy     = canopy
        self.fbs        = fbs
        self.fv         = fv
        self.xit        = xit