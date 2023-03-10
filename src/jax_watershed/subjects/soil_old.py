"""This implements the basic properties of soil using equinox."""

import jax.numpy as jnp

from jaxtyping import Array, Float, Int
from typing import Union, Optional, List
from ..shared_utilities.types import Float_0D, Float_1D
from typeguard import typechecked

import equinox as eqx
from equinox.module import static_field


# @typechecked
class Soil:

    def __init__(
        self, 
        ndim: int,
        theta_sat: Float_1D,
        theta_r  : Union[Float_0D, Float_1D],
        theta_wp : Union[Float_0D, Float_1D],
        theta_lim: Union[Float_0D, Float_1D],
        ksat     : Union[Float_1D, Float_1D],
        alpha    : Union[Float_0D, Float_1D],
        n        : Union[Float_0D, Float_1D],
        depths   : Float_1D,
        nsoil    : int,
        cs       : Optional[Union[Float_0D, Float_1D]]=None,
        kthermal : Optional[Union[Float_0D, Float_1D]]=None,
        rho      : Optional[Union[Float_0D, Float_1D]]=None,
        **kwargs
    ) -> None:
        """A class for a column of soil properties.

        Args:
            theta_sat (Float_1D): Saturated volumetric soil water content [-]
            theta_r (Union[Float_0D, Float_1D]): Residual water content [-]
            theta_wp (Union[Float_0D, Float_1D]): Wilting point [-]
            theta_lim (Union[Float_0D, Float_1D]): Limiting soil moisture for vegetation [-]
            ksat (Union[Float_0D, Float_1D]): Saturated hydraulic conductivity [m d-1]
            alpha (Union[Float_0D, Float_1D]): Van Genuchten parameter alpha
            n (Union[Float_0D, Float_1D]): Van Genuchten parameter n
            cs (Union[Float_0D, Float_1D]): Soil heat capacity [MJ kg-1 degC-1 or MJ kg-1 degK-1]
            kthermal (Union[Float_0D, Float_1D]):  Soil thermal conductivity [MJ m-1 d-1 degC-1 or MJ m-1 d-1 degK-1]
            rho (Union[Float_0D, Float_1D]): Soil density [kg m-3]
            depths (Float_1D): Soil depths of all cells [m]
            nsoil (int): Number of vertical cells
        """
        super().__init__(**kwargs)
        self.theta_sat = theta_sat
        self.theta_r   = theta_r
        self.theta_wp  = theta_wp
        self.theta_lim = theta_lim
        self.ksat      = ksat
        self.alpha     = alpha
        self.n         = n
        self.cs        = cs
        self.kthermal  = kthermal
        self.rho       = rho
        self.depths    = depths
        self.nsoil     = nsoil