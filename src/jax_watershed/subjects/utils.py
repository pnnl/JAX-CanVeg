"""
Some utility functions for creating the subject class instances.

Author: Peishi Jiang
Date: 2023. 3. 8.
"""

import jax.numpy as jnp

from typing import Optional

from ..shared_utilities.types import Float_general, Float_0D
from ..shared_utilities.domain import Time
from ..shared_utilities.domain import BaseSpace


def parameter_initialize(
    paraname: str, space: BaseSpace, para: Optional[Float_general] = None, default=0.0
) -> Float_general:
    """Initialize the parameter based on the spatial domain.

    Args:
        para (Float_general): The parameter values given by the user.
        space (BaseSpace): The spatial domain.
        paraname (str): The parameter name
        default (_type_, optional): The default parameter value. Defaults to 0..

    Returns:
        Float_general: The parameter values with shape identical to the spatial domain.
    """
    if para is None:
        return jnp.zeros(space.shape) + default
    elif isinstance(para, Float_0D):
        return jnp.zeros(space.shape) + para
    elif para.shape == space.shape and not isinstance(para, Float_0D):
        return jnp.array(para)
    # elif para.shape != space.shape:
    else:
        raise Exception(
            "The shape {} of is not idential to the shape of spatial domain {}".format(
                paraname, space.shape
            )
        )


def state_initialize(
    statename: str,
    time: Time,
    space: BaseSpace,
    state: Optional[Float_general] = None,
    default=0.0,
) -> Float_general:
    """Initialize the model state based on the spatio-temporal domain.

    Args:
        state (Float_general): The state values given by the user.
        time (Time): The temporal domain.
        space (BaseSpace): The spatial domain.
        statename (str): The model state name.
        default (_type_, optional): The default state value. Defaults to 0..

    Returns:
        Float_general: The state values with shape identical to the spatio-temporal domain.
    """  # noqa: E501
    shape = (time.nt,) + space.shape
    if state is None:
        return jnp.zeros(shape) + default
    elif isinstance(state, Float_0D):
        return jnp.zeros(shape) + state
    elif state.shape == shape and not isinstance(state, Float_0D):
        return jnp.array(state)
    # elif state.shape != shape:
    else:
        raise Exception(
            "The shape {} of is not idential to the shape of spatio-temporal domain {}".format(  # noqa: E501
                statename, shape
            )
        )
