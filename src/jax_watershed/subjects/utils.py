"""
Some utility functions for creating the subject class instances.

Author: Peishi Jiang
Date: 2023. 3. 8.
"""

import jax.numpy as jnp

from ..shared_utilities.types import Float_general
from ..shared_utilities.domain import Time
from ..shared_utilities.domain import BaseSpace

def parameter_initialize(para: Float_general, space:BaseSpace, paraname:str, default=0.) -> Float_general:
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
    elif isinstance(para, float):
        return jnp.zeros(space.shape) + para
    elif para.shape == space.shape:
        return jnp.array(para)
    elif para.shape != space.shape:
        raise Exception("The shape {} of is not idential to the shape of spatial domain {}".format(paraname, space.shape)) 

def state_initialize(state: Float_general, time:Time, space:BaseSpace, statename:str, default=0.) -> Float_general:
    """Initialize the model state based on the spatio-temporal domain.

    Args:
        state (Float_general): The state values given by the user.
        time (Time): The temporal domain.
        space (BaseSpace): The spatial domain.
        statename (str): The model state name.
        default (_type_, optional): The default state value. Defaults to 0..

    Returns:
        Float_general: The state values with shape identical to the spatio-temporal domain.
    """
    shape = (time.nt,) + space.shape
    if state is None:
        return jnp.zeros(shape) + default
    elif isinstance(state, float):
        return jnp.zeros(shape) + state
    elif state.shape == shape:
        return jnp.array(state)
    elif state.shape != shape:
        raise Exception("The shape {} of is not idential to the shape of spatio-temporal domain {}".format(statename, shape)) 
