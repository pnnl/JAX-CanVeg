"""
Calculate sensitivity and directional deriviate.

Author: Peishi Jiang
Date: 2024.09.13
"""

# TODO: Need a better documentation

from functools import partial

import jax
import jax.tree_util as jtu
from equinox.internal import ω
import equinox as eqx

################################################################
# Functions for sensitivity analysis
################################################################
@eqx.filter_jit
@partial(jax.grad, argnums=0)
def df_canveg(para, met, canveg_eqx, update_func, get_func):
    canveg_eqx = eqx.tree_at(lambda t: t.para, canveg_eqx, para)
    result = canveg_eqx.get_fixed_point_states(met, update_func, get_func)
    return result.sum()


@eqx.filter_jit
def df_canveg_le_batched(para, batched_met, canveg_eqx, update_func, get_func):
    def compute_grad(para, met):
        grad = df_canveg(para, met, canveg_eqx, update_func, get_func)
        return para, grad

    _, gradients = jax.lax.scan(compute_grad, para, xs=batched_met)
    return gradients


################################################################
# Functions for computing IFT-based directional derivative
################################################################
# Define directional vector for the parameters of interest
def get_partial_directional_vector(subpara, para):
    para_tangents = jtu.tree_map(lambda _: 0.0, para)
    para_tangents = eqx.tree_at(
        lambda t: tuple(getattr(t, p) for p in subpara),
        para,
        replace=tuple(1.0 for _ in subpara),
    )
    return para_tangents


@eqx.filter_jit
def tree_dot(a, b):
    """
    Computes the dot product of two pytrees
    Args:
        a, b: pytrees with the same treedef
    Returns:
        A scalar equal the dot product of of the flattened arrays of a and b.
    """
    return jax.tree_util.tree_reduce(
        jax.numpy.add,
        jax.tree_util.tree_map(
            jax.numpy.nansum, jax.tree_util.tree_map(jax.numpy.multiply, a, b)
        ),
    )


def AD_directional_derivative(
    para, met, canveg_eqx, para_tangents, update_func, get_func
):
    gradient = df_canveg(para, met, canveg_eqx, update_func, get_func)
    dir_grad = tree_dot(gradient, para_tangents)
    # jax.debug.print("bprime: {x}", x=gradient.bprime)
    # jax.debug.print("bprime tangent: {x}", x=para_tangents.bprime)
    # jax.debug.print("dir_grad: {x}", x=dir_grad)
    return dir_grad


@eqx.filter_jit
def AD_directional_derivative_batched(
    para, batched_met, canveg_eqx, para_tangents, update_func, get_func
):
    def compute_grad(para, met):
        grad = AD_directional_derivative(
            para, met, canveg_eqx, para_tangents, update_func, get_func
        )
        return para, grad

    _, gradients = jax.lax.scan(compute_grad, para, xs=batched_met)
    return gradients


################################################################
# Functions for computing FD-based directional derivative
################################################################
@eqx.filter_jit
def FD_directional_derivative(
    para, met, canveg_eqx, para_tangents, update_func, get_func, h=1e-2
):
    # assert jax.config.jax_enable_x64  # pyright: ignore
    canveg_eqx = eqx.tree_at(lambda t: t.para, canveg_eqx, para)
    out = canveg_eqx.get_fixed_point_states(met, update_func, get_func)

    # Choose ε to trade-off truncation error and floating-point rounding error.
    para_h = (ω(para) + h * ω(para_tangents)).ω
    canveg_eqx = eqx.tree_at(lambda t: t.para, canveg_eqx, para_h)
    out_h = canveg_eqx.get_fixed_point_states(met, update_func, get_func)
    dir_grad = jtu.tree_map(lambda x, y: (x - y) / h, out_h, out)
    # We actually return the perturbed primal.
    # This should still be within all tolerance checks, and means that we have aceesss
    # to both the true primal and the perturbed primal when debugging.
    return dir_grad


@eqx.filter_jit
def FD_directional_derivative_batched(
    para, batched_met, canveg_eqx, para_tangents, update_func, get_func, h=1e-2
):
    def compute_grad(para, met):
        grad = FD_directional_derivative(
            para, met, canveg_eqx, para_tangents, update_func, get_func, h
        )
        return para, grad

    _, gradients = jax.lax.scan(compute_grad, para, xs=batched_met)
    return gradients.flatten()
