import jax
import jax.numpy as jnp

from jaxtyping import Array

import numpy as np

import hydroeval as he
from sklearn.metrics import r2_score  # pyright: ignore

from ..shared_utilities.types import Float_0D, Float_1D, Float_2D


# Subroutine to compute scalar concentrations from source estimates
# and the Lagrangian dispersion matrix
def conc(
    cref: Float_0D,
    soilflux: Float_0D,
    factor: Float_0D,
    met_zl: Float_0D,
    delz: Float_0D,
    # izref: jnp.integer,
    izref: int,
    ustar_ref: Float_0D,
    ustar: Float_0D,
    source: Float_1D,
    dispersion: Float_2D,
) -> Float_1D:

    jtot3, jtot = dispersion.shape

    # Compute concentration profiles from Dispersion matrix
    ustfact = ustar_ref / ustar  # factor to adjust Dij with alternative u* values
    # Note that dispersion matrix was computed using u* = 1.00

    def compute_disperzl(disper):
        disperzl = jax.lax.cond(
            met_zl < 0,
            lambda x: x[0] * (0.97 * -0.7182) / (x[1] - 0.7182),
            lambda x: x[0] * (-0.31 * x[1] + 1.00),
            [disper, met_zl],
        )
        return disperzl

    # Calculate disperzl matrix
    disper = ustfact * dispersion
    disperzl = jnp.vectorize(compute_disperzl)(disper)

    # Calculate sumcc
    sumcc = jnp.sum(delz * disperzl * source, axis=1)  # type: ignore[code]
    # jax.debug.print("source: {b}", b=source)
    # jax.debug.print("delz: {b}; sumcc: {a}", a=sumcc, b=delz)
    # jax.debug.print("delz: {b}; disperzl: {a}", a=disperzl, b=delz)

    # Calculate cc
    # jax.debug.print("sumcc: {a}", a=sumcc)
    def compute_each_cc(carry, i):
        # Scale dispersion matrix according to Z/L
        disper = ustfact * dispersion[i, 0]
        disperzl = compute_disperzl(disper)
        # Add soil flux to the lowest boundary condition
        soilbnd = soilflux * disperzl / factor
        each_cc = sumcc[i] / factor + soilbnd
        # jax.debug.print("each_cc: {a}; sumcc[i]: {b}; factor: {c}; soilflux: {d}",
        #                 a=each_cc, b=sumcc[i], c=factor, d=soilflux)
        return carry, each_cc

    _, cc = jax.lax.scan(f=compute_each_cc, init=None, xs=jnp.arange(jtot3))
    # jax.debug.print('space ..')
    # jax.debug.print("cref: {b}; soilflux: {a}", a=soilflux, b=cref)

    # Compute scalar profile below reference
    def compute_each_cncc(carry, i):
        each_cncc = cc[i] + cref - cc[izref]
        return carry, each_cncc

    _, cncc = jax.lax.scan(f=compute_each_cncc, init=None, xs=jnp.arange(jtot3))

    return cncc


def filter_array(
    array: Float_1D, a_min: Float_0D, a_max: Float_0D, replace: Float_0D
) -> Float_1D:
    # nsize = array.size
    def update_array(c, a):
        # jax.debug.print("a: {a}", a=a)
        a_new = jax.lax.cond((a < a_min) | (a > a_max), lambda: replace, lambda: a)
        return c, a_new

    _, array_new = jax.lax.scan(update_array, init=None, xs=array)
    return array_new


def compute_metrics(
    pred: Array,
    true: Array,
):
    """
    Computing a bunch of evaluation metrics
    """
    pred, true = np.array(pred), np.array(true)  # pyright: ignore
    assert pred.shape == true.shape

    rmse = he.evaluator(he.rmse, pred, true)[0]
    kge = he.evaluator(he.kge, pred, true)[0][0]
    mkge_all = he.evaluator(he.kgeprime, pred, true).flatten()
    mkge = mkge_all[0]
    cc = mkge_all[1]
    beta = mkge_all[2]
    alpha = mkge_all[3]
    nse = he.evaluator(he.nse, pred, true)[0]
    r2 = r2_score(pred, true)
    return {
        "rmse": rmse,
        "mse": rmse**2,
        "r2": r2,
        "kge": kge,
        "nse": nse,
        "mkge": mkge,
        "cc": cc,
        "alpha": alpha,
        "beta": beta,
    }


# dot product: (n) x (n,m) -> (n,m)
dot = jax.vmap(lambda x, y: x * y, in_axes=(None, 1), out_axes=1)

# plus: (n) + (n,m) -> (n,m)
add = jax.vmap(lambda x, y: x + y, in_axes=(None, 1), out_axes=1)

# minus: (n) + (n,m) -> (n,m)
minus = jax.vmap(lambda x, y: x - y, in_axes=(None, 1), out_axes=1)

# divide: (n) + (n,m) -> (n,m)
divide = jax.vmap(lambda x, y: x / y, in_axes=(None, 1), out_axes=1)


# Function for configuring the on or off of debugging nans and infs
def tune_jax_naninfs_for_debug(value=True):
    jax.config.update("jax_debug_nans", value)
    jax.config.update("jax_debug_infs", value)
    jax.config.update("jax_disable_jit", value)
