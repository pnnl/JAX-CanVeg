"""
Functions used for performing optimization using optax.

Author: Peishi Jiang
Date: 2023.08.22.
"""

import optax
import equinox as eqx
import jax.numpy as jnp

from typing import Tuple, List
from jaxtyping import Array

from ...models import CanoakBase
from ...subjects import Met


# Define the loss function
@eqx.filter_value_and_grad
def loss_func(diff_model: CanoakBase, static_model: CanoakBase, y: Array, met: Met):
    model = eqx.combine(diff_model, static_model)
    pred_y = model(met)
    return jnp.mean((y - pred_y) ** 2)


def perform_optimization(
    model: CanoakBase,
    filter_model_spec: CanoakBase,
    optim: optax._src.base.GradientTransformation,
    y: Array,
    met: Met,
    nsteps: int,
) -> Tuple[CanoakBase, List]:
    @eqx.filter_jit
    def make_step(model, filter_model_spec, y, opt_state, met):
        diff_model, static_model = eqx.partition(model, filter_model_spec)
        loss, grads = loss_func(diff_model, static_model, y, met)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, grads

    loss_set = []
    opt_state = optim.init(model)
    for i in range(nsteps):
        model, opt_state, loss, grads = make_step(
            model, filter_model_spec, y, opt_state, met
        )
        loss_set.append(loss)
        print(f"The loss of step {i}: {loss}")
        print("par_reflect:{}".format(model.para.par_reflect))

    return model, loss_set
