"""
Functions used for performing optimization using optax.

Author: Peishi Jiang
Date: 2023.08.22.
"""

import jax
import jax.tree_util as jtu

import optax
import equinox as eqx
import jax.numpy as jnp

from typing import Tuple, List, Optional, Callable
from jaxtyping import Array

import logging

from .loss import mse
from ...models import CanvegBase
from ...subjects import Met, BatchedMet, Para


# # Define a mean square error function
# def mse(y: Array, pred_y: Array):
#     """Function for calculating mean square error function

#     Args:
#         y (Array): _description_
#         pred_y (Array): _description_

#     Returns:
#         _type_: _description_
#     """
#     return jnp.mean((y - pred_y) ** 2)


def loss_func(model: CanvegBase, y: Array, met: Met, loss: Callable, *model_args):
    """Calculate the loss value for a given Canveg model.

    Args:
        model (CanvegBase): _description_
        y (Array): _description_
        met (Met): _description_
        loss (callable): _description_

    Returns:
        _type_: _description_
    """
    pred_y = model(met, *model_args)
    return loss(y, pred_y)


# Define the loss function
@eqx.filter_value_and_grad
def loss_func_optim(
    diff_model: CanvegBase,
    static_model: CanvegBase,
    y: Array,
    met: Met,
    loss: Callable,
    *model_args,
):
    """Calculating the gradient with respect to diff_model.
       Note that diff_model and static_model has the same type and
       can be generated by using the filtering strategy. See an example here:
       https://docs.kidger.site/equinox/examples/frozen_layer/.

    Args:
        diff_model (CanvegBase): _description_
        static_model (CanvegBase): _description_
        y (Array): _description_
        met (Met): _description_

    Returns:
        _type_: _description_
    """
    model = eqx.combine(diff_model, static_model)
    return loss_func(model, y, met, loss, *model_args)
    # jax.debug.print("args: {x}", x=args)
    # pred_y = model(met, *args)
    # jax.debug.print("pred_y: {x}", x=pred_y)
    # L2-loss
    # return jnp.mean((y - pred_y) ** 2)
    # return jnp.mean((pred_y - pred_y) ** 2)
    # return jnp.array(0.)
    # return loss(y, pred_y)

    # # Relative L2-loss
    # return jnp.mean((y - pred_y) ** 2 / (y ** 2))


def perform_optimization(
    model: CanvegBase,
    filter_model_spec: CanvegBase,
    optim: optax._src.base.GradientTransformation,
    y: Array,
    met: Met,
    nsteps: int,
    loss: Callable = mse,
    *args,
) -> Tuple[CanvegBase, List]:
    """A wrapped function for performing optimization using optax.

    Args:
        model (CanvegBase): _description_
        filter_model_spec (CanvegBase): _description_
        optim (optax._src.base.GradientTransformation): _description_
        y (Array): _description_
        met (Met): _description_
        nsteps (int): _description_

    Returns:
        Tuple[CanvegBase, List]: _description_
    """

    @eqx.filter_jit
    def make_step(model, filter_model_spec, y, opt_state, met, loss, *args):
        diff_model, static_model = eqx.partition(model, filter_model_spec)
        loss_value, grads = loss_func_optim(
            diff_model, static_model, y, met, loss, *args
        )
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value, grads

    loss_set = []
    # opt_state = optim.init(model)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    for i in range(nsteps):
        model, opt_state, loss_value, grads = make_step(
            model, filter_model_spec, y, opt_state, met, loss, *args
        )
        loss_set.append(loss_value)
        # print(f"The loss of step {i}: {loss_value}")
        logging.info(f"The loss of step {i}: {loss_value}")

    return model, loss_set


def perform_optimization_batch(
    model: CanvegBase,
    filter_model_spec: CanvegBase,
    optim: optax._src.base.GradientTransformation,
    nsteps: int,
    loss: Callable,
    batched_y: Array,
    batched_met: BatchedMet,
    batched_y_test: Optional[Array] = None,
    batched_met_test: Optional[BatchedMet] = None,
    para_min: Optional[Para] = None,
    para_max: Optional[Para] = None,
    *args,
    # ) -> Tuple[CanvegBase, List]:
):
    """A wrapped function for performing optimization in batch using optax.

    Args:
        model (CanvegBase): _description_
        filter_model_spec (CanvegBase): _description_
        optim (optax._src.base.GradientTransformation): _description_
        y (Array): _description_
        met (Met): _description_
        nsteps (int): _description_

    Returns:
        Tuple[CanvegBase, List]: _description_
    """

    # Function for making the step
    @eqx.filter_jit
    def make_step(
        model, filter_model_spec, batched_y, opt_state, batched_met, loss, *args
    ):
        print("Compiling make_step ...")
        diff_model, static_model = eqx.partition(model, filter_model_spec)
        # loss, grads = loss_func(diff_model, static_model, y, met)
        def loss_func_batch(c, batch):
            met, y = batch
            loss_value, grads = loss_func_optim(
                diff_model, static_model, y, met, loss, *args
            )
            return c, [loss_value, grads]

        _, results = jax.lax.scan(loss_func_batch, None, xs=[batched_met, batched_y])
        loss_value = results[0].mean()
        # # Print out the gradients of each parameter for check
        # jax.debug.print("bprime: {x}", x=results[1].__self__.para.bprime)
        # jax.debug.print("par_soil_refl: {x}",x=results[1].__self__.para.par_soil_refl)
        # jax.debug.print("nir_trans: {x}", x=results[1].__self__.para.par_soil_refl)
        # jax.debug.print("q10b: {x}", x=results[1].__self__.para.par_soil_refl)
        # jax.debug.print(
        #   "leaf_clumping_factor: {x}", x=results[1].__self__.para.leaf_clumping_factor
        # )
        # TODO: Need a better way to check nan occurring in the gradients
        grads = jtu.tree_map(lambda x: jnp.nanmean(x), results[1])
        # grads = results[1].mean()
        # grads = jtu.tree_map(lambda x: x.mean(), results[1])

        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value, grads

    # Function for calculating the loss of the test dataset
    @eqx.filter_jit
    def calculate_test_loss(model, batched_y_test, batched_met_test, loss, *args):
        # print("Compiling calculate_test_loss ...")
        # def loss_func_batch(c, batch):
        #     met, y = batch
        #     loss_value = loss_func(model, y, met, loss, *args)
        #     return c, loss_value
        # _, results = jax.lax.scan(
        #    loss_func_batch, None, xs=[batched_met_test, batched_y_test])
        def loss_func_batch(met, y):
            loss_value = loss_func(model, y, met, loss, *args)
            return loss_value

        loss_value_test = jax.vmap(loss_func_batch, in_axes=[0, 0])(
            batched_met_test, batched_y_test
        )
        # jax.debug.print("loss_value_test: {x}", x=loss_value_test)
        loss_value_test = loss_value_test.mean()
        return loss_value_test

    # Function for returning the training result
    def get_trained_result(model, loss_set):
        if batched_met_test is not None and batched_y_test is not None:
            loss_set_train = [l_value[0] for l_value in loss_set]
            loss_set_test = [l_value[1] for l_value in loss_set]
            return model, loss_set_train, loss_set_test
        else:
            return model, loss_set

    loss_set = []
    # opt_state = optim.init(model)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    for i in range(nsteps):
        check_arg(model, "model")
        check_arg(filter_model_spec, "filter_model_spec")
        check_arg(batched_y, "batched_y")
        check_arg(opt_state, "opt_state")
        check_arg(batched_met, "batched_met")
        # Update the model parameters
        try:
            model_updated, opt_state, loss_value, grads = make_step(
                model, filter_model_spec, batched_y, opt_state, batched_met, loss, *args
            )
        except Exception as e:
            # If there is error in the update, return the current model
            # and report the error
            logging.error('Fail to update the model with the following error \m: ' + str(e))
            return get_trained_result(model, loss_set)
            # if batched_met_test is not None and batched_y_test is not None:
            #     loss_set_train = [l_value[0] for l_value in loss_set]
            #     loss_set_test = [l_value[1] for l_value in loss_set]
            #     return model, loss_set_train, loss_set_test
            # else:
            #     return model, loss_set

        # para_list = [
        #     "bprime", "ep", "lleaf", "qalpha", "LeafRHDL", "kball",
        #     "leaf_clumping_factor",
        #     "vcopt", "jmopt", "rd25", "toptvc", "toptjm", "epsoil",
        #     "par_reflect", "par_trans", "par_soil_refl",
        #     "nir_reflect", "nir_trans", "nir_soil_refl",
        #     "q10a", "q10b", "q10c"
        # ]
        # for para in para_list:
        #     print(para)
        #     print(getattr(model.__self__.para, para))
        #     print(getattr(model_updated.__self__.para, para))
        #     print("")

        # Check NaN in loss_value
        if jnp.isnan(loss_value):
            logging.error(f"Encountered NaN in step {i}. Stop training.")
            return get_trained_result(model, loss_set)
            # if batched_met_test is not None and batched_y_test is not None:
            #     loss_set_train = [l_value[0] for l_value in loss_set]
            #     loss_set_test = [l_value[1] for l_value in loss_set]
            #     return model, loss_set_train, loss_set_test
            # else:
            #     return model, loss_set

        # Check the magnitude of loss value
        # If the loss is way much larger than the previous loss, something wrong happens.
        # Stop training and return the current model.
        if batched_met_test is not None and batched_y_test is not None:
            current_loss_set_train = [l_value[0] for l_value in loss_set]
        else:
            current_loss_set_train = loss_set
        current_loss_set_train = jnp.array(current_loss_set_train)
        if (loss_value / current_loss_set_train[-10:].mean()) > 1e3:
            logging.warning(f"The current loss value {loss_value} is way larger than the previous ones. Stop training.")  # noqa: E501
            return get_trained_result(model, loss_set)
            # if batched_met_test is not None and batched_y_test is not None:
            #     loss_set_train = [l_value[0] for l_value in loss_set]
            #     loss_set_test = [l_value[1] for l_value in loss_set]
            #     return model, loss_set_train, loss_set_test
            # else:
            #     return model, loss_set

        # If the previous check go through, update the model.
        model = model_updated

        # Check model parameters upper and lower bounds
        para = model.__self__.para  # pyright: ignore
        if para_min is not None:
            para = jtu.tree_map(lambda p, u: jnp.clip(p, a_min=u), para, para_min)
        if para_max is not None:
            para = jtu.tree_map(lambda p, u: jnp.clip(p, a_max=u), para, para_max)
        model = eqx.tree_at(
            lambda t: (t.__self__.para,), model, replace=(para,)  # pyright: ignore
        )

        # Calculate the loss for the test dataset
        if batched_met_test is not None and batched_y_test is not None:
            loss_value_test = calculate_test_loss(
                model, batched_y_test, batched_met_test, loss, *args
            )
            loss_set.append([loss_value, loss_value_test])
            logging.info(
                f"The training loss of step {i}: {loss_value}; the test loss of step {i}: {loss_value_test}."  # noqa: E501
            )
        else:
            loss_set.append(loss_value)
            logging.info(f"The loss of step {i}: {loss_value}")

    return get_trained_result(model, loss_set)
    # if batched_met_test is not None and batched_y_test is not None:
    #     loss_set_train = [l_value[0] for l_value in loss_set]
    #     loss_set_test = [l_value[1] for l_value in loss_set]
    #     return model, loss_set_train, loss_set_test
    # else:
    #     return model, loss_set


@eqx.filter_jit
def check_arg(arg, name):
    print(f"Argument {name} is triggering a compile.")
