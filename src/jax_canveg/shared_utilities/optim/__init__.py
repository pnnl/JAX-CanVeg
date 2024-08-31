from .optim import perform_optimization  # noqa: F401
from .optim import perform_optimization_batch  # noqa: F401
from .optim import mse  # noqa: F401

import logging
import optax
from typing import Optional, Dict, Any


def get_loss_function(loss_func_arg: str):
    """Get the loss function.

    Args:
        loss_func_arg (str): The loss function
    """
    if loss_func_arg.lower() == "mse":
        return mse
    else:
        raise Exception(f"Unknown loss function: {loss_func_arg}")


def get_optimzer(optim_configs: Optional[Dict] = None):
    """Get the optimizer.

    Args:
        optim_configs (Optional[Dict], optional): Optimization configuration.
            Defaults to None.
    """
    if optim_configs is None:
        logging.info(
            "Getting the default optimizer using Adam with learning rate 0.001..."
        )
        return optax.adam(learning_rate=0.001)

    else:
        optim_type = check_and_get_keyword(optim_configs, "type", "optimizer")
        optim_type = get_optimzer_type(optim_type)
        optim_args = check_and_get_keyword(optim_configs, "args", "optimizer")

        learning_configs = check_and_get_keyword(
            optim_configs, "learning_scheduler", "optimizer"
        )
        learning_scheduler = get_learning_scheduler(learning_configs)

        if optim_configs is None or optim_configs == {}:
            return optim_type(learning_scheduler)
        else:
            return optim_type(learning_scheduler, **optim_args)


def get_optimzer_type(optim_type: str):
    if optim_type.lower() == "adam":
        return optax.adam

    elif optim_type.lower() == "adamw":
        return optax.adamw

    else:
        raise Exception(
            f"Unknown optimizer type {optim_type}. Double check optax optimization library."  # noqa: E501
        )


def get_learning_scheduler(learning_configs: Optional[Dict] = None):
    if learning_configs is None:
        logging.info(
            "Getting the default learning scheduler with learning rate 0.001..."
        )
        return optax.constant_schedule(0.001)

    else:
        learning_type = check_and_get_keyword(
            learning_configs, "type", "learning scheduler"
        )
        learning_args = check_and_get_keyword(
            learning_configs, "args", "learning scheduler"
        )
        if learning_type.lower() == "constant":
            return optax.constant_schedule(learning_args)
        elif learning_type.lower() == "piecewise constant":
            init_value = check_and_get_keyword(
                learning_args, "init_value", "piecewise constant scheculer", True, 0.01
            )
            boundaries_and_scales = check_and_get_keyword(
                learning_args,
                "boundaries_and_scales",
                "piecewise constant scheculer",
                True,
                None,
            )
            if boundaries_and_scales is None:
                return optax.piecewise_constant_schedule(
                    init_value, boundaries_and_scales
                )
            else:
                boundaries_and_scales2 = {}
                for k, v in boundaries_and_scales.items():
                    key = int(k)
                    boundaries_and_scales2[key] = v
                return optax.piecewise_constant_schedule(
                    init_value, boundaries_and_scales2
                )
        else:
            raise Exception(
                f"Unknown learning scheduler type {learning_type}. Double check optax optimization library."  # noqa: E501
            )


def check_and_get_keyword(
    configs: dict,
    key: str,
    config_type: str = "Unknown",
    return_default: bool = False,
    default: Any = None,
):
    if key in configs:
        return configs[key]
    else:
        if return_default:
            logging.info(
                f"{key} is not found in configuration of {config_type} and return {default}."  # noqa: E501
            )
            return default
        else:
            raise Exception(f"{key} is not found in configuration of {config_type}.")
