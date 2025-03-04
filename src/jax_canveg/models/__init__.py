from .canveg import (
    canveg,  # noqa: F401
    canveg_each_iteration,  # noqa: F401
    canveg_initialize_states,  # noqa: F401
)
from .canveg import (
    get_all,  # noqa: F401
    get_canle,  # noqa: F401
    get_cannee,  # noqa: F401
    get_canlenee,  # noqa
    get_soilresp,  # noqa: F401
    update_all,  # noqa: F401
    update_canle,  # noqa: F401
    update_cannee,  # noqa: F401
    update_canlenee,  # noqa: F401
    update_soilresp,  # noqa: F401
)
from .canveg_eqx import CanvegBase, Canveg, CanvegIFT  # noqa: F401

from .utils import load_model_check, load_model, save_model  # noqa: F401

import logging
from typing import Optional


def get_canveg_eqx_class(model_type: Optional[str] = None):
    if model_type is None:
        logging.info("Getting model type CanvegIFT ...")
        return CanvegIFT
    elif model_type.lower() == "canvegift":
        logging.info("Getting model type CanvegIFT ...")
        return CanvegIFT
    elif model_type.lower() == "canveg":
        logging.info("Getting model type Canveg...")
        return Canveg
    else:
        raise Exception("Unknown model type: %s", model_type)


# def get_canveg_eqx_class(model_type: Optional[str] = None):
#     if model_type is None:
#         logging.info("Getting model type CanvegIFT ...")
#         return CanvegIFT
#     elif model_type.lower() == "canvegift":
#         logging.info("Getting model type CanvegIFT ...")
#         return CanvegIFT
#     elif model_type.lower() == "canvegleafrhhybridift":
#         logging.info("Getting model type CanvegLeafRHHybridIFT ...")
#         return CanvegLeafRHHybridIFT
#     elif model_type.lower() == "canvegrsoilhybridift":
#         logging.info("Getting model type CanvegRsoilHybridIFT ...")
#         return CanvegRsoilHybridIFT
#     else:
#         raise Exception("Unknown model type: %s", model_type)


def get_output_function(output_function_type: Optional[str] = None):
    if output_function_type is None:
        logging.info("Getting output function that gives all results ...")
        return update_all, get_all
    elif (
        output_function_type.lower() == "canle"
        or output_function_type.lower() == "canopy le"
    ):
        logging.info("Getting output function that gives canopy latent heat flux ...")
        return update_canle, get_canle
    elif (
        output_function_type.lower() == "cannee"
        or output_function_type.lower() == "canopy nee"
    ):
        logging.info(
            "Getting output function that gives canopy net ecosystem exchange ..."
        )
        return update_cannee, get_cannee
    elif (
        output_function_type.lower() == "canlenee"
        or output_function_type.lower() == "canopy le nee"
    ):
        logging.info(
            "Getting output function that gives canopy latent heat fluxes and net ecosystem exchange ..."  # noqa: E501
        )
        return update_canlenee, get_canlenee
    elif (
        output_function_type.lower() == "soilresp"
        or output_function_type.lower() == "soil respiration"
    ):
        logging.info("Getting output function that gives soil respiration ...")
        return update_soilresp, get_soilresp
    else:
        raise Exception("Unknown output function type: %s", output_function_type)
