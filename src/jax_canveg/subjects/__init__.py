from .parameters import Para, Setup  # noqa: F401
from .meterology import Met  # noqa: F401
from .states import ParNir, Ir, Rnet, SunShadedCan  # noqa: F401
from .states import BoundLayerRes, Qin, Veg, Soil, Can  # noqa: F401
from .states import SunAng, LeafAng, Prof, Ps, Lai, Obs  # noqa: F401
from .batched_meterology import BatchedMet  # noqa: F401
from .batched_meterology import convert_batchedmet_to_met  # noqa: F401
from .batched_meterology import convert_met_to_batched_met  # noqa: F401
from .batched_states import (
    BatchedParNir,  # noqa: F401
    BatchedIr,  # noqa: F401
    BatchedRnet,  # noqa: F401
    BatchedSunShadedCan,  # noqa: F401
)
from .batched_states import (
    BatchedBoundLayerRes,  # noqa: F401
    BatchedQin,  # noqa: F401
    BatchedVeg,  # noqa: F401
    BatchedSoil,  # noqa: F401
    BatchedCan,  # noqa: F401
)
from .batched_states import (
    BatchedSunAng,  # noqa: F401
    BatchedLeafAng,  # noqa: F401
    BatchedProf,  # noqa: F401
    BatchedPs,  # noqa: F401
    BatchedLai,  # noqa: F401
    BatchedObs,  # noqa: F401
)
from .batched_states import convert_batchedstates_to_states  # noqa: F401
from .batched_states import convert_obs_to_batched_obs  # noqa: F401
from .initialization_update import (
    initialize_parameters,  # noqa: F401
    initialize_met,  # noqa: F401
    get_met_forcings,  # noqa: F401
    get_obs,  # noqa: F401
    initialize_profile,  # noqa: F401
    update_profile,  # noqa: F401
    calculate_veg,  # noqa: F401
    calculate_var_stats,  # noqa: F401
    initialize_model_states,  # noqa: F401
    calculate_can,  # noqa: F401
)

from .dnn import (
    MLP,  # noqa: F401
    MLP2,  # noqa: F401
    MLP3,  # noqa: F401
)

# from .parameters import initialize_parameters  # noqa: F401
# from .meterology import initialize_met, get_met_forcings  # noqa: F401
# from .states import initialize_profile  # noqa: F401
# from .states import update_profile  # noqa: F401
# from .states import calculate_veg  # noqa: F401
# from .states import initialize_model_states  # noqa: F401

import logging
import jax.tree_util as jtu
import equinox as eqx


def get_filter_para_spec(para: Para, tunable_para: list):
    filter_para_spec = jtu.tree_map(lambda _: False, para)
    if tunable_para is None or len(tunable_para) == 0:
        logging.info(
            """No parameters are given. So we tune the following parameters:
               bprime, ep, lleaf, qalpha, kball, leaf_clumping factor"""
        )
        filter_para_spec = eqx.tree_at(
            lambda t: (
                t.bprime,
                t.ep,
                t.lleaf,
                t.qalpha,
                t.kball,
                t.leaf_clumping_factor,
            ),
            filter_para_spec,
            replace=(True, True, True, True),
        )
    else:
        # Filter the parameters to be estimated
        filter_para_spec = eqx.tree_at(
            lambda t: tuple(getattr(t, para) for para in tunable_para),
            filter_para_spec,
            replace=tuple(True for _ in tunable_para),
        )

    return filter_para_spec
