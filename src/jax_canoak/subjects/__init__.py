from .parameters import Para  # noqa: F401
from .parameters import initialize_parameters  # noqa: F401
from .meterology import Met  # noqa: F401
from .meterology import initialize_met  # noqa: F401
from .states import ParNir, Ir, Rnet, SunShadedCan  # noqa: F401
from .states import BoundLayerRes, Qin, Veg, Soil  # noqa: F401
from .states import SunAng, LeafAng, Prof, Ps, Lai  # noqa: F401
from .states import initialize_profile_mx  # noqa: F401
from .states import update_profile_mx  # noqa: F401
from .states import calculate_veg_mx  # noqa: F401
from .states import initialize_model_states  # noqa: F401

# from jax import tree_util

# Let's register these classes to pytrees
# tree_util.register_pytree_node(
#     Para, Para._tree_flatten, Para._tree_unflatten  # pyright: ignore
# )
# tree_util.register_pytree_node(
#     Soil, Soil._tree_flatten, Soil._tree_unflatten  # pyright: ignore
# )
# tree_util.register_pytree_node(
#     Met, Met._tree_flatten, Met._tree_unflatten  # pyright: ignore
# )
# tree_util.register_pytree_node(
#     Prof, Prof._tree_flatten, Prof._tree_unflatten  # pyright: ignore
# )
# tree_util.register_pytree_node(
#     SunAng, SunAng._tree_flatten, SunAng._tree_unflatten  # pyright: ignore
# )
# tree_util.register_pytree_node(
#     LeafAng, LeafAng._tree_flatten, LeafAng._tree_unflatten  # pyright: ignore
# )
# tree_util.register_pytree_node(
#     ParNir, ParNir._tree_flatten, ParNir._tree_unflatten  # pyright: ignore
# )
# tree_util.register_pytree_node(
#     Ir, Ir._tree_flatten, Ir._tree_unflatten  # pyright: ignore
# )
# tree_util.register_pytree_node(
#     Rnet, Rnet._tree_flatten, Rnet._tree_unflatten  # pyright: ignore
# )
# tree_util.register_pytree_node(
#     SunShadedCan,
#     SunShadedCan._tree_flatten,  # pyright: ignore
#     SunShadedCan._tree_unflatten,
# )
# tree_util.register_pytree_node(
#     BoundLayerRes,
#     BoundLayerRes._tree_flatten,  # pyright: ignore
#     BoundLayerRes._tree_unflatten,
# )
# tree_util.register_pytree_node(
#     Qin, Qin._tree_flatten, Qin._tree_unflatten  # pyright: ignore
# )
# tree_util.register_pytree_node(
#     Veg, Veg._tree_flatten, Veg._tree_unflatten  # pyright: ignore
# )
# tree_util.register_pytree_node(
#     Lai, Lai._tree_flatten, Lai._tree_unflatten  # pyright: ignore
# )
# tree_util.register_pytree_node(
#     Ps, Ps._tree_flatten, Ps._tree_unflatten  # pyright: ignore
# )
