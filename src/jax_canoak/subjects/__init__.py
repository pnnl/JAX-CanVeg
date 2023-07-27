from .parameters import Para
from .meterology import Met
from .states import ParNir, Ir, Rnet, SunShadedCan, BoundLayerRes, Qin, Veg, Soil

from jax import tree_util

# Let's register these classes to pytrees
tree_util.register_pytree_node(
    Para, Para._tree_flatten, Para._tree_unflatten  # pyright: ignore
)
tree_util.register_pytree_node(
    Met, Met._tree_flatten, Met._tree_unflatten  # pyright: ignore
)
tree_util.register_pytree_node(
    ParNir, ParNir._tree_flatten, ParNir._tree_unflatten  # pyright: ignore
)
tree_util.register_pytree_node(
    Ir, Ir._tree_flatten, Ir._tree_unflatten  # pyright: ignore
)
tree_util.register_pytree_node(
    Rnet, Rnet._tree_flatten, Rnet._tree_unflatten  # pyright: ignore
)
tree_util.register_pytree_node(
    SunShadedCan,
    SunShadedCan._tree_flatten,  # pyright: ignore
    SunShadedCan._tree_unflatten,
)
tree_util.register_pytree_node(
    BoundLayerRes,
    BoundLayerRes._tree_flatten,  # pyright: ignore
    BoundLayerRes._tree_unflatten,
)
tree_util.register_pytree_node(
    Qin, Qin._tree_flatten, Qin._tree_unflatten  # pyright: ignore
)
tree_util.register_pytree_node(
    Veg, Veg._tree_flatten, Veg._tree_unflatten  # pyright: ignore
)
tree_util.register_pytree_node(
    Soil, Soil._tree_flatten, Soil._tree_unflatten  # pyright: ignore
)
