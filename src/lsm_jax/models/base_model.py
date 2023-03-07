"""This is a model template, implementing the basic functionalities of a model have."""

from typing import Any

import equinox as eqx


class BaseModel(eqx.Module):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
