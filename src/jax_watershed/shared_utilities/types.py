from typing import Union
from jaxtyping import Array, Float, Int

Float_0D = Float[Array, ""]
Float_1D = Float[Array, "dim1"]
Float_2D = Float[Array, "dim1 dim2"]
Float_3D = Float[Array, "dim1 dim2 dim3"]
Float_general = Union[Float_0D, Float_1D, Float_2D, Float_3D]
# Float_Array_1D = Float[Array, "dim0"]

Numeric_ND = [Array, "var time ..."]