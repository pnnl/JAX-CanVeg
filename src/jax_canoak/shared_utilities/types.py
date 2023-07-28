from typing import Union
from jaxtyping import Array, Float, Int

Int_0D = Union[int, Int[Array, ""]]  # noqa: F722
Int_1D = Int[Array, "dim1"]
Int_2D = Int[Array, "dim1 dim2"]

Float_0D = Union[float, Float[Array, ""]]  # noqa: F722
Float_1D = Float[Array, "dim1"]
Float_2D = Float[Array, "dim1 dim2"]
Float_3D = Float[Array, "dim1 dim2 dim3"]
Float_ND = Float[Array, "var time ..."]
# Float_general = Union[Float_0D, Float_1D, Float_2D, Float_3D]
Float_general = Union[Float_1D, Float_2D, Float_3D, Float_ND]
# Float_Array_1D = Float[Array, "dim0"]

Numeric_ND = Union[Int[Array, "var time ..."], Float_ND]  # noqa: F722
