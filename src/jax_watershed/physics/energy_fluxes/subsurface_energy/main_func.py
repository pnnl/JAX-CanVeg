"""
The main function for solving the subsurface energy balance for each time step.

Author: Peishi Jiang
Date: 2023.04.10.

"""

import diffrax as dr
from .soil_temp import Tsoil_vector_field

# from .soil_temp import calculate_Tg_from_Tsoil1

from typing import Tuple
from ....shared_utilities.types import Float_1D, Float_0D


def solve_subsurface_energy(
    Tsoil: Float_1D, κ: Float_1D, cv: Float_1D, Δz: Float_1D, Δt: Float_0D, G: Float_0D
) -> Tuple[Float_1D, Float_0D]:
    """Update the column soil temperature profile and the ground temperature based on the ground heat flux.

    Args:
        Tsoil (Float_1D): The soil temperature with nsoil layers at the previous time step [degK]
        cv (Float_1D): The volumetric heat capacity with nsoil layers [J m-3 K-1]
        κ (Float_1D): The thermal conductivity with nsoil layers [W m-1 K-1]
        Δz (Float_1D): The thickness of the nsoil layers [m]
        Δt (Float_1D): The time interval [day]
        G (Float_0D): The ground heat flux with positive values indicating the upward direction [W m-2]

    Returns:
        Float_1D: The updated soil temperature and the ground temperature at the next time step.
    """  # noqa: E501
    # Convert the time interval from day to second
    Δt = Δt * 86400

    # Create the vector field term
    def Tsoil_vector_field_dr_each_step(t, y, args):
        return Tsoil_vector_field(
            Tsoil=y, κ=args["κ"], cv=args["cv"], Δz=args["Δz"], G=args["G"]
        )

    term = dr.ODETerm(Tsoil_vector_field_dr_each_step)

    # Create the solver
    solver = dr.ImplicitEuler(dr.NewtonNonlinearSolver(rtol=1e-5, atol=1e-5))
    # solver = dr.ImplicitEuler(dr.NewtonNonlinearSolver(rtol=1e-5, atol=1e-5))

    # Combine all the arguments
    args = dict(κ=κ, cv=cv, Δz=Δz, G=G)

    # Solve the soil temperature profile
    state = solver.init(terms=term, t0=0.0, t1=Δt, y0=Tsoil, args=args)
    Tsoilnew, _, _, state, _ = solver.step(
        terms=term,
        t0=0.0,
        t1=Δt,
        y0=Tsoil,
        args=args,
        solver_state=state,
        made_jump=False,
    )

    return Tsoilnew

    # # Update the ground surface temperature
    # Tg = calculate_Tg_from_Tsoil1(
    #     Tsoil1=Tsoilnew[0],
    #     G=G,
    #     Δz=Δz[0] / 2.0,
    #     κ=κ[0],
    # )

    # return Tsoilnew, Tg
