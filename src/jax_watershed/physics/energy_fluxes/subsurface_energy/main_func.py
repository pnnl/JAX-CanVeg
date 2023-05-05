"""
The main function for solving the subsurface energy balance for each time step.

Author: Peishi Jiang
Date: 2023.04.10.

"""

import jax

import diffrax as dr
from .soil_temp import Tsoil_vector_field, Tsoil_vector_field_varyingG

from typing import Tuple
from ....shared_utilities.types import Float_1D, Float_0D
from ....shared_utilities.constants import λ_VAP as λ

from ..surface_energy.monin_obukhov import perform_most_dual_source
from ..turbulent_fluxes import calculate_E, calculate_H
from ..radiative_transfer import calculate_ground_longwave_fluxes
from ...water_fluxes import qsat_from_temp_pres

# from .soil_temp import calculate_Tg_from_Tsoil1


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


def solve_subsurface_energy_varyingG_laxscan(
    Tsoil: Float_1D,
    κ: Float_1D,
    cv: Float_1D,
    Δz: Float_1D,
    Δt: Float_0D,
    l: Float_0D,  # noqa: E741
    S_g: Float_0D,
    L_down: Float_0D,
    pres: Float_0D,
    ρ_atm_t2: Float_0D,
    T_v_t1: Float_0D,
    T_v_t2: Float_0D,
    T_g_t1: Float_0D,
    T_a_t2: Float_0D,
    u_a_t2: Float_0D,
    q_a_t2: Float_0D,
    L: Float_0D,
    S: Float_0D,
    ε_g: Float_0D,
    ε_v: Float_0D,
    z_a: Float_0D,
    z0m: Float_0D,
    z0c: Float_0D,
    d: Float_0D,
    gstomatal: Float_0D,
    gsoil: Float_0D,
) -> Float_1D:
    """Update the column soil temperature profile and the ground temperature based on the ground heat flux.

    Args:
        Tsoil (Float_1D): The soil temperature with nsoil layers [degK]
        cv (Float_1D): The volumetric heat capacity with nsoil layers [J m-3 K-1]
        κ (Float_1D): The thermal conductivity with nsoil layers [W m-1 K-1]
        Δz (Float_1D): The thickness of the nsoil layers [m]
        Δt (Float_1D): The time interval [day]
        l (Float_0D): The Obuhkov length [m]
        S_g (Float_0D): The incoming solar radiation on the ground [W m-2]
        L_down (Float_0D): The incoming longwave radiation at the current time step t2 [W m-2]
        pres (Float_0D): The air pressure at the reference height at the current time step t2 [Pa]
        ρ_atm_t2 (Float_0D): The air density at the reference height at the current time step t2 [kg m-3]
        T_v_t1 (Float_0D): The vegetation temperature at the previous time step t1 [degK]
        T_v_t2 (Float_0D): The vegetation temperature at the current time step t2 [degK]
        T_g_t1 (Float_0D): The ground temperature at the previous time step t1 [degK]
        T_a_t2 (Float_0D): The air temperature at the reference height at the current time step t2 [degK]
        u_a_t2 (Float_0D): The wind velocity at the reference height at the current time step t2 [m s-1]
        q_a_t2 (Float_0D): The specific humidity at the reference height at the current time step t2 [kg kg-1]
        L (Float_0D): The leaf area index [m2 m-2]
        S (Float_0D): The steam area index [m2 m-2]
        ε_g (Float_0D): The ground emissivities [-]
        ε_v (Float_0D): The vegetation emissivities [-]
        z_a (Float_0D): The reference height of the atmosphere [m]
        z0m (Float_0D): The roughness length for momentum [m]
        z0c (Float_0D): The roughness length for scalars [m]
        d (Float_0D): The displacement height [m]
        gstomatal (Float_0D): The stomatal conductance [m s-1]
        gsoil (Float_0D): The soil conductance [m s-1]

    Returns:
        Float_1D: The updated soil temperature and the ground temperature at the next time step.
    """  # noqa: E501
    # Convert the time interval from day to second
    Δt = Δt * 86400

    # Create the vector field term and the solver
    def Tsoil_vector_field_dr_each_step(t, y, args):
        return Tsoil_vector_field(
            Tsoil=y, κ=args["κ"], cv=args["cv"], Δz=args["Δz"], G=args["G"]
        )

    term = dr.ODETerm(Tsoil_vector_field_dr_each_step)
    solver = dr.ImplicitEuler(dr.NewtonNonlinearSolver(rtol=1e-5, atol=1e-5))

    def update_soil_temperature(Tsoil, x=None):
        # Calculate the ground heat flux
        T_g_t2 = Tsoil[0]
        q_g_t2_sat = qsat_from_temp_pres(T=T_g_t2, pres=pres)
        q_g_t2 = q_g_t2_sat
        (_, _, _, _, _, ggm, ggw, _, T_s_t2, q_s_t2,) = perform_most_dual_source(
            L_guess=l,
            pres=pres,
            T_v=T_v_t2,
            T_g=T_g_t2,
            T_a=T_a_t2,
            u_a=u_a_t2,
            q_a=q_a_t2,
            q_g=q_g_t2,
            L=L,
            S=S,
            z_a=z_a,
            z0m=z0m,
            z0c=z0c,
            d=d,
            gstomatal=gstomatal,
            gsoil=gsoil,
        )
        L_g = calculate_ground_longwave_fluxes(
            L_down=L_down,
            ε_v=ε_v,
            ε_g=ε_g,
            L=L,
            S=S,
            T_v_t1=T_v_t1,
            T_v_t2=T_v_t2,
            T_g_t1=T_g_t1,
            T_g_t2=T_g_t2,
        )
        H_g = calculate_H(T_1=T_g_t2, T_2=T_s_t2, ρ_atm=ρ_atm_t2, gh=ggm)
        E_g = calculate_E(
            q_1=q_g_t2, q_2=q_s_t2, ρ_atm=ρ_atm_t2, ge=ggw
        )  # [kg m-2 s-1]
        λE_g = λ * E_g
        G = S_g - L_g - H_g - λE_g
        G = -G
        # Solve the soil temperature profile
        args_solver = dict(κ=κ, cv=cv, Δz=Δz, G=G)
        state = solver.init(terms=term, t0=0.0, t1=Δt, y0=Tsoil, args=args_solver)
        Tsoilnew, _, _, state, _ = solver.step(
            terms=term,
            t0=0.0,
            t1=Δt,
            y0=Tsoil,
            args=args_solver,
            solver_state=state,
            made_jump=False,
        )
        return Tsoilnew, G

    Tsoilnew, G_list = jax.lax.scan(
        update_soil_temperature, init=Tsoil, xs=None, length=100
    )
    # jax.debug.print('The updated G_list: {}', G_list)

    return Tsoilnew


def solve_subsurface_energy_varyingG(
    Tsoil: Float_1D,
    κ: Float_1D,
    cv: Float_1D,
    Δz: Float_1D,
    Δt: Float_0D,
    l: Float_0D,  # noqa: E741
    S_g: Float_0D,
    L_down: Float_0D,
    pres: Float_0D,
    ρ_atm_t2: Float_0D,
    T_v_t1: Float_0D,
    T_v_t2: Float_0D,
    T_g_t1: Float_0D,
    T_a_t2: Float_0D,
    u_a_t2: Float_0D,
    q_a_t2: Float_0D,
    L: Float_0D,
    S: Float_0D,
    ε_g: Float_0D,
    ε_v: Float_0D,
    z_a: Float_0D,
    z0m: Float_0D,
    z0c: Float_0D,
    d: Float_0D,
    gstomatal: Float_0D,
    gsoil: Float_0D,
) -> Tuple[Float_1D, Float_0D]:
    """Update the column soil temperature profile and the ground temperature based on the ground heat flux.

    Args:
        Tsoil (Float_1D): The soil temperature with nsoil layers [degK]
        cv (Float_1D): The volumetric heat capacity with nsoil layers [J m-3 K-1]
        κ (Float_1D): The thermal conductivity with nsoil layers [W m-1 K-1]
        Δz (Float_1D): The thickness of the nsoil layers [m]
        Δt (Float_1D): The time interval [day]
        l (Float_0D): The Obuhkov length [m]
        S_g (Float_0D): The incoming solar radiation on the ground [W m-2]
        L_down (Float_0D): The incoming longwave radiation at the current time step t2 [W m-2]
        pres (Float_0D): The air pressure at the reference height at the current time step t2 [Pa]
        ρ_atm_t2 (Float_0D): The air density at the reference height at the current time step t2 [kg m-3]
        T_v_t1 (Float_0D): The vegetation temperature at the previous time step t1 [degK]
        T_v_t2 (Float_0D): The vegetation temperature at the current time step t2 [degK]
        T_g_t1 (Float_0D): The ground temperature at the previous time step t1 [degK]
        T_a_t2 (Float_0D): The air temperature at the reference height at the current time step t2 [degK]
        u_a_t2 (Float_0D): The wind velocity at the reference height at the current time step t2 [m s-1]
        q_a_t2 (Float_0D): The specific humidity at the reference height at the current time step t2 [kg kg-1]
        L (Float_0D): The leaf area index [m2 m-2]
        S (Float_0D): The steam area index [m2 m-2]
        ε_g (Float_0D): The ground emissivities [-]
        ε_v (Float_0D): The vegetation emissivities [-]
        z_a (Float_0D): The reference height of the atmosphere [m]
        z0m (Float_0D): The roughness length for momentum [m]
        z0c (Float_0D): The roughness length for scalars [m]
        d (Float_0D): The displacement height [m]
        gstomatal (Float_0D): The stomatal conductance [m s-1]
        gsoil (Float_0D): The soil conductance [m s-1]

    Returns:
        Float_1D: The updated soil temperature and the ground temperature at the next time step.
    """  # noqa: E501
    # Convert the time interval from day to second
    Δt = Δt * 86400

    # Create the vector field term
    def Tsoil_vector_field_dr_each_step(t, y, args):
        return Tsoil_vector_field_varyingG(Tsoil=y, **args)

    term = dr.ODETerm(Tsoil_vector_field_dr_each_step)

    # Create the solver
    solver = dr.ImplicitEuler(dr.NewtonNonlinearSolver(rtol=1e-5, atol=1e-5))
    # solver = dr.ImplicitEuler(dr.NewtonNonlinearSolver(rtol=1e-5, atol=1e-5))

    # Combine all the arguments
    args = dict(
        κ=κ,
        cv=cv,
        Δz=Δz,
        l=l,
        S_g=S_g,
        L_down=L_down,
        pres=pres,
        ρ_atm_t2=ρ_atm_t2,
        T_v_t1=T_v_t1,
        T_v_t2=T_v_t2,
        T_g_t1=T_g_t1,
        T_a_t2=T_a_t2,
        u_a_t2=u_a_t2,
        q_a_t2=q_a_t2,
        L=L,
        S=S,
        ε_g=ε_g,
        ε_v=ε_v,
        z_a=z_a,
        z0m=z0m,
        z0c=z0c,
        d=d,
        gstomatal=gstomatal,
        gsoil=gsoil,
    )

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
