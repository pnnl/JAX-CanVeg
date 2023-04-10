import unittest

import diffrax as dr

import jax.numpy as jnp

from jax_watershed.physics.energy_fluxes.subsurface_energy import (
    Tsoil_vector_field,
    calculate_Tg_from_Tsoil1,
)
from jax_watershed.shared_utilities.forcings import ushn2

# import matplotlib.pyplot as plt

nz = 10
Tsoil = 268.0 + jnp.linspace(start=5, stop=0, num=nz)
cv = 2.5e6 + jnp.zeros(nz)
# cv = 2.5 + jnp.zeros(nz)
κ = 2.0 + jnp.zeros(nz)
Δz = jnp.array([0.05] * 3 + [0.1] * 3 + [0.2] * 4)


# ---------------------------- Process ushn2 data ---------------------------- #
ts = ushn2.df_obs["G"].index.values - ushn2.df_obs["G"].index.values[0]
ts = jnp.array(ts.astype("timedelta64[s]").astype("float"))
G_all = ushn2.df_obs["G"].values
G_dr = dr.LinearInterpolation(ts=ts, ys=G_all)

# -------------------- Functions/terms required by diffrax ------------------- #
def Tsoil_vector_field_dr_each_step(t, y, args):
    return Tsoil_vector_field(
        Tsoil=y, κ=args["κ"], cv=args["cv"], Δz=args["Δz"], G=args["G"]
    )


def Tsoil_vector_field_dr_time_series(t, y, args):
    return Tsoil_vector_field(
        Tsoil=y, κ=args["κ"], cv=args["cv"], Δz=args["Δz"], G=-G_dr.evaluate(t)
    )


# solver = dr.ImplicitEuler(dr.NewtonNonlinearSolver(rtol=1e-6, atol=1e-7))
solver = dr.ImplicitEuler(dr.NewtonNonlinearSolver(rtol=1e-5, atol=1e-5))
# solver = dr.ImplicitEuler(dr.NewtonNonlinearSolver(rtol=1e-5, atol=1e-5))
term1 = dr.ODETerm(Tsoil_vector_field_dr_each_step)
term2 = dr.ODETerm(Tsoil_vector_field_dr_time_series)

# -------------------------------- Unit tests -------------------------------- #
class TestSoilTemperature(unittest.TestCase):
    def test_Tsoil_vector_functions(self):
        print("Performing test_Tsoil_vector_functions()...")
        G = -100.0
        dT = Tsoil_vector_field(Tsoil=Tsoil, κ=κ, cv=cv, Δz=Δz, G=G)
        # print(dT)
        print("")
        self.assertEqual(dT.size, nz)

    def test_calculate_Tg_from_Tsoil1(self):
        print("Performing test_calculate_Tg_from_Tsoil1()...")
        G, Tsoil1 = -100.0, 280
        Tg = calculate_Tg_from_Tsoil1(Tsoil1=Tsoil1, κ=2.0, Δz=0.05, G=G)
        print("The estimated ground temperature is: {}".format(Tg))
        print("")
        self.assertTrue(Tg > Tsoil1)

    def test_estimate_soil_temperature_one_step(self):
        print("Performing test_estiamte_soil_temperature_one_step()...")
        G = -100.0
        args = dict(κ=κ, cv=cv, Δz=Δz, G=G)
        state = solver.init(terms=term1, t0=0.0, t1=3600.0, y0=Tsoil, args=args)
        Tsoilnew, _, _, state, _ = solver.step(
            terms=term1,
            t0=0.0,
            t1=3600.0,
            y0=Tsoil,
            args=args,
            solver_state=state,
            made_jump=False,
        )
        print("Initial soil temperature: ", Tsoil)
        print("Updated soil temperature: ", Tsoilnew)
        print("Soil temperature differences: ", Tsoilnew - Tsoil)
        print("")
        self.assertEqual(Tsoilnew.size, nz)

    def test_estimate_soil_temperature_time_series(self):
        print("Performing test_estiamte_soil_temperature_time_series()...")
        nday = 365
        t0, t1, dt0 = 0.0, nday * 86400, 86400
        args = dict(κ=κ, cv=cv, Δz=Δz)

        saveat = dr.SaveAt(ts=jnp.linspace(t0, t1, nday))
        sol = dr.diffeqsolve(
            terms=term2,
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=Tsoil,
            args=args,
            saveat=saveat,
        )
        # print(sol.ys)
        print("")
        # # Plot the evolution of the temperature
        # z = jnp.cumsum(Δz)
        # c = plt.cm.winter(jnp.linspace(0.0, 1.0, nday))
        # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        # for i in range(nday):
        #     ax.plot(
        #         sol.ys[i, :],
        #         z,
        #         c=c[i],
        #         label="t={:.2f} day".format(sol.ts[i] / 86400) if i % 50 == 0 else None,  # noqa: E501
        #     )
        # ax.set(
        #     xlabel="Soil temperature [degK]",
        #     ylabel="Soil depth [m]",
        #     ylim=[z[-1], z[0]],
        # )
        # ax.legend()
        # plt.show()
        # print(sol.stats)
        # print(sol.solver_state)
        # print(sol)
        self.assertEqual(sol.stats["num_rejected_steps"], 0)
