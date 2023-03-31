import unittest

from jax_watershed.physics.energy_fluxes.turbulent_fluxes import (
    calculate_E,
    calculate_H,
)
from jax_watershed.shared_utilities.constants import λ_VAP as λ


class TestHeatFluxes(unittest.TestCase):
    def test_sensible_heat_flux(self):
        print("Performing test_sensible_heat_flux()...")
        H = calculate_H(T_2=295, T_1=290, ρ_atm=1.293, gh=1.0 / 100.0)
        print("The sensible heat flux is {} W m-2.".format(H))
        print("")
        self.assertTrue(H < 0)

    def test_latent_heat_flux(self):
        print("Performing test_latent_heat_flux()...")
        E = calculate_E(q_2=0.01, q_1=0.006, ρ_atm=1.293, ge=1.0 / 100.0)
        λE = λ * E
        print("The latent heat flux is {} W m-2.".format(λE))
        print("")
        self.assertTrue(λE < 0)
