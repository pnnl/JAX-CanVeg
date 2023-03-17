import unittest

from jax_watershed.physics.energy_fluxes.turbulent_fluxes import calculate_conductance_ground_canopy, calculate_conductance_leaf_boundary

class TestConductanceCalculation(unittest.TestCase):

    def test_conductance_ground_canopy(self):
        print("Performing test_conductance_ground_canopy()...")
        g = calculate_conductance_ground_canopy(L=3., S=0., ustar=3., z0m=0.03)
        print("The conductance is {} m s-1.".format(g))
        print("")
        self.assertTrue(g > 0)

    def test_conductance_leaf_boundary(self):
        print("Performing test_conductance_leaf_boundary()...")
        g = calculate_conductance_leaf_boundary(ustar=3.)
        print("The conductance is {} m s-1.".format(g))
        print("")
        self.assertTrue(g > 0)