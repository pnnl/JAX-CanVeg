import unittest

import jax
import numpy as np
import jax.numpy as jnp
from jax import config

import canoak  # noqa: E402

from jax_canoak.physics.energy_fluxes import soil_sfc_resistance  # type: ignore
from jax_canoak.physics.energy_fluxes import set_soil  # type: ignore
from jax_canoak.physics.energy_fluxes import soil_energy_balance  # type: ignore

from jax_canoak.shared_utilities.constants import epsoil, mass_air

config.update("jax_enable_x64", True)

jtot = 3
jtot3 = 5
sze = jtot + 2
sze3 = jtot3 + 2
soilsze = 12
nsoil = soilsze - 3
szeang = 19


class TestSoilEnergy(unittest.TestCase):
    def test_soil_sfc_resistance(self):
        print("Performing test_soil_sfc_resistance()...")
        # Inputs
        wg = 0.27

        # CANOAK
        rs_soil = canoak.soil_sfc_resistance(wg)  # type: ignore

        # JAX
        soil_sfc_resistance_jit = jax.jit(soil_sfc_resistance)
        rs_soil_jit = soil_sfc_resistance_jit(wg)

        # print(rs_soil_jit, rs_soil)
        print("")
        self.assertTrue(np.allclose(rs_soil_jit, rs_soil))

    def test_set_soil(self):
        print("Performing test_set_soil()...")
        # Inputs
        dt, total_t, water_content_15cm, soil_T_base = 20.0, 3600.0, 0.15, 21.3
        air_temp, air_density, air_press_Pa = 25.1, 1.3, 101122.4
        air_density_mole = air_density / mass_air * 1000.0
        T_soil_np = np.zeros(soilsze)
        z_soil_np = np.zeros(soilsze)
        soil_bulk_density_np = np.zeros(soilsze)
        cp_soil_np = np.zeros(soilsze)
        k_conductivity_soil_np = np.zeros(soilsze)

        # CANOAK
        soil_mtime_np = canoak.set_soil(  # type: ignore
            dt,
            total_t,
            nsoil,
            water_content_15cm,
            soil_T_base,
            air_temp,
            air_density,
            air_density_mole,
            air_press_Pa,
            T_soil_np,
            z_soil_np,
            soil_bulk_density_np,
            cp_soil_np,
            k_conductivity_soil_np,
        )

        # JAX
        set_soil_jit = jax.jit(set_soil)
        (
            soil_mtime_jnp,
            T_soil_jnp,
            z_soil_jnp,
            soil_bulk_density_jnp,
            cp_soil_jnp,
            k_conductivity_soil_jnp,
        ) = set_soil_jit(
            dt,
            total_t,
            jnp.zeros(nsoil + 1),
            water_content_15cm,
            soil_T_base,
            air_temp,
            air_density,
            air_density_mole,
            air_press_Pa,
        )

        # print(T_soil_np)
        # print(z_soil_np)
        # print(z_soil_jnp, z_soil_np)
        # print(cp_soil_jnp, cp_soil_np)
        # print(k_conductivity_soil_jnp, k_conductivity_soil_np)
        print("")
        self.assertTrue(np.allclose(soil_mtime_jnp, soil_mtime_np))
        self.assertTrue(np.allclose(T_soil_jnp, T_soil_np[: nsoil + 1]))
        self.assertTrue(np.allclose(z_soil_jnp, z_soil_np[: nsoil + 2]))
        self.assertTrue(
            np.allclose(soil_bulk_density_jnp, soil_bulk_density_np[: nsoil + 2])
        )
        self.assertTrue(np.allclose(cp_soil_jnp, cp_soil_np[: nsoil + 1]))
        self.assertTrue(
            np.allclose(k_conductivity_soil_jnp, k_conductivity_soil_np[: nsoil + 1])
        )

    def test_soil_energy_balance(self):
        print("Performing test_soil_energy_balance()...")
        # Inputs
        dt, total_t, ht, wnd = 10, 1800.0, 2.0, 1.5
        air_density, air_relative_humidity = 1.2, 0.4
        air_density_mole = air_density / mass_air * 1000.0
        air_press_Pa, air_temp = 101377.0, 25.2
        water_content_15cm, initial_tsoil, soil_T_base = 0.15, 23.2, 23.3
        beam_flux_par_sfc, par_down_sfc, par_up_sfc = 30.0, 300.0, 0.0
        beam_flux_nir_sfc, nir_dn_sfc, nir_up_sfc = 15.0, 250.0, 10.0
        # beam_flux_par_sfc, par_down_sfc, par_up_sfc = 0., 0., 0.
        # beam_flux_nir_sfc, nir_dn_sfc, nir_up_sfc = 0., 0., 10.
        ir_dn_sfc, delz = 18.0, ht / jtot
        tair_filter_sfc, soil_sfc_temperature, rhov_filter_sfc = 23.3, 23.3, 0.008
        soil_bulk_density_sfc, soil_mtime, iter_step = 0.1, 0, 3
        soil_rnet, soil_lout, soil_evap, soil_heat = 0.0, 0.0, 0.0, 0.0
        T_soil_np = np.zeros(soilsze)
        z_soil_np = np.zeros(soilsze)
        soil_bulk_density_np = np.zeros(soilsze)
        cp_soil_np = np.zeros(soilsze)
        k_conductivity_soil_np = np.zeros(soilsze)

        # CANOAK
        soil_mtime = canoak.set_soil(  # type: ignore
            dt,
            total_t,
            nsoil,
            water_content_15cm,
            soil_T_base,
            air_temp,
            air_density,
            air_density_mole,
            air_press_Pa,
            T_soil_np,
            z_soil_np,
            soil_bulk_density_np,
            cp_soil_np,
            k_conductivity_soil_np,
        )
        (
            soil_rnet_np,
            soil_lout_np,
            soil_heat_np,
            soil_evap_np,
            soil_sfc_temperature_np,
        ) = canoak.soil_energy_balance(  # type: ignore
            soilsze,
            epsoil,
            delz,
            ht,
            wnd,
            air_density,
            air_relative_humidity,
            air_press_Pa,
            air_temp,
            water_content_15cm,
            initial_tsoil,
            beam_flux_par_sfc,
            par_down_sfc,
            par_up_sfc,
            beam_flux_nir_sfc,
            nir_dn_sfc,
            nir_up_sfc,
            ir_dn_sfc,
            tair_filter_sfc,
            soil_sfc_temperature,
            rhov_filter_sfc,
            soil_bulk_density_sfc,
            soil_mtime,
            iter_step,
            soil_rnet,
            soil_lout,
            soil_evap,
            soil_heat,
            T_soil_np,
            k_conductivity_soil_np,
            cp_soil_np,
        )

        # JAX
        set_soil_jit = jax.jit(set_soil)
        (
            soil_mtime_jnp,
            T_soil_jnp,
            z_soil_jnp,
            soil_bulk_density_jnp,
            cp_soil_jnp,
            k_conductivity_soil_jnp,
        ) = set_soil_jit(
            dt,
            total_t,
            jnp.zeros(nsoil + 1),
            water_content_15cm,
            soil_T_base,
            air_temp,
            air_density,
            air_density_mole,
            air_press_Pa,
        )
        soil_energy_balance_jit = jax.jit(soil_energy_balance)
        (
            soil_rnet_jnp,
            soil_lout_jnp,
            soil_heat_jnp,
            soil_evap_jnp,
            soil_sfc_temperature_jnp,
            T_soil_jnp,
        ) = soil_energy_balance_jit(
            epsoil,
            delz,
            ht,
            wnd,
            air_density,
            air_relative_humidity,
            air_press_Pa,
            air_temp,
            water_content_15cm,
            initial_tsoil,
            beam_flux_par_sfc,
            par_down_sfc,
            par_up_sfc,
            beam_flux_nir_sfc,
            nir_dn_sfc,
            nir_up_sfc,
            ir_dn_sfc,
            tair_filter_sfc,
            soil_sfc_temperature,
            rhov_filter_sfc,
            soil_bulk_density_sfc,
            soil_lout,
            soil_heat,
            soil_evap,
            soil_mtime,
            iter_step,
            k_conductivity_soil_jnp,
            cp_soil_jnp,
        )

        # print(soil_rnet_np, soil_lout_np, soil_evap_np,
        #       soil_heat_np, soil_sfc_temperature_np)
        print(
            soil_rnet_jnp,
            soil_lout_jnp,
            soil_evap_jnp,
            soil_heat_jnp,
            soil_sfc_temperature_jnp,
        )
        # print(k_conductivity_soil_np)
        print(T_soil_jnp, T_soil_np)
        print("")
        self.assertTrue(np.allclose(T_soil_jnp, T_soil_np[: nsoil + 2]))
        self.assertTrue(np.allclose(soil_rnet_jnp, soil_rnet_np))
        self.assertTrue(np.allclose(soil_lout_jnp, soil_lout_np))
        self.assertTrue(np.allclose(soil_heat_jnp, soil_heat_np))
        self.assertTrue(np.allclose(soil_evap_jnp, soil_evap_np))
        self.assertTrue(np.allclose(soil_sfc_temperature_jnp, soil_sfc_temperature_np))
