import unittest

import jax
import numpy as np
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

import canoak  # noqa: E402
from jax_canoak.physics.energy_fluxes import rnet  # noqa: E402
from jax_canoak.physics.energy_fluxes import par  # noqa: E402
from jax_canoak.physics.energy_fluxes import nir  # noqa: E402
from jax_canoak.physics.energy_fluxes import sky_ir  # noqa: E402
from jax_canoak.physics.energy_fluxes import irflux  # noqa: E402
from jax_canoak.physics.energy_fluxes import diffuse_direct_radiation  # noqa: E402
from jax_canoak.physics.energy_fluxes import g_func_diffuse  # noqa: E402

# jtot = 300
# jtot3 = 1500
# jtot = 3
# jtot3 = 5
jtot = 1
jtot3 = 3
sze = jtot + 2
sze3 = jtot3 + 2
szeang = 19
met_zl = 1.5
delz = 0.5
izref = 1
cref = 2.2
soilflux = 5.4
factor = 4.5
ustar_ref = 0.5
ustar = 1.8

parin = -1.2
par_beam = 0.4
par_reflect = 0.3
par_trans = 0.6
par_soil_refl = 0.3
par_absorbed = 0.8

nir_beam, nir_diffuse, nir_reflect = 2.0, 0.3, 0.3
nir_trans, nir_soil_refl, nir_absorbed = 0.5, 0.9, 0.1


class TestRadiationTransfer(unittest.TestCase):
    def test_rnet(self):
        print("Performing test_rnet()...")
        # inputs
        ir_dn_np = np.random.random(jtot)
        ir_up_np = np.random.random(jtot)
        par_sun_np = np.random.random(jtot)
        nir_sun_np = np.random.random(jtot)
        par_sh_np = np.random.random(jtot)
        nir_sh_np = np.random.random(jtot)
        rnet_sh_np = np.random.random(jtot)
        rnet_sun_np = np.random.random(jtot)

        ir_dn_jnp = jnp.array(ir_dn_np)
        ir_up_jnp = jnp.array(ir_up_np)
        par_sun_jnp = jnp.array(par_sun_np)
        nir_sun_jnp = jnp.array(nir_sun_np)
        par_sh_jnp = jnp.array(par_sh_np)
        nir_sh_jnp = jnp.array(nir_sh_np)

        # canoak conc
        # print("canoak...")
        canoak.rnet(  # type: ignore
            jtot,
            ir_dn_np,
            ir_up_np,
            par_sun_np,
            nir_sun_np,
            par_sh_np,
            nir_sh_np,
            rnet_sh_np,
            rnet_sun_np,
        )

        # jax conc
        # print("jax...")
        rnet_jit = jax.jit(rnet)
        rnet_sun_jnp, rnet_sh_jnp = rnet_jit(
            ir_dn_jnp,
            ir_up_jnp,
            par_sun_jnp,
            nir_sun_jnp,
            par_sh_jnp,
            nir_sh_jnp,
        )

        # print(rnet_sun_jnp)
        # print(rnet_sun_np)
        # print(dispersion_np)
        print("")
        # print(cncc_jnp[cncc_jnp != cncc_np])
        # print(cncc_np[cncc_jnp != cncc_np])
        self.assertTrue(np.allclose(rnet_sh_np, rnet_sh_jnp))
        self.assertTrue(np.allclose(rnet_sun_np, rnet_sun_jnp))

    def test_par_day(self):
        solar_sine_beta = 0.5
        print("Performing test_par_day()...")
        # inputs
        dLAIdz_np = np.random.random(sze)
        exxpdir_np = np.random.random(sze)
        Gfunc_solar_np = np.random.random(sze)
        sun_lai_np = np.zeros(sze)
        shd_lai_np = np.zeros(sze)
        prob_beam_np = np.zeros(sze)
        prob_sh_np = np.zeros(sze)
        par_up_np = np.zeros(sze)
        par_down_np = np.zeros(sze)
        beam_flux_par_np = np.zeros(sze)
        quantum_sh_np = np.zeros(sze)
        quantum_sun_np = np.zeros(sze)
        par_shade_np = np.zeros(sze)
        par_sun_np = np.zeros(sze)
        # dLAIdz_np = np.random.random(jtot)
        # exxpdir_np = np.random.random(jtot)
        # Gfunc_solar_np = np.random.random(jtot)
        # sun_lai_np = np.zeros(jtot)
        # shd_lai_np = np.zeros(jtot)
        # prob_beam_np = np.zeros(jtot)
        # prob_sh_np = np.zeros(jtot)
        # par_up_np = np.zeros(jtot)
        # par_down_np = np.zeros(jtot)
        # beam_flux_par_np = np.zeros(jtot)
        # quantum_sh_np = np.zeros(jtot)
        # quantum_sun_np = np.zeros(jtot)
        # par_shade_np = np.zeros(jtot)
        # par_sun_np = np.zeros(jtot)

        # canoak
        canoak.par(  # type: ignore
            jtot,
            sze,
            solar_sine_beta,
            parin,
            par_beam,
            par_reflect,
            par_trans,
            par_soil_refl,
            par_absorbed,
            dLAIdz_np,
            exxpdir_np,
            Gfunc_solar_np,
            sun_lai_np,
            shd_lai_np,
            prob_beam_np,
            prob_sh_np,
            par_up_np,
            par_down_np,
            beam_flux_par_np,
            quantum_sh_np,
            quantum_sun_np,
            par_shade_np,
            par_sun_np,
        )

        # jax
        par_jit = jax.jit(par)
        (
            sun_lai_jnp,
            shd_lai_jnp,
            prob_beam_jnp,
            prob_sh_jnp,
            par_up_jnp,
            par_down_jnp,
            beam_flux_par_jnp,
            quantum_sh_jnp,
            quantum_sun_jnp,
            par_shade_jnp,
            par_sun_jnp,
        ) = par_jit(
            solar_sine_beta,
            parin,
            par_beam,
            par_reflect,
            par_trans,
            par_soil_refl,
            par_absorbed,
            jnp.array(dLAIdz_np),
            jnp.array(exxpdir_np),
            jnp.array(Gfunc_solar_np),
        )

        # print(sun_lai_jnp)
        # print(sun_lai_np)
        # print(quantum_sh_jnp)
        # print(quantum_sh_np)
        print("")
        self.assertTrue(np.allclose(sun_lai_jnp, sun_lai_np))
        self.assertTrue(np.allclose(shd_lai_jnp, shd_lai_np))
        self.assertTrue(np.allclose(prob_beam_jnp, prob_beam_np))
        self.assertTrue(np.allclose(prob_sh_jnp, prob_sh_np))
        self.assertTrue(np.allclose(par_up_jnp, par_up_np))
        self.assertTrue(np.allclose(par_down_jnp, par_down_np))
        self.assertTrue(np.allclose(beam_flux_par_jnp, beam_flux_par_np))
        self.assertTrue(np.allclose(quantum_sh_jnp, quantum_sh_np))
        self.assertTrue(np.allclose(quantum_sun_jnp, quantum_sun_np))
        self.assertTrue(np.allclose(par_shade_jnp, par_shade_np))
        self.assertTrue(np.allclose(par_sun_jnp, par_sun_np))

    def test_par_night(self):
        solar_sine_beta = 0.001
        print("Performing test_par_night()...")
        # inputs
        dLAIdz_np = np.random.random(sze)
        exxpdir_np = np.random.random(sze)
        Gfunc_solar_np = np.random.random(sze)
        sun_lai_np = np.zeros(sze)
        shd_lai_np = np.zeros(sze)
        prob_beam_np = np.zeros(sze)
        prob_sh_np = np.zeros(sze)
        par_up_np = np.zeros(sze)
        par_down_np = np.zeros(sze)
        beam_flux_par_np = np.zeros(sze)
        quantum_sh_np = np.zeros(sze)
        quantum_sun_np = np.zeros(sze)
        par_shade_np = np.zeros(sze)
        par_sun_np = np.zeros(sze)

        # canoak
        canoak.par(  # type: ignore
            jtot,
            sze,
            solar_sine_beta,
            parin,
            par_beam,
            par_reflect,
            par_trans,
            par_soil_refl,
            par_absorbed,
            dLAIdz_np,
            exxpdir_np,
            Gfunc_solar_np,
            sun_lai_np,
            shd_lai_np,
            prob_beam_np,
            prob_sh_np,
            par_up_np,
            par_down_np,
            beam_flux_par_np,
            quantum_sh_np,
            quantum_sun_np,
            par_shade_np,
            par_sun_np,
        )

        # jax
        par_jit = jax.jit(par)
        (
            sun_lai_jnp,
            shd_lai_jnp,
            prob_beam_jnp,
            prob_sh_jnp,
            par_up_jnp,
            par_down_jnp,
            beam_flux_par_jnp,
            quantum_sh_jnp,
            quantum_sun_jnp,
            par_shade_jnp,
            par_sun_jnp,
        ) = par_jit(
            solar_sine_beta,
            parin,
            par_beam,
            par_reflect,
            par_trans,
            par_soil_refl,
            par_absorbed,
            jnp.array(dLAIdz_np),
            jnp.array(exxpdir_np),
            jnp.array(Gfunc_solar_np),
        )

        # print(prob_sh_jnp)
        print("")
        self.assertTrue(np.allclose(prob_sh_jnp, prob_sh_np))

    def test_diffuse_direct_radiation(self):
        print("Performing test_rnet()...")
        # Inputs
        solar_sine_beta = 0.1
        rglobal, press_kpa = 24.0, 101.325
        # Assume 50% is PAR, and unit convertion from W m-2 to umol m-2 s-1
        parin = 4.6 * rglobal / 2.0

        # CANOAK
        (
            ratrad,
            par_beam,
            par_diffuse,
            nir_beam,
            nir_diffuse,
        ) = canoak.diffuse_direct_radiation(  # type: ignore
            solar_sine_beta,
            rglobal,
            parin,
            press_kpa,
        )

        # JAX-CANOAK
        diffuse_direct_radiation_jit = jax.jit(diffuse_direct_radiation)
        (
            ratrad_jnp,
            par_beam_jnp,
            par_diffuse_jnp,
            nir_beam_jnp,
            nir_diffuse_jnp,
        ) = diffuse_direct_radiation_jit(
            solar_sine_beta,
            rglobal,
            parin,
            press_kpa,
        )

        # print(ratrad, par_beam, par_diffuse, nir_beam, nir_diffuse)
        # print(ratrad_jnp, par_beam_jnp, par_diffuse_jnp, nir_beam_jnp,nir_diffuse_jnp)
        self.assertTrue(np.allclose(ratrad, ratrad_jnp))
        self.assertTrue(np.allclose(par_beam, par_beam_jnp))
        self.assertTrue(np.allclose(par_diffuse, par_diffuse_jnp))
        self.assertTrue(np.allclose(nir_beam, nir_beam_jnp))
        self.assertTrue(np.allclose(nir_diffuse, nir_diffuse_jnp))

    def test_nir_day(self):
        solar_sine_beta = 0.5
        print("Performing test_nir_day()...")
        # inputs
        dLAIdz_np = np.random.random(sze)
        exxpdir_np = np.random.random(sze)
        Gfunc_solar_np = np.random.random(sze)
        nir_dn_np = np.zeros(sze)
        nir_up_np = np.zeros(sze)
        beam_flux_nir_np = np.zeros(sze)
        nir_sh_np = np.zeros(sze)
        nir_sun_np = np.zeros(sze)

        # canoak
        canoak.nir(  # type: ignore
            jtot,
            sze,
            solar_sine_beta,
            nir_beam,
            nir_diffuse,
            nir_reflect,
            nir_trans,
            nir_soil_refl,
            nir_absorbed,
            dLAIdz_np,
            exxpdir_np,
            Gfunc_solar_np,
            nir_dn_np,
            nir_up_np,
            beam_flux_nir_np,
            nir_sh_np,
            nir_sun_np,
        )

        # JAX
        nir_jit = jax.jit(nir)
        nir_dn_jnp, nir_up_jnp, beam_flux_nir_jnp, nir_sh_jnp, nir_sun_jnp = nir_jit(
            solar_sine_beta,
            nir_beam,
            nir_diffuse,
            nir_reflect,
            nir_trans,
            nir_soil_refl,
            nir_absorbed,
            jnp.array(dLAIdz_np),
            jnp.array(exxpdir_np),
            jnp.array(Gfunc_solar_np),
        )

        # print(nir_dn_jnp, nir_dn_np)
        # print(nir_up_jnp, nir_up_np)
        # print(beam_flux_nir_jnp, beam_flux_nir_np)
        # print(nir_sh_jnp, nir_sh_np)
        # print(nir_sun_jnp, nir_sun_np)
        print("")
        self.assertTrue(np.allclose(nir_dn_jnp, nir_dn_np))
        self.assertTrue(np.allclose(nir_up_jnp, nir_up_np))
        self.assertTrue(np.allclose(beam_flux_nir_jnp, beam_flux_nir_np))
        self.assertTrue(np.allclose(nir_sh_jnp, nir_sh_np))
        self.assertTrue(np.allclose(nir_sun_jnp, nir_sun_np))
        # self.assertTrue(1 == 1)

    def test_nir_night(self):
        solar_sine_beta = 0.0
        print("Performing test_nir_night()...")
        # inputs
        dLAIdz_np = np.random.random(sze)
        exxpdir_np = np.random.random(sze)
        Gfunc_solar_np = np.random.random(sze)
        nir_dn_np = np.zeros(sze)
        nir_up_np = np.zeros(sze)
        beam_flux_nir_np = np.zeros(sze)
        nir_sh_np = np.zeros(sze)
        nir_sun_np = np.zeros(sze)

        # canoak
        canoak.nir(  # type: ignore
            jtot,
            sze,
            solar_sine_beta,
            nir_beam,
            nir_diffuse,
            nir_reflect,
            nir_trans,
            nir_soil_refl,
            nir_absorbed,
            dLAIdz_np,
            exxpdir_np,
            Gfunc_solar_np,
            nir_dn_np,
            nir_up_np,
            beam_flux_nir_np,
            nir_sh_np,
            nir_sun_np,
        )

        # JAX
        nir_jit = jax.jit(nir)
        nir_dn_jnp, nir_up_jnp, beam_flux_nir_jnp, nir_sh_jnp, nir_sun_jnp = nir_jit(
            solar_sine_beta,
            nir_beam,
            nir_diffuse,
            nir_reflect,
            nir_trans,
            nir_soil_refl,
            nir_absorbed,
            jnp.array(dLAIdz_np),
            jnp.array(exxpdir_np),
            jnp.array(Gfunc_solar_np),
        )

        # print(nir_up_np)
        print("")
        self.assertTrue(np.allclose(nir_dn_jnp, nir_dn_np))
        self.assertTrue(np.allclose(nir_up_jnp, nir_up_np))
        self.assertTrue(np.allclose(beam_flux_nir_jnp, beam_flux_nir_np))
        self.assertTrue(np.allclose(nir_sh_jnp, nir_sh_np))
        self.assertTrue(np.allclose(nir_sun_jnp, nir_sun_np))

    def test_sky_ir(self):
        print("Performing test_sky_ir()...")
        # Inputs
        T, ratrad = 285.0, 3.4

        # CANOAK
        radiation = canoak.sky_ir(T, ratrad)  # type: ignore

        # JAX
        sky_ir_jit = jax.jit(sky_ir)
        radiation_jax = sky_ir_jit(T, ratrad)

        print("")
        self.assertTrue(np.allclose(radiation, radiation_jax))

    def test_irflux(self):
        print("Performing test_irflux()...")
        # Inputs
        T_Kelvin, ratrad = 285.0, 3.4
        sfc_temperature = 296.0
        exxpdir_np = np.random.random(sze)
        sun_T_filter_np = np.random.random(sze)
        shd_T_filter_np = np.random.random(sze)
        prob_beam_np = np.random.random(sze)
        prob_sh_np = np.random.random(sze)
        ir_dn_np, ir_up_np = np.zeros(sze), np.zeros(sze)

        # CANOAK
        canoak.irflux(  # type: ignore
            jtot,
            sze,
            T_Kelvin,
            ratrad,
            sfc_temperature,
            exxpdir_np,
            sun_T_filter_np,
            shd_T_filter_np,
            prob_beam_np,
            prob_sh_np,
            ir_dn_np,
            ir_up_np,
        )

        # JAX
        irflux_jit = jax.jit(irflux)
        ir_up_jnp, ir_dn_jnp = irflux_jit(
            T_Kelvin,
            ratrad,
            sfc_temperature,
            jnp.array(exxpdir_np),
            jnp.array(sun_T_filter_np),
            jnp.array(shd_T_filter_np),
            jnp.array(prob_beam_np),
            jnp.array(prob_sh_np),
        )

        # print(ir_dn_np, ir_dn_jnp)
        # print(ir_up_np, ir_up_jnp)
        print("")
        self.assertTrue(np.allclose(ir_up_jnp, ir_up_np))
        self.assertTrue(np.allclose(ir_dn_jnp, ir_dn_np))

    def test_g_func_diffuse(self):
        print("Performing test_g_func_diffuse()...")
        # Inputs
        dLAIdz_np = np.random.random(sze)
        bdens_np = np.zeros(9)
        Gfunc_sky_np = np.zeros([sze, szeang])

        # CANOAK
        canoak.g_func_diffuse(jtot, dLAIdz_np, bdens_np, Gfunc_sky_np)  # type: ignore

        # JAX
        g_func_diffuse_jit = jax.jit(g_func_diffuse)
        Gfunc_sky_jnp = g_func_diffuse_jit(jnp.array(dLAIdz_np))
        # bdens_jnp = freq(lflai)

        # print(Gfunc_sky_np, Gfunc_sky_jnp)
        print("")
        self.assertTrue(np.allclose(Gfunc_sky_jnp, Gfunc_sky_np, atol=1e-04))
