import unittest

import jax
import numpy as np
import jax.numpy as jnp
from jax import config

import canoak  # noqa: E402

from jax_canoak.physics.energy_fluxes import es  # type: ignore
from jax_canoak.physics.energy_fluxes import sfc_vpd  # type: ignore
from jax_canoak.physics.energy_fluxes import llambda  # type: ignore
from jax_canoak.physics.energy_fluxes import desdt  # type: ignore
from jax_canoak.physics.energy_fluxes import des2dt  # type: ignore
from jax_canoak.physics.energy_fluxes import energy_balance_amphi  # type: ignore
from jax_canoak.physics.combined_fluxes import energy_and_carbon_fluxes  # type: ignore

from jax_canoak.shared_utilities.constants import lleaf, nnu  # type: ignore
from jax_canoak.shared_utilities.constants import nuvisc, dh  # type: ignore
from jax_canoak.shared_utilities.constants import dv, dc  # type: ignore

config.update("jax_enable_x64", True)

jtot = 3
jtot3 = 5
sze = jtot + 2
sze3 = jtot3 + 2
soilsze = 12
szeang = 19


class TestLeafEnergy(unittest.TestCase):
    def test_es(self):
        print("Performing test_es()...")
        # Inputs
        tk = 292.0

        # CANOAK
        es1 = canoak.es(tk)  # type: ignore

        # JAX
        es_jit = jax.jit(es)
        es2 = es_jit(tk)

        print("")
        self.assertTrue(np.allclose(es1, es2))

    def test_sfc_vpd(self):
        print("Performing test_sfc_vpd()...")
        # Inputs
        Z, hz, tlk = 1.5, 2.0, 292.0
        delz = hz / jtot
        leleafpt, latent, vapor = 2.3, 1.4, 5.6
        rhov_air_np = np.random.random(sze3)
        ind_z = int(Z / delz) - 1
        rhov_air_z = rhov_air_np[ind_z]

        # CANOAK
        vpd_np = canoak.sfc_vpd(  # type: ignore
            delz, tlk, Z, leleafpt, latent, vapor, rhov_air_np
        )

        # JAX
        sfc_vpd_jit = jax.jit(sfc_vpd)
        vpd_jnp = sfc_vpd_jit(tlk, leleafpt, latent, vapor, rhov_air_z)

        # print(vpd_np, vpd_jnp)
        print("")
        self.assertTrue(np.allclose(vpd_np, vpd_jnp))

    def test_llambda(self):
        print("Performing test_llambda()...")
        # Inputs
        tak = 300.0

        # CANOAK
        lnp = canoak.llambda(tak)  # type: ignore

        # JAX
        llambda_jit = jax.jit(llambda)
        ljnp = llambda_jit(tak)

        # print(lnp, ljnp)
        print("")
        self.assertTrue(np.allclose(lnp, ljnp))

    def test_desdt(self):
        print("Performing test_desdt()...")
        # Inputs
        tk, latent18 = 300.0, 15.4

        # CANOAK
        des_np = canoak.desdt(tk, latent18)  # type: ignore

        # JAX
        desdt_jit = jax.jit(desdt)
        des_jnp = desdt_jit(tk, latent18)

        # print(des_np, des_jnp)
        print("")
        self.assertTrue(np.allclose(des_np, des_jnp))

    def test_des2dt(self):
        print("Performing test_des2dt()...")
        # Inputs
        tk, latent18 = 300.0, 15.4

        # CANOAK
        des2_np = canoak.des2dt(tk, latent18)  # type: ignore

        # JAX
        des2dt_jit = jax.jit(des2dt)
        des2_jnp = des2dt_jit(tk, latent18)

        # print(des2_np, des2_jnp)
        print("")
        self.assertTrue(np.allclose(des2_np, des2_jnp))

    def test_energy_balance_amphi(self):
        print("Performing test_energy_balance_amphi()...")
        # Inputs
        # qrad, taa = 2.3, 300.0
        # rhovva, rvsfc, stomsfc = 0.003, 12.3, 23.45
        # air_density, latent, press_Pa, heat = 1.2, 56.7, 101876., 34.2
        # qrad, taa = 0.7254, 293.3600
        # rhovva, rvsfc, stomsfc = 0.003, 12.3, 23.45
        # air_density, latent, press_Pa, heat = 1.2, 2453736.8000, 101876., 101.6457
        qrad, taa = 0.7254, 293.3600
        rhovva, rvsfc, stomsfc = 0.008, 12.3, 23.45
        air_density, latent, press_Pa, heat = 1.2, 2453736.8000, 101876.0, 44.1750

        # CANOAK
        tlk, le, h, lout = canoak.energy_balance_amphi(  # type: ignore
            qrad, taa, rhovva, rvsfc, stomsfc, air_density, latent, press_Pa, heat
        )

        # JAX
        energy_balance_amphi_jit = jax.jit(energy_balance_amphi)
        tlk_j, le_j, h_j, lout_j = energy_balance_amphi_jit(
            qrad, taa, rhovva, rvsfc, stomsfc, air_density, latent, press_Pa, heat
        )

        # print(tlk, le, h, lout)
        # print(tlk_j, le_j, h_j, lout_j)
        print("")
        self.assertTrue(np.allclose(tlk, tlk_j))
        self.assertTrue(np.allclose(le, le_j))
        self.assertTrue(np.allclose(h, h_j))
        self.assertTrue(np.allclose(lout, lout_j))

    def test_energy_and_carbon_fluxes(self):
        print("Performing test_energy_and_carbon_fluxes()...")
        # Inputs
        wnd, zzz, ht = 1.2, 1.5, 2.0
        delz, grasshof = ht / jtot, 9.8 * np.power(lleaf, 3) / np.power(nnu, 2)
        press_kPa = 101.234
        lai, pai, press_Pa = 4.1, 0.0, press_kPa * 1000.0
        kballstr, co2air = 4.0, 440.0
        rhovva = 0.008
        air_density, pstat273 = 1.2, 102.0
        pr33 = np.power(nuvisc / dh, 0.33)
        sc33 = np.power(nuvisc / dv, 0.33)
        scc33 = np.power(nuvisc / dc, 0.33)

        # tair_filter_np = 278.2 + np.zeros(sze3)
        tair_filter_np = 20.2 + np.zeros(sze3)
        zzz_ht_np = delz * np.arange(1, jtot + 1)
        can_co2_air_np = np.random.random(sze3)
        rhov_air_np = np.random.random(sze3)
        rhov_filter_np = np.random.random(sze3)
        dLAIdz_np = np.random.random(sze)
        prob_beam_np = np.random.random(sze)
        prob_sh_np = np.random.random(sze)
        rnet_sun_np = np.random.random(sze)
        rnet_sh_np = np.random.random(sze)
        quantum_sun_np = np.random.random(sze)
        quantum_sh_np = np.random.random(sze)
        sun_tleaf_np = 20.0 + np.random.random(sze)
        shd_tleaf_np = 20.0 + np.random.random(sze)
        sun_rs_np = np.random.uniform(size=sze)
        shd_rs_np = np.random.uniform(size=sze)
        sun_gs_np = 1.0 / sun_rs_np
        shd_gs_np = 1.0 / shd_rs_np
        sun_resp_np = np.zeros(sze)
        shd_resp_np = np.zeros(sze)
        sun_wj_np = np.zeros(sze)
        shd_wj_np = np.zeros(sze)
        sun_wc_np = np.zeros(sze)
        shd_wc_np = np.zeros(sze)
        sun_A_np = np.zeros(sze)
        shd_A_np = np.zeros(sze)
        sun_rbh_np = np.zeros(sze)
        shd_rbh_np = np.zeros(sze)
        sun_rbv_np = np.zeros(sze)
        shd_rbv_np = np.zeros(sze)
        sun_rbco2_np = np.zeros(sze)
        shd_rbco2_np = np.zeros(sze)
        sun_ci_np = np.zeros(sze)
        shd_ci_np = np.zeros(sze)
        sun_cica_np = np.zeros(sze)
        shd_cica_np = np.zeros(sze)
        dLEdz_np = np.zeros(sze)
        dHdz_np = np.zeros(sze)
        dRNdz_np = np.zeros(sze)
        dPsdz_np = np.zeros(sze)
        Ci_np = np.zeros(sze)
        drbv_np = np.zeros(sze)
        dRESPdz_np = np.zeros(sze)
        dStomCondz_np = np.zeros(sze)

        tair_filter_jnp = jnp.array(tair_filter_np.copy())
        zzz_ht_jnp = jnp.array(zzz_ht_np.copy())
        prob_beam_jnp = jnp.array(prob_beam_np.copy())
        prob_sh_jnp = jnp.array(prob_sh_np.copy())
        rnet_sun_jnp = jnp.array(rnet_sun_np.copy())
        rnet_sh_jnp = jnp.array(rnet_sh_np.copy())
        quantum_sun_jnp = jnp.array(quantum_sun_np.copy())
        quantum_sh_jnp = jnp.array(quantum_sh_np.copy())
        can_co2_air_jnp = jnp.array(can_co2_air_np.copy())
        rhov_air_jnp = jnp.array(rhov_air_np.copy())
        rhov_filter_jnp = jnp.array(rhov_filter_np.copy())
        dLAIdz_jnp = jnp.array(dLAIdz_np.copy())
        sun_rs_jnp = jnp.array(sun_rs_np.copy())
        shd_rs_jnp = jnp.array(shd_rs_np.copy())
        sun_tleaf_jnp = jnp.array(sun_tleaf_np.copy())
        shd_tleaf_jnp = jnp.array(shd_tleaf_np.copy())

        # CANOAK
        canoak.energy_and_carbon_fluxes(  # type: ignore
            jtot,
            delz,
            zzz,
            ht,
            grasshof,
            press_kPa,
            co2air,
            wnd,
            pr33,
            sc33,
            scc33,
            rhovva,
            air_density,
            press_Pa,
            lai,
            pai,
            pstat273,
            kballstr,
            # Input arrays
            tair_filter_np,
            zzz_ht_np,
            prob_beam_np,
            prob_sh_np,
            rnet_sun_np,
            rnet_sh_np,
            quantum_sun_np,
            quantum_sh_np,
            can_co2_air_np,
            rhov_air_np,
            rhov_filter_np,
            dLAIdz_np,
            # Output arrays
            sun_rs_np,
            shd_rs_np,
            sun_gs_np,
            shd_gs_np,
            sun_tleaf_np,
            shd_tleaf_np,
            sun_resp_np,
            shd_resp_np,
            sun_wj_np,
            shd_wj_np,
            sun_wc_np,
            shd_wc_np,
            sun_A_np,
            shd_A_np,
            sun_rbh_np,
            shd_rbh_np,
            sun_rbv_np,
            shd_rbv_np,
            sun_rbco2_np,
            shd_rbco2_np,
            sun_ci_np,
            shd_ci_np,
            sun_cica_np,
            shd_cica_np,
            dLEdz_np,
            dHdz_np,
            dRNdz_np,
            dPsdz_np,
            Ci_np,
            drbv_np,
            dRESPdz_np,
            dStomCondz_np,
        )

        # JAX
        energy_and_carbon_fluxes_jit = jax.jit(energy_and_carbon_fluxes)
        (
            sun_rs_jnp,
            shd_rs_jnp,
            sun_gs_jnp,
            shd_gs_jnp,
            sun_resp_jnp,
            shd_resp_jnp,
            sun_wj_jnp,
            shd_wj_jnp,
            sun_wc_jnp,
            shd_wc_jnp,
            sun_A_jnp,
            shd_A_jnp,
            sun_rbh_jnp,
            shd_rbh_jnp,
            sun_rbv_jnp,
            shd_rbv_jnp,
            sun_rbco2_jnp,
            shd_rbco2_jnp,
            sun_ci_jnp,
            shd_ci_jnp,
            sun_cica_jnp,
            shd_cica_jnp,
            sun_tleaf_jnp,
            shd_tleaf_jnp,
            dLEdz_jnp,
            dHdz_jnp,
            dRNdz_jnp,
            dPsdz_jnp,
            Ci_jnp,
            drbv_jnp,
            dRESPdz_jnp,
            dStomCondz_jnp,
        ) = energy_and_carbon_fluxes_jit(
            ht,
            grasshof,
            press_kPa,
            co2air,
            wnd,
            pr33,
            sc33,
            scc33,
            rhovva,
            air_density,
            lai,
            pai,
            pstat273,
            kballstr,
            tair_filter_jnp,
            zzz_ht_jnp,
            prob_beam_jnp,
            prob_sh_jnp,
            rnet_sun_jnp,
            rnet_sh_jnp,
            quantum_sun_jnp,
            quantum_sh_jnp,
            can_co2_air_jnp,
            rhov_air_jnp,
            rhov_filter_jnp,
            dLAIdz_jnp,
            sun_rs_jnp,
            shd_rs_jnp,
            sun_tleaf_jnp,
            shd_tleaf_jnp,
        )

        # print(shd_rs_jnp, shd_rs_np)
        # print(sun_rs_jnp, sun_rs_np)
        # print(sun_tleaf_jnp, sun_tleaf_np)
        # print(shd_tleaf_jnp, shd_tleaf_np)
        # print(rnet_sun_jnp, rnet_sun_np)
        print("")
        self.assertTrue(np.allclose(sun_tleaf_jnp, sun_tleaf_np[:jtot]))
        self.assertTrue(np.allclose(shd_tleaf_jnp, shd_tleaf_np[:jtot]))
        self.assertTrue(np.allclose(sun_rs_jnp, sun_rs_np[:jtot]))
        self.assertTrue(np.allclose(shd_rs_jnp, shd_rs_np[:jtot]))
        self.assertTrue(np.allclose(sun_gs_jnp, sun_gs_np[:jtot]))
        self.assertTrue(np.allclose(shd_gs_jnp, shd_gs_np[:jtot]))
        self.assertTrue(np.allclose(sun_resp_jnp, sun_resp_np[:jtot]))
        self.assertTrue(np.allclose(shd_resp_jnp, shd_resp_np[:jtot]))
        self.assertTrue(np.allclose(sun_wj_jnp, sun_wj_np[:jtot]))
        self.assertTrue(np.allclose(shd_wj_jnp, shd_wj_np[:jtot]))
        self.assertTrue(np.allclose(sun_wc_jnp, sun_wc_np[:jtot]))
        self.assertTrue(np.allclose(shd_wc_jnp, shd_wc_np[:jtot]))
        self.assertTrue(np.allclose(sun_A_jnp, sun_A_np[:jtot]))
        self.assertTrue(np.allclose(shd_A_jnp, shd_A_np[:jtot]))
        self.assertTrue(np.allclose(sun_rbh_jnp, sun_rbh_np[:jtot]))
        self.assertTrue(np.allclose(shd_rbh_jnp, shd_rbh_np[:jtot]))
        self.assertTrue(np.allclose(sun_rbv_jnp, sun_rbv_np[:jtot]))
        self.assertTrue(np.allclose(shd_rbv_jnp, shd_rbv_np[:jtot]))
        self.assertTrue(np.allclose(sun_rbco2_jnp, sun_rbco2_np[:jtot]))
        self.assertTrue(np.allclose(shd_rbco2_jnp, shd_rbco2_np[:jtot]))
        self.assertTrue(np.allclose(sun_ci_jnp, sun_ci_np[:jtot]))
        self.assertTrue(np.allclose(shd_ci_jnp, shd_ci_np[:jtot]))
        self.assertTrue(np.allclose(sun_cica_jnp, sun_cica_np[:jtot]))
        self.assertTrue(np.allclose(shd_cica_jnp, shd_cica_np[:jtot]))
        self.assertTrue(np.allclose(dLEdz_jnp, dLEdz_np[:jtot]))
        self.assertTrue(np.allclose(dHdz_jnp, dHdz_np[:jtot]))
        self.assertTrue(np.allclose(dRNdz_jnp, dRNdz_np[:jtot]))
        self.assertTrue(np.allclose(dPsdz_jnp, dPsdz_np[:jtot]))
        self.assertTrue(np.allclose(Ci_jnp, Ci_np[:jtot]))
        self.assertTrue(np.allclose(drbv_jnp, drbv_np[:jtot]))
        self.assertTrue(np.allclose(dRESPdz_jnp, dRESPdz_np[:jtot]))
        self.assertTrue(np.allclose(dStomCondz_jnp, dStomCondz_np[:jtot]))
