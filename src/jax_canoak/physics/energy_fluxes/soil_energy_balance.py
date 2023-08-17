"""
Soil energy balance functions and subroutines, including:
- soil_energy_balance()

Author: Peishi Jiang
Date: 2023.07.25.
"""

import jax
import jax.numpy as jnp

import equinox as eqx

# from typing import Tuple

from ...subjects import ParNir, Ir, Met, Prof, Para, Soil
from ...subjects.utils import desdt as fdesdt
from ...subjects.utils import des2dt as fdes2dt
from ...subjects.utils import es as fes
from ...subjects.utils import llambda as fllambda


# @eqx.filter_jit
def soil_energy_balance(
    quantum: ParNir,
    nir: ParNir,
    ir: Ir,
    met: Met,
    prof: Prof,
    prm: Para,
    soil: Soil,
    soil_mtime: int,
) -> Soil:
    """The soil energy balance model of Campbell has been adapted to
       compute soil energy fluxes and temperature profiles at the soil
       surface.  The model has been converted from BASIC to C.  We
       also use an analytical version of the soil surface energy
       balance to solve for LE, H and G.

       Combine surface energy balance calculations with soil heat
       transfer model to calculate soil conductive and convective heat
       transfer and evaporation rates.  Here, only the deep temperature
       is needed and G, Hs and LEs can be derived from air temperature
       and energy inputs.

       Soil evaporation models by Kondo, Mafouf et al. and
       Dammond and Simmonds are used. Dammond and Simmonds for example
       have a convective adjustment to the resistance to heat transfer.
       Our research in Oregon and Canada have shown that this consideration
       is extremely important to compute G and Rn_soil correctly.

    Args:
        quantum (ParNir): _description_
        nir (ParNir): _description_
        ir (Ir): _description_
        met (Met): _description_
        prof (Prof): _description_
        prm (Para): _description_
        soil (Soil): _description_

    Returns:
        Soil: _description_
    """
    # ntime = prm.ntime
    ntime = soil.T_soil.shape[0]

    # radiation balance at soil in PAR band, W m-2
    soil_par = (
        quantum.beam_flux[:, 0] + quantum.dn_flux[:, 0] - quantum.up_flux[:, 0]
    ) / 4.6
    # radiation balance at soil in NIR band, W m-2
    soil_nir = nir.beam_flux[:, 0] + nir.dn_flux[:, 0] - nir.up_flux[:, 0]
    # net incoming solar radiation balance at soil  and incoming terrestrial, W m-2
    soil_Qin = soil_par + soil_nir + ir.ir_dn[:, 0] * prm.epsoil

    # set air temperature over soil with lowest air layer, filtered
    soil_T_air = prof.Tair_K[:, 1]

    # Compute Rh_soil and Rv_soil from wind log profile for lowest layer
    u_soil = prof.wind[:, 1]  #  wind speed one layer above soil
    soil_z0 = 0.02  # one millimeter
    Ram_soil = jnp.power(jnp.log(prm.zht[1] / soil_z0), 2.0) / (0.4 * 0.4 * u_soil)
    # Stability factor from Daamen and Simmonds
    stabdel = (
        5.0
        * 9.8
        * (prm.zht[1])
        * (soil.sfc_temperature - soil_T_air)
        / (soil_T_air * u_soil * u_soil)
    )

    @jnp.vectorize
    def calculate_facstab(stabdel_e):
        return jax.lax.cond(
            stabdel_e > 0,
            lambda: jnp.power(1 + stabdel_e, -0.75),
            lambda: jnp.power(1.0 + stabdel_e, -2.0),
        )

    facstab = calculate_facstab(stabdel)
    facstab = jnp.clip(facstab, a_min=0.1, a_max=5.0)
    Rh_soil = Ram_soil * facstab
    Rh_soil = jnp.clip(Rh_soil, a_min=25.0, a_max=1000.0)
    Rv_soil = Rh_soil

    # kcsoil is the convective transfer coeff for the soil. (W m-2 K-1)
    kcsoil = (prm.Cp * met.air_density) / Rh_soil
    # jax.debug.print("kcsoil: {a}", a=kcsoil[10])
    # jax.debug.print("Rh_soil: {a}", a=Rh_soil[10])
    # jax.debug.print("air_density: {a}", a=met.air_density[10])
    # soil surface conductance to water vapor transfer
    kv_soil = 1.0 / (Rv_soil + soil.resistance_h2o)

    # Compute products of absolute air temperature...or Tsoil..check
    # derivation  It is air temperature from the linearization
    tk1 = prof.Tair_K[:, 0]
    tk2 = tk1 * tk1
    tk3 = tk2 * tk1
    # tk4 = tk3 * tk1

    # Slope of the vapor pressure-temperature curve, Pa/C
    # evaluate as function of Tk
    dest = fdesdt(tk1)

    # Second derivative of the vapor pressure-temperature curve, Pa/C
    # Evaluate as function of Tk
    d2est = fdes2dt(tk1)
    fact_latent = fllambda(tk1)
    est = fes(tk1)
    vpdsoil = est - prof.eair_Pa[:, 0]

    # call finite difference routine to solve Fourier Heat Transfer equation for soil
    soil = finite_difference_matrix(soil, prm, soil_mtime)

    # compute storage
    # tmparray=jnp.zeros(prm.ntime)
    # tmparray(2:prm.nn,1) =(soil.T_soil(2:prm.nn,1) - soil.T_soil_old(1:prm.nn-1,1));
    tmparray = jnp.concatenate(
        [
            jnp.zeros([1]),
            soil.T_soil[1:ntime, 0] - soil.T_soil_old[: ntime - 1, 0],
            # soil.T_soil[1 : prm.ntime, 0] - soil.T_soil[: prm.ntime - 1, 0],
        ]
    )

    # soil.gsoil is computed in FiniteDifferenceMatrix
    storage = soil.cp_soil[:, 0] * tmparray
    gsoil = soil.gsoil + storage
    # soil.gsoil = soil.gsoil + storage

    # coefficients for latent heat flux density
    lecoef = met.air_density * 0.622 * fact_latent * kv_soil / (met.P_kPa * 1000)

    # The quadratic coefficients for the solution to
    #   a LE^2 + b LE +c =0
    repeat = kcsoil + 4.0 * prm.epsoil * prm.sigma * tk3
    acoeff = lecoef * d2est / (2.0 * repeat)
    acoef = acoeff
    bcoef = (
        -repeat
        - lecoef * dest
        + acoeff * (-2 * soil_Qin + 2 * soil.llout + 2.0 * gsoil)
    )
    ccoef = (
        repeat * lecoef * vpdsoil
        + lecoef * dest * (soil_Qin - soil.llout - gsoil)
        + acoeff
        * (
            soil_Qin * soil_Qin
            + soil.llout * soil.llout
            + gsoil * gsoil
            - 2 * soil_Qin * soil.llout
            - 2 * soil_Qin * gsoil
            + 2.0 * gsoil * soil.llout
        )
    )
    product = bcoef * bcoef - 4 * acoef * ccoef
    # le1= (-bcoef + jnp.power(product,.5)) / (2*acoef)
    le2 = (-bcoef - jnp.power(product, 0.5)) / (2 * acoef)
    evap = jnp.real(le2)
    # soil.evap = jnp.real(le2)

    # # solve for Ts using quadratic solution
    # att = 6 * prm.epsoil * prm.sigma * tk2 + d2est * lecoef / 2
    # btt = 4 * prm.epsoil * prm.sigma * tk3 + kcsoil + lecoef * dest
    # ctt = -soil_Qin + soil.llout+soil.gsoil + lecoef * vpdsoil
    # product = btt * btt - 4. * att * ctt
    # @jnp.vectorize
    # def calculate_sfc_temp(product_e, T_air_K_e, att_e):
    #     return jax.lax.cond(
    #         product_e > 0,
    #         lambda: T_air_K_e + (-btt + jnp.sqrt(product_e)) / (2. * att_e),
    #         lambda: T_air_K_e
    #     )
    # soil.sfc_temperature = calculate_sfc_temp(product, met.T_air_K, att)
    # soil.T_Kelvin=soil.sfc_temperature
    # soil.T_soil_up_boundary=soil.sfc_temperature
    # soil.del_Tk =soil.sfc_temperature-soil.T_air
    # soil.lout_sfc = prm.epsoil * prm.sigma * jnp.power(soil.sfc_temperature,4)

    # dT = (Q -LE - Gsoil -  ep sigma Ta^4)/( rho Cp gh + 4 ep sigma Ta^3)
    # del_Tk = (soil_Qin - soil.evap - soil.gsoil - soil.llout) / repeat
    del_Tk = (soil_Qin - evap - gsoil - soil.llout) / repeat
    # jax.debug.print("soil Qin: {a}", a=soil_Qin[10])
    # jax.debug.print("soil evap: {a}", a=soil.evap[10])
    # jax.debug.print("soil gsoil: {a}", a=soil.gsoil[10])
    # jax.debug.print("soil llout: {a}", a=soil.llout[10])
    # jax.debug.print("repeat: {a}", a=repeat[10])
    # jax.debug.print("soil T: {a}", a=soil.T_soil[10,:])
    sfc_temperature = soil_T_air + del_Tk
    # soil.T_Kelvin=soil.sfc_temperature
    T_soil_up_boundary = sfc_temperature
    lout_sfc = prm.epsoil * prm.sigma * jnp.power(sfc_temperature, 4)

    # Sensible heat flux density over soil, W m-2
    # soil.heat = del_Tk * kcsoil
    heat = soil_Qin - lout_sfc - evap - gsoil
    rnet = soil_Qin - lout_sfc

    soil = eqx.tree_at(
        lambda t: (
            t.gsoil,
            t.evap,
            t.heat,
            t.rnet,
            t.sfc_temperature,
            t.T_soil_up_boundary,
        ),
        soil,
        (gsoil, evap, heat, rnet, sfc_temperature, T_soil_up_boundary),
    )

    return soil


# @eqx.filter_jit
def finite_difference_matrix(soil: Soil, prm: Para, soil_mtime: int) -> Soil:
    """Convert Soil Physics with Basic from Campbell for soil heat flux

       Using finite difference approach to solve soil heat transfer
       equations, but using geometric spacing due to the exponential decay in
       heat and temperature with depth into the soil

       Soil has N layers and we increment from 1 to N+1

       old code from python and basic based computations on N layers
       from levels 0 to N

       Lower Boundary is level 1, upper Boundary is level N+1

    Args:
        soil (Soil): _description_
        prm (Para): _description_

    Returns:
        Soil: _description_
    """
    # soil_mtime = prm.soil_mtime
    # soil_nsoil = soil.n_soil
    soil_nsoil = soil.dz.size
    # soil_mtime = soil.mtime
    # soil_mtime = 180
    # tolerance = 1.e-2

    # ntime = prm.ntime
    ntime = soil.T_soil.shape[0]

    Fst = 0.6  # (0: explicit, 1: implicit Euler)
    Gst = 1.0 - Fst
    # energyBalance = 1.

    # Looping to solve the Fourier heat transfer equation
    # def update_tsoil(i, c):
    def update_tsoil(c, i):
        T_soil = c
        # Calculate the coef for each soil layers
        # Top boundary conditions: The first soil layer
        a_soil_0, b_soil_0 = jnp.zeros([ntime, 1]), jnp.ones([ntime, 1])
        c_soil_0, d_soil_0 = jnp.zeros([ntime, 1]), soil.T_soil_up_boundary
        d_soil_0 = jnp.expand_dims(d_soil_0, axis=-1)

        @jnp.vectorize
        def calculate_abcd(
            k_cond_up, k_cond_cu, cp_cu, T_soil_up, T_soil_cu, T_soil_dn
        ):
            a_soil_e = -k_cond_up * Fst
            b_soil_e = Fst * (k_cond_cu + k_cond_up) + cp_cu
            c_soil_e = -k_cond_cu * Fst
            d_soil_e = (
                T_soil_up * k_cond_up * Gst
                + T_soil_cu * cp_cu
                - T_soil_cu * k_cond_cu * Gst
                - T_soil_cu * k_cond_up * Gst
                + T_soil_dn * k_cond_cu * Gst
            )
            return a_soil_e, b_soil_e, c_soil_e, d_soil_e

        a_soil, b_soil, c_soil, d_soil = calculate_abcd(
            soil.k_conductivity_soil[:, :-2],
            soil.k_conductivity_soil[:, 1:-1],
            soil.cp_soil[:, 1:],
            T_soil[:, :-3],
            T_soil[:, 1:-2],
            T_soil[:, 2:-1],
        )  # (ntime, nsoil-1) for each
        # Bottom boundary conditions
        c_soil_n = jnp.zeros([ntime, 1])
        d_soil_n = (
            d_soil[:, -1]
            + Fst * soil.k_conductivity_soil[:, soil_nsoil - 1] * soil.T_soil_low_bound
        )  # noqa: E501
        d_soil_n = jnp.expand_dims(d_soil_n, axis=-1)
        a_soil = jnp.concatenate([a_soil_0, a_soil], axis=1)
        b_soil = jnp.concatenate([b_soil_0, b_soil], axis=1)
        c_soil = jnp.concatenate([c_soil_0, c_soil[:, :-1], c_soil_n], axis=1)
        d_soil = jnp.concatenate([d_soil_0, d_soil[:, :-1], d_soil_n], axis=1)
        # jax.debug.print('T_soil: {a}', a=T_soil[3,:])
        # jax.debug.print('b_soil: {a}', a=b_soil[3,:])
        # jax.debug.print('d_soil: {a}', a=d_soil[3,:])

        # Use Thomas algorithm to solve for simultaneous equ
        def update_bd(c2, j):
            b_soil_e, d_soil_e = c2[0], c2[1]
            mm = a_soil[:, j] / b_soil_e
            b_soil_e_new = b_soil[:, j] - mm * c_soil[:, j - 1]
            d_soil_e_new = d_soil[:, j] - mm * d_soil_e
            c2new = [b_soil_e_new, d_soil_e_new]
            return c2new, c2new

        _, out = jax.lax.scan(
            update_bd, [b_soil[:, 0], d_soil[:, 0]], jnp.arange(1, soil_nsoil)
        )
        b_soil_update, d_soil_update = out[0].T, out[1].T
        b_soil = jnp.concatenate([b_soil[:, [0]], b_soil_update], axis=1)
        d_soil = jnp.concatenate([d_soil[:, [0]], d_soil_update], axis=1)

        # Calculate the soil temperature by back substitution
        def calculate_Tsoilnew(c, i):
            Tsoil_dn = c
            Tsoil_cu = (d_soil[:, i] - c_soil[:, i] * Tsoil_dn) / b_soil[:, i]
            cnew = Tsoil_cu
            return cnew, cnew

        _, out = jax.lax.scan(
            calculate_Tsoilnew,
            T_soil[:, soil_nsoil],
            xs=jnp.arange(soil_nsoil - 1, -1, -1),
        )
        T_soil_new = out.T[:, ::-1]  # (ntime, n_soil)
        T_soil_new = jnp.concatenate(
            [T_soil_new, T_soil[:, -2:]], axis=(-1)
        )  # (ntime, n_soil+2)
        # jax.debug.print('a_soil: {a}', a=a_soil[3,:])
        # jax.debug.print('c_soil: {a}', a=c_soil[3,:])
        # jax.debug.print('b_soil: {a}', a=b_soil[3,:])
        # jax.debug.print('d_soil: {a}', a=d_soil[3,:])
        # jax.debug.print('T_soil: {a}', a=T_soil[3,:])
        # jax.debug.print('T_soil_new: {a}', a=T_soil_new[3,:])
        # jax.debug.print('k_cond: {a}', a=soil.k_conductivity_soil[3,:])
        # jax.debug.print('cp: {a}', a=soil.cp_soil[3,:])
        cnew = T_soil_new
        return cnew, None

    # carry = jax.lax.fori_loop(
    #     0,
    #     soil_mtime,
    #     # 3,
    #     update_tsoil,
    #     soil.T_soil,
    # )
    carry, _ = jax.lax.scan(update_tsoil, soil.T_soil, xs=None, length=soil_mtime)

    # Update T_soil
    T_soil = carry

    # Update T_soil_old
    T_soil_old = T_soil

    # Update the soil heat flux
    gsoil = soil.k_conductivity_soil[:, 0] * (T_soil[:, 0] - T_soil[:, 1])

    soil = eqx.tree_at(
        lambda t: (t.T_soil, t.T_soil_old, t.gsoil), soil, (T_soil, T_soil_old, gsoil)
    )

    return soil
