"""
Soil energy balance functions and subroutines, including:
- set_soil()
- soil_energy_balance()
- soil_sfc_resistance()

Author: Peishi Jiang
Date: 2023.07.17.
"""

import jax
import jax.numpy as jnp

from typing import Tuple

from .leaf_energy_balance import llambda, es
from .leaf_energy_balance import desdt as desdt_func
from .leaf_energy_balance import des2dt as des2dt_func
from .turbulence_leaf_boundary_layer import uz
from ...shared_utilities.types import Float_0D, Float_1D
from ...shared_utilities.constants import cp, sigma


def set_soil(
    dt: Float_0D,
    total_t: Float_0D,
    T_soil: Float_1D,
    water_content_15cm: Float_0D,
    soil_T_base: Float_0D,
    air_temp: Float_0D,
    air_density: Float_0D,
    air_density_mole: Float_0D,
    air_press_Pa: Float_0D,
    # ) -> Tuple[int, Float_1D, Float_1D, Float_1D, Float_1D, Float_1D]:
) -> Tuple:
    """Routines, algorithms and parameters for soil moisture were from
       Campbell, G.S. 1985. Soil physics with basic. Elsevier
       updated to algorithms in Campbell and Norman and derived from
       Campbell et al 1994 Soil Science.

       Need to adjust for clay and organic fractions.
       Need to adjust heat capacity and conductivity for peat.

    Args:
        dt (_type_): _description_
        total_t (_type_): _description_
        n_soil (_type_): _description_
        water_content_15cm (_type_): _description_
        soil_T_base (_type_): _description_
        air_temp (_type_): _description_
        air_density (_type_): _description_
        air_density_mole (_type_): _description_
        air_press_Pa (_type_): _description_

    Returns:
        Tuple[int, Float_1D, Float_1D, Float_1D, Float_1D, Float_1D]: _description_
    """
    n_soil = T_soil.size - 1
    # Calculate the number of time steps
    soil_mtime = jnp.array(total_t / dt)
    soil_mtime = soil_mtime.astype(int)
    # Water content of litter. Values ranged between 0.02 and 0.126
    # water_content_litter = 0.0  # assumed constant but needs to vary

    # soil content
    # clay_fraction = 0.3  # Clay fraction
    peat_fraction = 0.129  # SOM = a C; C = 7.5%, a = 1.72
    pore_fraction = 0.687  # from alfalfa, 1 minus ratio bulk density 0.83 g cm-3/2.65 g cm-3, density of solids  # noqa: E501
    mineral_fraction = 0.558  # from bulk density asssuming density of solids is 2.65
    soil_air_fraction = pore_fraction - water_content_15cm

    # J kg-1 K-1, heat capacity
    Cp_water = 4180
    Cp_air = 1065
    Cp_org = 1920
    Cp_mineral = 870

    # W m-1 K-1, thermal conductivity
    K_mineral = 2.5
    K_org = 0.8
    K_water = 0.25

    # Thermal conductivity code from Campbell and Norman
    # terms for Stefan flow as water evaporates in the pores
    fw = 1.0 / (1 + jnp.power((water_content_15cm / 0.15), -4))
    latent = llambda(air_temp + 273.15)
    latent18 = latent * 18.0
    desdt = desdt_func(air_temp + 273.15, latent18)
    K_air = 0.024 + 44100 * 2.42e-5 * fw * air_density_mole * desdt / air_press_Pa
    k_fluid = K_air + fw * (K_water - K_air)
    wt_air = 2 / (3 * (1 + 0.2 * (K_air / k_fluid - 1))) + 1 / (
        3 * (1 + (1 - 2 * 0.2) * (K_air / k_fluid - 1))
    )
    wt_water = 2 / (3 * (1 + 0.2 * (K_water / k_fluid - 1))) + 1 / (
        3 * (1 + (1 - 2 * 0.2) * (K_water / k_fluid - 1))
    )
    wt_mineral = 2 / (3 * (1 + 0.2 * (K_mineral / k_fluid - 1))) + 1 / (
        3 * (1 + (1 - 2 * 0.2) * (K_mineral / k_fluid - 1))
    )
    wt_org = 2 / (3 * (1 + 0.2 * (K_org / k_fluid - 1))) + 1 / (
        3 * (1 + (1 - 2 * 0.2) * (K_org / k_fluid - 1))
    )
    Cp_soil_num = (
        air_density * Cp_air * soil_air_fraction
        + 1000.000 * Cp_water * water_content_15cm
        + 1300.000 * Cp_org * peat_fraction
        + 2650.000 * Cp_mineral * mineral_fraction
    )
    Cp_soil = Cp_soil_num / (
        air_density * soil_air_fraction
        + 1000.000 * water_content_15cm
        + 1300.000 * peat_fraction
        + 2650.000 * mineral_fraction
    )
    K_soil_num = (
        mineral_fraction * wt_mineral * K_mineral
        + soil_air_fraction * wt_air * K_air
        + water_content_15cm * wt_water * K_water
        + peat_fraction * wt_org * K_mineral
    )
    K_soil = K_soil_num / (
        mineral_fraction * wt_mineral
        + soil_air_fraction * wt_air
        + water_content_15cm * wt_water
        + peat_fraction * wt_org
    )

    # Assign soil layers and initial temperatures
    # and compute layer heat capacities and conductivities
    z_soil_0 = 0.0
    soil_bulk_density_0 = 0.0

    def assign_tsoil_bulk(c, i):
        z_soil_up = c
        z_soil_down = z_soil_up + 0.005 * jnp.power(1.5, i - 1)
        t_soil_down = soil_T_base
        soil_b_down = 0.83
        # soil_b_down = jax.lax.cond(
        #     z_soil_down < z_litter,
        #     lambda: 0.074, lambda: 0.83
        # )
        out_each = [z_soil_down, t_soil_down, soil_b_down]
        c_new = z_soil_down
        return c_new, out_each

    _, out = jax.lax.scan(assign_tsoil_bulk, z_soil_0, jnp.arange(n_soil + 1))
    z_soil, T_soil, soil_bulk_density = out[0], out[1], out[2]  # nsoil+1
    z_soil = jnp.concatenate([jnp.array([z_soil_0]), z_soil])  # nsoil+2
    soil_bulk_density = jnp.concatenate(
        [jnp.array([soil_bulk_density_0]), soil_bulk_density]
    )  # nsoil+2

    # Heat capacity and conductivity
    def calculate_cp_k(c, i):
        cp_soil_each = Cp_soil * (z_soil[i + 1] - z_soil[i - 1]) / (2.0 * dt)
        K_soil_each = K_soil / (z_soil[i + 1] - z_soil[i])
        out_each = [cp_soil_each, K_soil_each]
        return c, out_each

    _, out = jax.lax.scan(calculate_cp_k, None, jnp.arange(1, n_soil + 1))
    cp_soil, k_conductivity_soil = out[0], out[1]
    cp_soil = jnp.concatenate([jnp.array([0.0]), cp_soil])  # nsoil+1
    k_conductivity_soil = jnp.concatenate(
        [jnp.array([0.0]), k_conductivity_soil]
    )  # nsoil+1

    return soil_mtime, T_soil, z_soil, soil_bulk_density, cp_soil, k_conductivity_soil


def soil_energy_balance(
    epsoil: Float_0D,
    delz: Float_0D,
    ht: Float_0D,
    wnd: Float_0D,
    air_density: Float_0D,
    air_relative_humidity: Float_0D,
    air_press_Pa: Float_0D,
    air_temp: Float_0D,
    water_content_15cm: Float_0D,
    tsoil_init: Float_0D,
    beam_flux_par_sfc: Float_0D,
    par_down_sfc: Float_0D,
    par_up_sfc: Float_0D,
    beam_flux_nir_sfc: Float_0D,
    nir_dn_sfc: Float_0D,
    nir_up_sfc: Float_0D,
    ir_dn_sfc: Float_0D,
    tair_filter_sfc: Float_0D,
    soil_sfc_temperature: Float_0D,
    rhov_filter_sfc: Float_0D,
    soil_bulk_density_sfc: Float_0D,
    soil_lout: Float_0D,
    soil_heat: Float_0D,
    soil_evap: Float_0D,
    soil_mtime: int,
    iter_step: int,
    k_conductivity_soil: Float_1D,
    cp_soil: Float_1D
    # soil_rnet, soil_lout, soil_evap, soil_heat,
    # py::array_t<double, py::array::c_style> T_soil_np,
    # py::array_t<double, py::array::c_style> k_conductivity_soil_np,
    # py::array_t<double, py::array::c_style> cp_soil_np
) -> Tuple:
    # ) -> Tuple[
    #     Float_0D, Float_0D, Float_0D, Float_0D, Float_0D, Float_1D, Float_1D, Float_1D
    # ]:
    # TODO:
    # n_soil, n_soil_1 = soilsze-3, soilsze-2
    n_soil_1 = k_conductivity_soil.size
    n_soil = n_soil_1 - 1

    # at the soil surface
    # water_content_sfc = 0.0

    # soil surface resistance to water vapor transfer
    # updated and revisited the soil resistance model
    soil_resistance_h2o = soil_sfc_resistance(water_content_15cm)

    # Compute soilevap as a function of energy balance at the soil
    # surface. Net incoming short and longwave energy
    # radiation balance at soil in PAR band, W m-2
    soil_par = (beam_flux_par_sfc + par_down_sfc - par_up_sfc) / 4.6

    # radiation balance at soil in NIR band, W m-2
    soil_nir = beam_flux_nir_sfc + nir_dn_sfc - nir_up_sfc

    # incoming radiation balance at soil, solar and terrestrial, W m-2
    # the Q value of Rin - Rout + Lin
    soil_rnet = soil_par + soil_nir + ir_dn_sfc * epsoil

    # initialize T profile
    soil_T_base = tsoil_init

    # set air temperature over soil with lowest air layer, filtered
    soil_T_air = tair_filter_sfc

    # initialize the soil temperature profile at the lower bound.
    T_soil = jnp.concatenate(
        [jnp.array([soil_T_air]), jnp.zeros(n_soil_1) + soil_T_base]
    )  # nsoil+2

    # Compute Rh_soil and rv_soil from wind log profile for lowest layer
    u_soil = uz(delz, ht, wnd)  # wind speed one layer above soil

    # Stability factor from Daamen and Simmonds
    stabdel = (
        5.0
        * 9.8
        * delz
        * (soil_sfc_temperature - soil_T_air)
        / ((soil_T_air + 273.0) * u_soil * u_soil)
    )
    facstab = jax.lax.cond(
        stabdel > 0,
        lambda: jnp.power(1.0 + stabdel, -0.75),
        lambda: jnp.power(1.0 + stabdel, -2),
    )
    facstab = jax.lax.cond(iter_step <= 1, lambda: 1.0, lambda: facstab)
    facstab = jnp.clip(facstab, a_min=0.1, a_max=5.0)
    Rh_soil = 98.0 * facstab / u_soil
    Rh_soil = jnp.clip(Rh_soil, a_min=5.0, a_max=5000.0)
    Rv_soil = Rh_soil

    # kcsoil is the convective transfer coeff for the soil. (W m-2 K-1)
    kcsoil = (cp * air_density) / Rh_soil

    # soil surface conductance to water vapor transfer
    kv_soil = 1.0 / (Rv_soil + soil_resistance_h2o)

    # Boundary layer conductance at soil surface, W m-2 K-1
    k_conductivity_soil = jnp.concatenate(
        [jnp.array([kcsoil]), k_conductivity_soil[1:]]
    )

    # initialize absolute temperature
    soil_T_Kelvin = soil_T_air + 273.16

    # evaluate latent heat of evaporation at T_soil
    latent = llambda(soil_T_Kelvin)
    latent18 = latent * 18.0

    # evaluate saturation vapor pressure in the energy balance equation
    # it is est of air layer
    est = es(soil_T_Kelvin)  #  es(T) f Pa
    # ea = rhov_filter_sfc * soil_T_Kelvin * 461.89

    # Use a pedo transfer function
    # to convert volumetric water content to matric potential
    # then solve for RH;  psi = R Tk/ Mw ln(RH)
    psi = -12.56 - 12.49 * jnp.log(water_content_15cm / soil_bulk_density_sfc)  # - MPa
    psiPa = -psi * 1000000  # Pa
    rhsoil = jnp.exp(psiPa * 1.805e-5 / (8.314 * soil_T_Kelvin))  # relative humidity
    vpdsoil = (1 - rhsoil) * est  # vpd of the soil
    dest = desdt_func(soil_T_Kelvin, latent18)
    d2est = des2dt_func(soil_T_Kelvin, latent18)

    #  Compute products of absolute air temperature
    tk2 = soil_T_Kelvin * soil_T_Kelvin
    tk3 = tk2 * soil_T_Kelvin
    tk4 = tk3 * soil_T_Kelvin

    # Longwave emission at air temperature, W m-2
    llout = epsoil * sigma * tk4

    # coefficients for latent heat flux density
    lecoef = air_density * 0.622 * latent * kv_soil / air_press_Pa

    # Weighting factors for solving diff eq.
    Fst = 0.6
    Gst = 1.0 - Fst

    # solve by looping through the d[]/dt term of the Fourier
    # heat transfer equation
    # def update_tsoil(c, i):
    def update_tsoil(i, c):
        soil_rnet, soil_lout, soil_heat, soil_evap, soil_sfc_temperature, T_soil = c

        # Define coef for each soil layer
        def calculate_abcd(c, i1):
            ip1, im1 = i1 + 1, i1 - 1
            c_soil = -k_conductivity_soil[i1] * Fst
            a_soil = c_soil
            b_soil = (
                Fst * (k_conductivity_soil[i1] + k_conductivity_soil[im1]) + cp_soil[i1]
            )
            d_soil = (
                Gst * k_conductivity_soil[im1] * T_soil[im1]
                + (
                    cp_soil[i1]
                    - Gst * (k_conductivity_soil[i1] + k_conductivity_soil[im1])
                )
                * T_soil[i1]
                + Gst * k_conductivity_soil[i1] * T_soil[ip1]
            )
            return c, [a_soil, b_soil, c_soil, d_soil]

        _, abcd = jax.lax.scan(calculate_abcd, None, jnp.arange(1, n_soil + 1))
        a_soil, b_soil, c_soil, d_soil = abcd
        d_soil_top = (
            d_soil[0]
            + k_conductivity_soil[0] * T_soil[0] * Fst
            + soil_rnet
            - soil_lout
            - soil_evap
        )
        d_soil_bot = d_soil[-1] + k_conductivity_soil[n_soil] * Fst * T_soil[n_soil_1]
        d_soil = jnp.concatenate(
            [jnp.array([d_soil_top]), d_soil[1:-1], jnp.array([d_soil_bot])]
        )

        mm1 = n_soil - 1

        def update_abcd(c, i2):
            new_b, new_d = c
            ip2 = i2 + 1
            new_c = c_soil[i2] / new_b
            new_d = new_d / new_b
            new_b_next = b_soil[ip2] - a_soil[i2] * new_c
            new_d_next = d_soil[ip2] - a_soil[i2] * new_d
            cnew = [new_b_next, new_d_next]
            return cnew, [new_c, new_b_next, new_d]

        carry_i2, bcd_mm1 = jax.lax.scan(
            update_abcd, [b_soil[0], d_soil[0]], jnp.arange(mm1)
        )  # noqa: E501
        d_soil_bot = carry_i2[1]
        c_soil_mm1, b_soil_mm1, d_soil_mm1 = bcd_mm1
        c_soil = jnp.concatenate([c_soil_mm1, c_soil[-1:]])
        b_soil = jnp.concatenate([b_soil[:1], b_soil_mm1])
        d_soil = jnp.concatenate([d_soil_mm1, jnp.array([d_soil_bot])])
        # jax.debug.print(
        #     "{a}", a=b_soil,
        # )

        T_new_soil_bot = d_soil[-1] / b_soil[-1]

        def update_tsoil_z(c, i3):
            t_soil_dn = c
            t_soil_up = d_soil[i3] - c_soil[i3] * t_soil_dn
            return t_soil_up, t_soil_up

        _, T_new_soil_mm1 = jax.lax.scan(
            update_tsoil_z, T_new_soil_bot, jnp.arange(mm1 - 1, -1, -1)
        )
        T_new_soil = jnp.concatenate(
            [
                jnp.array([T_soil[0]]),
                jnp.flip(T_new_soil_mm1),
                jnp.array([T_new_soil_bot, T_soil[n_soil_1]]),
            ]
        )  # nsoil+2
        # jax.debug.print(
        #     "{a}", a=T_new_soil,
        # )

        # compute soil conductive heat flux density, W m-2
        gsoil = k_conductivity_soil[1] * (T_new_soil[1] - T_new_soil[2])
        storage = cp_soil[1] * (T_new_soil[1] - T_soil[1])
        gsoil += storage

        # test if gsoil is in bounds??
        gsoil = jax.lax.cond(
            (gsoil < -500.0) | (gsoil > 500.0), lambda: 0.0, lambda: gsoil
        )

        # The quadratic coefficients for the solution to
        #   a LE^2 + b LE +c =0
        repeat = kcsoil + 4.0 * epsoil * sigma * tk3
        acoeff = lecoef * d2est / (2.0 * repeat)
        acoef = acoeff
        bcoef = (
            -(repeat)
            - lecoef * dest
            + acoeff * (-2.0 * soil_rnet + 2 * llout + 2 * gsoil)
        )
        ccoef = (
            (repeat) * lecoef * vpdsoil
            + lecoef * dest * (soil_rnet - llout - gsoil)
            + acoeff
            * (
                (soil_rnet * soil_rnet)
                + llout * llout
                + gsoil * gsoil
                - 2.0 * soil_rnet * llout
                - 2.0 * soil_rnet * gsoil
                + 2 * gsoil * llout
            )
        )
        product = bcoef * bcoef - 4 * acoef * ccoef
        # latent energy flux density over soil, W m-2
        soil_evap = jax.lax.cond(
            product >= 0,
            lambda: (-bcoef - jnp.power(product, 0.5)) / (2 * acoef),
            lambda: 0.0,
        )
        # solve for Ts using quadratic solution
        att = 6 * epsoil * sigma * tk2 + d2est * lecoef / 2
        btt = 4 * epsoil * sigma * tk3 + kcsoil + lecoef * dest
        ctt = -soil_rnet + llout + gsoil + lecoef * vpdsoil
        product = btt * btt - 4.0 * att * ctt
        soil_sfc_temperature = jax.lax.cond(
            product >= 0,
            lambda: soil_T_air + (-btt + jnp.sqrt(product)) / (2.0 * att),
            lambda: soil_T_air,
        )
        # Soil surface temperature, K
        soil_T_Kelvin = soil_sfc_temperature + 273.16
        # IR emissive flux density from soil, W m-2
        soil_lout = epsoil * sigma * jnp.power(soil_T_Kelvin, 4)
        # Sensible heat flux density over soil, W m-2
        soil_heat = kcsoil * (T_soil[1] - soil_T_air)

        fluxes = [soil_rnet, soil_lout, soil_heat, soil_evap, soil_sfc_temperature]
        T_soil = T_new_soil

        cnew = fluxes + [T_soil]
        return cnew
        # return cnew, i

    # carry, _ = jax.lax.scan(
    #     update_tsoil,
    #     [soil_rnet, soil_lout, soil_heat, soil_evap, soil_sfc_temperature, T_soil],
    #     xs=None, length=soil_mtime
    # )
    carry = jax.lax.fori_loop(
        0,
        soil_mtime,
        update_tsoil,
        [soil_rnet, soil_lout, soil_heat, soil_evap, soil_sfc_temperature, T_soil],
    )
    soil_rnet, soil_lout, soil_heat, soil_evap, soil_sfc_temperature, T_soil = carry

    return soil_rnet, soil_lout, soil_heat, soil_evap, soil_sfc_temperature, T_soil


def soil_sfc_resistance(wg: Float_0D) -> Float_0D:
    """Calculate the soil surface resistance.

    Args:
        wg (Float_0D): _description_

    Returns:
        _type_: _description_
    """
    # Camillo and Gurney model for soil resistance
    # Rsoil= 4104 (ws-wg)-805, ws=.395, wg=0
    # ws= 0.395
    # wg is at 10 cm, use a linear interpolation to the surface, top cm, mean
    # between 0 and 2 cm
    wg0 = 1.0 * wg / 10.0
    # y=4104.* (0.395-wg0)-805.;

    # model of Kondo et al 1990, JAM for a 2 cm thick soil
    y = 3.0e10 * jnp.power((0.395 - wg0), 16.6)

    return y
