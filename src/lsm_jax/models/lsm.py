"""Implementation of the column-based land surface model."""
# TODO: Revise the calculation of the actual ET ratio
# TODO: Enrich the documentation

import jax
import jax.numpy as jnp

from functools import partial
from typing import List
from ..types import Float_1D, Float_0D
# from jaxtyping import Array, Float, Int
# from typeguard import typechecked

import equinox as eqx
from equinox.module import static_field

from ..physics.constants import L as l
from ..physics.constants import C_TO_K as c2k
from ..physics.constants import PI as pi
from ..physics.constants import RHO_WATER as rho_water

from ..physics import Surface
from ..physics import Soil
from ..physics import Canopy, calculate_canopy_resistance

from ..physics import van_genuchten_model
from ..physics import calculate_infiltration_greenampt
from ..physics import calculate_evaporation_pm, calculate_evaporation_p
from ..physics import calculate_net_radiation, calculate_sensible_heat_flux
from ..physics import calculate_rhs_surface_temperature

from ..data import PointData

class LSMVectorMassOnly(eqx.Module):

    soil: Soil
    canopy: Canopy
    fbs: Float_0D
    fv: Float_0D
    xit: Float_0D

    def __init__(
        self,
        theta_sat_sl: Float_0D, theta_sat_fr: Float_0D,
        theta_r_sl: Float_0D, theta_r_fr: Float_0D,
        ksat_sl: Float_0D, ksat_fr: Float_0D,
        alpha_sl: Float_0D, alpha_fr: Float_0D,
        n_sl: Float_0D, n_fr: Float_0D,
        rsmin: Float_0D, theta_wp: Float_0D, theta_lim: Float_0D,
        tamin: Float_0D, tamax: Float_0D, taopt: Float_0D, w: Float_0D,
        dh: Float_0D, zh: Float_0D, zm: Float_0D, zoh: Float_0D, zom: Float_0D,
        fbs: Float_0D, fv: Float_0D, xit: Float_0D, dsl: Float_0D, dfr: Float_0D,
        **kwargs
    ) -> None:
    
        super().__init__(**kwargs)
        self.fbs, self.fv  = jnp.array(fbs), jnp.array(fv)
        self.xit           = jnp.array(xit)

        self.soil = Soil(
            theta_sat=jnp.array([theta_sat_sl, theta_sat_fr]),
            theta_r=jnp.array([theta_r_sl, theta_r_fr]),
            theta_wp=jnp.array([theta_wp]*2),
            theta_lim=jnp.array([theta_lim]*2),
            ksat=jnp.array([ksat_sl, ksat_fr]),
            alpha=jnp.array([alpha_sl, alpha_fr]),
            n=jnp.array([n_sl, n_fr]),
            depths=jnp.array([dsl, dfr]),
            nsoil=2,
        )

        self.canopy = Canopy(
            rsmin=jnp.array(rsmin), tamin=jnp.array(tamin), tamax=jnp.array(tamax), taopt=jnp.array(taopt), w=jnp.array(w),
            dh=jnp.array(dh), zh=jnp.array(zh), zm=jnp.array(zm), zoh=jnp.array(zoh), zom=jnp.array(zom),
        )

    # @partial(jax.jit, static_argnums=[3])
    # @partial(jax.jit, static_argnums=[3])
    def __call__(
        self,
        t: Float_0D,
        y: Float_1D,
        forcings: PointData,
    ) -> Float_1D:
        theta_sl, theta_fr, cew, ci = y

        # Get the atmospheric forcings
        # p    = forcings['P'].evaluate(t)      # precipitation [m]
        # g    = forcings['G'].evaluate(t)      # ground heat flux [MJ m-2 day-1]
        # rn   = forcings['NETRAD'].evaluate(t) # net radiation [MJ m-2 day-1]
        # vpd  = forcings['VPD'].evaluate(t)    # vapor pressure deficit [kPa]
        # ta   = forcings['TA'].evaluate(t)     # air temperature [degC]
        # uz   = forcings['WS'].evaluate(t)     # wind speed [m s-1]
        # lai  = forcings['LAI'].evaluate(t)    # leaf area index [-]
        # dt_obs = forcings['dt']               # the resolution of dt
        # p    = forcings.evaluate('P', t)      # precipitation [m]
        # g    = forcings.evaluate('G', t)      # ground heat flux [MJ m-2 day-1]
        # rn   = forcings.evaluate('NETRAD', t) # net radiation [MJ m-2 day-1]
        # vpd  = forcings.evaluate('VPD', t)    # vapor pressure deficit [kPa]
        # ta   = forcings.evaluate('TA', t)     # air temperature [degC]
        # uz   = forcings.evaluate('WS', t)     # wind speed [m s-1]
        # lai  = forcings.evaluate('LAI', t)    # leaf area index [-]
        # dt_obs   = forcings.dt               # the resolution of dt

        # TODO: Check whether the indices of these forcings in forcings.varn_list
        # p  : precipitation [m]
        # g  : ground heat flux [MJ m-2 day-1]
        # rn : net radiation [MJ m-2 day-1]
        # vpd: vapor pressure deficit [kPa]
        # ta # air temperature [degC]
        # uz : wind speed [m s-1]
        # lai: leaf area index [-]
        f = forcings.interpolate_time(t)
        p, rn, g, vpd = f[0], f[1], f[2], f[3]
        ta, uz, lai   = f[4], f[5], f[-1]
        # jax.debug.print("forcings: {}", f)
        dt_obs = forcings.ts[1] - forcings.ts[0]

        # Correct theta_sl and theta_fr so that they are within their saturated and residual values
        theta_sl = jnp.min(jnp.array([theta_sl, self.soil.theta_sat[0]]))
        theta_fr = jnp.min(jnp.array([theta_fr, self.soil.theta_sat[1]]))
        theta_sl = jnp.max(jnp.array([theta_sl, self.soil.theta_r[0]]))
        theta_fr = jnp.max(jnp.array([theta_fr, self.soil.theta_r[1]]))

        # Water mass change of wet canopy evaporation or interception, ew
        # The canopy interception can't exceed the maximum threshold
        cewmax = 0.2 * (lai) * 0.001     # [m]
        cew    = jnp.min(jnp.array([cew, cewmax])) # [m]
        # The dt_obs here refers to the rainfall resolution
        ew     = (jnp.min(jnp.array([cew+p*dt_obs, cewmax])) - cew) / dt_obs  #[m d-1]

        # Compute throughfall
        tf = jnp.max(jnp.array([p-ew, 0]))   # [m d-1]

        # Compute the soil water potential and hydraulic conductivity in the two layers 
        # using soil water retention curve model
        psi_sl, k_sl = van_genuchten_model(
            theta_sl, self.soil.theta_sat[0], self.soil.theta_r[0], 
            self.soil.ksat[0], self.soil.alpha[0], self.soil.n[0]
        )
        psi_fr, k_fr = van_genuchten_model(
            theta_fr, self.soil.theta_sat[1], self.soil.theta_r[1], 
            self.soil.ksat[1], self.soil.alpha[1], self.soil.n[1]
        )

        # Compute the infiltration term, i
        # istar = self.infiltration(ci, k_sl, psi_sl, theta_sl)  # [m d-1]
        istar = calculate_infiltration_greenampt(ci, k_sl, psi_sl, theta_sl, self.soil.theta_sat[0])  # [m d-1]
        i     = jnp.min(jnp.array([tf, istar]))                # [m d-1]

        # Compute bare soil evaporation, ebs
        # TODO: use soil temperature
        alpha_sl = -9.815*theta_sl**3+9.173*theta_sl**2-0.082*theta_sl  # Parlange et al. 1999
        e_bs  = alpha_sl * calculate_evaporation_p(rn, ta, uz, vpd)     # [m d-1]

        # Compute canopy resistance
        rc_sl = calculate_canopy_resistance(
            lai, theta_sl, ta, vpd, 
            rsmin=self.canopy.rsmin, theta_wp=self.soil.theta_wp[0], theta_lim=self.soil.theta_lim[0], 
            tamin=self.canopy.tamin, tamax=self.canopy.tamax, taopt=self.canopy.taopt, w=self.canopy.w
        )
        rc_fr = calculate_canopy_resistance(
            lai, theta_fr, ta, vpd, 
            rsmin=self.canopy.rsmin, theta_wp=self.soil.theta_wp[1], theta_lim=self.soil.theta_lim[1], 
            tamin=self.canopy.tamin, tamax=self.canopy.tamax, taopt=self.canopy.taopt, w=self.canopy.w
        )

        # Compute vegetation transpiration
        # TODO: computation of actual ET
        alpha_fr = -9.815*theta_fr**3+9.173*theta_fr**2-0.082*theta_fr  # Parlange et al. 1999
        et_sl = alpha_sl * calculate_evaporation_pm(
            rn, g, ta, uz, vpd, rc_sl, 
            self.canopy.dh, self.canopy.zh, self.canopy.zm, self.canopy.zoh, self.canopy.zom) # [m d-1]
        et_fr = alpha_fr * calculate_evaporation_pm(
            rn, g, ta, uz, vpd, rc_fr,
            self.canopy.dh, self.canopy.zh, self.canopy.zm, self.canopy.zoh, self.canopy.zom) # [m d-1]

        # Compute the hydraulic redistribution, hr
        hr = 3.3e-9 * (psi_fr-psi_sl) / self.soil.depths[1] * 86400.   # [m d-1], Montaldo et al (2021)

        # Compute the drainage term, dr, and the leakage term, le
        dr = k_sl  # [m d-1]
        le = k_fr  # [m d-1]

        # Water mass change of surface soil layer
        dtheta_sl = (
            i - self.fbs*e_bs - self.fv*self.xit*et_sl - dr - hr
            # fvg*(1-xigrh)*egnrh - ew - dr - hr
        ) / self.soil.depths[0]

        # Water mass change of fractured rock sublayer
        dtheta_fr = (
            dr - self.fv*(1-self.xit)*et_fr + hr -le
        ) / self.soil.depths[1]

        out = jnp.array([dtheta_sl, dtheta_fr, ew, i])

        return out


class LSMVectorMassEnergy(eqx.Module):

    soil        : Soil
    surface     : Surface
    # canopy      : Canopy
    # forcing_list: List[str] = static_field
    # albedo      : Float_0D
    # emissivity  : Float_0D
    # fbs         : Float_0D = static_field()
    # fv          : Float_0D = static_field()
    # xit         : Float_0D = static_field()

    def __init__(
        self,
        # forcing_list: List[str],
        albedo      : Float_0D,  emissivity  : Float_0D,
        theta_sat_sl: Float_0D,  theta_sat_fr: Float_0D,
        theta_r_sl  : Float_0D,  theta_r_fr  : Float_0D,
        ksat_sl     : Float_0D,  ksat_fr     : Float_0D,
        alpha_sl    : Float_0D,  alpha_fr    : Float_0D,
        n_sl        : Float_0D,  n_fr        : Float_0D,
        kthermal    : Float_0D,  rhos        : Float_0D, cs       : Float_0D,
        rsmin       : Float_0D,  theta_wp    : Float_0D, theta_lim: Float_0D,
        tamin       : Float_0D,  tamax       : Float_0D, taopt    : Float_0D, w  : Float_0D,
        dh          : Float_0D,  zh          : Float_0D, zm       : Float_0D, zoh: Float_0D, zom: Float_0D,
        fbs         : Float_0D,  fv          : Float_0D, xit      : Float_0D, dsl: Float_0D, dfr: Float_0D,
        **kwargs
    ) -> None:
        """RHS of a column-based land surface model for simulating hydrological process with mass/energy balance.

        Args:
            forcing_list (List[str]): A list of forcing variables.
            albedo (Float_0D): Albeda [-]
            emissivity (Float_0D): Emissivity [-]
            theta_sat_sl (Float_0D): Saturated volumetric soil water content in the soil surface layer [-]
            theta_sat_fr (Float_0D): Saturated volumetric soil water content in the fractured rock layer [-]
            theta_r_sl (Float_0D): Residual soil moisture in the soil surface layer [-] 
            theta_r_fr (Float_0D): Residual soil moisture in the fractured rock layer [-] 
            ksat_sl (Float_0D): Saturated hydraulic conductivity in the soil surface layer [m d-1]
            ksat_fr (Float_0D):Saturated hydraulic conductivity in the fractured rock layer [m d-1] 
            alpha_sl (Float_0D): Van Genuchten parameter alpha in the soil surface layer [-]
            alpha_fr (Float_0D): Van Genuchten parameter alpha in the fractured rock layer [-]
            n_sl (Float_0D):  Van Genuchten parameter n in the soil surface layer [-]
            n_fr (Float_0D):  Van Genuchten parameter n in the fractured rock layer [-]
            kthermal (Float_0D): Soil thermal conductivity [MJ m-1 d-1 degC-1 or MJ m-1 d-1 degK-1]
            rhos (Float_0D): Soil density [kg m-3]
            cs (Float_0D): Soil heat capacity [MJ kg-1 degC-1 or MJ kg-1 degK-1]
            rsmin (Float_0D): Minimum stomatal resistance [s m-1]
            theta_wp (Float_0D): Wilting point [-]
            theta_lim (Float_0D): Limiting soil moisture for vegetation [-]
            tamin (Float_0D): Minimum temperature [degC]
            tamax (Float_0D): Maximum temperature [degC]
            taopt (Float_0D): Optimal temperature [degC]
            w (Float_0D): Slope of the f3 relation [kPa-1]
            dh (Float_0D):  Zero plane displacement height [m]
            zh (Float_0D): Height of the humidity measurements [m]
            zm (Float_0D): Height of the wind speed measurements [m]
            zoh (Float_0D): Roughness length governing transfer of heat and vapour [m]
            zom (Float_0D): Roughness length governing momentum transfer [m]
            fbs (Float_0D): Fraction of the bare soil [-]
            fv (Float_0D): Fraction of the vegetation [-]
            xit (Float_0D): Percentage of vegetation ET from the surface layer [-]
            dsl (Float_0D): Depth of the soil surface layer [m]
            dfr (Float_0D): Depth of the fractured rock layer [m]
        """
    
        super().__init__(**kwargs)
        # self.fbs   , self.fv         = jnp.array(fbs), jnp.array(fv)
        # self.albedo, self.emissivity = jnp.array(albedo), jnp.array(emissivity)
        # self.xit           = jnp.array(xit)

        self.soil = Soil(
            theta_sat=jnp.array([theta_sat_sl, theta_sat_fr]),
            theta_r=jnp.array([theta_r_sl, theta_r_fr]),
            theta_wp=jnp.array([theta_wp]*2),
            theta_lim=jnp.array([theta_lim]*2),
            ksat=jnp.array([ksat_sl, ksat_fr]),
            alpha=jnp.array([alpha_sl, alpha_fr]),
            n=jnp.array([n_sl, n_fr]),
            kthermal=jnp.array(kthermal),
            # rho=jnp.array(rhos),
            # cs=jnp.array(cs),
            # Below are fixed parameters
            rho=rhos,
            cs=cs,
            depths=[dsl, dfr],
            nsoil=2,
        )

        # self.canopy = Canopy(
        canopy = Canopy(
            rsmin=jnp.array(rsmin), tamin=jnp.array(tamin), tamax=jnp.array(tamax), taopt=jnp.array(taopt), w=jnp.array(w),
            dh=jnp.array(dh), zh=jnp.array(zh), zm=jnp.array(zm), zoh=jnp.array(zoh), zom=jnp.array(zom),
        )

        self.surface = Surface(
            albedo=jnp.array(albedo), emissivity=jnp.array(emissivity),
            canopy=canopy,
            # fbs=jnp.array(fbs), fv=jnp.array(fv), xit=jnp.array(xit)
            fbs=float(fbs), fv=float(fv), xit=float(xit)
        )

    def __call__(
        self,
        t: Float_0D,
        y: Float_1D,
        forcings: PointData,
    ) -> Float_1D:
        out, aux = calculate_lsm_vector_mass_energy(t, y, self.soil, self.surface, forcings)
        return out


# Function for calculating the vector field of a column-based land surface model
def calculate_lsm_vector_mass_energy(
    t: Float_0D,
    y: Float_1D,
    soil: Soil,
    surface: Surface,
    forcings: PointData,
):
    theta_sl, theta_fr, cew, ci, ts, ts_avg = y

    # Get the atmospheric forcings
    # p    : precipitation [m]
    # sw_in: incoming short-wave radiation [MJ m-2 day-1]
    # lw_in: incoming long-wave radiation [MJ m-2 day-1]
    # rn   : net radiation [MJ m-2 day-1]
    # vpd  : vapor pressure deficit [kPa]
    # ta   : air temperature [degC]
    # uz   : wind speed [m s-1]
    # lai  : leaf area index [-]
    f      = forcings.interpolate_time(t)
    p    , vpd   = f[forcings.varn_list.index('P')],     f[forcings.varn_list.index('VPD')]
    sw_in, lw_in = f[forcings.varn_list.index('SW_IN')], f[forcings.varn_list.index('LW_IN')]
    ta   , uz    = f[forcings.varn_list.index('TA')], f[forcings.varn_list.index('WS')]
    lai          = f[forcings.varn_list.index('LAI')]
    # jax.debug.print("forcings: {}", f)
    dt_obs = forcings.ts[1] - forcings.ts[0]

    # Correct theta_sl and theta_fr so that they are within their saturated and residual values
    theta_sl = jnp.min(jnp.array([theta_sl, soil.theta_sat[0]]))
    theta_fr = jnp.min(jnp.array([theta_fr, soil.theta_sat[1]]))
    theta_sl = jnp.max(jnp.array([theta_sl, soil.theta_r[0]]))
    theta_fr = jnp.max(jnp.array([theta_fr, soil.theta_r[1]]))

    # Calculate the net radiation
    ts_k = ts + c2k  # conversion from [degC] to [degK]
    rn   = calculate_net_radiation(sw_in, lw_in, ts_k, surface.albedo, surface.emissivity) # [MJ m-2 d-1]
    # jax.debug.print('Net rad components: {}', jnp.array([sw_in, lw_in, ts_k, surface.albedo, surface.emissivity]))
    # jax.debug.print('Energy components: {}', jnp.array([rn]))

    # Calculate the sensible heat flux
    sh = calculate_sensible_heat_flux(ts, ta, uz, surface.canopy.zm, surface.canopy.zom) # [MJ m-2 d-1]
    # jax.debug.print('Sensible heat: {}', jnp.array([sh]))

    # Water mass change of wet canopy evaporation or interception, ew
    # The canopy interception can't exceed the maximum threshold
    cewmax = 0.2 * (lai) * 0.001     # [m]
    cew    = jnp.min(jnp.array([cew, cewmax])) # [m]
    # The dt_obs here refers to the rainfall resolution
    ew     = (jnp.min(jnp.array([cew+p*dt_obs, cewmax])) - cew) / dt_obs  #[m d-1]

    # Compute throughfall
    tf = jnp.max(jnp.array([p-ew, 0]))   # [m d-1]

    # Compute the soil water potential and hydraulic conductivity in the two layers 
    # using soil water retention curve model
    psi_sl, k_sl = van_genuchten_model(
        theta_sl, soil.theta_sat[0], soil.theta_r[0], 
        soil.ksat[0], soil.alpha[0], soil.n[0]
    )
    psi_fr, k_fr = van_genuchten_model(
        theta_fr, soil.theta_sat[1], soil.theta_r[1], 
        soil.ksat[1], soil.alpha[1], soil.n[1]
    )

    # Compute the infiltration term, i
    istar = calculate_infiltration_greenampt(ci, k_sl, psi_sl, theta_sl, soil.theta_sat[0])  # [m d-1]
    i     = jnp.min(jnp.array([tf, istar]))                # [m d-1]

    # Compute bare soil evaporation, ebs
    # TODO: use soil temperature
    alpha_bs = -9.815*theta_sl**3+9.173*theta_sl**2-0.082*theta_sl  # Parlange et al. 1999
    e_bs     = alpha_bs * calculate_evaporation_p(rn, ta, uz, vpd)     # [m d-1]

    # Compute canopy resistance
    rc_sl = calculate_canopy_resistance(
        lai, theta_sl, ta, vpd, 
        rsmin=surface.canopy.rsmin, theta_wp=soil.theta_wp[0], theta_lim=soil.theta_lim[0], 
        tamin=surface.canopy.tamin, tamax=surface.canopy.tamax, taopt=surface.canopy.taopt, w=surface.canopy.w
    )
    rc_fr = calculate_canopy_resistance(
        lai, theta_fr, ta, vpd, 
        rsmin=surface.canopy.rsmin, theta_wp=soil.theta_wp[1], theta_lim=soil.theta_lim[1], 
        tamin=surface.canopy.tamin, tamax=surface.canopy.tamax, taopt=surface.canopy.taopt, w=surface.canopy.w
    )

    # Compute vegetation transpiration
    # TODO: computation of actual ET
    # alpha_fr = -9.815*theta_fr**3+9.173*theta_fr**2-0.082*theta_fr  # Parlange et al. 1999
    # alpha_gv_sl = -9.815*theta_sl**3+9.173*theta_sl**2-0.082*theta_sl  # Parlange et al. 1999
    alpha_gv_sl = (theta_sl - soil.theta_wp[0]) / (soil.theta_lim[0] - soil.theta_wp[0])
    # et_sl = alpha_sl * calculate_evaporation_pm(
    et_sl = alpha_gv_sl * calculate_evaporation_pm(
        rn, 0, ta, uz, vpd, rc_sl, 
        surface.canopy.dh, surface.canopy.zh, surface.canopy.zm, surface.canopy.zoh, surface.canopy.zom) # [m d-1]
    # et_fr = alpha_fr * calculate_evaporation_pm(
    # alpha_gv_fr = -9.815*theta_fr**3+9.173*theta_fr**2-0.082*theta_fr  # Parlange et al. 1999
    alpha_gv_fr = (theta_fr - soil.theta_wp[1]) / (soil.theta_lim[1] - soil.theta_wp[1])
    et_fr = alpha_gv_fr * calculate_evaporation_pm(
        rn, 0, ta, uz, vpd, rc_fr,
        surface.canopy.dh, surface.canopy.zh, surface.canopy.zm, surface.canopy.zoh, surface.canopy.zom) # [m d-1]
    
    # Convert the et components to zero if negative
    # jax.debug.print('ET proportion coefficients: {}', jnp.array([alpha_bs, alpha_gv_sl, alpha_gv_fr]))
    # jax.debug.print('ET components (before zeroing): {}', jnp.array([e_bs, et_fr, et_sl]))
    et_sl = jnp.max(jnp.array([0, et_sl]))
    et_fr = jnp.max(jnp.array([0, et_fr]))
    e_bs  = jnp.max(jnp.array([0, e_bs]))
    
    # Calculate the total ET and latent heat flux
    total_et = surface.fbs*e_bs + surface.fv*surface.xit*et_sl + surface.fv*(1-surface.xit)*et_fr # [m d-1]
    lh = rho_water*total_et*l  # [MJ m-2 d-1]
    # jax.debug.print('ET components: {}', jnp.array([e_bs, et_fr, et_sl, total_et]))
    # jax.debug.print('ET proportion coefficients: {}', jnp.array([alpha_bs, alpha_gv_sl, alpha_gv_fr]))
    # jax.debug.print('latent heat flux: {}', lh)
    
    # Calculate the ground heat flux from energy balance
    g = rn - sh - lh

    # Calculate the rhs of the surface temperature
    dts, dts_avg = calculate_rhs_surface_temperature(
        ts, ts_avg, g, soil.kthermal, soil.cs, soil.rho, omega=2*pi/1.)

    # Compute the hydraulic redistribution, hr
    hr = 3.3e-9 * (psi_fr-psi_sl) / soil.depths[1] * 86400.   # [m d-1], Montaldo et al (2021)

    # Compute the drainage term, dr, and the leakage term, le
    dr = k_sl  # [m d-1]
    le = k_fr  # [m d-1]

    # Water mass change of surface soil layer
    dtheta_sl = (
        i - surface.fbs*e_bs - surface.fv*surface.xit*et_sl - dr - hr
        # fvg*(1-xigrh)*egnrh - ew - dr - hr
    ) / soil.depths[0]

    # Water mass change of fractured rock sublayer
    dtheta_fr = (
        dr - surface.fv*(1-surface.xit)*et_fr + hr -le
    ) / soil.depths[1]

    out = jnp.array([dtheta_sl, dtheta_fr, ew, i, dts, dts_avg])

    aux = jnp.array([rn, lh, g, sh])

    # jax.debug.print("Time: {}", t)
    # jax.debug.print('Wind speed, resistance, etc. : {}', jnp.array([rc_fr, rc_sl, uz]))
    # jax.debug.print('ET proportion coefficients: {}', jnp.array([alpha_bs, alpha_gv_sl, alpha_gv_fr]))
    # jax.debug.print('ET components : {}', jnp.array([e_bs, et_fr, et_sl]))
    # jax.debug.print("Energy components: {}", jnp.array([rn, lh, g, sh]))
    # jax.debug.print("out: {}", out)
    # jax.debug.print("aux: {}", aux)
    # jax.debug.print(" ")

    return out, aux


    # def __call__(
    #     self,
    #     t: Float_0D,
    #     y: Float_1D,
    #     forcings: PointData,
    # ) -> Float_1D:
    #     theta_sl, theta_fr, cew, ci, ts, ts_avg = y

    #     # Get the atmospheric forcings
    #     # p    : precipitation [m]
    #     # sw_in: incoming short-wave radiation [MJ m-2 day-1]
    #     # lw_in: incoming long-wave radiation [MJ m-2 day-1]
    #     # rn   : net radiation [MJ m-2 day-1]
    #     # vpd  : vapor pressure deficit [kPa]
    #     # ta   : air temperature [degC]
    #     # uz   : wind speed [m s-1]
    #     # lai  : leaf area index [-]
    #     f      = forcings.interpolate_time(t)
    #     p    , vpd   = f[forcings.varn_list.index('P')],     f[forcings.varn_list.index('VPD')]
    #     sw_in, lw_in = f[forcings.varn_list.index('SW_IN')], f[forcings.varn_list.index('LW_IN')]
    #     ta   , uz    = f[forcings.varn_list.index('TA')], f[forcings.varn_list.index('WS')]
    #     lai          = f[forcings.varn_list.index('LAI')]
    #     # jax.debug.print("forcings: {}", f)
    #     dt_obs = forcings.ts[1] - forcings.ts[0]

    #     # Correct theta_sl and theta_fr so that they are within their saturated and residual values
    #     theta_sl = jnp.min(jnp.array([theta_sl, self.soil.theta_sat[0]]))
    #     theta_fr = jnp.min(jnp.array([theta_fr, self.soil.theta_sat[1]]))
    #     theta_sl = jnp.max(jnp.array([theta_sl, self.soil.theta_r[0]]))
    #     theta_fr = jnp.max(jnp.array([theta_fr, self.soil.theta_r[1]]))

    #     # Calculate the net radiation
    #     ts_k = ts + c2k  # conversion from [degC] to [degK]
    #     rn   = calculate_net_radiation(sw_in, lw_in, ts_k, self.albedo, self.emissivity) # [MJ m-2 d-1]

    #     # Calculate the sensible heat flux
    #     sh = calculate_sensible_heat_flux(ts, ta, uz, self.canopy.zm, self.canopy.zom) # [MJ m-2 d-1]

    #     # Water mass change of wet canopy evaporation or interception, ew
    #     # The canopy interception can't exceed the maximum threshold
    #     cewmax = 0.2 * (lai) * 0.001     # [m]
    #     cew    = jnp.min(jnp.array([cew, cewmax])) # [m]
    #     # The dt_obs here refers to the rainfall resolution
    #     ew     = (jnp.min(jnp.array([cew+p*dt_obs, cewmax])) - cew) / dt_obs  #[m d-1]

    #     # Compute throughfall
    #     tf = jnp.max(jnp.array([p-ew, 0]))   # [m d-1]

    #     # Compute the soil water potential and hydraulic conductivity in the two layers 
    #     # using soil water retention curve model
    #     psi_sl, k_sl = van_genuchten_model(
    #         theta_sl, self.soil.theta_sat[0], self.soil.theta_r[0], 
    #         self.soil.ksat[0], self.soil.alpha[0], self.soil.n[0]
    #     )
    #     psi_fr, k_fr = van_genuchten_model(
    #         theta_fr, self.soil.theta_sat[1], self.soil.theta_r[1], 
    #         self.soil.ksat[1], self.soil.alpha[1], self.soil.n[1]
    #     )

    #     # Compute the infiltration term, i
    #     istar = calculate_infiltration_greenampt(ci, k_sl, psi_sl, theta_sl, self.soil.theta_sat[0])  # [m d-1]
    #     i     = jnp.min(jnp.array([tf, istar]))                # [m d-1]

    #     # Compute bare soil evaporation, ebs
    #     # TODO: use soil temperature
    #     alpha_sl = -9.815*theta_sl**3+9.173*theta_sl**2-0.082*theta_sl  # Parlange et al. 1999
    #     e_bs     = alpha_sl * calculate_evaporation_p(rn, ta, uz, vpd)     # [m d-1]

    #     # Compute canopy resistance
    #     rc_sl = calculate_canopy_resistance(
    #         lai, theta_sl, ta, vpd, 
    #         rsmin=self.canopy.rsmin, theta_wp=self.soil.theta_wp[0], theta_lim=self.soil.theta_lim[0], 
    #         tamin=self.canopy.tamin, tamax=self.canopy.tamax, taopt=self.canopy.taopt, w=self.canopy.w
    #     )
    #     rc_fr = calculate_canopy_resistance(
    #         lai, theta_fr, ta, vpd, 
    #         rsmin=self.canopy.rsmin, theta_wp=self.soil.theta_wp[1], theta_lim=self.soil.theta_lim[1], 
    #         tamin=self.canopy.tamin, tamax=self.canopy.tamax, taopt=self.canopy.taopt, w=self.canopy.w
    #     )

    #     # Compute vegetation transpiration
    #     # TODO: computation of actual ET
    #     alpha_fr = -9.815*theta_fr**3+9.173*theta_fr**2-0.082*theta_fr  # Parlange et al. 1999
    #     et_sl = alpha_sl * calculate_evaporation_pm(
    #         rn, 0, ta, uz, vpd, rc_sl, 
    #         self.canopy.dh, self.canopy.zh, self.canopy.zm, self.canopy.zoh, self.canopy.zom) # [m d-1]
    #     et_fr = alpha_fr * calculate_evaporation_pm(
    #         rn, 0, ta, uz, vpd, rc_fr,
    #         self.canopy.dh, self.canopy.zh, self.canopy.zm, self.canopy.zoh, self.canopy.zom) # [m d-1]
        
    #     # Calculate the total ET
    #     total_et = self.fbs*e_bs + self.fv*self.xit*et_sl + self.fv*(1-self.xit)*et_fr # [m d-1]
        
    #     # Calculate the ground heat flux from energy balance
    #     g = rn - sh - l*total_et

    #     # Calculate the rhs of the surface temperature
    #     dts, dts_avg = calculate_rhs_surface_temperature(
    #         ts, ts_avg, g, self.soil.kthermal, self.soil.cs, self.soil.rho, omega=2*pi/1.)

    #     # Compute the hydraulic redistribution, hr
    #     hr = 3.3e-9 * (psi_fr-psi_sl) / self.soil.depths[1] * 86400.   # [m d-1], Montaldo et al (2021)

    #     # Compute the drainage term, dr, and the leakage term, le
    #     dr = k_sl  # [m d-1]
    #     le = k_fr  # [m d-1]

    #     # Water mass change of surface soil layer
    #     dtheta_sl = (
    #         i - self.fbs*e_bs - self.fv*self.xit*et_sl - dr - hr
    #         # fvg*(1-xigrh)*egnrh - ew - dr - hr
    #     ) / self.soil.depths[0]

    #     # Water mass change of fractured rock sublayer
    #     dtheta_fr = (
    #         dr - self.fv*(1-self.xit)*et_fr + hr -le
    #     ) / self.soil.depths[1]

    #     out = jnp.array([dtheta_sl, dtheta_fr, ew, i, dts, dts_avg])

    #     # jax.debug.print("Time: {}", t)
    #     # jax.debug.print("out: {}", out)

    #     return out