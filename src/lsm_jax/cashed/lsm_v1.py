# TODO: Add units and documentation
# TODO: Add surface temperature & sensible heat flux components
# TODO: Add an energy balance component

import jax
import jax.numpy as jnp
from jaxtyping import Array

import equinox as eqx
from equinox.module import static_field

# from .et import calculate_evaporation_p, calculate_evaporation_pm
# from .infiltration import calculate_infiltration_greenampt
# from .utils import calculate_canopy_resistance, van_genuchten_model

from .et import calculate_evaporation_p, PenmanMonteith
from .infiltration import GreenAmpt
from .retention import VanGenuchten
from .utils import CanopyResistance

class LSMVector(eqx.Module):
    sl_water_retention: VanGenuchten
    fr_water_retention: VanGenuchten
    canopy_resistance: CanopyResistance
    infiltration: GreenAmpt
    potential_et: PenmanMonteith

    fbs: Array = static_field()
    fv: Array = static_field()
    xit: Array = static_field()
    dsl: Array = static_field()
    dfr: Array = static_field()

    def __init__(
        self,
        theta_sat_sl: jnp.array, theta_sat_fr: jnp.array,
        theta_r_sl: jnp.array, theta_r_fr: jnp.array,
        ksat_sl: jnp.array, ksat_fr: jnp.array,
        alpha_sl: jnp.array, alpha_fr: jnp.array,
        n_sl: jnp.array, n_fr: jnp.array,
        rsmin: jnp.array, theta_wp: jnp.array, theta_lim: jnp.array,
        tamin: jnp.array, tamax: jnp.array, taopt: jnp.array, w: jnp.array,
        dh: jnp.array, zh: jnp.array, zm: jnp.array, zoh: jnp.array, zom: jnp.array,
        fbs: jnp.array, fv: jnp.array, xit: jnp.array, dsl: jnp.array, dfr: jnp.array,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.fbs, self.fv  = fbs, fv
        self.dsl, self.dfr = dsl, dfr
        self.xit           = xit

        self.sl_water_retention = VanGenuchten(
            theta_sat_sl, theta_r_sl, ksat_sl, alpha_sl, n_sl
        )
        self.fr_water_retention = VanGenuchten(
            theta_sat_fr, theta_r_fr, ksat_fr, alpha_fr, n_fr
        )
        self.canopy_resistance = CanopyResistance(
            rsmin, theta_wp, theta_lim, tamin, tamax, taopt, w
        )
        self.infiltration = GreenAmpt(theta_sat_sl)
        self.potential_et = PenmanMonteith(
            dh, zh, zm, zoh, zom
        )
        

    def __call__(
        self,
        t: float,
        y: Array,
        forcings: dict,
    ) -> Array:
        theta_sl, theta_fr, cew, ci = y

        # Get the atmospheric forcings
        # TODO: Need a class for the forcing data
        p    = forcings['P'].evaluate(t)      # precipitation [m]
        g    = forcings['G'].evaluate(t)      # ground heat flux [MJ m-2 day-1]
        rn   = forcings['NETRAD'].evaluate(t) # net radiation [MJ m-2 day-1]
        vpd  = forcings['VPD'].evaluate(t)    # vapor pressure deficit [kPa]
        ta   = forcings['TA'].evaluate(t)     # air temperature [degC]
        uz   = forcings['WS'].evaluate(t)     # wind speed [m s-1]
        lai  = forcings['LAI'].evaluate(t)    # leaf area index [-]
        dt_obs = forcings['dt']               # the resolution of dt

        # Correct theta_sl and theta_fr so that they are within their saturated and residual values
        theta_sl = jnp.min(jnp.array([theta_sl, self.sl_water_retention.theta_sat]))
        theta_fr = jnp.min(jnp.array([theta_fr, self.fr_water_retention.theta_sat]))
        theta_sl = jnp.max(jnp.array([theta_sl, self.sl_water_retention.theta_r]))
        theta_fr = jnp.max(jnp.array([theta_fr, self.fr_water_retention.theta_r]))

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
        psi_sl, k_sl = self.sl_water_retention(theta_sl)
        psi_fr, k_fr = self.fr_water_retention(theta_fr)

        # Compute the infiltration term, i
        istar = self.infiltration(ci, k_sl, psi_sl, theta_sl)  # [m d-1]
        i     = jnp.min(jnp.array([tf, istar]))                # [m d-1]

        # Compute bare soil evaporation, ebs
        # TODO: use soil temperature
        alpha_sl = -9.815*theta_sl**3+9.173*theta_sl**2-0.082*theta_sl  # Parlange et al. 1999
        e_bs  = alpha_sl * calculate_evaporation_p(rn, ta, uz, vpd)     # [m d-1]

        # Compute canopy resistance
        # TODO: use soil temperature
        rc_sl = self.canopy_resistance(lai, theta_sl, ta, vpd)
        rc_fr = self.canopy_resistance(lai, theta_fr, ta, vpd)

        # Compute vegetation transpiration
        # TODO: use soil temperature
        # TODO: computation of actual ET
        alpha_fr = -9.815*theta_fr**3+9.173*theta_fr**2-0.082*theta_fr  # Parlange et al. 1999
        et_sl = alpha_sl * self.potential_et(rn, g, ta, uz, vpd, rc_sl) # [m d-1]
        et_fr = alpha_fr * self.potential_et(rn, g, ta, uz, vpd, rc_fr) # [m d-1]

        # # TODO: Energy balance
        # # Compute total ET
        # et = self.fbs*e_bs + self.fv*self.xit*et_sl + self.fv*(1-self.xit)*et_fr
        # # TODO: Calculate the soil/ground heat flux
        # # TODO: Calcualte the sensible heat flux
        # h  = rn - et*l - g
        # # TODO: Calculate the surface temperature
        # ts = ta + h*self.ra/(rho_a*cp)

        # Compute the hydraulic redistribution, hr
        hr = 3.3e-9 * (psi_fr-psi_sl) / self.dfr * 86400.   # [m d-1], Montaldo et al (2021)

        # Compute the drainage term, dr, and the leakage term, le
        dr = k_sl  # [m d-1]
        le = k_fr  # [m d-1]

        # Water mass change of surface soil layer
        dtheta_sl = (
            i - self.fbs*e_bs - self.fv*self.xit*et_sl - dr - hr
            # fvg*(1-xigrh)*egnrh - ew - dr - hr
        ) / self.dsl

        # Water mass change of fractured rock sublayer
        dtheta_fr = (
            dr - self.fv*(1-self.xit)*et_fr + hr -le
        ) / self.dfr

        out = jnp.array([dtheta_sl, dtheta_fr, ew, i])

        return out

# def lsm(t, y, forcings, args):
#     theta_sl, theta_fr, cew, ci = y
    
#     # Get the atmospheric forcings
#     # [m], [MJ m-2 day-1], [kPa], [degC], [m/s]
#     # p, rn, vpd, ta, uz, lai = args['p'], args['rn'], args['vpd'], args['ta'], args['uz'], args['lai']
#     p    = forcings['P'].evaluate(t)      # precipitation [m]
#     g    = forcings['G'].evaluate(t)      # ground heat flux [MJ m-2 day-1]
#     rn   = forcings['NETRAD'].evaluate(t) # net radiation [MJ m-2 day-1]
#     vpd  = forcings['VPD'].evaluate(t)    # vapor pressure deficit [kPa]
#     ta   = forcings['TA'].evaluate(t)     # air temperature [degC]
#     uz   = forcings['WS'].evaluate(t)     # wind speed [m s-1]
#     lai  = forcings['LAI'].evaluate(t)    # leaf area index [-]

#     # Get the parameters
#     dt_obs, dfr, dsl             = args['dt_obs'], args['dfr'], args['dsl']
#     # psisat_sl, psisat_fr         = args['psisat_sl'], args['psisat_fr'] 
#     # ksat_sl, ksat_fr, b          = args['ksat_sl'], args['ksat_fr'], args['b']
#     alpha_sl, alpha_fr           = args['alpha_sl'], args['alpha_fr']
#     n_sl, n_fr                   = args['n_sl'], args['n_fr']
#     ksat_sl, ksat_fr             = args['ksat_sl'], args['ksat_fr']
#     theta_r_sl, theta_r_fr       = args['theta_r_sl'], args['theta_r_fr']
#     theta_sat_sl, theta_sat_fr   = args['theta_sat_sl'], args['theta_sat_fr']
#     psisat_sl, psisat_fr, b      = args['psisat_sl'], args['psisat_fr'], args['b']
#     theta_wp, theta_lim          = args['theta_wp'], args['theta_lim'] 
#     tamin, tamax, taopt, w       = args['tamin'], args['tamax'], args['taopt'], args['w']
#     rsmin, dh                    = args['rsmin'], args['dh']
#     zh, zm, zoh, zom             = args['zh'], args['zm'], args['zoh'], args['zom'] 
#     xit, fbs, fv                 = args['xit'], args['fbs'], args['fv']
    
#     # Correct theta_sl and theta_fr so that they are within their saturated and residual values
#     theta_sl = jnp.min(jnp.array([theta_sl, theta_sat_sl]))
#     theta_fr = jnp.min(jnp.array([theta_fr, theta_sat_fr]))
#     theta_sl = jnp.max(jnp.array([theta_sl, theta_r_sl]))
#     theta_fr = jnp.max(jnp.array([theta_fr, theta_r_fr]))
    
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
#     psi_sl, k_sl = van_genuchten_model(theta_sl, theta_sat_sl, theta_r_sl, ksat_sl, alpha_sl, n_sl)
#     psi_fr, k_fr = van_genuchten_model(theta_fr, theta_sat_fr, theta_r_fr, ksat_fr, alpha_fr, n_fr)
#     # psi_sl, k_sl = clapp_hornberger_model(theta_sl, theta_sat_sl, ksat_sl, psisat_sl, b)
#     # psi_fr, k_fr = clapp_hornberger_model(theta_fr, theta_sat_fr, ksat_fr, psisat_fr, b)
    
#     # Compute the infiltration term, i
#     istar = calculate_infiltration_greenampt(ci, k_sl, psi_sl, theta_sl, theta_sat_sl)  # [m d-1]
#     i     = jnp.min(jnp.array([tf, istar])) # [m d-1]

#     # Compute bare soil evaporation, ebs
#     alpha_sl = -9.815*theta_sl**3+9.173*theta_sl**2-0.082*theta_sl  # Parlange et al. 1999
#     e_bs  = alpha_sl * calculate_evaporation_p(rn, ta, uz, vpd)     # [m d-1]
#     # jax.debug.print("rc terms: {}", jnp.array([rcg_sl, rct_sl, rct_fr]))
    
#     # Compute canopy resistance
#     # based on soil moisture, temperature, and VPD dependencies
#     rc_sl = calculate_canopy_resistance(rsmin, lai, theta_sl, ta, vpd, theta_wp, theta_lim, tamin, tamax, taopt, w)
#     rc_fr = calculate_canopy_resistance(rsmin, lai, theta_fr, ta, vpd, theta_wp, theta_lim, tamin, tamax, taopt, w)
    
#     # Compute vegetation transpiration 
#     alpha_fr = -9.815*theta_fr**3+9.173*theta_fr**2-0.082*theta_fr  # Parlange et al. 1999
#     # from the surface layer, etrh
#     et_sl = alpha_sl*calculate_evaporation_pm(rn, g, ta, uz, vpd, rc_sl, dh, zh, zm, zoh, zom)  # [m d-1]
#     # from the fractured rock layer, etfr
#     et_fr = alpha_fr*calculate_evaporation_pm(rn, g, ta, uz, vpd, rc_fr, dh, zh, zm, zoh, zom)  # [m d-1]
    
#     # Compute the hydraulic redistribution, hr
#     hr = 3.3e-9 * (psi_fr-psi_sl) / dfr * 86400.   # [m d-1], Montaldo et al (2021)
    
#     # Compute the drainage term, dr, and the leakage term, le
#     dr = k_sl  # [m d-1]
#     le = k_fr  # [m d-1]
    
#     # Water mass change of surface soil layer
#     dtheta_sl = (
#         i - fbs*e_bs - fv*xit*et_sl - dr - hr
#         # fvg*(1-xigrh)*egnrh - ew - dr - hr
#     ) / dsl
    
#     # Water mass change of fractured rock sublayer
#     dtheta_fr = (
#         dr - fv*(1-xit)*et_fr + hr -le
#     ) / dfr
    
#     out = jnp.array([dtheta_sl, dtheta_fr, ew, i])
    
#     # jax.debug.print('yt: {}', jnp.array([theta_sl, theta_fr, cew, ci]))
#     # jax.debug.print('forcings: {}', jnp.array([p, rn, vpd, ta, uz, lai]))
#     # jax.debug.print("canopy interception terms: {}", jnp.array([cewmax, ew, tf]))
#     # jax.debug.print("et component: {}", jnp.array([alpha_sl, alpha_fr, e_bs, et_sl, et_fr]))
#     # jax.debug.print("infiltration terms: {}", jnp.array([tf, istar, i]))
#     # jax.debug.print("rc terms: {}", jnp.array([rc_sl, rc_fr]))
#     # jax.debug.print("psi terms: {}", jnp.array([psi_sl, psi_fr]))
#     # jax.debug.print("hydraulic conductivity terms: {}", jnp.array([k_sl, k_fr]))
#     # jax.debug.print("dtheta_sl terms: {}", jnp.array([i, fbs*e_bs, fv*xit*et_sl, dr, hr, dsl]))
#     # jax.debug.print("dtheta_fr terms: {}", jnp.array([dr, fv*(1-xit)*et_fr, hr, le, dfr]))
#     # jax.debug.print("out: {} \n", jnp.array([out]))
    
#     return out
