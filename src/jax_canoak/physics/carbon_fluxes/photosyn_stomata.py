"""
Photosynthesis/stomatal conductance/respiratin, including:
- stomata()
- photosynthesis_amphi()
- temp_func()
- tboltz()
- soil_respiration()

Author: Peishi Jiang
Date: 2023.07.06.
"""

import jax
import jax.numpy as jnp

from typing import Tuple
from ..energy_fluxes import sfc_vpd
from ...shared_utilities.types import Float_0D, Float_1D
from ...shared_utilities.constants import brs, rsm, rugc, hkin, tk_25
from ...shared_utilities.constants import kc25, ekc, ko25, eko, tau25, ektau, o2
from ...shared_utilities.constants import jmopt, vcopt, erd, ejm, toptjm, evc, toptvc
from ...shared_utilities.constants import bprime, bprime16, qalpha, qalpha2
from ...shared_utilities.constants import PI2


def stomata(
    lai: Float_0D,
    pai: Float_0D,
    rcuticle: Float_0D,
    par_sun: Float_1D,
    par_shade: Float_1D,
) -> Tuple[Float_1D, Float_1D]:
    """First guess of rstom to run the energy balance model.
       It is later updated with the Ball-Berry model.

    Args:
        lai (Float_0D): _description_
        pai (Float_0D): _description_
        rcuticle (Float_0D): _description_
        par_sun (Float_1D): _description_
        par_shade (Float_1D): _description_

    Returns:
        Tuple[Float_1D, Float_1D]: _description_
    """
    sze = par_sun.size
    jtot = sze - 2
    sun_rs, shd_rs = jnp.zeros(sze), jnp.zeros(sze)

    rsfact = brs * rsm

    def update_rs(c, i):
        sun_rs_each = jax.lax.cond(
            (lai == pai) | (par_sun[i] < 5.0),
            lambda: rcuticle,
            lambda: rsm + rsfact / par_sun[i],
        )
        shd_rs_each = jax.lax.cond(
            (lai == pai) | (par_shade[i] < 5.0),
            lambda: rcuticle,
            lambda: rsm + rsfact / par_shade[i],
        )
        return c, [sun_rs_each, shd_rs_each]

    _, rs_update = jax.lax.scan(update_rs, None, jnp.arange(jtot))

    # jax.debug.print("rs_update: {x}", x=rs_update)

    # sun_rs = sun_rs.at[:jtot].set(rs_update[0])
    # shd_rs = shd_rs.at[:jtot].set(rs_update[1])
    sun_rs = jnp.concatenate([rs_update[0], sun_rs[jtot:]])
    shd_rs = jnp.concatenate([rs_update[1], shd_rs[jtot:]])

    return sun_rs, shd_rs


def photosynthesis_amphi(
    Iphoton: Float_0D,
    cca: Float_0D,
    tlk: Float_0D,
    leleaf: Float_0D,
    vapor: Float_0D,
    pstat273: Float_0D,
    kballstr: Float_0D,
    latent: Float_0D,
    co2air: Float_0D,
    co2bound_res: Float_0D,
    rhov_air_z: Float_0D,
) -> Tuple[Float_0D, Float_0D, Float_0D, Float_0D, Float_0D, Float_0D]:
    # temperature difference
    tprime25 = tlk - tk_25
    # # product of universal gas constant and abs temperature
    # rt = rugc * tlk
    # # denominator term
    # ttemp = jnp.exp((skin * tlk - hkin) / rt) + 1.0
    # # Initialize min and max roots
    # minroot, midroot, maxroot= 1.e10, 0., -1.e10
    # root1, root2, root3, aphoto = 0., 0., 0., 0.

    # KC and KO are solely a function of the Arrhenius Eq.
    kct = temp_func(kc25, ekc, tprime25, tk_25, tlk)
    ko = temp_func(ko25, eko, tprime25, tk_25, tlk)
    tau = temp_func(tau25, ektau, tprime25, tk_25, tlk)
    bc = kct * (1.0 + o2 / ko)

    Iphoton = jnp.clip(Iphoton, a_min=0.0)

    # gammac is the CO2 compensation point due to photorespiration, umol mol-1
    # Recalculate gammac with the new temperature dependent KO and KC
    # coefficients
    gammac = 500.0 * o2 / tau

    # Temperature corrections for Jmax and Vcmax
    # Scale jmopt and VCOPT with a surrogate for leaf nitrogen
    # specific leaf weight (Gutschick and Weigel).
    # Normalized leaf wt is 1 at top of canopy and is 0.35
    # at forest floor.  Leaf weight scales linearly with height
    # and so does jmopt and vcmax
    # zoverh=0.65/HT=zh65
    jmaxz, vcmaxz = jmopt, vcopt

    # Scale rd with height via vcmax and apply temperature
    # correction for dark respiration
    rdz = vcmaxz * 0.004657
    rdz = jax.lax.cond(Iphoton > 10, lambda: rdz * 0.4, lambda: rdz)
    rd = temp_func(rdz, erd, tprime25, tk_25, tlk)

    # Apply temperature correction to JMAX and vcmax
    jmax = tboltz(jmaxz, ejm, toptjm, tlk)
    vcmax = tboltz(vcmaxz, evc, toptvc, tlk)

    # Compute the leaf boundary layer resistance
    # gb_mole leaf boundary layer conductance for CO2 exchange,
    # mol m-2 s-1
    # RB has units of s/m, convert to mol-1 m2 s1 to be consistant with R.
    rb_mole = co2bound_res * tlk * pstat273
    gb_mole = 1.0 / rb_mole
    dd = gammac
    b8_dd = 8 * dd

    # APHOTO = PG - rd, net photosynthesis is the difference
    # between gross photosynthesis and dark respiration. Note
    # photorespiration is already factored into PG.
    #
    # coefficients for Ball-Berry stomatal conductance model
    # Gs = k A rh/cs + b'
    # rh is relative humidity, which comes from a coupled
    # leaf energy balance model
    # rh_leaf  = sfc_vpd(delz, tlk, zzz, leleaf, latent, vapor, rhov_air)
    rh_leaf = sfc_vpd(tlk, leleaf, latent, vapor, rhov_air_z)
    k_rh = rh_leaf * kballstr

    # Gs from Ball-Berry is for water vapor.  It must be divided
    # by the ratio of the molecular diffusivities to be valid for A
    k_rh = k_rh / 1.6  # adjust the coefficient for the diffusion of CO2 rather than H2O
    gb_k_rh = gb_mole * k_rh
    ci_guess = cca * 0.7  # initial guess of internal CO2 to estimate Wc and Wj

    # cubic coefficients that are only dependent on CO2 levels
    # factors of 2 are included
    alpha_ps = 1.0 + (bprime16 / (gb_mole)) - k_rh
    bbeta = cca * (2 * gb_k_rh - 3.0 * bprime16 - 2 * gb_mole)
    gamma = cca * cca * gb_mole * bprime16 * 4
    theta_ps = 2 * gb_k_rh - 2 * bprime16

    # Test for the minimum of Wc and Wj.  Both have the form:
    # W = (a ci - ad)/(e ci + b)
    # after the minimum is chosen set a, b, e and d for the cubic solution.
    # estimate of J according to Farquhar and von Cammerer (1981)
    # J photon from Harley
    j_photon = jax.lax.cond(
        jmax > 0,
        lambda: qalpha
        * Iphoton
        / jnp.sqrt(1.0 + (qalpha2 * Iphoton * Iphoton / (jmax * jmax))),
        lambda: 0.0,
    )
    wj = j_photon * (ci_guess - dd) / (4.0 * ci_guess + b8_dd)
    wc = vcmax * (ci_guess - dd) / (ci_guess + bc)
    # psguess = jax.lax.cond(wj<wc,lambda:wj,lambda:wc)
    B_ps = jax.lax.cond(wj < wc, lambda: b8_dd, lambda: bc)
    a_ps = jax.lax.cond(wj < wc, lambda: j_photon, lambda: vcmax)
    E_ps = jax.lax.cond(wj < wc, lambda: 4.0, lambda: 1.0)

    # If wj or wc are less than rd then A would probably be less than zero.
    # This would yield a negative stomatal conductance.
    # In this case, assume gs equals the cuticular value. This
    # assumptions yields a quadratic rather than cubic solution for A
    def cubic():
        # cubic solution:
        # A^3 + p A^2 + q A + r = 0
        denom = E_ps * alpha_ps
        Pcube = E_ps * bbeta + B_ps * theta_ps - a_ps * alpha_ps + E_ps * rd * alpha_ps
        Pcube /= denom
        Qcube = (
            E_ps * gamma
            + (B_ps * gamma / cca)
            - a_ps * bbeta
            + a_ps * dd * theta_ps
            + E_ps * rd * bbeta
            + rd * B_ps * theta_ps
        )
        Qcube /= denom
        Rcube = (
            -a_ps * gamma
            + a_ps * dd * (gamma / cca)
            + E_ps * rd * gamma
            + rd * B_ps * gamma / cca
        )
        Rcube /= denom
        # Use solution from Numerical Recipes from Press
        P2 = Pcube * Pcube
        P3 = P2 * Pcube
        Q = (P2 - 3.0 * Qcube) / 9.0
        R = (2.0 * P3 - 9.0 * Pcube * Qcube + 27.0 * Rcube) / 54.0
        # Test = Q ^ 3 - R ^ 2
        # if test >= O then all roots are real
        # rr=R*R
        qqq = Q * Q * Q
        arg_U = R / jnp.sqrt(qqq)
        ang_L = jnp.arccos(arg_U)
        root1 = -2.0 * jnp.sqrt(Q) * jnp.cos(ang_L / 3.0) - Pcube / 3.0
        root2 = -2.0 * jnp.sqrt(Q) * jnp.cos((ang_L + PI2) / 3.0) - Pcube / 3.0
        root3 = -2.0 * jnp.sqrt(Q) * jnp.cos((ang_L - PI2) / 3.0) - Pcube / 3.0
        # Rank roots #1,#2 and #3 according to the minimum, intermediate and maximum
        # value
        sroots = jnp.sort(jnp.array([root1, root2, root3]))
        minroot, midroot, maxroot = sroots[0], sroots[1], sroots[2]
        # find out where roots plop down relative to the x-y axis
        conds = jnp.array(
            [
                (minroot > 0.0) & (midroot > 0.0) & (maxroot > 0.0),
                (minroot < 0.0) & (midroot < 0.0) & (maxroot > 0.0),
                (minroot < 0.0) & (midroot > 0.0) & (maxroot > 0.0),
            ]
        )
        index = jnp.where(conds, size=1)[0][0]
        aphoto = jax.lax.switch(
            index, [lambda: minroot, lambda: maxroot, lambda: midroot]
        )
        # also test for sucrose limitation of photosynthesis, as suggested by
        # Collatz.  Js=Vmax/2
        j_sucrose = vcmax / 2.0 - rd
        aphoto = jax.lax.cond(j_sucrose < aphoto, lambda: j_sucrose, lambda: aphoto)
        cs = cca - aphoto / (2 * gb_mole)
        cs = jax.lax.cond(cs > 1000, lambda: co2air, lambda: cs)
        # Stomatal conductance for water vapor
        # alfalfa is amphistomatous...be careful on where the factor of two is applied
        # just did on LE on energy balance..dont want to double count
        # this version should be for an amphistomatous leaf since A is
        # considered on both sides
        gs_leaf_mole = (kballstr * rh_leaf * aphoto / cs) + bprime
        gs_co2 = gs_leaf_mole / 1.6
        # jax.debug.print("aphoto: {a}; gs_co2: {b}; cs: {c}; Pcube: {d}.",
        #                 a=aphoto, b=gs_co2, c=cs, d=Pcube)
        # Stomatal conductance is mol m-2 s-1
        # convert back to resistance (s/m) for energy balance routine
        gs_m_s = gs_leaf_mole * tlk * pstat273
        rstompt = 1.0 / gs_m_s
        ci = cs - aphoto / gs_co2
        # Recompute wj and wc with ci
        wj = j_photon * (ci - dd) / (4.0 * ci + b8_dd)
        wc = vcmax * (ci - dd) / (ci + bc)
        a, b, c = 0.98, -(wj + wc), wj * wc
        wp1 = (-b + jnp.sqrt(b * b - 4 * a * c)) / (2 * a)
        wp2 = (-b - jnp.sqrt(b * b - 4 * a * c)) / (2 * a)
        wp = jnp.minimum(wp1, wp2)
        aa, bb, cc = 0.95, -(wp + j_sucrose), wp * j_sucrose
        Aps1 = (-bb + jnp.sqrt(bb * bb - 4 * aa * cc)) / (2 * aa)
        Aps2 = (-bb - jnp.sqrt(bb * bb - 4 * aa * cc)) / (2 * aa)
        Aps = jnp.minimum(Aps1, Aps2)
        aphoto = jax.lax.cond(
            (Aps < aphoto) & (Aps > 0), lambda: Aps - rd, lambda: aphoto
        )
        A_mgpt = aphoto * 0.044
        # If A < 0 then gs should go to cuticular value and recalculate A
        # using quadratic solution
        rstompt, A_mgpt, ci, wj, wc = jax.lax.cond(
            aphoto <= 0.0, quad, lambda: (rstompt, A_mgpt, ci, wj, wc)
        )
        return rstompt, A_mgpt, ci, wj, wc

    def quad():
        gs_leaf_mole = bprime
        gs_co2 = gs_leaf_mole / 1.6
        # stomatal conductance is mol m-2 s-1
        # convert back to resistance (s/m) for energy balance routine
        gs_m_s = gs_leaf_mole * tlk * pstat273
        rstompt = 1.0 / gs_m_s
        # a quadratic solution of A is derived if gs=ax, but a cubic form occurs
        # if gs =ax + b.  Use quadratic case when A is less than zero because gs will be
        # negative, which is nonsense
        ps_1 = cca * gb_mole * gs_co2
        delta_1 = gs_co2 + gb_mole
        denom = gb_mole * gs_co2
        Aquad1 = delta_1 * E_ps
        Bquad1 = -ps_1 * E_ps - a_ps * delta_1 + E_ps * rd * delta_1 - B_ps * denom
        Cquad1 = a_ps * ps_1 - a_ps * dd * denom - E_ps * rd * ps_1 - rd * B_ps * denom
        product = Bquad1 * Bquad1 - 4.0 * Aquad1 * Cquad1
        sqrprod = jnp.sqrt(product)
        aphoto = (-Bquad1 - sqrprod) / (2.0 * Aquad1)
        # Tests suggest that APHOTO2 is the correct photosynthetic root when
        # light is zero because root 2, not root 1 yields the dark respiration
        # value rd.
        cs = cca - aphoto / gb_mole
        ci = cs - aphoto / gs_co2
        A_mgpt = aphoto * 0.044
        return rstompt, A_mgpt, ci, wj, wc

    # jax.debug.print("wj: {a}; wc: {b}; rd: {c}; j_photon: {d}.",
    #                 a=wj, b=wc, c=rd, d=j_photon)
    rstompt, A_mgpt, cipnt, wjpnt, wcpnt = jax.lax.cond(
        (wj <= rd) | (wc <= rd), quad, cubic
    )
    resppt = rd

    return rstompt, A_mgpt, resppt, cipnt, wjpnt, wcpnt


def temp_func(
    rate: Float_0D, eact: Float_0D, tprime: Float_0D, tref: Float_0D, t_lk: Float_0D
) -> Float_0D:
    """Arhennius temperature function.

    Args:
        rate (Float_0D): _description_
        eact (Float_0D): _description_
        tprime (Float_0D): _description_
        tref (Float_0D): _description_
        t_lk (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    return rate * jnp.exp(tprime * eact / (tref * rugc * t_lk))


def tboltz(rate: Float_0D, eakin: Float_0D, topt: Float_0D, tl: Float_0D) -> Float_0D:
    """Boltzmann temperature distribution for photosynthesis

    Args:
        rate (Float_0D): _description_
        eakin (Float_0D): _description_
        topt (Float_0D): _description_
        tl (Float_0D): _description_

    Returns:
        Float_0D: _description_
    """
    dtlopt = tl - topt
    prodt = rugc * topt * tl
    numm = rate * hkin * jnp.exp(eakin * (dtlopt) / (prodt))
    denom = hkin - eakin * (1.0 - jnp.exp(hkin * (dtlopt) / (prodt)))
    return numm / denom


def soil_respiration(
    Ts: Float_0D, base_respiration: Float_0D = 8.0
) -> Tuple[Float_0D, Float_0D]:
    """Compute soil respiration

    Args:
        Ts (Float_0D): _description_
        base_respiration (Float_0D, optional): _description_. Defaults to 8..
    """
    # After Hanson et al. 1993. Tree Physiol. 13, 1-15
    # reference soil respiration at 20 C, with value of about 5 umol m-2 s-1
    # from field studies

    # assume Q10 of 1.4 based on Mahecha et al Science 2010, Ea = 25169
    respiration_mole = base_respiration * jnp.exp(
        (25169.0 / 8.314) * ((1.0 / 295.0) - 1.0 / (Ts + 273.16))
    )

    # soil wetness factor from the Hanson model, assuming constant and wet soils
    respiration_mole *= 0.86

    # convert soilresp to mg m-2 s-1 from umol m-2 s-1
    respiration_mg = respiration_mole * 0.044

    return respiration_mole, respiration_mg
