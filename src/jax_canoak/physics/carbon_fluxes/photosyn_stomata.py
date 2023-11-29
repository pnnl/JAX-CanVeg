"""
Photosynthesis/stomatal conductance/respiratin, including:
- leaf_ps()
- temp_func()
- specificity()
- tboltz()
- soil_respiration()

Author: Peishi Jiang
Date: 2023.08.01.
"""

import jax
import jax.numpy as jnp

import equinox as eqx

from typing import Tuple
from ...shared_utilities.types import Float_2D, Float_1D, Float_0D
from ...subjects import Para, Ps
from ...subjects.utils import es

from ...shared_utilities.utils import dot, add

PI2 = 3.1415926 * 2


def leaf_ps(
    Iphoton: Float_2D,
    cca: Float_2D,
    Tlk: Float_2D,
    rb_co2: Float_2D,
    P_kPa: Float_1D,
    eair_Pa: Float_2D,
    theta_soil: Float_1D,
    prm: Para,
    stomata: int,
) -> Ps:
    """This program solves a cubic equation to calculate
          leaf photosynthesis.  This cubic expression is derived from solving
          five simultaneous equations for A, PG, cs, CI and GS.
          Stomatal conductance is computed with the Ball-Berry model.
          The cubic derivation assumes that b', the intercept of the Ball-Berry
          stomatal conductance model, is non-zero.

          Gs = k A rh/cs + b'

          We also found that the solution for A can be obtained by a quadratic equation
          when Gs is constant or b' is zero.

          The derivation is published in:

          Baldocchi, D.D. 1994. An analytical solution for coupled leaf photosynthesis
          and stomatal conductance models. Tree Physiology 14: 1069-1079.

       -----------------------------------------------------------------------

              A Biochemical Model of C3 Photosynthesis

                After Farquhar, von Caemmerer and Berry (1980) Planta.
                149: 78-90.

            The original program was modified to incorporate functions and parameters
            derived from gas exchange experiments of Harley, who paramertized Vc and J in
            terms of optimal temperature, rather than some reference temperature, eg 25C.

            Program calculates leaf photosynthesis from biochemical parameters

            rd25 - Dark respiration at 25 degrees C (umol m-2 s-1)
           tlk - leaf temperature, Kelvin
            jmax - optimal rate of electron transport
            vcopt - maximum rate of RuBP Carboxylase/oxygenase
            iphoton - incident photosynthetically active photon flux (mmols m-2 s-1)

                note: Harley parameterized the model on the basis of incident PAR

            gs - stomatal conductance (mol m-2 s-1), typically 0.01-0.20
           pstat-station pressure, bars
            aphoto - net photosynthesis  (umol m-2 s-1)
            ps - gross photosynthesis (umol m-2 s-1)
            aps - net photosynthesis (mg m-2 s-1)
            aphoto (umol m-2 s-1)

    --------------------------------------------------

            iphoton is radiation incident on leaves

           The temperature dependency of the kinetic properties of
            RUBISCO are compensated for using the Arrhenius and
            Boltzmann equations.  From biochemistry, one observes that
            at moderate temperatures enzyme kinetic rates increase
            with temperature.  At extreme temperatures enzyme
            denaturization occurs and rates must decrease.

            Arrhenius Eq.

            f(T)=f(tk_25) exp(tk -298)eact/(298 R tk)), where eact is the
           activation energy.

            Boltzmann distribution

            F(T)=tboltz)


            Define terms for calculation of gross photosynthesis, PG

            PG is a function of the minimum of RuBP saturated rate of
            carboxylation, Wc, and the RuBP limited rate of carboxylation, Wj.
            Wj is limiting when light is low and electron transport, which
            re-generates RuBP, is limiting.  Wc is limiting when plenty of RuBP is
            available compared to the CO2 that is needed for carboxylation.

            Both equations take the form:

            PG-photorespiration= (a CI-a d)/(e CI + b)

            PG-photorespiration=min[Wj,Wc] (1-gamma/Ci)

            Wc=Vcmax Ci/(Ci + Kc(1+O2/Ko))

            Wj=J Ci/(4 Ci + 8 gamma)

            Ps kinetic coefficients from Harley at WBW.

            Gamma is the CO2 compensation point

         Information on the leaf photosynthetic parameters can be found in:

         Harley, P.C. and Baldocchi, 1995.Scaling carbon dioxide and water vapor exchange
         from leaf to canopy in a deciduous forest:leaf level parameterization.
         Plant, Cell and Environment. 18: 1146-1156.

         Wilson, K.B., D.D. Baldocchi and P.J. Hanson. 2000. Spatial and seasonal variability of
         photosynthesis parameters and their relationship to leaf nitrogen in a deciduous forest.
         Tree Physiology. 20, 565-587.


         Tests of the model are reported in:

         Baldocchi, D.D. 1997. Measuring and modeling carbon dioxide and water vapor
         exchange over a temperate broad-leaved forest during the 1995 summer drought.
         Plant, Cell and Environment. 20: 1108-1122

        Baldocchi, D.D. and P.C. Harley. 1995. Scaling carbon dioxide and water vapor
         exchange from leaf to canopy in a deciduous forest: model testing and application.
         Plant, Cell and Environment. 18: 1157-1173.

         Baldocchi, D.D and T.P. Meyers. 1998. On using eco-physiological, micrometeorological
         and biogeochemical theory to evaluate carbon dioxide, water vapor and gaseous deposition
        fluxes over vegetation. Agricultural and Forest Meteorology 90: 1-26.

         Baldocchi, D.D. Fuentes, J.D., Bowling, D.R, Turnipseed, A.A. Monson, R.K. 1999. Scaling
         isoprene fluxes from leaves to canopies: test cases over a boreal aspen and a mixed species temperate
         forest. J. Applied Meteorology. 38, 885-898.

         Baldocchi, D.D. and K.B.Wilson. 2001. Modeling CO2 and water vapor exchange of a
         temperate broadleaved forest across hourly to decadal time scales. Ecological Modeling
              142: 155-184

       Args:
           Iphoton (Float_2D): _description_
           cca (Float_2D): _description_
           Tlk (Float_2D): _description_
           rb_co2 (Float_2D): _description_
           P_kPa (Float_1D): _description_
           eair_Pa (Float_2D): _description_
           theta_soil (Float_1D): _description_
           prm (Para): _description_

       Returns:
           Ps: _description_
    """  # noqa: E501
    _, jtot = cca.shape
    # Rank roots #1,#2 and #3 according to the minimum, intermediate and maximum
    default_root = 0.0
    pstat273 = 101.3 * 0.022624 / (273.15 * P_kPa)
    pstat273 = jnp.tile(pstat273, (jtot, 1)).T

    TlC = Tlk - 273.15
    # rt = prm.rugc * Tlk  #  product of universal gas constant and abs temperature
    tprime25 = Tlk - prm.tk_25  # temperature difference
    # ttemp = jnp.exp((prm.skin * Tlk - prm.hkin) / rt) + 1.0  #  denominator term

    # KC and KO are solely a function of the Arrhenius Eq.
    kct = temp_func(prm.kc25, prm.ekc, tprime25, prm.rugc, prm.tk_25, Tlk)
    # ko = temp_func(prm.ko25, prm.eko, tprime25, prm.rugc, prm.tk_25, Tlk)
    tau = specificity(TlC)

    # fix the Ko and O2 values with same units
    ko25_Pa = prm.ko25 * 100  # Pa
    o2_Pa = prm.o2 * 101.3  # Pa
    # bc = kct * (1.0 + o2 / ko)
    bc = kct * (1.0 + o2_Pa / ko25_Pa)

    # gammac is the CO2 compensation point due to photorespiration, umol mol-1
    # Recalculate gammac with the new temperature dependent KO and KC
    # coefficients..C at Vc = 0.5 Vo
    # gammac = O2/(2 tau)
    # O2 has units of kPa so multiplying 0.5 by 1000 gives units of Pa
    gammac = 500.0 * prm.o2 / tau  # umol/mol

    # Scale rd with vcmax and apply temperature
    # correction for dark respiration
    rdzref = prm.vcopt * 0.004657
    rd = temp_func(rdzref, prm.erd, tprime25, prm.rugc, prm.tk_25, Tlk)
    # jax.debug.print("rd: {a}", a=rd.mean(axis=0))
    # jax.debug.print("rdzref: {a}", a=rdzref)
    # jax.debug.print("Tlk: {a}", a=Tlk.mean(axis=0))

    # Reduce respiration by 40% in light according to Amthor
    @jnp.vectorize
    def func1(rd_e, Iphoton_e):
        return jax.lax.cond(Iphoton_e > 10.0, lambda: rd_e * 0.4, lambda: rd_e)

    rd = func1(rd, Iphoton)
    # jax.debug.print("rd: {a}", a=rd.mean(axis=0))

    # Apply temperature correction to JMAX and vcmax
    jmax = tboltz(prm.jmopt, prm.ejm, prm.toptjm, prm.rugc, prm.hkin, Tlk)
    vcmax = tboltz(prm.vcopt, prm.evc, prm.toptvc, prm.rugc, prm.hkin, Tlk)

    # T leaf boundary layer resistance
    # gb_mole leaf boundary layer conductance for CO2 exchange, mol m-2 s-1
    # RB has units of s/m, convert to mol-1 m2 s1 to be consistant with R.
    # rb_mole = rb_co2 * Tlk * 101.3* 0.022624 / (273.15 * P_kPa)
    rb_mole = rb_co2 * Tlk * pstat273
    gb_mole = 1.0 / rb_mole
    dd = gammac
    b8_dd = 8 * dd

    # Compute the soil moisture impact coefficient
    # Eq.(7) in Wang and Leuning (1998)
    # fw = 10 * (theta_soil - prm.theta_min) / (3 * (prm.theta_max - prm.theta_min))
    # # fw = (theta_soil - prm.theta_min) / (prm.theta_max-prm.theta_min)
    # fw = jnp.clip(fw, a_max=1.0, a_min=0.0)

    # APHOTO = PG - rd, net photosynthesis is the difference
    # between gross photosynthesis and dark respiration. Note
    # photorespiration is already factored into PG.
    #
    # coefficients for Ball-Berry stomatal conductance model
    # Gs = k A rh/cs + b'
    # rh is relative humidity, which comes from a coupled
    # leaf energy balance model
    rh_leaf = eair_Pa / es(Tlk)  # need to transpose matrix
    # rh_leaf = dot(fw, rh_leaf)  # include the impact of soil moisture
    k_rh = rh_leaf * prm.kball  # combine product of rh and K ball-berry
    # jax.debug.print('eair_Pa: {a}', a=eair_Pa[11254:11257,:3])
    # jax.debug.print('rh_leaf: {a}', a=rh_leaf[11254:11257,:3])
    # jax.debug.print('es: {a}', a=es(Tlk)[11254:11257,:3])
    # jax.debug.print("rh_leaf: {a}", a=rh_leaf[0,:3])
    # jax.debug.print("eair_Pa: {a}", a=eair_Pa[0,:3])

    # Gs from Ball-Berry is for water vapor.  It must be divided
    # by the ratio of the molecular diffusivities to be valid for A
    k_rh = k_rh / 1.6  # adjust the coefficient for the diffusion of CO2 rather than H2O
    gb_k_rh = gb_mole * k_rh
    ci_guess = cca * 0.7  # initial guess of internal CO2 to estimate Wc and Wj

    # cubic coefficients that are only dependent on CO2 levels
    def cubic_coef_hypo():
        alpha_ps = 1.0 + (prm.bprime16 / gb_mole) - k_rh
        bbeta = cca * (gb_k_rh - 2.0 * prm.bprime16 - gb_mole)
        gamma = cca * cca * gb_mole * prm.bprime16
        theta_ps = gb_k_rh - prm.bprime16
        return alpha_ps, bbeta, gamma, theta_ps

    def cubic_coef_amphi():
        alpha_ps = 1.0 + (prm.bprime16 / (gb_mole)) - k_rh
        bbeta = cca * (2 * gb_k_rh - 3.0 * prm.bprime16 - 2 * gb_mole)
        gamma = cca * cca * gb_mole * prm.bprime16 * 4
        theta_ps = 2 * gb_k_rh - 2 * prm.bprime16
        return alpha_ps, bbeta, gamma, theta_ps

    alpha_ps, bbeta, gamma, theta_ps = jax.lax.switch(
        stomata,
        [
            cubic_coef_hypo,  # hypostomatous = 0
            cubic_coef_amphi,  # amphistomatous = 1
        ],
    )

    # Test for the minimum of Wc and Wj.  Both have the form:
    # W = (a ci - ad)/(e ci + b)
    # after the minimum is chosen set a, b, e and d for the cubic solution.
    # estimate of J according to Farquhar and von Cammerer (1981)
    # J photon from Harley
    j_photon = (
        prm.qalpha
        * Iphoton
        / jnp.sqrt(1.0 + (prm.qalpha2 * Iphoton * Iphoton / (jmax * jmax)))
    )
    wj = j_photon * (ci_guess - dd) / (4.0 * ci_guess + b8_dd)
    wc = vcmax * (ci_guess - dd) / (ci_guess + bc)

    @jnp.vectorize
    def func2(wj_e, wc_e, b8_dd_e, bc_e, j_photon_e, vcmax_e):
        return jax.lax.cond(
            wj_e < wc_e,
            lambda: (wj_e, b8_dd_e, j_photon_e, 4.0),
            lambda: (wc_e, bc_e, vcmax_e, 1.0),
        )

    psguess, B_ps, a_ps, E_ps = func2(wj, wc, b8_dd, bc, j_photon, vcmax)

    # If wj or wc are less than rd then A would probably be less than zero.
    # This would yield a negative stomatal conductance.
    # In this case, assume gs equals the cuticular value. This
    # assumptions yields a quadratic rather than cubic solution for A
    # cubic solution:
    # A^3 + p A^2 + q A + r = 0
    denom = E_ps * alpha_ps
    Pcube = E_ps * bbeta + B_ps * theta_ps - a_ps * alpha_ps + E_ps * rd * alpha_ps
    Pcube = Pcube / denom
    Qcube = (
        E_ps * gamma
        + (B_ps * gamma / cca)
        - a_ps * bbeta
        + a_ps * dd * theta_ps
        + E_ps * rd * bbeta
        + rd * B_ps * theta_ps
    )
    Qcube = Qcube / denom
    Rcube = (
        -a_ps * gamma
        + a_ps * dd * gamma / cca
        + E_ps * rd * gamma
        + rd * B_ps * gamma / cca
    )
    Rcube = Rcube / denom
    # Use solution from Numerical Recipes from Press
    P2 = Pcube * Pcube
    P3 = P2 * Pcube
    Q = (P2 - 3.0 * Qcube) / 9.0
    R = (2.0 * P3 - 9.0 * Pcube * Qcube + 27.0 * Rcube) / 54.0
    # Test = Q ^ 3 - R ^ 2
    # if test >= O then all roots are real
    rr = R * R
    qqq = Q * Q * Q
    tstroots = qqq - rr
    # jax.debug.print('tstroots: {a}', a=tstroots[18950, 28:36])
    # jax.debug.print('denom: {a}', a=denom[18950, 28:36])
    # jax.debug.print('R: {a}', a=R[18950, 28:36])

    # Peishi made some value checks to make sure numerical stability
    qqq = jnp.clip(qqq, a_min=0.01)  # Added by Peishi..
    arg_U = R / jnp.sqrt(qqq)
    # arg_U = jnp.clip(arg_U, a_min=-1, a_max=1)  # Added by Peishi..
    arg_U = jnp.clip(arg_U, a_min=-0.99, a_max=0.99)  # Added by Peishi..
    ang_L = jnp.arccos(arg_U)
    Q = jnp.clip(Q, a_min=0.01)  # Added by Peishi..
    sqrtQ = jnp.sqrt(Q)

    @jnp.vectorize
    def calculate_root(tstroot, sqrtQ_e, ang_L_e, Pcube_e):
        return jax.lax.cond(
            tstroot > 0,
            lambda: (
                -2.0 * sqrtQ_e * jnp.cos(ang_L_e / 3.0) - Pcube_e / 3.0,
                -2.0 * sqrtQ_e * jnp.cos((ang_L_e + PI2) / 3.0) - Pcube_e / 3.0,
                -2.0 * sqrtQ_e * jnp.cos((ang_L_e - PI2) / 3.0) - Pcube_e / 3.0,
            ),
            lambda: (default_root, default_root, default_root),
        )

    root1, root2, root3 = calculate_root(tstroots, sqrtQ, ang_L, Pcube)

    # Rank roots #1,#2 and #3 according to the minimum, intermediate and maximum
    # values
    root = jnp.stack([root1, root2, root3])  # (3,ntime,jtot)
    root = root.sort(axis=0)
    minroot, midroot, maxroot = root[0], root[1], root[2]
    # find out where roots plop down relative to the x-y axis
    # TODO:
    @jnp.vectorize
    def get_aphoto(minr, midr, maxr):
        conds = jnp.array(
            [
                (minr > 0.0) & (midr > 0.0) & (maxr > 0.0),
                (minr < 0.0) & (midr < 0.0) & (maxr > 0.0),
                (minr < 0.0) & (midr > 0.0) & (maxr > 0.0),
                # The following conditions are added by Peishi --
                (minr < 0.0) & (midr < 0.0) & (maxr < 0.0),
            ]
        )
        index = jnp.where(conds, size=1)[0][0]
        return jax.lax.switch(
            index, [lambda: minr, lambda: maxr, lambda: midr, lambda: maxr]
        )

    aphoto = get_aphoto(minroot, midroot, maxroot)
    aphoto = aphoto - rd
    # also test for sucrose limitation of photosynthesis, as suggested by
    # Collatz.  Js=Vmax/2
    # jax.debug.print('aphoto: {a}', a=aphoto[18950, 28:36])
    # jax.debug.print('rd: {a}', a=rd[18950, 28:36])
    # jax.debug.print('minroot: {a}', a=minroot[18950, 28:36])
    # jax.debug.print('midroot: {a}', a=midroot[18950, 28:36])
    # jax.debug.print('maxroot: {a}', a=maxroot[18950, 28:36])
    j_sucrose = vcmax / 2.0 - rd

    @jnp.vectorize
    def update_aphoto(aphoto_e, j_sucrose_e):
        return jax.lax.cond(
            j_sucrose_e < aphoto_e, lambda: j_sucrose_e, lambda: aphoto_e
        )

    aphoto = update_aphoto(aphoto, j_sucrose)
    # Stomatal conductance for water vapor
    # alfalfa is amphistomatous...be careful on where the factor of two is applied
    # just did on LE on energy balance..dont want to double count
    cs = jax.lax.switch(
        stomata,
        [
            lambda: cca - aphoto / gb_mole,  # hypostomatous = 0
            lambda: cca - aphoto / (2 * gb_mole),  # amphistomatous = 1
        ],
    )
    rh_leaf = jnp.clip(rh_leaf, a_max=1.0)
    # TODO: the following should be computed after aphoto is corrected based on Aps
    # Stomatal conductance is mol m-2 s-1
    # convert back to resistance (s/m) for energy balance routine
    # gs_m_s = gs_leaf_mole * Tlk * 101.3* 0.022624 /(273.15 * P_kPa)
    gs_leaf_mole = (prm.kball * rh_leaf * aphoto / cs) + prm.bprime
    gs_m_s = gs_leaf_mole * Tlk * pstat273
    rstom = 1.0 / gs_m_s
    gs_co2 = gs_leaf_mole / 1.6
    ci = cs - aphoto / gs_co2
    # recompute wj and wc with ci
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
    # print(Aps.shape, rd.shape, aphoto.shape)
    # @jnp.vectorize
    # def update_aphoto2(aphoto_e, Aps_e, rd_e):
    #     tst = (Aps_e < aphoto_e) & (Aps_e > 0)
    #     return jax.lax.cond(tst, lambda: Aps_e - rd_e, lambda: aphoto_e)
    # aphoto = update_aphoto2(aphoto, Aps, rd)
    # jax.debug.print('gs_m_s-before quad: {a}', a=gs_m_s.mean(axis=0))
    @jnp.vectorize
    def update_aphoto_gs(
        aphoto_e,
        Aps_e,
        rd_e,
        rh_leaf_e,
        cs_e,
        tlk_e,
        PkPa_e,
        gs_leaf_mole_e,
        gs_co2_e,
        gs_m_s_e,
        rstom_e,
    ):
        tst = (Aps_e < aphoto_e) & (Aps_e > 0)
        aphoto_e_new = Aps_e - rd_e
        gs_leaf_mole_e_new = (
            prm.kball * rh_leaf_e * aphoto_e_new / cs_e
        ) / 1.0 + prm.bprime
        # ) / 1.6 + prm.bprime16
        gs_co2_e = gs_leaf_mole_e_new / 1.6
        gs_m_s_e_new = (
            gs_leaf_mole_e_new
            * tlk_e
            * 101.3
            * 0.022624
            / (273.15 * PkPa_e)
            # 1.6 * gs_leaf_mole_e_new * tlk_e * 101.3 * 0.022624 / (273.15 * PkPa_e)
        )
        rstom_e = 1.0 / gs_m_s_e_new
        return jax.lax.cond(
            tst,
            lambda: (
                aphoto_e_new,
                gs_leaf_mole_e_new,
                gs_co2_e,
                gs_m_s_e_new,
                rstom_e,
            ),
            lambda: (aphoto_e, gs_leaf_mole_e, gs_co2_e, gs_m_s_e, rstom_e),
        )

    PkPa = dot(P_kPa, jnp.ones(Tlk.shape))  # (ntime, jtot)
    aphoto, gs_leaf_mole, gs_co2, gs_m_s, rstom = update_aphoto_gs(
        aphoto, Aps, rd, rh_leaf, cs, Tlk, PkPa, gs_leaf_mole, gs_co2, gs_m_s, rstom
    )
    # jax.debug.print('gs_m_s-before quad-2: {a}', a=gs_m_s.mean(axis=0))
    # jax.debug.print('aphoto: {a}', a=aphoto[18950, :36])
    # jax.debug.print('rd: {a}', a=rd[18950, :36])
    # jax.debug.print('Aps: {a}', a=Aps[18950, :36])
    # jax.debug.print('Iphoton: {a}', a=Iphoton[:2,:])
    # jax.debug.print('wc: {a}', a=wc[:2,:])
    # jax.debug.print('rd: {a}', a=rd[:2,:])

    # Eliminate few conditions with negative conductance
    @jnp.vectorize
    def clean_negative_gs_leaf(gs_leaf_mole_e):
        return jax.lax.cond(
            gs_leaf_mole_e < 0, lambda: prm.bprime16, lambda: gs_leaf_mole_e
        )

    gs_leaf_mole = clean_negative_gs_leaf(gs_leaf_mole)

    # Correct the solution using quadratic method if necessary
    @jnp.vectorize
    def update_quad(
        aphoto_e,
        aps_e,
        rstompt_e,
        ci_e,
        cs_e,
        gs_co2_e,
        gs_m_s_e,
        wj_e,
        wc_e,
        rd_e,
        tlk_e,
        pstat273_e,
        cca_e,
        gb_mole_e,
        E_ps_e,
        a_ps_e,
        B_ps_e,
        dd_e,
    ):
        def quad():
            gs_leaf_mole = prm.bprime
            gs_co2_e = gs_leaf_mole / 1.6
            # stomatal conductance is mol m-2 s-1
            # convert back to resistance (s/m) for energy balance routine
            gs_m_s_e = gs_leaf_mole * tlk_e * pstat273_e
            rstompt_e = 1.0 / gs_m_s_e
            # a quadratic solution of A is derived if gs=ax, but a cubic form occurs
            # if gs =ax + b.  Use quadratic case when A is less than zero because gs
            # will be negative, which is nonsense
            ps_1 = cca_e * gb_mole_e * gs_co2_e
            delta_1 = gs_co2_e + gb_mole_e
            denom = gb_mole_e * gs_co2_e
            Aquad1 = delta_1 * E_ps_e
            Bquad1 = (
                -ps_1 * E_ps_e
                - a_ps_e * delta_1
                + E_ps_e * rd_e * delta_1
                - B_ps_e * denom
            )
            Cquad1 = (
                a_ps_e * ps_1
                - a_ps_e * dd_e * denom
                - E_ps_e * rd_e * ps_1
                - rd_e * B_ps_e * denom
            )
            product = Bquad1 * Bquad1 - 4.0 * Aquad1 * Cquad1

            # (Peishi) To ensure numerical stability, we force product to be nonnegative
            product = jnp.clip(product, a_min=0)

            sqrprod = jnp.sqrt(product)
            aphoto_e = (-Bquad1 - sqrprod) / (2.0 * Aquad1)
            aps_e = aphoto_e * 0.044
            # Tests suggest that APHOTO2 is the correct photosynthetic root when
            # light is zero because root 2, not root 1 yields the dark respiration
            # value rd.
            cs_e = cca_e - aphoto_e / gb_mole_e
            ci_e = cs_e - aphoto_e / gs_co2_e
            return (aphoto_e, aps_e, ci_e, cs_e, gs_co2_e, gs_m_s_e, rstompt_e)

        return jax.lax.cond(
            (wj_e <= rd_e) | (wc_e <= rd_e),
            quad,
            lambda: (aphoto_e, aps_e, ci_e, cs_e, gs_co2_e, gs_m_s_e, rstompt_e),
        )

    aphoto, Aps, ci, cs, gs_co2, gs_m_s, rstom = update_quad(
        aphoto,
        Aps,
        rstom,
        ci,
        cs,
        gs_co2,
        gs_m_s,
        wj,
        wc,
        rd,
        Tlk,
        pstat273,
        cca,
        gb_mole,
        E_ps,
        a_ps,
        B_ps,
        dd,
    )
    # jax.debug.print('gs_m_s: {a}', a=gs_m_s.mean(axis=0))

    ps = Ps(
        aphoto,
        ci,
        gs_co2,
        gs_m_s,
        wj,
        wc,
        wp,
        j_sucrose,
        Aps,
        root1,
        root2,
        root3,
        Pcube,
        Qcube,
        Rcube,
        rd,
        rstom,
        rh_leaf,
    )
    return ps


def temp_func(
    rate: Float_0D,
    eact: Float_0D,
    tprime: Float_0D,
    rugc: Float_0D,
    tref: Float_0D,
    t_lk: Float_2D,
) -> Float_2D:
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


def specificity(T: Float_2D) -> Float_2D:
    # partioning coefficients for CO2 and O2
    # Roupsard et al 1996 Ann Sci Forest 53: 243-256
    # Henry Law coefficience mol l-1 bar-1
    Kh_co2_a0 = 78.5e-3
    Kh_co2_a1 = -2.89e-3
    Kh_co2_a2 = 54.7e-6
    Kh_co2_a3 = -0.417e-6

    Kh_o2_a0 = 2.1e-3
    Kh_o2_a1 = -57.1e-6
    Kh_o2_a2 = 1.024e-6
    Kh_o2_a3 = -7.503e-9

    T2 = T * T
    T3 = T2 * T

    Kh_co2_T = Kh_co2_a0 + Kh_co2_a1 * T + Kh_co2_a2 * T2 + Kh_co2_a3 * T3
    Kh_o2_T = Kh_o2_a0 + Kh_o2_a1 * T + Kh_o2_a2 * T2 + Kh_o2_a3 * T3

    # Specificity
    # A = Vc - 0.5 Vo - Rd
    # A = Vc (1- 0.5 Vo/Vc) -Rd
    S = 102.0  # specificity at 25 C from Rouspard et al
    tau = S * Kh_co2_T / Kh_o2_T

    return tau


def tboltz(
    rate: Float_0D,
    eakin: Float_0D,
    topt: Float_0D,
    rugc: Float_0D,
    hkin: Float_0D,
    tl: Float_2D,
) -> Float_2D:
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
    Ts: Float_1D, base_respiration: Float_0D = 8.0
) -> Tuple[Float_1D, Float_1D]:
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


def soil_respiration_alfalfa(
    Ac: Float_1D,
    Tsoil: Float_1D,
    soilmoisture: Float_1D,
    veght: Float_1D,
    Rd: Float_1D,
    prm: Para,
) -> Float_1D:
    # Use statistical model derived from ACE analysis on alfalfa
    x = jnp.array([Ac + Rd, Tsoil, soilmoisture, veght])  # (4, ntime)

    b1 = jnp.array(
        [-1.51316616076344, 0.673139978230031, -59.2947930385706, -3.33294857624960]
    )
    b2 = jnp.array(
        [1.38034307225825, 3.73823712105636, 59.9066980644239, 1.36701108005293]
    )
    b3 = jnp.array(
        [2.14475255616910, 19.9136298988773, 0.000987230986585085, 0.0569682453563841]
    )

    temp = x / add(b3, x)
    temp = dot(b2, temp)
    phi = add(b1, temp)  # (4, ntime)
    phisum = phi.sum(axis=0)  # (ntime,)

    b0_0 = 4.846927939437475
    b0_1 = 2.22166756601498
    b0_2 = -0.0281417120818586

    # Reco-Rd = Rsoil
    resp = b0_0 + b0_1 * phisum + b0_2 * phisum * phisum - Rd

    return resp


def soil_respiration_dnn(
    Tsoil_K: Float_2D,
    swc: Float_1D,
    prm: Para,
    RsoilDL: eqx.Module,
) -> Float_1D:
    # Normalize the inputs
    Tsoil = Tsoil_K - 273.15
    # Tsfc_norm = (Tsfc - prm.var_mean.Tsoil) / prm.var_std.Tsoil
    # swc_norm = (swc - prm.var_mean.soilmoisture) / prm.var_std.soilmoisture
    Tsoil_norm = (Tsoil - prm.var_min.T_air) / (  # pyright: ignore
        prm.var_max.T_air - prm.var_min.T_air  # pyright: ignore
    )  # pyright: ignore
    swc_norm = (swc - prm.var_min.soilmoisture) / (  # pyright: ignore
        prm.var_max.soilmoisture - prm.var_min.soilmoisture  # pyright: ignore
    )  # pyright: ignore

    # Get the inputs
    # x = Tsoil_norm
    # jax.debug.print("Tsoil shape: {a}", a=x.shape)
    x = jnp.array([Tsoil_norm[:, 0], swc_norm]).T
    # x = jnp.expand_dims(Tsfc_norm, axis=-1)

    # Perform the Rsoil calculation
    rsoil_norm = jax.vmap(RsoilDL)(x)  # pyright: ignore
    rsoil_norm = rsoil_norm.flatten()

    # jax.debug.print("rsoil_norm: {x}", x=rsoil_norm)
    # jax.debug.print("rsoil_norm: {x}", x=RsoilDL.layers[0].weight)

    # Transform it back
    # rsoil = rsoil_norm * prm.var_std.rsoil + prm.var_mean.rsoil
    rsoil = (
        rsoil_norm * (prm.var_max.rsoil - prm.var_min.rsoil)  # pyright: ignore
        + prm.var_min.rsoil  # pyright: ignore
    )

    return rsoil
