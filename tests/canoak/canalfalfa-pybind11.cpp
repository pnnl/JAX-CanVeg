#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
// #include "stdafx.h"
// #include <io.h>
#include <sstream>
#include <float.h>
#include <string.h>
#include <errno.h>
#include <algorithm>


// #define sze 32                                  // canopy layers is 30 with flex
// #define sze3 152                                // 5 times canopy layers is 150 with flex
// #define szeang 19                               // number of sky angle classes is 18 with flex
// #define soilsze 12                              // number of soil layers is 10 with flex
#define PI 3.14159                              // pi
#define PI180 0.017453292                                               // pi divided by 180, radians per degree
#define PI9 2.864788976
#define PI2 6.283185307     // 2 time pi

// canopy structure variables
const double ht = 1;             //   0.55 Canopy height, m
const double pai = .0;            //    Plant area index
const double lai = 4;      //  1.65 Leaf area index data are from clip plots and correspond with broadband NDVI estimates

// Constants for leaf energy balance

const double sigma = 5.67e-08;   // Stefan-Boltzmann constant W M-2 K-4
const double cp = 1005.;         // Specific heat of air, J KG-1 K-1
const double mass_air = 29.;     // Molecular weight of air, g mole-1
const double mass_CO2=44.;               // molecular weight of CO2, g mole-1
const double dldt = -2370.;      // Derivative of the latent heat of vaporization

const double ep = .98;                    // emissivity of leaves
const double epm1=0.02;                   // 1- ep
const double epsoil = .98;                // Emissivity of soil
const double epsigma=5.5566e-8;           // ep*sigma
const double epsigma2 = 11.1132e-8;       // 2*ep*sigma
const double epsigma4 = 22.2264e-8;       //  4.0 * ep * sigma
const double epsigma6 = 33.3396e-8;       //  6.0 * ep * sigma
const double epsigma8 = 44.448e-8;        //  8.0 * ep * sigma
const double epsigma12= 66.6792e-8;       // 12.0 * ep * sigma

const double betfact=1.5;                 // multiplication factor for aerodynamic
// sheltering, based on work by Grace and Wilson

//  leaf clumping factor

const double markov = 1.00;

namespace py = pybind11;


void RNET(
        int jtot, 
        py::array_t<double, py::array::c_style> ir_dn_np,
        py::array_t<double, py::array::c_style> ir_up_np,
        py::array_t<double, py::array::c_style> par_sun_np,
        py::array_t<double, py::array::c_style> nir_sun_np,
        py::array_t<double, py::array::c_style> par_sh_np,
        py::array_t<double, py::array::c_style> nir_sh_np,
        py::array_t<double, py::array::c_style> rnet_sh_np,
        py::array_t<double, py::array::c_style> rnet_sun_np
)
{
    double ir_shade,q_sun,qsh;
//     int JJ, JJP1;
    int JJ;

    auto ir_dn = ir_dn_np.unchecked<1>();
    auto ir_up = ir_up_np.unchecked<1>();
    auto par_sun = par_sun_np.unchecked<1>();
    auto nir_sun = nir_sun_np.unchecked<1>();
    auto par_shade = par_sh_np.unchecked<1>();
    auto nir_sh = nir_sh_np.unchecked<1>();
    auto rnet_sh = rnet_sh_np.mutable_unchecked<1>();
    auto rnet_sun = rnet_sun_np.mutable_unchecked<1>();
    /*

     Radiation layers go from jtot+1 to 1.  jtot+1 is the top of the
     canopy and level 1 is at the soil surface.

     Energy balance and photosynthesis are performed for vegetation
     between levels and based on the energy incident to that level

    */
    for(JJ = 1; JJ<= jtot; JJ++)
    {

        // JJP1=JJ+1;

        //Infrared radiation on leaves


        // ir_shade = ir_dn[JJ] + ir_up[JJ];
        ir_shade = ir_dn(JJ-1) + ir_up(JJ-1);

        ir_shade = ir_shade * ep;  //  this is the absorbed IR
        /*

                 Available energy on leaves for evaporation.
                 Values are average of top and bottom levels of a layer.
                 The index refers to the layer.  So layer 3 is the average
                 of fluxes at level 3 and 4.  Level 1 is soil and level
                 j+1 is the top of the canopy. layer jtot is the top layer

                 Sunlit, shaded values
        */

        // q_sun = par_sun[JJ] + nir_sun[JJ] + ir_shade;  // These are the Q values of Rin - Rout + Lin
        // qsh = par_shade[JJ] + nir_sh[JJ] + ir_shade;
        // rnet_sun[JJ] = q_sun;
        // rnet_sh[JJ] = qsh;
        q_sun = par_sun(JJ-1) + nir_sun(JJ-1) + ir_shade;  // These are the Q values of Rin - Rout + Lin
        qsh = par_shade(JJ-1) + nir_sh(JJ-1) + ir_shade;
        // printf("%5.4f %6.4f  %5.4f  %7.4f  %7.4f \n", ir_shade, par_sun(JJ-1), nir_sun(JJ-1), q_sun, qsh);
        rnet_sun(JJ-1) = q_sun;
        rnet_sh(JJ-1) = qsh;

    } // next JJ

    return;
}


void PAR(
    int jtot, int sze, double solar_sine_beta, double parin, double par_beam, 
    double par_reflect, double par_trans, double par_soil_refl, double par_absorbed,
    py::array_t<double, py::array::c_style> dLAIdz_np,
    py::array_t<double, py::array::c_style> exxpdir_np,
    py::array_t<double, py::array::c_style> Gfunc_solar_np,
    py::array_t<double, py::array::c_style> sun_lai_np,
    py::array_t<double, py::array::c_style> shd_lai_np,
    py::array_t<double, py::array::c_style> prob_beam_np,
    py::array_t<double, py::array::c_style> prob_sh_np,
    py::array_t<double, py::array::c_style> par_up_np,
    py::array_t<double, py::array::c_style> par_down_np,
    py::array_t<double, py::array::c_style> beam_flux_par_np,
    py::array_t<double, py::array::c_style> quantum_sh_np,
    py::array_t<double, py::array::c_style> quantum_sun_np,
    py::array_t<double, py::array::c_style> par_shade_np,
    py::array_t<double, py::array::c_style> par_sun_np
)
{


    /*
    -------------------------------------------------------------
                         SUBROUTINE PAR

            This subroutine computes the flux densities of direct and
            diffuse radiation using the measured leaf distrib.
                We apply the Markov model to account for clumping of leaves.

            The model is based on the scheme of Norman.

                    Norman, J.M. 1979. Modeling the complete crop canopy.
            Modification of the Aerial Environment of Crops.
            B. Barfield and J. Gerber, Eds. American Society of Agricultural Engineers, 249-280.

    -----------------------------------------------------------

    */

    double SUP[sze], SDN[sze], transmission_layer[sze], reflectance_layer[sze], beam[sze];
    double ADUM[sze], TBEAM[sze];
    double fraction_beam,sumlai, llai;
    double PEN1, PEN2, dff, par_normal_quanta;
    double exp_direct,QU,TLAY2;
    double par_normal_abs_energy, par_normal_abs_quanta;
    double DOWN, UP;

    long int J,JJP1;
    long int I,JM1, JJ, par_total;
    long int IREP, ITER;
    long int jktot = jtot+1;

    auto dLAIdz = dLAIdz_np.unchecked<1>();
    auto exxpdir = exxpdir_np.unchecked<1>();
    auto Gfunc_solar = Gfunc_solar_np.unchecked<1>();
    auto sun_lai = sun_lai_np.mutable_unchecked<1>();
    auto shd_lai = shd_lai_np.mutable_unchecked<1>();
    auto prob_beam = prob_beam_np.mutable_unchecked<1>();
    auto prob_sh = prob_sh_np.mutable_unchecked<1>();
    auto par_up = par_up_np.mutable_unchecked<1>();
    auto par_down = par_down_np.mutable_unchecked<1>();
    auto beam_flux_par = beam_flux_par_np.mutable_unchecked<1>();
    auto quantum_sh = quantum_sh_np.mutable_unchecked<1>();
    auto quantum_sun = quantum_sun_np.mutable_unchecked<1>();
    auto par_shade = par_shade_np.mutable_unchecked<1>();
    auto par_sun = par_sun_np.mutable_unchecked<1>();


    if(solar_sine_beta <= 0.1)
        goto par_night;

    fraction_beam = par_beam / parin;

    beam[jktot] = fraction_beam;
    TBEAM[jktot] = fraction_beam;

    SDN[1] = 0;


    /*
           Compute probability of penetration for direct and
           diffuse radiation for each layer in the canopy

           Level 1 is the soil surface and level jktot is the
           top of the canopy.  layer 1 is the layer above
           the soil and layer jtot is the top layer.
    */
    sumlai = 0;
    PEN1 = 1.;

    for(J = 1; J<= jtot; J++)
    {
        JJ = jktot - J;

        /*
               Diffuse PAR reflected by each layer

               compute the probability of diffuse radiation penetration through the
               hemisphere.  this computation is not affected by penubra
               since we are dealing only with diffuse radiation from a sky
               sector.

               The probability of beam penetration is computed with a
               negative binomial distriubution.
        */

        // dff = dLAIdz[JJ];  /* + prof.dPAIdz[JJ] */
        dff = dLAIdz(JJ-1);  /* + prof.dPAIdz[JJ] */
        sumlai += dff;

        // reflectance_layer[JJ] = (1. - exxpdir[JJ]) * par_reflect;
        reflectance_layer[JJ] = (1. - exxpdir(JJ-1)) * par_reflect;


        //   DIFFUSE RADIATION TRANSMITTED THROUGH LAYER

        // transmission_layer[JJ] = (1. - exxpdir[JJ]) * par_trans + exxpdir[JJ];
        transmission_layer[JJ] = (1. - exxpdir(JJ-1)) * par_trans + exxpdir(JJ-1);

    } // next J

    /*
    '       COMPUTE THE PROBABILITY OF beam PENETRATION
    */
    sumlai = 0;

    for(J = 1; J<= jtot; J++)
    {
        JJ = jktot - J;
        JJP1 = JJ + 1;

        /*
                Probability of beam penetration.  This is that which
                is not umbral.  Hence, this radiation
                is attenuated by the augmented leaf area: DF+PA.
        */
        // dff = dLAIdz[JJ];  /* + prof.dPAIdz[JJ] */
        dff = dLAIdz(JJ-1);  /* + prof.dPAIdz[JJ] */

        sumlai += dff;

        exp_direct = exp(-dff*markov*Gfunc_solar(JJ-1)/ solar_sine_beta);

        PEN2 = exp(-sumlai*markov*Gfunc_solar(JJ-1)/ solar_sine_beta);

        /* lai Sunlit and shaded  */

        sun_lai(JJ-1)=solar_sine_beta * (1 - PEN2)/ (markov*Gfunc_solar(JJ-1));

        shd_lai(JJ-1)=sumlai - sun_lai(JJ-1);


        /* note that the integration of the source term time solar.prob_beam with respect to
           leaf area will yield the sunlit leaf area, and with respect to solar.prob_sh the
           shaded leaf area.


        In terms of evaluating fluxes for each layer

        Fcanopy = sum {fsun psun + fshade pshade}  (see Leuning et al. Spitters et al.)

        psun is equal to exp(-lai G markov/sinbet)

        pshade = 1 - psun


        */

        prob_beam(JJ-1) = markov*PEN2;

        // if(prob_beam == 0)
        //     PEN1 = 0;


        // probability of beam

        beam[JJ] = beam[JJP1] * exp_direct;

        QU = 1.0 - prob_beam(JJ-1);

        if(QU > 1)
            QU=1;

        if (QU < 0)
            QU=0;


        // probability of umbra

        prob_sh(JJ-1) = QU;

        TBEAM[JJ] = beam[JJ];


        // beam PAR that is reflected upward by a layer

        SUP[JJP1] = (TBEAM[JJP1] - TBEAM[JJ]) * par_reflect;


        // beam PAR that is transmitted downward


        SDN[JJ] = (TBEAM[JJP1] - TBEAM[JJ]) * par_trans;

    } // next J

    /*
         initiate scattering using the technique of NORMAN (1979).
         scattering is computed using an iterative technique.

         Here Adum is the ratio up/down diffuse radiation.

    */
    SUP[1] = TBEAM[1] * par_soil_refl;

    par_down(jktot-1) = 1.0 - fraction_beam;
    ADUM[1] = par_soil_refl;

    for(J = 2,JM1=1; J<=jktot; J++,JM1++)
    {
        TLAY2 = transmission_layer[JM1] * transmission_layer[JM1];
        ADUM[J] = ADUM[JM1] * TLAY2 / (1. - ADUM[JM1] * reflectance_layer[JM1]) + reflectance_layer[JM1];
        // printf("%5.4f \n", TLAY2);
    } /* NEXT J */

    for(J = 1; J<= jtot; J++)
    {
        JJ = jtot - J + 1;
        JJP1 = JJ + 1;
        par_down(JJ-1) = par_down(JJP1-1) * transmission_layer[JJ] / (1. - ADUM[JJP1] * reflectance_layer[JJ]) + SDN[JJ];
        par_up(JJP1-1) = ADUM[JJP1] * par_down(JJP1-1) + SUP[JJP1];
    } // next J

    // lower boundary: upward radiation from soil

//     par_up[1] = par_soil_refl * par_down[1] + SUP[1];
    par_up(0) = par_soil_refl * par_down(0) + SUP[1];

    /*
        Iterative calculation of upward diffuse and downward beam +
        diffuse PAR.

        This section has been commented out for the negative binomial
        model.  It seems not to apply and incorrectly calculates
        scattering.  When I ignore this section, I get perfect
        agreement between measured and calculated Rn.

    */

    // Scattering

    ITER = 0;
    IREP=1;

    // while (IREP==1)
    while (ITER<5) // TODO: Peishi
    {
        IREP = 0;

        ITER += 1;

        // // Printing by looping through the array elements
        // for (J = 1; J <= sze; J++) {
        //     // std::cout << transmission_layer[J] << ' ';
        //     std::cout << par_down(J-1) << ' ';
        // }
        // std::cout << '\n';
        for(J = 2; J <= jktot; J++)
        {
            JJ = jktot - J + 1;
            JJP1 = JJ + 1;
        //     DOWN = transmission_layer[JJ] * par_down[JJP1] + par_up[JJ] * reflectance_layer[JJ] + SDN[JJ];
            DOWN = transmission_layer[JJ] * par_down(JJP1-1) + par_up(JJ-1) * reflectance_layer[JJ] + SDN[JJ];

            // printf("%5.4f %5.4f %5.4f %5.4f %5.4f \n", transmission_layer[JJ], par_down(JJP1-1), par_up(JJ-1), reflectance_layer[JJ], SDN[JJ]);

            if((fabs(DOWN - par_down(JJ-1))) > .01)
                IREP = 1;

        //     par_down[JJ] = DOWN;
            par_down(JJ-1) = DOWN;
        }  // next J

        // for (J = 1; J <= sze; J++) {
        //     std::cout << par_down(J-1) << ' ';
        // }
        // std::cout << '\n';

        //  upward radiation at soil is reflected beam and downward diffuse  */

        // par_up[1] = (par_down[1] + TBEAM[1]) * par_soil_refl;
        par_up(0) = (par_down(0) + TBEAM[1]) * par_soil_refl;

        for(JJ = 2; JJ <=jktot; JJ++)
        {
            JM1 = JJ - 1;
        //     UP = reflectance_layer[JM1] * par_down[JJ] + par_up[JM1] * transmission_layer[JM1] + SUP[JJ];
            UP = reflectance_layer[JM1] * par_down(JJ-1) + par_up(JM1-1) * transmission_layer[JM1] + SUP[JJ];

            // printf("%5.4f %5.4f %5.4f %5.4f %5.4f \n", transmission_layer[JM1], par_down(JJ-1), par_up(JM1-1), reflectance_layer[JM1], SUP[JJ]);
        //     if((fabs(UP - par_up[JJ])) > .01)
            if((fabs(UP - par_up(JJ-1))) > .01)
                IREP = 1;

        //     par_up[JJ] = UP;
            par_up(JJ-1) = UP;
        }  // next JJ

    }
    // printf("%5.4f \n", ITER);
    // printf("%ld \n", ITER);


    // Compute flux density of PAR

    llai = 0;

    for(J = 1; J<=jktot; J++)
    {
        // llai += dLAIdz[J];
        llai += dLAIdz(J-1);

        // upward diffuse PAR flux density, on the horizontal

        // par_up[J] *= input.parin;
        // if(par_up[J] <= 0)
        //     par_up[J] = .001;
        par_up(J-1) *= parin;
        // if(par_up(J-1) <= 0)
        if(par_up(J-1) <= 0.001) // TODO: Peishi
            par_up(J-1) = .001;

        // downward beam PAR flux density, incident on the horizontal

        // beam_flux_par[J] = beam[J] * input.parin;
        // if(beam_flux_par[J] <= 0)
        //     beam_flux_par[J] = .001;
        beam_flux_par(J-1) = beam[J] * parin;
        // if(beam_flux_par(J-1) <= 0)
        if(beam_flux_par(J-1) <= 0.001) // TODO: Peishi
            beam_flux_par(J-1) = .001;


        // Downward diffuse radiatIon flux density on the horizontal

        // par_down[J] *= input.parin;
        // if(par_down[J] <= 0)
        //     par_down[J] = .001;
        par_down(J-1) *= parin;
        // if(par_down(J-1) <= 0)
        if(par_down(J-1) <= 0.001) // TODO: Peishi
            par_down(J-1) = .001;

        //   Total downward PAR, incident on the horizontal

        // par_total = (int)beam_flux_par[J] +(int)par_down[J];
        par_total = (int)beam_flux_par(J-1) +(int)par_down(J-1);


    } // next J

    if(par_beam <= 0)
       par_beam = .001;


    //  PSUN is the radiation incident on the mean leaf normal

    for(JJ = 1; JJ<=jtot; JJ++)
    {


        if (solar_sine_beta > 0.1)
        //     par_normal_quanta = par_beam * Gfunc_solar[JJ] / (solar_sine_beta);
            par_normal_quanta = par_beam * Gfunc_solar(JJ-1) / (solar_sine_beta);
        else
            par_normal_quanta=0;

        // amount of energy absorbed by sunlit leaf */

        par_normal_abs_energy = par_normal_quanta*par_absorbed / 4.6;   // W m-2

        par_normal_abs_quanta = par_normal_quanta * par_absorbed;      // umol m-2 s-1

        /*
                  Convert PAR to W m-2 for energy balance computations
                  but remember umol m-2 s-1 is needed for photosynthesis.
                  Average fluxes for the values on the top and bottom of
                  each layer that drives energy fluxes.

                  Energy balance computations are on the basis of
                  absorbed energy.  Harley's photosynthesis
                  parameterizations are on the basis of incident
                  PAR
        */
        // quantum_sh[JJ] = (par_down[JJ] + par_up[JJ])*par_absorbed;     /* umol m-2 s-1  */
        // quantum_sun[JJ] = quantum_sh[JJ] + par_normal_abs_quanta;
        quantum_sh(JJ-1) = (par_down(JJ-1) + par_up(JJ-1))*par_absorbed;     /* umol m-2 s-1  */
        quantum_sun(JJ-1) = quantum_sh(JJ-1) + par_normal_abs_quanta;


        // calculate absorbed par


        // par_shade[JJ] = quantum_sh[JJ] / 4.6;    /* W m-2 */
        par_shade(JJ-1) = quantum_sh(JJ-1) / 4.6;    /* W m-2 */

        /*
                 solar.par_sun is the total absorbed radiation on a sunlit leaf,
                 which consists of direct and diffuse radiation
        */

        // par_sun[JJ] = par_normal_abs_energy + par_shade[JJ];
        par_sun(JJ-1) = par_normal_abs_energy + par_shade(JJ-1);


    } // next JJ

par_night:

    if(solar_sine_beta <= 0.1)
    {

        for(I = 1; I<=jtot; I++)
        {
        //     prob_sh[I] = 1.;
        //     prob_beam[I] = 0.;
            prob_sh(I-1) = 1.;
            prob_beam(I-1) = 0.;
        }
    }
    return;
}


std::tuple<double, double, double, double, double> DIFFUSE_DIRECT_RADIATION(
    double solar_sine_beta, double rglobal, double parin, double press_kpa
    // double ratrad, double par_beam, double par_diffuse, double nir_beam, double nir_diffuse
)
{
    double fand, fir,fv;
    double rdir, rdvis,rsvis,wa;
    double ru, rsdir, rvt,rit, nirx;
    double xvalue,fvsb,fvd,fansb;

    double ratrad, par_beam, par_diffuse, nir_beam, nir_diffuse;
    /*
     This subroutine uses the Weiss-Norman ( 1985, Agric. forest Meteorol. 34: 205-213)
     routine tocompute direct and diffuse PAR from total par


    There were two typos in Weiss and Norman (1985).
    Equation (3) should be
    Rdv=0.4(600 cos(theta) - RDV)
    Equation (5) should be
    Rdn=0.6(720 - RDN/cos(theta) -w) cos(theta).

      Weiss and Normam assume a solar constant of 1320, which is much lower than 1373 commonly used


           fractions of NIR and PAR (visible)
    */
    if(parin == 0)
        goto par_night;

    fir = .54;
    fv = .46;

    ru = press_kpa / (101.3 * solar_sine_beta);

//         visible direct PAR


    rdvis = 624.0 * exp(-.185 * ru) * solar_sine_beta;


//      potential diffuse PAR

//        rsvis = .4 * (600.0 - rdvis) * solar.sine_beta;

// corrected version

    rsvis = 0.4 * (624. *solar_sine_beta -rdvis);



    /*
            solar constant was assumed to be: 1320 W m-2

            it is really 1373  W m-2

            water absorption in NIR for 10 mm precip water
    */

    wa = 1373.0 * .077 * pow((2. * ru),.3);

    /*

            direct beam NIR
    */
    rdir = (748.0 * exp(-.06 * ru) - wa) * solar_sine_beta;

    if(rdir < 0)
        rdir=0;

    /*
            potential diffuse NIR
    */

//       rsdir = .6 * (720 - wa - rdir) * solar.sine_beta;  // Eva asks if we should correct twice for angles?

// corrected version, Rdn=0.6(720 - RDN/cos(theta) -w) cos(theta).


    rsdir = 0.6* (748. -rdvis/solar_sine_beta-wa)*solar_sine_beta;

    if (rsdir < 0)
        rsdir=0;


    rvt = rdvis + rsvis;
    rit = rdir + rsdir;

    // TODO: Peishi
    // if(rit <= 0)
    if(rit <= 0.1)
        rit = .1;

    // TODO: Peishi
    // if(rvt <= 0)
    if(rvt <= 0.1)
        rvt = .1;

    // printf("%5.4f %6.4f  %5.4f  %7.4f  %7.4f  %7.4f\n", rdvis, rsvis, rdir, rsdir, ru, wa);
    ratrad = rglobal / (rvt + rit);


    // if (local_time_hr >= 12.00 && local_time_hr <=13.00)
    //     solar.ratradnoon=ratrad;
    /*
            ratio is the ratio between observed and potential radiation

            NIR flux density as a function of PAR

            since NIR is used in energy balance calculations
            convert it to W m-2: divide PAR by 4.6
    */


    nirx = rglobal - (parin / 4.6);


//        ratio = (PARIN / 4.6 + NIRX) / (rvt + rit)

    // TODO: Peishi
    // if (ratrad >= .9)
    if (ratrad >= .89)
        ratrad = .89;

    // TODO: Peishi
    // if (ratrad <= 0)
    if (ratrad <= 0.22)
        ratrad=0.22;


//     fraction PAR direct and diffuse

    xvalue=(0.9-ratrad)/.70;

    fvsb = rdvis / rvt * (1. - pow(xvalue,.67));

    if(fvsb < 0)
        fvsb = 0.;

    if (fvsb > 1)
        fvsb=1.0;


    fvd = 1. - fvsb;


//      note PAR has been entered in units of uE m-2 s-1

    par_beam = fvsb * parin;
    par_diffuse = fvd * parin;

    if(par_beam <= 0)
    {
        par_beam = 0;
        par_diffuse = parin;
    }

    if(parin == 0)
    {
        par_beam=0.001;
        par_diffuse=0.001;
    }

    xvalue=(0.9-ratrad)/.68;
    fansb = rdir / rit * (1. - pow(xvalue,.67));

    if(fansb < 0)
        fansb = 0;

    if(fansb > 1)
        fansb=1.0;


    fand = 1. - fansb;


//      NIR beam and diffuse flux densities

    nir_beam = fansb * nirx;
    nir_diffuse = fand * nirx;

    if(nir_beam <= 0)
    {
        nir_beam = 0;
        nir_diffuse = nirx;
    }

    if (nirx == 0)
    {
        nir_beam=0.1;
        nir_diffuse=0.1;
    }


    nir_beam= nirx-nir_diffuse;
    par_beam = parin-par_diffuse;


par_night:
    if (parin == 0){
        ratrad = 0.;
        par_beam = 0.;
        par_diffuse = 0.;
        nir_beam = 0.;
        nir_diffuse = 0.;
    }
    // return;

    return std::make_tuple(ratrad, par_beam, par_diffuse, nir_beam, nir_diffuse);
}


void NIR(
    int jtot, int sze, double solar_sine_beta, double nir_beam, double nir_diffuse, 
    double nir_reflect, double nir_trans, double nir_soil_refl, double nir_absorbed,
    py::array_t<double, py::array::c_style> dLAIdz_np,
    py::array_t<double, py::array::c_style> exxpdir_np,
    py::array_t<double, py::array::c_style> Gfunc_solar_np,
    py::array_t<double, py::array::c_style> nir_dn_np,
    py::array_t<double, py::array::c_style> nir_up_np,
    py::array_t<double, py::array::c_style> beam_flux_nir_np,
    py::array_t<double, py::array::c_style> nir_sh_np,
    py::array_t<double, py::array::c_style> nir_sun_np
)
{



    /*
    ------------------------------------------------------------
                         SUBROUTINE NIR

        This subroutine computes the flux density of direct and diffuse
        radiation in the near infrared waveband.  The Markov model is used
        to compute the probability of beam penetration through clumped foliage.

          The algorithms of Norman (1979) are used.

            Norman, J.M. 1979. Modeling the complete crop canopy.
            Modification of the Aerial Environment of Crops.
            B. Barfield and J. Gerber, Eds. American Society of Agricultural Engineers, 249-280.
    -----------------------------------------------------------------
    */

    long int J, JM1, JJ, JJP1, ITER,IREP;
    long int jktot = jtot+1;

    double nir_incoming,fraction_beam;
    double SUP[sze], SDN[sze], transmission_layer[sze], reflectance_layer[sze], beam[sze];
    double TBEAM[sze];
    double ADUM[sze];
    double exp_direct,sumlai, dff;
    double TLAY2,nir_normal, NSUNEN;
    double llai,NIRTT,DOWN, UP;
    double nir_total;

    auto dLAIdz = dLAIdz_np.unchecked<1>();
    auto exxpdir = exxpdir_np.unchecked<1>();
    auto Gfunc_solar = Gfunc_solar_np.unchecked<1>();
    auto nir_dn = nir_dn_np.mutable_unchecked<1>();
    auto nir_up = nir_up_np.mutable_unchecked<1>();
    auto beam_flux_nir = beam_flux_nir_np.mutable_unchecked<1>();
    auto nir_sh = nir_sh_np.mutable_unchecked<1>();
    auto nir_sun = nir_sun_np.mutable_unchecked<1>();

    nir_incoming = nir_beam + nir_diffuse;

    if(nir_incoming <= 1. || solar_sine_beta <=0)
        goto NIRNIGHT;

    fraction_beam = nir_beam / (nir_beam + nir_diffuse);
    beam[jktot] = fraction_beam;
    TBEAM[jktot] = fraction_beam;


    SDN[1] = 0;

    /*

           Compute probability of penetration for direct and
           diffuse radiation for each layer in the canopy

           Level 1 is the soil surface and level jktot is the
           top of the canopy.  layer 1 is the layer above
           the soil and layer jtot is the top layer.
    */

    sumlai = 0;

    for(J = 2; J<= jktot; J++)
    {
        JJ = jktot - J + 1;

        /*
               diffuse NIR reflected by each layer

               compute the probability of diffuse radiation penetration through the
               hemisphere.  this computation is not affected by penubra
               since we are dealing only with diffuse radiation from a sky
               sector.

               The probability of beam penetration is computed with a
               negative binomial distriubution for LAI.

               Radiation attenuation is a function of leaf and woody
               biomass
        */

        // sumlai += dLAIdz[JJ];  // + prof.dPAIdz[JJ]
        sumlai += dLAIdz(JJ-1);  // + prof.dPAIdz[JJ]

        /*
             'Itegrated probability of diffuse sky radiation penetration
             'for each layer
             '
            EXPDIF[JJ] is computed in PAR and can be used in NIR and IRFLUX
        */

        // reflectance_layer[JJ] = (1. - exxpdir[JJ]) * nir_reflect;
        reflectance_layer[JJ] = (1. - exxpdir(JJ-1)) * nir_reflect;


        //       DIFFUSE RADIATION TRANSMITTED THROUGH LAYER

        // transmission_layer[JJ] = (1. - exxpdir[JJ]) * nir_trans + exxpdir[JJ];
        transmission_layer[JJ] = (1. - exxpdir(JJ-1)) * nir_trans + exxpdir(JJ-1);
    } // next J


//       COMPUTE THE PROBABILITY OF beam PENETRATION


    sumlai = 0;

    for(J = 2; J <= jktot; J++)
    {
        JJ = jktot - J + 1;
        JJP1 = JJ + 1;

        // Probability of beam penetration.


        // dff = dLAIdz[JJ]; /* '+ prof.dPAIdz[JJ] */
        // sumlai += dLAIdz[JJ];
        dff = dLAIdz(JJ-1); /* '+ prof.dPAIdz[JJ] */
        sumlai += dLAIdz(JJ-1);


        // exp_direct = exp(-dff * markov*Gfunc_solar[JJ] / solar_sine_beta);
        exp_direct = exp(-dff * markov*Gfunc_solar(JJ-1) / solar_sine_beta);

        // PEN1 = exp(-llai * prof.Gfunc_solar[JJ] / solar.sine_beta)
        // exp_direct = exp(-DFF * prof.Gfunc_solar[JJ] / solar.sine_beta)

        // Beam transmission

        beam[JJ] = beam[JJP1] * exp_direct;

        TBEAM[JJ] = beam[JJ];


        SUP[JJP1] = (TBEAM[JJP1] - TBEAM[JJ]) * nir_reflect;

        SDN[JJ] = (TBEAM[JJP1] - TBEAM[JJ]) * nir_trans;
    }  // next J

    /*
        initiate scattering using the technique of NORMAN (1979).
        scattering is computed using an iterative technique
    */

    SUP[1] = TBEAM[1] * nir_soil_refl;
    // nir_dn[jktot] = 1. - fraction_beam;
    nir_dn(jktot-1) = 1. - fraction_beam;
    ADUM[1] = nir_soil_refl;

    for(J = 2; J<=jktot; J++)
    {
        JM1 = J - 1;
        TLAY2 = transmission_layer[JM1] * transmission_layer[JM1];
        ADUM[J] = ADUM[JM1] * TLAY2 / (1. - ADUM[JM1] * reflectance_layer[JM1]) + reflectance_layer[JM1];
    } // NEXT J

    for(J = 1; J<=jtot; J++)
    {
        JJ = jtot - J + 1;
        JJP1 = JJ + 1;

        // nir_dn[JJ] = nir_dn[JJP1] * transmission_layer[JJ] / (1. - ADUM[JJP1] * reflectance_layer[JJ]) + SDN[JJ];
        // nir_up[JJP1] = ADUM[JJP1] * nir_dn[JJP1] + SUP[JJP1];
        nir_dn(JJ-1) = nir_dn(JJP1-1) * transmission_layer[JJ] / (1. - ADUM[JJP1] * reflectance_layer[JJ]) + SDN[JJ];
        nir_up(JJP1-1) = ADUM[JJP1] * nir_dn(JJP1-1) + SUP[JJP1];
    }

    // lower boundary: upward radiation from soil

    // nir_up[1] = nir_soil_refl * nir_dn[1] + SUP[1];
    nir_up(0) = nir_soil_refl * nir_dn(0) + SUP[1];

    /*
        Iterative calculation of upward diffuse and downward beam +
        diffuse NIR to compute scattering
    */

    ITER = 0;
    IREP = 1;

    // ITER += 1;

    // while (IREP==1)
    while (ITER<5) // TODO: Peishi
    {

        IREP=0;
        ITER += 1;

        for (J = 2; J<=jktot; J++)
        {
            JJ = jktot - J + 1;
            JJP1 = JJ + 1;
            // DOWN = transmission_layer[JJ] * nir_dn[JJP1] + nir_up[JJ] * reflectance_layer[JJ] + SDN[JJ];
            DOWN = transmission_layer[JJ] * nir_dn(JJP1-1) + nir_up(JJ-1) * reflectance_layer[JJ] + SDN[JJ];

            // if ((fabs(DOWN - nir_dn[JJ])) > .01)
            if ((fabs(DOWN - nir_dn(JJ-1))) > .01)
                IREP = 1;

            // nir_dn[JJ] = DOWN;
            nir_dn(JJ-1) = DOWN;
        }

        // upward radiation at soil is reflected beam and downward diffuse

        // nir_up[1] = (nir_dn[1] + TBEAM[1]) * nir_soil_refl;
        nir_up(0) = (nir_dn(0) + TBEAM[1]) * nir_soil_refl;

        for (JJ = 2; JJ <=jktot; JJ++)
        {
            JM1 = JJ - 1;

            // UP = reflectance_layer[JM1] * nir_dn[JJ] + nir_up[JM1] * transmission_layer[JM1] + SUP[JJ];
            UP = reflectance_layer[JM1] * nir_dn(JJ-1) + nir_up(JM1-1) * transmission_layer[JM1] + SUP[JJ];

            // if ((fabs(UP - nir_up[JJ])) > .01)
            if ((fabs(UP - nir_up(JJ-1))) > .01)
                IREP = 1;

            // nir_up[JJ] = UP;
            nir_up(JJ-1) = UP;
        }

    }

    // Compute NIR flux densities


    nir_total = nir_beam + nir_diffuse;
    llai = lai;

    for(J = 1; J<=jktot; J++)
    {
        // llai -= dLAIdz[J];   // decrement LAI
        llai -= dLAIdz(J-1);   // decrement LAI


        // upward diffuse NIR flux density, on the horizontal

        // nir_up[J] *= nir_total;
        nir_up(J-1) *= nir_total;

        // if(nir_up[J] <= 0.)
            // nir_up[J] = .1;
        if(nir_up(J-1) <= 0.001) // TODO: Peishi
            nir_up(J-1) = .001;


        // downward beam NIR flux density, incident on the horizontal


        // beam_flux_nir[J] = beam[J] * nir_total;
        beam_flux_nir(J-1) = beam[J] * nir_total;

        // if(beam_flux_nir[J] <= 0.)
        //     beam_flux_nir[J] = .1;
        if(beam_flux_nir(J-1) <= 0.001) // TODO: Peishi
            beam_flux_nir(J-1) = .001;


        // downward diffuse radiaiton flux density on the horizontal

        // nir_dn[J] *= nir_total;
        nir_dn(J-1) *= nir_total;

        // if(nir_dn[J] <= 0.)
        //     nir_dn[J] = .1;
        if(nir_dn(J-1) <= 0.001) // TODO: Peishi
            nir_dn(J-1) = .001;


        // total downward NIR, incident on the horizontal


        // NIRTT = beam_flux_nir[J] + nir_dn[J];
        NIRTT = beam_flux_nir(J-1) + nir_dn(J-1);

    } // next J

    for(J = 1; J<=jtot; J++)
    {



        // normal radiation on sunlit leaves

        if(solar_sine_beta > 0.1)
            // nir_normal = nir_beam * Gfunc_solar[J] / solar_sine_beta;
            nir_normal = nir_beam * Gfunc_solar(J-1) / solar_sine_beta;
        else
            nir_normal=0;

        NSUNEN = nir_normal * nir_absorbed;

        /*
                 ' Diffuse radiation received on top and bottom of leaves
                 ' drive photosynthesis and energy exchanges
        */

        // nir_sh[J] = (nir_dn[J] + nir_up[J]);
        nir_sh(J-1) = (nir_dn(J-1) + nir_up(J-1));


        // absorbed radiation, shaded

        // nir_sh[J] *= nir_absorbed;
        nir_sh(J-1) *= nir_absorbed;


        // plus diffuse component


        // nir_sun[J] = NSUNEN + nir_sh[J];  // absorbed NIR on sun leaves
        nir_sun(J-1) = NSUNEN + nir_sh(J-1);  // absorbed NIR on sun leaves
    } // next J

NIRNIGHT:  // jump to here at night since fluxes are zero
    return;
}


double SKY_IR (double T, double ratrad)
{

    // Infrared radiation from sky, W m-2, using algorithm from Norman

    double y;

    y = sigma * pow(T,4.) * ((1. - .261 * exp(-.000777 * pow((273.16 - T), 2.))) * ratrad + 1 - ratrad);

    return y;
}


void IRFLUX(
    int jtot, int sze, double T_Kelvin, double ratrad, double sfc_temperature,
    py::array_t<double, py::array::c_style> exxpdir_np,
    py::array_t<double, py::array::c_style> sun_T_filter_np,
    py::array_t<double, py::array::c_style> shd_T_filter_np,
    py::array_t<double, py::array::c_style> prob_beam_np,
    py::array_t<double, py::array::c_style> prob_sh_np,
    py::array_t<double, py::array::c_style> ir_dn_np,
    py::array_t<double, py::array::c_style> ir_up_np
)
{
    int J, JJ,JJP1,jktot1,JM1,K;
    double ir_in, abs_IR,reflc_lay_IR;
    double Tk_sun_filt,Tk_shade_filt, IR_source_sun,IR_source_shade,IR_source;
    double SDN[sze], SUP[sze];
    double emiss_IR_soil;

    long int jktot = jtot+1;

    auto exxpdir = exxpdir_np.unchecked<1>();
    auto sun_T_filter = sun_T_filter_np.unchecked<1>();
    auto shd_T_filter = shd_T_filter_np.unchecked<1>();
    auto prob_beam = prob_beam_np.unchecked<1>();
    auto prob_sh = prob_sh_np.unchecked<1>();
    auto ir_dn = ir_dn_np.mutable_unchecked<1>();
    auto ir_up = ir_up_np.mutable_unchecked<1>();

    ir_in = SKY_IR(T_Kelvin, ratrad);


    /*  This subroutine is adapted from:

    Norman, J.M. 1979. Modeling the complete crop canopy.
    Modification of the Aerial Environment of Crops.
    B. Barfield and J. Gerber, Eds. American Society of Agricultural Engineers, 249-280.


    Compute probability of penetration for diffuse
    radiation for each layer in the canopy .
    IR radiation is isotropic.

    Level 1 is the soil surface and level jtot+1 is the
    top of the canopy.  LAYER 1 is the layer above
    the soil and LAYER jtot is the top layer.

    IR down flux at top of canopy
    */


    abs_IR = 1.;
    // ir_dn[jktot] = ir_in;
    ir_dn(jktot-1) = ir_in;

    for(J = 1; J<= jtot; J++)
    {
        JJ = jktot - J;
        JJP1 = JJ + 1;
        /*
                  Loop from layers jtot to 1

                Integrated probability of diffuse sky radiation penetration
                EXPDIF[JJ] is computed in RAD

                compute IR radiative source flux as a function of
                leaf temperature weighted according to
                sunlit and shaded fractions

                source=ep*sigma*(laisun*tksun^4 + laish*tksh^4)
                 remember energy balance is done on layers not levels.
                 so level jtot+1 must use tl from layer jtot
        */



        // Tk_sun_filt = sun_T_filter[JJ]+273.16;
        // Tk_shade_filt = shd_T_filter[JJ]+273.16;
        Tk_sun_filt = sun_T_filter(JJ-1)+273.16;
        Tk_shade_filt = shd_T_filter(JJ-1)+273.16;

        // IR_source_sun = prob_beam[JJ] *pow(Tk_sun_filt,4.);
        // IR_source_shade = prob_sh[JJ] * pow(Tk_shade_filt,4.);
        IR_source_sun = prob_beam(JJ-1) *pow(Tk_sun_filt,4.);
        IR_source_shade = prob_sh(JJ-1) * pow(Tk_shade_filt,4.);

        IR_source = epsigma * (IR_source_sun + IR_source_shade);

        /*
                ' Intercepted IR that is radiated up
        */

        // SUP[JJP1] = IR_source * (1. - exxpdir[JJ]);
        SUP[JJP1] = IR_source * (1. - exxpdir(JJ-1));

        /*
                'Intercepted IR that is radiated downward
        */

        // SDN[JJ] = IR_source * (1. - exxpdir[JJ]);
        SDN[JJ] = IR_source * (1. - exxpdir(JJ-1));

    }  /* NEXT J  */

    jktot1 = jktot + 1;

    for(J = 2; J <= jktot; J++)
    {
        JJ = jktot1 - J;
        JJP1 = JJ + 1;
        /*

                Downward IR radiation, sum of that from upper layer that is transmitted
                and the downward source generated in the upper layer.

                 REMEMBER LEVEL JJ IS AFFECTED BY temperature OF LAYER
                 ABOVE WHICH IS JJ
        */

        // ir_dn[JJ] = exxpdir[JJ] * ir_dn[JJP1] + SDN[JJ];
        ir_dn(JJ-1) = exxpdir(JJ-1) * ir_dn(JJP1-1) + SDN[JJ];

    } // next J

    emiss_IR_soil = epsigma * pow((sfc_temperature + 273.16),4.);

    // SUP[1] = ir_dn[1] * (1. - epsoil);
    // ir_up[1] = emiss_IR_soil + SUP[1];
    SUP[1] = ir_dn(0) * (1. - epsoil);
    ir_up(0) = emiss_IR_soil + SUP[1];

    for(J = 2; J<=jktot ; J++)
    {
        JM1 = J - 1;
        /*
                 '
                 ' REMEMBER THE IR UP IS FROM THE LAYER BELOW
        */

        // solar.ir_up[J] = solar.exxpdir[JM1] * solar.ir_up[JM1] + SUP[J];
        ir_up(J-1) = exxpdir(JM1-1) * ir_up(JM1-1) + SUP[J];

    } /* NEXT J  */

    // // Printing by looping through the array elements
    // for (J = 1; J <= sze; J++) {
    //     std::cout << ir_up(J-1) << ' ';
    //     // std::cout << ir_up(J-1) << ' ';
    // }
    // std::cout << '\n';

    for (K = 1; K<=2; K++)
    {

        for (J = 2; J<=jktot; J++)
        {
            JJ = jktot - J + 1;
            JJP1 = JJ + 1;

            // reflc_lay_IR = (1 - solar.exxpdir[JJ]) * (epm1);
            // ir_dn[JJ] = solar.exxpdir[JJ] * ir_dn[JJP1] + solar.ir_up[JJ] * reflc_lay_IR + SDN[JJ];
            reflc_lay_IR = (1 - exxpdir(JJ-1)) * (epm1);
            ir_dn(JJ-1) = exxpdir(JJ-1) * ir_dn(JJP1-1) + ir_up(JJ-1) * reflc_lay_IR + SDN[JJ];
        }        // next J

        // SUP[1] = ir_dn[1] * (1 - epsoil);
        // solar.ir_up[1] = emiss_IR_soil + SUP[1];
        SUP[1] = ir_dn(0) * (1 - epsoil);
        ir_up(0) = emiss_IR_soil + SUP[1];

        for (J = 2; J<= jktot; J++)
        {
            JM1 = J - 1;
            // reflc_lay_IR = (1 - solar.exxpdir[JM1]) * (epm1);
            // solar.ir_up[J] = reflc_lay_IR * ir_dn[J] + solar.ir_up[JM1] * solar.exxpdir[JM1] + SUP[J];
            reflc_lay_IR = (1 - exxpdir(JM1-1)) * (epm1);
            ir_up(J-1) = reflc_lay_IR * ir_dn(J-1) + ir_up(JM1-1) * exxpdir(JM1-1) + SUP[J];
        }   // next J

    } // next K


    return;
}


double GAMMAF(double x)
{

    //  gamma function

    double y,gam;

    gam= (1.0 / (12.0 * x)) + (1.0 / (288.0 * x*x)) - (139.0 / (51840.0 * pow(x,3.0)));
    gam = gam + 1.0;


    if (x > 0)
        y = sqrt(2.0 * PI / x) * pow(x,x) * exp(-x) * gam;
    else
        printf("gamma/n");

    return y;
}


void FREQ (double lflai, py::array_t<double, py::array::c_style> bdens_np)
{
    int I;

    double STD, MEAN, CONS;
    double VAR,nuu,SUM,MU,FL1,MU1,nu1;
    double ANG,FL2,FL3;

    auto bdens = bdens_np.mutable_unchecked<1>();

    /*
            THIS PROGRAM USES THE BETA DISTRIBUTION
            TO COMPUTE THE PROBABILITY FREQUENCY
            DISTRIBUTION FOR A KNOWN MEAN LEAF INCLINATION ANGLE
            STARTING FROM THE TOP OF THE CANOPY, WHERE llai=0

            AFTER GOEL AND STREBEL (1984)

    */

    MEAN=57.4;  // spherical leaf angle

    STD = 26;

    MEAN = MEAN;
    STD = STD;
    VAR = STD * STD + MEAN * MEAN;
    nuu = (1. - VAR / (90. * MEAN)) / (VAR / (MEAN * MEAN) - 1.);
    MU = nuu * ((90. / MEAN) - 1.);
    SUM = nuu + MU;


    FL1 = GAMMAF(SUM) / (GAMMAF(nuu) * GAMMAF(MU));
    MU1 = MU - 1.;
    nu1 = nuu - 1.;

    CONS = 1. / 9.;

    /*

            COMPUTE PROBABILITY DISTRIBUTION FOR 9 ANGLE CLASSES
            BETWEEN 5 AND 85 DEGREES, WITH INCREMENTS OF 10 DEGREES
    */
    for (I=1; I <= 9; I++)
    {
        ANG = (10. * I - 5.);
        FL2 =pow((1. - ANG / 90.),MU1);

        FL3 = pow((ANG / 90.), nu1);
        // canopy.bdens[I] = CONS * FL1 * FL2 * FL3;
        bdens(I-1) = CONS * FL1 * FL2 * FL3;
    }
    return;
}


void G_FUNC_DIFFUSE(
    int jtot,
    py::array_t<double, py::array::c_style> dLAIdz_np,
    py::array_t<double, py::array::c_style> bdens_np,
    py::array_t<double, py::array::c_style> Gfunc_sky_np
)
{


    /*
    ------------------------------------------------------------
            This subroutine computes the G Function according to
                    the algorithms of Lemeur (1973, Agric. Meteorol. 12: 229-247).

                    The original code was in Fortran and was converted to C

            This program computs G for each sky sector, as
                    needed to compute the transmission of diffuse light

            G varies with height to account for vertical variations
                    in leaf angles

    -------------------------------------------------------
    */


    int IJ,IN, J, K, KK,I,II, IN1;

    double aden[18], TT[18], PGF[18], sin_TT[18], del_TT[18], del_sin[18];
    double PPP;
    double ang,dang, aang;
    double cos_A,cos_B,sin_A,sin_B,X,Y;
    double T0,TII,TT0,TT1;
    double R,S, PP,bang;
    double sin_TT1, sin_TT0, square;
    double llai;

    auto dLAIdz = dLAIdz_np.unchecked<1>();
    auto Gfunc_sky = Gfunc_sky_np.mutable_unchecked<2>();


    llai = 0.0;

    ang = 5.0* PI180;
    dang =2.0*ang;

    // Midpoints for azimuth intervals


    for (I = 1; I <= 16; I++)
        aden[I] = .0625;


    /* 3.1415/16 =0.1963  */

    for(IN = 1; IN <= 17; IN++)
    {
        K = 2 * IN - 3;
        TT[IN] = .1963* K;
        sin_TT[IN] = sin(TT[IN]);
    }

    for (IN = 1,IN1=2; IN <= 16; IN++,IN1++)
    {
        del_TT[IN] = TT[IN1] - TT[IN];
        del_sin[IN] = sin_TT[IN1] - sin_TT[IN];
    }

    for(KK = 1; KK <= 9; KK++)
    {
        bang = ang;

        // COMPUTE G FUNCTION FOR EACH LAYER IN THE CANOPY


        llai = 0.0;

        for(IJ = 1; IJ <= jtot; IJ++)
        {
            II = jtot - IJ + 1;

            /*
            'top of the canopy is jtot.  Its cumulative LAI starts with 0.
            */

            // llai += prof.dLAIdz[II];
            llai += dLAIdz(II-1);


            // CALCULATE PROBABILITY FREQUENCY DISTRIBUTION, BDENS


            FREQ(llai, bdens_np);
            auto bdens = bdens_np.mutable_unchecked<1>();

            // LEMEUR DEFINES BDENS AS DELTA F/(PI/N), WHERE HERE N=9

            PPP = 0.0;
            for(I = 1; I <= 9; I++)
            {
                aang = ((I - 1) * 10.0 + 5.0) * PI180;
                cos_B = cos(bang);
                sin_B = sin(bang);
                cos_A=cos(aang);
                sin_A=sin(aang);
                X = cos_A * sin_B;
                Y = sin_A * cos_B;

                if((aang - bang) <= 0.0)
                {   /* if ab  { */
                    // printf("-- %5.6f %5.6f %5.6f \n", bang, aang, aang-bang);
                    for(IN = 1; IN <= 16; IN++)
                    {   /* for IN { */
                        PGF[IN] = X * del_TT[IN] + Y * del_sin[IN];
                    }                            /* for IN } */
                    goto LOOPOUT;
                }                               /* if ab   } */
                else
                {   /* else ab { */
                    T0 = 1.0 + X /Y;
                    TII = 1.0 - X / Y;

                    if(T0/TII > 0)
                        square= sqrt(T0/TII);
                    else
                        printf("bad sqrt in ggfun\n");



                    TT0 = 2.0 * atan(square);
                    sin_TT0 = sin(TT0);
                    TT1 = PI2 - TT0;
                    sin_TT1 = sin(TT1);
                    // printf("-- %5.6f %5.6f %5.6f \n", bang, aang, aang-bang);
                    // printf("%5.6f %5.6f %5.6f %5.6f \n", TT0, sin_TT0, TT1, sin_TT1);

                    for(IN = 1,IN1=2; IN <= 16; IN1++, IN++)                /* for IN { */
                    {
                        if((TT[IN1] - TT0) <= 0)
                        {   /* if 1a { */
                            PGF[IN] = X * del_TT[IN] + Y * del_sin[IN];
                            continue;
                        }                                           /* if 1a  } */
                        else
                        {   /* else 1a { */


                            if((TT[IN1] - TT1) <= 0)                 /* if 1 { */
                            {
                                if((TT0 - TT[IN]) <= 0)                     /* if 2 { */
                                {
                                    PGF[IN] = -X * del_TT[IN] - Y * del_sin[IN];
                                    continue;
                                }                                           /* if 2  } */
                                else                                       /* else 2 { */
                                {
                                    R = X * (TT0 - TT[IN]) + Y * (sin_TT0 - sin_TT[IN]);
                                    S = X * (TT[IN1] - TT0) + Y * (sin_TT[IN1] - sin_TT0);
                                    PGF[IN] = R - S;
                                    continue;
                                }                                          /* else 2 } */
                            }                                             /* if 1 } */

                            else
                            {   /* else 1 { */
                                if((TT1 - TT[IN]) <= 0.0)
                                {   /* if 3  { */
                                    PGF[IN] = X * del_TT[IN] + Y * del_sin[IN];
                                    continue;
                                }                                         /* if 3  } */
                                else                                      /* else 3 { */
                                {
                                    R = X * (TT1 - TT[IN]) + Y * (sin_TT1 - sin_TT[IN]);
                                    S = X * (TT[IN1] - TT1) + Y * (sin_TT[IN1] - sin_TT1);
                                    PGF[IN] = S - R;
                                }                                         /* else 3 } */
                            }                                           /* else 1 } */
                        }                                             /* else 1a } */
                    }                                               /* else ab } */

                    // for (J = 1; J <= 18; J++) {
                    //     std::cout << PGF[J] << ' ';
                    // }
                    // std::cout << '\n';
                }                                                 /*  next IN } */

LOOPOUT:


//       Compute the integrated leaf orientation function, Gfun



                PP = 0.0;

                for(IN = 1; IN <= 16; IN++)
                    PP += (PGF[IN] * aden[IN]);

                // for (J = 1; J <= 16; J++) {
                //     // std::cout << PGF[J] << ' ';
                //     std::cout << PGF[J] * aden[J] << ' ';
                // }
                // std::cout << '\n';
                // printf("%5.4f \n", PP);

                // PPP += (PP * canopy.bdens[I] * PI9);
                PPP += (PP * bdens(I-1) * PI9);
                // printf("%d %5.6f %5.6f \n", (aang-bang)<=0.0, bang, aang);
                // printf("%d %5.6f %5.6f \n", aang-bang<=0.0, aang, bang);
                // printf("%5.4f ", PP);

            }  // next I

            // prof.Gfunc_sky[II][KK] = PPP;
            // printf("%5.4f ", PPP);
            Gfunc_sky(II-1,KK-1) = PPP;

        }  // next IJ

        ang += dang;

    }  // NEXT KK

    return;
}


void GFUNC(
    int jtot, double solar_beta_rad,
    py::array_t<double, py::array::c_style> dLAIdz_np,
    py::array_t<double, py::array::c_style> bdens_np,
    py::array_t<double, py::array::c_style> Gfunc_solar_np
)
{


    /*
    ------------------------------------------------------------
            This subroutine computes the G function according to the
            algorithms of:

                      Lemeur, R. 1973.  A method for simulating the direct solar
                      radiaiton regime of sunflower, Jerusalem artichoke, corn and soybean
                      canopies using actual stand structure data. Agricultural Meteorology. 12, 229-247

                    This progrom computes G for a given
            sun angle.  G changes with height due to change leaf angles
    ------------------------------------------------------------
    */

    int IJ,IN, J, K, I,II, IN1;

    double aden[19], TT[19], pgg[19];
    double sin_TT[19], del_TT[19],del_sin[19];
    double PPP, PP, aang;
    double cos_A,cos_B,sin_A,sin_B,X,Y, sin_TT0, sin_TT1;
    double T0,TII,TT0,TT1;
    double R,S,square;

    double llai = 0.0;

    auto dLAIdz = dLAIdz_np.unchecked<1>();
    auto Gfunc_solar = Gfunc_solar_np.mutable_unchecked<1>();


    // Midpoint of azimuthal intervals


    for(IN=1; IN <= 17; IN++)
    {
        K = 2 * IN - 3;
        TT[IN] = 3.14159265 / 16.0 * K;
        sin_TT[IN] = sin(TT[IN]);
    }

    for(IN=1,IN1=2; IN <= 16; IN++,IN1++)
    {
        del_TT[IN] = TT[IN1] - TT[IN];
        del_sin[IN]=sin_TT[IN1]-sin_TT[IN];
    }

    for(I=1; I <= 16; I++)
        aden[I] = .0625;


    // Compute the G function for each layer


    for(IJ = 1; IJ <= jtot; IJ++)
    {
        II = jtot - IJ + 1;

        /* need LAI from LAIZ */

        // llai += prof.dLAIdz[II];
        llai += dLAIdz(II-1);


        // Calculate the leaf angle probabilty distribution, bdens


        // FREQ(llai);
        FREQ(llai, bdens_np);
        auto bdens = bdens_np.mutable_unchecked<1>();
        // // Printing by looping through the array elements
        // for (J = 1; J <= 9; J++) {
        //     // std::cout << transmission_layer[J] << ' ';
        //     std::cout << bdens(J-1) << ' ';
        // }
        // std::cout << '\n';

        // Lemeur defines bdens as delta F/(PI/N), WHERE HERE N=9

        PPP = 0.0;

        for(I = 1; I <= 9; I++)
        {
            aang = ((I - 1.0) * 10.0 + 5.0) * PI180;

            cos_A = cos(aang);
            cos_B = cos(solar_beta_rad);
            sin_A = sin(aang);
            sin_B = sin(solar_beta_rad);

            X = cos_A * sin_B;
            Y = sin_A * cos_B;

            if((aang - solar_beta_rad) <= 0.0)
            {
                for(IN = 1; IN <= 16; IN++)
                {
                    pgg[IN] = X * del_TT[IN] + Y * del_sin[IN];
                }
                goto OUTGF;
            }
            else
            {
                T0 = (1.0 + X/Y);
                TII = (1.0 - X/Y);

                if(T0/TII > 0)
                    square=sqrt(T0/TII);
                else
                    printf("bad T0/TII \n");


                TT0 = 2.0 * atan(square);
                TT1 = 2.0 * 3.14159 - TT0;
                sin_TT0=sin(TT0);
                sin_TT1=sin(TT1);

                for(IN = 1,IN1=2; IN <= 16; IN++,IN1++)
                {
                    if ((TT[IN1] - TT0) <= 0.0)
                    {
                        pgg[IN] = X * del_TT[IN] + Y *del_sin[IN];
                    }
                    else
                    {
                        if ((TT[IN1] - TT1) <= 0.0)
                        {
                            if((TT0 - TT[IN]) <= 0.0)
                            {
                                pgg[IN] = -X * del_TT[IN] - Y * del_sin[IN];
                            }
                            else
                            {
                                R = X * (TT0 - TT[IN]) + Y * (sin_TT0 - sin_TT[IN]);
                                S = X * (TT[IN + 1] - TT0) + Y * (sin_TT[IN + 1] - sin_TT0);
                                pgg[IN] = R - S;
                            }
                        }
                        else
                        {
                            if((TT1 - TT[IN]) <= 0.0)
                            {

                                pgg[IN] = X * del_TT[IN] + Y * del_sin[IN];
                            }
                            else
                            {
                                R = X * (TT1 - TT[IN]) + Y * (sin_TT1 - sin_TT[IN]);
                                S = X * (TT[IN1] - TT1) + Y * (sin_TT[IN1] - sin_TT1);
                                pgg[IN] = S - R;
                            }
                        }
                    }
                }
            }                                     // next IN

OUTGF:

            // Compute the integrated leaf orientation function

            PP = 0.0;
            for(IN = 1; IN <= 16; IN++)
                PP += (pgg[IN] * aden[IN]);


            // PPP += (PP * canopy.bdens[I] * 9. / PI);
            PPP += (PP * bdens(I-1) * 9. / PI);

        } // next I

        // prof.Gfunc_solar[II] = PPP;
        // if(prof.Gfunc_solar[II] <= 0.0)
        //     prof.Gfunc_solar[II] = .01;
        Gfunc_solar(II-1) = PPP;
        // if(Gfunc_solar(II-1) <= 0.0)
        if(Gfunc_solar(II-1) <= 0.01) // TODO: Peishi
            Gfunc_solar(II-1) = .01;
    }                           // next IJ


    return;
}


std::tuple<double, double, double> ANGLE(
    double latitude, double longitude, double zone,
    int year, int day_local, double hour_local
)
{

//       ANGLE computes solar elevation angles,

//       This subroutine is based on algorithms in Walraven. 1978. Solar Energy. 20: 393-397



    double theta_angle,G,EL,EPS,sin_el,A1,A2,RA;
    double delyr,leap_yr,T_local,time_1980,leaf_yr_4, delyr4;
    // double day_savings_time, day_local;
    double day_savings_time;
    double S,HS,phi_lat_radians,value,declination_ang,ST,SSAS;
    double E_ang, zenith, elev_ang_deg, cos_zenith;

    double radd = .017453293;
    double twopi = 6.28318;

    // matlab code

    double lat_rad, long_rad, std_meridian, delta_long, delta_hours, declin;
    double cos_hour, sunrise, sunset, daylength, f, Et, Lc_deg, Lc_hr, T0, hour;
    double sin_beta, beta_deg, beta_rad, day, time_zone, lat_deg, long_deg;

    // // Twitchell Island, CA
    // double latitude = 38.1;     // latitude
    // double longitude= 121.65;    // longitude

    // // Eastern Standard TIME
    // double zone = 8.0;          // Five hour delay from GMT


    // delyr = time_var.year - 1980.0;
    delyr = year - 1980.0;
    delyr4=delyr/4.0;
    leap_yr=fmod(delyr4,4.0);
    day_savings_time=0.0;

    // Daylight Savings Time, Dasvtm =1
    // Standard time, Dasvtm= 0


    // T_local = time_var.local_time;
    T_local = hour_local;

    // day_local = time_var.days;
    // day_local = day;
    time_1980 = delyr * 365.0 + leap_yr + (day_local - 1) + T_local / 24.0;

    leaf_yr_4=leap_yr*4.0;

    if(delyr == leaf_yr_4)
        time_1980 -= 1.0;

    if(delyr < 0.0)
    {
        if (delyr < leaf_yr_4 || delyr > leaf_yr_4)
            time_1980 -= 1.0;
    }

    theta_angle = (360.0 * time_1980 / 365.25) * radd;
    G = -.031272 - 4.53963E-7 * time_1980 + theta_angle;
    EL = 4.900968 + 3.6747E-7 * time_1980 + (.033434 - 2.3E-9 * time_1980) * sin(G) + .000349 * sin(2. * G) + theta_angle;
    EPS = .40914 - 6.2149E-9 * time_1980;
    sin_el = sin(EL);
    A1 = sin_el * cos(EPS);
    A2 = cos(EL);

    RA = atan(A1/A2);

    /* for ATAN2

    RA = atan2(A1,A2);
    if(RA < 0)
    RA=RA+twopi;

    */

    /*
             The original program was in FORTRAN. It used the ATAN2 function.

             In C we must find the correct quadrant and evaluate the arctan
             correctly.

             Note ATAN2 is -PI TO PI, while ATN is from PI/2 TO -PI/2

    */

    //     QUAD II, TAN theta_angle < 0

    if(A1 > 0)
    {
        if(A2 <= 0)
            RA += 3.1415;
    }

    //  QUAD III, TAN theta_angle > 0  /

    if(A1 <= 0)
    {
        if(A2 <= 0)
            RA += 3.14159;
    }

    value = sin_el * sin(EPS);

    if (1.-value * value >= 0)
        declination_ang = atan(value/ sqrt(1. - value * value));
    else
        printf(" bad declination_ang\n");


//         declination_ang=asin(value)


    ST = 1.759335 + twopi * (time_1980 / 365.25 - delyr) + 3.694E-7 * time_1980;

    if(ST >= twopi)
        ST = ST - twopi;

    S = ST - longitude * radd + 1.0027379 * (zone - day_savings_time + T_local) * 15. * radd;

    if(S >= twopi)
        S = S - twopi;

    HS = RA - S;
    phi_lat_radians = latitude * radd;


    // DIRECTION COSINE

    SSAS = (sin(phi_lat_radians) * sin(declination_ang) + cos(phi_lat_radians) * cos(declination_ang) * cos(HS));


    if(1. - SSAS * SSAS >=0)
        E_ang = atan(sqrt(1. - (SSAS * SSAS))/ SSAS);
    else
        printf(" bad SSAS \n");

    if(SSAS < 0)
        E_ang=E_ang+3.1415;


    //      E=asin(SSAS);

    if(E_ang < 0)
        E_ang=3.1415/2.;

    zenith = E_ang / radd;
    elev_ang_deg = 90. - zenith;
    // solar.beta_rad = elev_ang_deg * radd;
    // solar.sine_beta = sin(solar.beta_rad);
    cos_zenith = cos(E_ang);
    // solar.beta_deg = solar.beta_rad / PI180;

    // enter Matlab version

    time_zone=zone;
    lat_deg=latitude;
    long_deg=longitude;

    lat_rad=lat_deg*PI/180;  // latitude, radians
    long_rad=long_deg*PI/180; // longitude, radians

    std_meridian = 0;
    delta_long=(long_deg - std_meridian)*PI/180;

    delta_hours=delta_long*12/PI;

    day=day_local;
    declin = -23.45*3.1415/180*cos(2*3.1415*(day+10)/365); // declination angle

    cos_hour=-tan(lat_rad)*tan(declin);

    sunrise=12- 12* acos(cos_hour)/PI; // time of sunrise

    sunset=12 + 12* acos(cos_hour)/PI; // time of sunset

    daylength=sunset-sunrise;  // hours of day length

    f=PI*(279.5+0.9856*day)/180;


    // equation of time, hours

    Et=(-104.7*sin(f)+596.2*sin(2*f)+4.3*sin(3*f)-12.7*sin(4*f)-429.3*cos(f)-2.0*cos(2*f)+19.3*cos(3*f))/3600;

    // longitudinal correction

    Lc_deg = long_deg - time_zone*15; // degrees from local meridian

    Lc_hr=Lc_deg*4/60;  // hours, 4 minutes/per degree

    T0 = 12-Lc_hr-Et;

    hour=PI*(T_local-T0)/12;  // hour angle, radians

    // sine of solar elevation, beta

    // printf("%5.4f %5.4f %5.4f %5.4f %5.4f \n", lat_rad, declin, hour, T_local, T0);
    sin_beta=sin(lat_rad)*sin(declin)+cos(lat_rad)*cos(declin)* cos(hour);

    // solar elevation, radians

    beta_rad=asin(sin_beta);

    // solar elevation, degrees

    beta_deg=beta_rad*180/PI;

    // printf("%5.4f %5.4f %5.4f \n", sin_beta, beta_rad, beta_deg);
    // printf("%5.4f %5.4f %5.4f \n", lat_rad, declin, hour);

    return std::make_tuple(beta_rad, sin_beta, beta_deg);

    // solar.beta_rad = beta_rad;
    // solar.sine_beta = sin(solar.beta_rad);
    // solar.beta_deg = solar.beta_rad / PI180;
    // return;
}


void LAI_TIME(
    int jtot, int sze, double tsoil, double lai, double ht,
    double par_reflect, double par_trans, double par_soil_refl, double par_absorbed,
    double nir_reflect, double nir_trans, double nir_soil_refl, double nir_absorbed,
    py::array_t<double, py::array::c_style> ht_midpt_np,
    py::array_t<double, py::array::c_style> lai_freq_np,
    py::array_t<double, py::array::c_style> bdens_np,
    py::array_t<double, py::array::c_style> Gfunc_sky_np,
    py::array_t<double, py::array::c_style> dLAIdz_np,
    py::array_t<double, py::array::c_style> exxpdir_np
)
{


    // Evaluate how LAI and other canopy structural variables vary
    // with time

    long int J,I, II, JM1;
    double lai_z[sze];
    double TF,MU1,MU2,integr_beta;
    double dx,DX2,DX4,X,P_beta,Q_beta,F1,F2,F3;
    double beta_fnc[sze];
    // double beta_fnc[sze],ht_midpt[6],lai_freq[6];
    double cum_lai,sumlai,dff,XX;
    double cum_ht;
    double AA,DA,dff_Markov;
    double cos_AA,sin_AA,exp_diffuse;
    double lagtsoil;
    double delz = ht/jtot;

    auto ht_midpt = ht_midpt_np.mutable_unchecked<1>();
    auto lai_freq = lai_freq_np.unchecked<1>();
    auto dLAIdz = dLAIdz_np.mutable_unchecked<1>();
    auto exxpdir = exxpdir_np.mutable_unchecked<1>();

    // lag is 100 days or 1.721 radians

    //  soil.T_base= 14.5 + 9. * sin((time_var.days * 6.283 / 365.) - 1.721);   */

    // compute seasonal trend of Tsoil Base at 85 cm, level 10 of the soil model




    // amplitude of the soil temperature at a reference depth

    // On average the mean annual temperature occurs around day 100 or 1.721 radians



    // soil.T_base= input.tsoil; // seasonal variation in reference soil temperature at 32 cm
    // soil.T_base= tsoil; // seasonal variation in reference soil temperature at 32 cm


    // full leaf


    // time_var.lai = lai;


    // optical properties PAR wave band
    // after Norman (1979) and NASA report

    // solar.par_reflect = .0377;  // spectrometer and from alfalfa, NASA report 1139, Bowker
    // solar.par_trans = .072;
    // solar.par_soil_refl = 0;    // black soil .3;
    // solar.par_absorbed = (1. - solar.par_reflect - solar.par_trans);


    // optical properties NIR wave band
    // after Norman (1979) and NASA report

    // solar.nir_reflect = .60;  // Strub et al IEEE...spectrometer from Alfalfa
    // solar.nir_trans = .26;
    // solar.nir_soil_refl = 0;    //  black soils 0.6;  // updated


    // value for variable reflectance

    //solar.nir_reflect = (15 * leaf.N + 5)/100;

    //leaf.Vcmax = 26.87 + 15.8 * leaf.N;


    // Absorbed NIR

    // solar.nir_absorbed = (1. - solar.nir_reflect - solar.nir_trans);


    // height of mid point of layer scaled to 0.55m tall alfalfa

    //ht_midpt[1] = 0.1;
    //ht_midpt[2]= 0.2;
    //ht_midpt[3]= 0.3;
    //ht_midpt[4] = 0.4;
    //ht_midpt[5] = 0.5;

    // // 3 m tule
    // ht_midpt[1] = 0.5;
    // ht_midpt[2] = 1.0;
    // ht_midpt[3] = 1.5;
    // ht_midpt[4] = 2.0;
    // ht_midpt[5] = 2.5;



    // lai of the layers at the midpoint of height, scaled to 1.65 LAI


    // lai_freq[1] = 0.05 * lai;
    // lai_freq[2] = 0.30 * lai;
    // lai_freq[3] = 0.30 * lai;
    //  lai_freq[4] = 0.30 * lai;
    //  lai_freq[5] = 0.05 * lai;

    // lai_freq[1] = 0.6 * lai;
    // lai_freq[2] = 0.60 * lai;
    // lai_freq[3] = 0.60 * lai;
    // lai_freq[4] = 0.60 * lai;
    // lai_freq[5] = 0.6 * lai;

    /*
       Beta distribution

       f(x) = x^(p-1) (1-x)^(q-1) / B(v,w)

       B(v,w) = int from 0 to 1 x^(p-1) (1-x)^(q-1) dx

      p = mean{[mean(1-mean)/var]-1}

      q =(1-mean){[mean(1-mean)/var]-1}

      *****************************************************************
    */
    TF = 0.;
    MU1 = 0.;
    MU2 = 0.;
    integr_beta = 0.;


//  Height at the midpoint


    for(I = 1; I<= 5; I++)
    {

        // Normalize height


        // ht_midpt[I] /= ht;   // was ht, but then it would divide by 24 or so
        ht_midpt(I-1) /= ht;   // was ht, but then it would divide by 24 or so


        // Total F in each layer. Should sum to LAI


        // TF += lai_freq[I];
        TF += lai_freq(I-1);


        // weighted mean lai

        // MU1 += (ht_midpt[I] * lai_freq[I]);
        MU1 += (ht_midpt(I-1) * lai_freq(I-1));


        // weighted variance

        // MU2 +=  (ht_midpt[I] * ht_midpt[I] * lai_freq[I]);
        MU2 +=  (ht_midpt(I-1) * ht_midpt(I-1) * lai_freq(I-1));
    }  // next I


    // normalize mu by lai

    MU1 /= TF;
    MU2 /= TF;


    // compute Beta parameters


    P_beta = MU1 * (MU1 - MU2) / (MU2 - MU1 * MU1);
    Q_beta = (1. - MU1) * (MU1 - MU2) / (MU2 - MU1 * MU1);
    P_beta -= 1.;
    Q_beta -= 1.;

    /*
    '  integrate Beta function, with Simpson's Approx.
    '
    '  The boundary conditions are level 1 is height of ground
    '  and level jtot+1 is height of canopy.  Layer 1 is between
    '  height levels 1 and 2.  Layer jtot is between levels
    '  jtot and jtot+1
    '
    '  Thickness of layer
    */

    dx = 1. / jtot;

    DX2 = dx / 2.;
    DX4 = dx / 4.;
    X = DX4;

    F2 = (pow(X,P_beta)) *pow((1. - X), Q_beta);
    X += DX4;
    F3 = pow(X, P_beta) *pow((1. - X),Q_beta);


    // start integration at lowest boundary


    beta_fnc[1] = DX4 * (4. * F2 + F3) / 3.;
    integr_beta += beta_fnc[1];

    JM1=jtot-1;

    for(I = 2; I <=JM1; I++)
    {
        F1 = F3;
        X += DX2;
        F2 = pow(X, P_beta) * pow((1. - X), Q_beta);
        X += DX2;
        F3 = pow(X, P_beta) * pow((1. - X), Q_beta);
        beta_fnc[I] = DX2 * (F1 + 4. * F2 + F3) / 3.;
        integr_beta += beta_fnc[I];
    }

    F1 = F3;
    X += DX4;
    F2 = pow(X, P_beta) * pow((1. - X),Q_beta);


    //  compute integrand at highest boundary


    beta_fnc[jtot] = DX4 * (F1 + 4. * F2) / 3.;
    integr_beta += beta_fnc[jtot];
    /*
            '   lai_z IS THE LEAF AREA AS A FUNCTION OF Z
            '
            '   beta_fnc is the pdf for the interval dx
    */

    lai_z[1] = beta_fnc[1] * lai / integr_beta;

    for(I = 2; I <= JM1; I++)
        lai_z[I] = beta_fnc[I] * lai / integr_beta;


    lai_z[jtot] = beta_fnc[jtot] * lai / integr_beta;

    cum_ht = 0;
    cum_lai = 0;


    for(I = 1; I <= jtot; I++)
    {
        /*
        ' re-index layers of lai_z.
        ' layer 1 is between ground and 1st level
        ' layer jtot is between level jtot and top of canopy (jtot+1)
        */

        cum_ht += delz;
        cum_lai += lai_z[I];


        // use prof.dLAIdz for radiative transfer model


        // prof.dLAIdz[I] = lai_z[I];
        dLAIdz(I-1) = lai_z[I];
    } // next I



    G_FUNC_DIFFUSE(jtot, dLAIdz_np, bdens_np, Gfunc_sky_np);   // Direction cosine for the normal between the mean
    auto Gfunc_sky = Gfunc_sky_np.mutable_unchecked<2>();
    // leaf normal projection and the sky sector.

    sumlai = 0;

    for(J = 1; J <= jtot; J++)
    {
        /*
        '
        '       compute the probability of diffuse radiation penetration through the
        '       hemisphere.  This computation is not affected by penubra
        '       since we are dealing only with diffuse radiation from a sky
        '       sector.
        '
        '       The probability of beam penetration is computed with a
        '       Markov distribution.
        */

        // dff = prof.dLAIdz[J];  //  + prof.dPAIdz[J]
        // sumlai += prof.dLAIdz[J];
        dff = dLAIdz(J-1);  //  + prof.dPAIdz[J]
        sumlai += dLAIdz(J-1);

        XX = 0;
        AA = .087;
        DA = .1745;

        // The leaf clumping coefficient. From Chason et al. 1990 and
        // studies on WBW


        dff_Markov = dff*markov;

        for(II = 1; II <= 9; II++)
        {
            cos_AA = cos(AA);
            sin_AA = sin(AA);

            // probability of photon transfer through a sky section
            // for clumped foliage and Markov model

            // exp_diffuse = exp(-dff_Markov * prof.Gfunc_sky[J][II] / cos_AA);
            exp_diffuse = exp(-dff_Markov * Gfunc_sky(J-1,II-1) / cos_AA);
            // printf("c++ %5.4f %5.4f %5.4f %5.4f\n", XX, AA, Gfunc_sky(J-1,II-1), dff_Markov);
            // printf("c++ %5.4f %5.4f %5.4f %5.4f %5.4f\n", cos_AA * sin_AA * exp_diffuse, exp_diffuse, dff_Markov, Gfunc_sky(J-1,II-1), cos_AA);

            // for spherical distribution
            // exp_diffuse = exp(-DFF * prof.Gfunc_sky(J, II) / cos_AA)

            // printf("c++ %5.4f %5.4f \n", XX, cos_AA * sin_AA * exp_diffuse);
            XX += (cos_AA * sin_AA * exp_diffuse);
            AA += DA;
            // printf("c++ %5.4f %5.4f %5.4f %5.4f %5.4f\n", cos_AA * sin_AA * exp_diffuse, exp_diffuse, dff_Markov, Gfunc_sky(J-1,II-1), cos_AA);
            // printf("c++ %5.4f %5.4f %5.4f %5.4f\n", XX, AA, Gfunc_sky(J-1,II-1), dff_Markov);
        }  // next II

        /*
        'Itegrated probability of diffuse sky radiation penetration
        'for each layer
        */

        // solar.exxpdir[J] = 2. * XX * DA;
        // if(solar.exxpdir[J] > 1.)
        //     solar.exxpdir[J] = .9999;
        exxpdir(J-1) = 2. * XX * DA;
        if(exxpdir(J-1) > 1.)
            exxpdir(J-1) = .9999;

    } // next J


    // printf("lai  day  time_var.lai\n");
    // printf("%5.2f  %4i  %5.2f\n", lai,time_var.days,time_var.lai);

    return;
}


void CONC(
        double cref, double soilflux, double factor,
        int sze3, int jtot, int jtot3, double met_zl, double delz, int izref,
        double ustar_ref, double ustar, 
        py::array_t<double, py::array::c_style> source_np,
        py::array_t<double, py::array::c_style> cncc_np,
        py::array_t<double, py::array::c_style> dispersion_np
)
{


        // Subroutine to compute scalar concentrations from source
        // estimates and the Lagrangian dispersion matrix


        double sumcc[sze3], cc[sze3];
        double disper, ustfact, disperzl, soilbnd;
        int i, j;

        // py::buffer_info source_buf = source_np.request()
        // py::buffer_info cncc_buf = cncc_np.request()
        // py::buffer_info dispersion_buf = dispersion_np.request()
        auto dispersion = dispersion_np.unchecked<2>();
        auto source = source_np.unchecked<1>();
        auto cncc = cncc_np.mutable_unchecked<1>();
        // double (*dispersion)[jtot];
        // dispersion = disp;


        // Compute concentration profiles from Dispersion matrix


        ustfact = ustar_ref / ustar;         // factor to adjust Dij with alternative u* values
        
                                        // Note that disperion matrix was computed using u* = 1.00

		
        for( i=1; i <=jtot3; i++)
{
        sumcc[i] = 0.0;


        for (j=1; j <= jtot; j++)
{

/*
        CC is the differential concentration (Ci-Cref)


        Ci-Cref = SUM (Dij S DELZ), units mg m-3 or mole m-3

         S = dfluxdz/DELZ

        note delz values cancel

        scale dispersion matrix according to friction velocity
*/

        disper = ustfact * dispersion(i-1,j-1);                    // units s/m
        // disper = ustfact * dispersion[i][j];                    // units s/m
        // disper = ustfact * *( *(dispersion + i) + j );                    // units s/m

        // scale dispersion matrix according to Z/L

              
        // if(met.zl < 0)
        // disperzl = disper * (0.679 (z/L) - 0.5455)/(z/L-0.5462);
        // else
        // disperzl=disper;

                // updated Dispersion matrix (Oct, 2015)..for alfalfa  runs for a variety of z?

                if(met_zl < 0)
                disperzl = disper * (0.97*-0.7182)/(met_zl -0.7182);
                else
                disperzl=disper * (-0.31 * met_zl + 1.00);

        // sumcc[i] += delz * disperzl * source[j];
        sumcc[i] += delz * disperzl * source(j-1);

        
        } // next j 


        // scale dispersion matrix according to Z/L


        disper = ustfact * dispersion(i-1,0);
        // disper = ustfact * dispersion[i][1];
        // disper = ustfact * *( *(dispersion + i) + 1 );
                
                      
                if(met_zl < 0)
                disperzl = disper * (0.97*-0.7182)/(met_zl -0.7182);
                else
                disperzl=disper * (-0.31 * met_zl + 1.00);

                
        // add soil flux to the lowest boundary condition

        soilbnd=soilflux*disperzl/factor;

        cc[i]=sumcc[i]/factor+soilbnd;

        // printf("%5.0f %6.2f  %5.2f  %7.4f  %7.4f \n", disper, ustfact, disperzl, soilbnd, soilflux);
        // printf("%5.2f %6.2f  %5.2f %5.2f %6.2f  %5.2f %5.2f \n", cc[i], sumcc[i], factor, soilbnd, disper, ustfact, dispersion(i-1,0));
        // printf("%5.4f %6.4f  %5.4f  %7.4f %5.4f \n", cc[i], sumcc[i], factor, disper, dispersion(i-1,0));

  
        } // next i


        //  Compute scalar profile below reference

        for(i=1; i<= jtot3; i++) {
            // cncc[i] = cc[i] + cref - cc[izref];
            cncc(i-1) = cc[i] + cref - cc[izref];
        //     printf("%5.4f %6.4f  %5.4f  %7.4f %5.4f \n", cncc(i-1), cc[i], cref, cc[izref], sumcc[i]);
        }

        
        return;
}


PYBIND11_MODULE(canoak, m) {
    m.doc() = "pybind11 plugins for CANOAK"; // optional module docstring

    m.def("rnet", &RNET, "Subroutine to energy balance and photosynthesis are performed for vegetation between levels and based on the energy incident to that level",
    py::arg("jtot"), py::arg("ir_dn_np"), py::arg("ir_up_np"),
    py::arg("par_sun_np"), py::arg("nir_sun_np"), py::arg("par_sh_np"),
    py::arg("nir_sh_np"), py::arg("rnet_sh_np"), py::arg("rnet_sun_np"));

    m.def("par", &PAR, "Subroutine to compute the flux densities of direct and diffuse radiation using the measured leaf distrib",
    py::arg("jtot"), py::arg("sze"), py::arg("solar_sine_beta"), py::arg("parin"),
    py::arg("par_beam"), py::arg("par_reflect"), py::arg("par_trans"),
    py::arg("par_soil_refl"), py::arg("par_absorbed"), 
    py::arg("dLAIdz_np"), py::arg("exxpdir_np"), py::arg("Gfunc_solar_np"),
    py::arg("sun_lai_np"), py::arg("shd_lai_np"), py::arg("prob_beam_np"),
    py::arg("prob_sh_np"), py::arg("par_up_np"), py::arg("par_down_np"),
    py::arg("beam_flux_par_np"), py::arg("quantum_sh_np"), py::arg("quantum_sun_np"),
    py::arg("par_shade_np"), py::arg("par_sun_np"));

    m.def("diffuse_direct_radiation", &DIFFUSE_DIRECT_RADIATION, "Subroutine to compute direct and diffuse PAR/NIR from total par and rglobal",
    py::arg("solar_sine_beta"), py::arg("rglobal"), py::arg("parin"), py::arg("press_kpa"));

    m.def("nir", &NIR, "Subroutine to compute the flux density of direct and diffuse radiation in the near infrared waveband",
    py::arg("jtot"), py::arg("sze"), py::arg("solar_sine_beta"), py::arg("parin"),
    py::arg("par_beam"), py::arg("par_reflect"), py::arg("par_trans"),
    py::arg("par_soil_refl"), py::arg("par_absorbed"), 
    py::arg("dLAIdz_np"), py::arg("exxpdir_np"), py::arg("Gfunc_solar_np"),
    py::arg("nir_dn_np"), py::arg("nir_up_np"), py::arg("beam_flux_nir_np"),
    py::arg("nir_sh_np"), py::arg("nir_sun_np"));

    m.def("sky_ir", &SKY_IR, "Subroutine to compute infrared radiation from sky using algorithm from Norman",
    py::arg("T"), py::arg("ratrad")); 

    m.def("irflux", &IRFLUX, "Subroutine to compute probability of penetration for diffuse radiation for each layer in the canopy",
    py::arg("jtot"), py::arg("sze"), py::arg("T_Kelvin"), py::arg("radtad"), py::arg("sfc_temperature"), 
    py::arg("exxpdir_np"), py::arg("sun_T_filter_np"), py::arg("shd_T_filter_np"), 
    py::arg("prob_beam_np"), py::arg("prob_sh_np"), py::arg("ir_dn_np"), py::arg("ir_up_np")); 

    m.def("g_func_diffuse", &G_FUNC_DIFFUSE, "Subroutine to compute the G Function according to the algorithms of Lemeur (1973, Agric. Meteorol. 12: 229-247).",
    py::arg("jtot"), py::arg("dLAIdz_np"), py::arg("bdens_np"), py::arg("Gfunc_sky_np")); 

    m.def("gfunc", &GFUNC, "Subroutine to computes G for a given sun angle.",
    py::arg("jtot"), py::arg("solar_beta_rad"), py::arg("dLAIdz_np"), py::arg("bdens_np"), py::arg("Gfunc_solar_np")); 

    m.def("gammaf", &GAMMAF, "Subroutine to compute gamma function",
    py::arg("x")); 

    m.def("freq", &FREQ, "Subroutine to compute the probability frequency distribution for a known mean leaf inclination angle",
    py::arg("lflai"), py::arg("bdens_np")); 

    m.def("angle", &ANGLE, "Subroutine to compute solar elevation angles",
    py::arg("latitude"), py::arg("longitude"), py::arg("zone"),
    py::arg("year"), py::arg("day_local"), py::arg("hour_local")); 

    m.def("lai_time", &LAI_TIME, "Subroutine to compute how LAI and other canopy structural variables vary with time",
    py::arg("jtot"), py::arg("sze"), py::arg("tsoil"), py::arg("lai"), py::arg("ht"),
    py::arg("par_reflect"), py::arg("par_trans"), py::arg("par_soil_refl"), py::arg("par_absorbed"),
    py::arg("nir_reflect"), py::arg("nir_trans"), py::arg("nir_soil_refl"), py::arg("nir_absorbed"),
    py::arg("ht_midpt_np"), py::arg("lai_freq_np"), py::arg("bdens_np"), py::arg("Gfunc_sky_np"),
    py::arg("dLAIdz_np"), py::arg("exxpdir_np")); 

    m.def("conc", &CONC, "Subroutine to compute scalar concentrations from source estimates and the Lagrangian dispersion matrix",
    py::arg("cref"), py::arg("soilflux"), py::arg("factor"),
    py::arg("sze3"), py::arg("jtot"), py::arg("jtot3"), py::arg("met_zl"), py::arg("delz"),
    py::arg("izref"), py::arg("ustar_ref"), py::arg("ustar"), 
    py::arg("source"), py::arg("cncc"), py::arg("dispersoin_np"));
}