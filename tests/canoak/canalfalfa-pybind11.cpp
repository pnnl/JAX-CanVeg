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

const double ep = .98;                    // emissivity of leaves

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

        sumlai += dLAIdz[JJ];  // + prof.dPAIdz[JJ]

        /*
             'Itegrated probability of diffuse sky radiation penetration
             'for each layer
             '
            EXPDIF[JJ] is computed in PAR and can be used in NIR and IRFLUX
        */

        reflectance_layer[JJ] = (1. - exxpdir[JJ]) * nir_reflect;


        //       DIFFUSE RADIATION TRANSMITTED THROUGH LAYER

        transmission_layer[JJ] = (1. - exxpdir[JJ]) * nir_trans + exxpdir[JJ];
    } // next J


//       COMPUTE THE PROBABILITY OF beam PENETRATION


    sumlai = 0;

    for(J = 2; J <= jktot; J++)
    {
        JJ = jktot - J + 1;
        JJP1 = JJ + 1;

        // Probability of beam penetration.


        dff = dLAIdz[JJ]; /* '+ prof.dPAIdz[JJ] */
        sumlai += dLAIdz[JJ];


        exp_direct = exp(-dff * markov*Gfunc_solar[JJ] / solar_sine_beta);

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
    nir_dn[jktot] = 1. - fraction_beam;
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

        nir_dn[JJ] = nir_dn[JJP1] * transmission_layer[JJ] / (1. - ADUM[JJP1] * reflectance_layer[JJ]) + SDN[JJ];
        nir_up[JJP1] = ADUM[JJP1] * nir_dn[JJP1] + SUP[JJP1];
    }

    // lower boundary: upward radiation from soil


    nir_up[1] = nir_soil_refl * nir_dn[1] + SUP[1];

    /*
        Iterative calculation of upward diffuse and downward beam +
        diffuse NIR to compute scattering
    */

    ITER = 0;
    IREP = 1;

    ITER += 1;

    while (IREP==1)
    {

        IREP=0;
        for (J = 2; J<=jktot; J++)
        {
            JJ = jktot - J + 1;
            JJP1 = JJ + 1;
            DOWN = transmission_layer[JJ] * nir_dn[JJP1] + nir_up[JJ] * reflectance_layer[JJ] + SDN[JJ];

            if ((fabs(DOWN - nir_dn[JJ])) > .01)
                IREP = 1;

            nir_dn[JJ] = DOWN;
        }

        // upward radiation at soil is reflected beam and downward diffuse

        nir_up[1] = (nir_dn[1] + TBEAM[1]) * nir_soil_refl;

        for (JJ = 2; JJ <=jktot; JJ++)
        {
            JM1 = JJ - 1;

            UP = reflectance_layer[JM1] * nir_dn[JJ] + nir_up[JM1] * transmission_layer[JM1] + SUP[JJ];

            if ((fabs(UP - nir_up[JJ])) > .01)
                IREP = 1;

            nir_up[JJ] = UP;
        }

    }


    // Compute NIR flux densities


    nir_total = nir_beam + nir_diffuse;
    llai = lai;

    for(J = 1; J<=jktot; J++)
    {
        llai -= dLAIdz[J];   // decrement LAI


        // upward diffuse NIR flux density, on the horizontal

        nir_up[J] *= nir_total;

        if(nir_up[J] <= 0.)
            nir_up[J] = .1;


        // downward beam NIR flux density, incident on the horizontal


        beam_flux_nir[J] = beam[J] * nir_total;

        if(beam_flux_nir[J] <= 0.)
            beam_flux_nir[J] = .1;


        // downward diffuse radiaiton flux density on the horizontal

        nir_dn[J] *= nir_total;

        if(nir_dn[J] <= 0.)
            nir_dn[J] = .1;


        // total downward NIR, incident on the horizontal


        NIRTT = beam_flux_nir[J] + nir_dn[J];

    } // next J

    for(J = 1; J<=jtot; J++)
    {



        // normal radiation on sunlit leaves

        if(solar_sine_beta > 0.1)
            nir_normal = nir_beam * Gfunc_solar[J] / solar_sine_beta;
        else
            nir_normal=0;

        NSUNEN = nir_normal * nir_absorbed;

        /*
                 ' Diffuse radiation received on top and bottom of leaves
                 ' drive photosynthesis and energy exchanges
        */

        nir_sh[J] = (nir_dn[J] + nir_up[J]);


        // absorbed radiation, shaded

        nir_sh[J] *= nir_absorbed;


        // plus diffuse component


        nir_sun[J] = NSUNEN + nir_sh[J];  // absorbed NIR on sun leaves
    } // next J

NIRNIGHT:  // jump to here at night since fluxes are zero
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

    m.def("conc", &CONC, "Subroutine to compute scalar concentrations from source estimates and the Lagrangian dispersion matrix",
    py::arg("cref"), py::arg("soilflux"), py::arg("factor"),
    py::arg("sze3"), py::arg("jtot"), py::arg("jtot3"), py::arg("met_zl"), py::arg("delz"),
    py::arg("izref"), py::arg("ustar_ref"), py::arg("ustar"), 
    py::arg("source"), py::arg("cncc"), py::arg("dispersoin_np"));
}