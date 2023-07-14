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
// const double ht = 1;             //   0.55 Canopy height, m
// const double pai = .0;            //    Plant area index
// const double lai = 4;      //  1.65 Leaf area index data are from clip plots and correspond with broadband NDVI estimates

// Gaetan et al 2012 IEEE, vcmax 70, jmax 123
// Erice et al physiologia planatar, 170, 278, alfalfa A-Ci, Vcmax and Jmax
const double vcopt = 170.0 ;   // carboxylation rate at optimal temperature, umol m-2 s-1; from lit
const double jmopt = 278.0;  // electron transport rate at optimal temperature, umol m-2 s-1
const double rd25 = .22;     // dark respiration at 25 C, rd25= 0.34 umol m-2 s-1

//  Universal gas constant

const double rugc = 8.314;              // J mole-1 K-1
const double rgc1000 = 8314;                    // gas constant times 1000.


// Consts for Photosynthesis model and kinetic equations.
// for Vcmax and Jmax.  Taken from Harley and Baldocchi (1995, PCE)


const double hkin = 200000.0;    // enthalpy term, J mol-1
const double skin = 710.0;       // entropy term, J K-1 mol-1
const double ejm = 55000.0;      // activation energy for electron transport, J mol-1
const double evc = 55000.0;      // activation energy for carboxylation, J mol-1


//  Enzyme constants & partial pressure of O2 and CO2
//  Michaelis-Menten K values. From survey of literature.


const double kc25 = 274.6;   // kinetic coef for CO2 at 25 C, microbars
const double ko25 = 419.8;   // kinetic coef for O2 at 25C,  millibars


const double o2 = 210.0;     // 210.0  oxygen concentration  mmol mol-1

// tau is computed on the basis of the Specificity factor (102.33)
// times Kco2/Kh2o (28.38) to convert for value in solution
// to that based in air/
// The old value was 2321.1.

// New value for Quercus robor from Balaguer et al. 1996
// Similar number from Dreyer et al. 2001, Tree Physiol, tau= 2710

const double tau25 = 2904.12;    //  tau coefficient


//  Arrhenius constants
//  Eact for Michaelis-Menten const. for KC, KO and dark respiration
//  These values are from Harley


const double ekc = 80500.0;     // Activation energy for K of CO2; J mol-1
const double eko = 14500.0;     // Activation energy for K of O2, J mol-1
const double erd = 38000.0;     // activation energy for dark respiration, eg Q10=2
const double ektau = -29000.0;  // J mol-1 (Jordan and Ogren, 1984)
const double tk_25 = 298.16;    // absolute temperature at 25 C
const double toptvc = 298.0;    // optimum temperature for maximum carboxylation, was 311
const double toptjm = 298.0;    // optimum temperature for maximum electron transport, was 311
const double eabole=45162;      // activation energy for bole respiration for Q10 = 2.02

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

//  constants for the polynomial equation for saturation vapor pressure-T function, es=f(t)

const double a1en=617.4;
const double a2en=42.22;
const double a3en=1.675;
const double a4en=0.01408;
const double a5en=0.0005818;


//  Ball-Berry stomatal coefficient for stomatal conductance

const double kball = 9.5;

// intercept of Ball-Berry model, mol m-2 s-1 

const double bprime = .0175;           // intercept for H2O

const double bprime16 = 0.0109375;      // intercept for CO2, bprime16 = bprime / 1.6;

// Minimum stomatal resistance, s m-1.  

const double rsm = 145.0;
const double brs=60.0;      // curvature coeffient for light response

//   leaf quantum yield, electrons  

const double qalpha = .22;
const double qalpha2 = 0.0484;   // qalpha squared, qalpha2 = pow(qalpha, 2.0);

//  leaf clumping factor

const double markov = 1.00;

//   Leaf dimension. geometric mean of length and width (m)
const double lleaf = .02;       // leaf length, m


// Diffusivity values for 273 K and 1013 mb (STP) using values from Massman (1998) Atmos Environment
// These values are for diffusion in air.  When used these values must be adjusted for
// temperature and pressure

// nu, Molecular viscosity


const double nuvisc = 13.27;    // mm2 s-1
const double nnu = 0.00001327;  // m2 s-1

// Diffusivity of CO2

const double dc = 13.81;         // mm2 s-1
const double ddc = 0.00001381;   // m2 s-1

//   Diffusivity of heat

const double dh = 18.69;         // mm2 s-1
const double ddh = 0.00001869;   // m2 s-1


//  Diffusivity of water vapor

const double dv = 21.78;         // mm2 s-1
const double ddv = 0.00002178;   // m2 s-1


// Diffusivity of ozone

const double do3=14.44;          // mm2 s-1
const double ddo3 = 0.00001444; // m2 s-1

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
    // llai = lai;

    for(J = 1; J<=jktot; J++)
    {
        // llai -= dLAIdz[J];   // decrement LAI
        // llai -= dLAIdz(J-1);   // decrement LAI


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


void STOMATA(
    int jtot, double lai, double pai, double rcuticle,
    py::array_t<double, py::array::c_style> par_sun_np,
    py::array_t<double, py::array::c_style> par_shade_np,
    py::array_t<double, py::array::c_style> sun_rs_np,
    py::array_t<double, py::array::c_style> shd_rs_np
)
{

    /* ----------------------------------------------

            SUBROUTINE STOMATA

            First guess of rstom to run the energy balance model.
                    It is later updated with the Ball-Berry model.
    -------------------------------------------------
    */

    int JJ;

    double rsfact;

    auto par_sun = par_sun_np.unchecked<1>();
    auto par_shade = par_shade_np.unchecked<1>();
    auto sun_rs = sun_rs_np.mutable_unchecked<1>();
    auto shd_rs = shd_rs_np.mutable_unchecked<1>();

    rsfact=brs*rsm;

    for(JJ = 1; JJ <=jtot; JJ++)
    {


        // compute stomatal conductance
        // based on radiation on sunlit and shaded leaves
        //  m/s.

        //  PAR in units of W m-2

        // if(time_var.lai == pai)
        if(lai == pai)
        {
            // prof.sun_rs[JJ] = sfc_res.rcuticle;
            // prof.shd_rs[JJ] = sfc_res.rcuticle;
            sun_rs(JJ-1) = rcuticle;
            shd_rs(JJ-1) = rcuticle;
        }
        else
        {
            // if(solar.par_sun[JJ] > 5.0)
            //     prof.sun_rs[JJ] = rsm + (rsfact) / solar.par_sun[JJ];
            // else
            //     prof.sun_rs[JJ] = sfc_res.rcuticle;
            if(par_sun(JJ-1) > 5.0)
                sun_rs(JJ-1) = rsm + (rsfact) / par_sun(JJ-1);
            else
                sun_rs(JJ-1) = rcuticle;

            // if(solar.par_shade[JJ] > 5.0)
            //     prof.shd_rs[JJ] = rsm + (rsfact) / solar.par_shade[JJ];
            // else
            //     prof.shd_rs[JJ] = sfc_res.rcuticle;
            if(par_shade(JJ-1) > 5.0)
                shd_rs(JJ-1) = rsm + (rsfact) / par_shade(JJ-1);
            else
                shd_rs(JJ-1) = rcuticle;
        }
    }
    return;
}


double TBOLTZ (double rate, double eakin, double topt, double tl)
{

    // Boltzmann temperature distribution for photosynthesis

    double y, dtlopt,prodt,numm,denom;

    dtlopt = tl - topt;
    prodt = rugc * topt * tl;
    numm = rate * hkin * exp(eakin * (dtlopt) / (prodt));
    denom = hkin - eakin * (1.0 - exp(hkin * (dtlopt) / (prodt)));
    y = numm / denom;
    return y;
}


double TEMP_FUNC(double rate,double eact,double tprime,double tref, double t_lk)
{

    //  Arhennius temperature function

    double y;
    y = rate * exp(tprime * eact / (tref * rugc*t_lk));
    return y;
}


std::tuple<double, double> SOIL_RESPIRATION(double Ts, double base_respiration)
{


    // Computes soil respiration

    /*

    After Hanson et al. 1993. Tree Physiol. 13, 1-15

    reference soil respiration at 20 C, with value of about 5 umol m-2 s-1 from field studies
    */
    double respiration_mole, respiration_mg;

    // soil.base_respiration=8.0;  // at base temp of 22 c, night values, minus plant respiration

    // assume Q10 of 1.4 based on Mahecha et al Science 2010, Ea = 25169

    // soil.respiration_mole = soil.base_respiration * exp((25169. / 8.314) * ((1. / 295.) - 1. / (soil.T_15cm + 273.16)));
    respiration_mole = base_respiration * exp((25169. / 8.314) * ((1. / 295.) - 1. / (Ts + 273.16)));

    // soil wetness factor from the Hanson model, assuming constant and wet soils

    respiration_mole *= 0.86;


    //  convert soilresp to mg m-2 s-1 from umol m-2 s-1

    respiration_mg = respiration_mole * .044;

    return std::make_tuple(respiration_mole, respiration_mg);
    // return;
}


double LAMBDA (double tak)
{
    // Latent heat of Vaporiation, J kg-1
    double y;
    y = 3149000. - 2370. * tak;
    // add heat of fusion for melting ice
    if(tak < 273.)
        y +=333;
    return y;
}


double ES(double tk)
{

    double y, y1, tc;

    tc=tk-273.15;

    // from Jones, es Pa, T oC

    y=613*exp(17.502 * tc/(240.97+tc));



    /*
    // saturation vapor pressure function (mb)
    // T is temperature in Kelvin
    if(t > 0)
    {
    y1 = (54.8781919 - 6790.4985 / t - 5.02808 * log(t));
    y = exp(y1);
    }
    else
    printf("bad es calc");

    */


    return y;
}


double DESDT (double t, double latent18)
{
    // first derivative of es with respect to tk
    //  Pa
    double y;
    // y = ES(t) * fact.latent18  / (rgc1000 * t * t);
    y = ES(t) * latent18  / (rgc1000 * t * t);
    return y;
}


double DES2DT(double T, double latent18)
{
    // The second derivative of the saturation vapor pressure
    // temperature curve, using the polynomial equation of Paw U
    //        a3en=1.675;
    //        a4en=0.01408;
    //        a5en=0.0005818;
    double y;
//        tcel=t-273.16;
//       y=2*a3en+6*a4en*tcel+12*a5en*tcel*tcel;
    // analytical equation seems better than Paw U's 4th order at low and high temperatures
    //  d(g(x)f(x))/dx = f'(x)g(x) + g'(x)f(x)
    // Tk
    y = -2. * ES(T) * LAMBDA(T) * 18. / (rgc1000 * T * T * T) +  DESDT(T,latent18) * LAMBDA(T) * 18. / (rgc1000 * T * T);
    return y;
}

double SFC_VPD (
    double delz, double tlk, double Z, double leleafpt, double latent, double vapor, 
    py::array_t<double, py::array::c_style> rhov_air_np
)
{

    // this function computes the relative humidity at the leaf surface for
    // application in the Ball Berry Equation

    //  latent heat flux, LE, is passed through the function, mol m-2 s-1
    //  and it solves for the humidity at leaf surface

    int J;
    double y, rhov_sfc,e_sfc,vpd_sfc,rhum_leaf;
    double es_leaf;

    auto rhov_air = rhov_air_np.unchecked<1>();

    es_leaf = ES(tlk);    // saturation vapor pressure at leaf temperature

    J = (int)(Z / delz);  // layer number
    // for (J = 1; J <= 7; J++) {
    //     std::cout << rhov_air(J-1) << ' ';
    // }
    // std::cout << '\n';
    // printf("%d %5.4f %5.4f %5.4f %5.4f \n", J, Z, delz, Z/delz, rhov_air(J-1));

    // rhov_sfc = (leleafpt / (fact.latent)) * vapor + rhov_air[J];  /* kg m-3 */
    rhov_sfc = (leleafpt / (latent)) * vapor + rhov_air(J-1);  /* kg m-3 */

    e_sfc = 1000* rhov_sfc * tlk / 2.165;    // Pa
    vpd_sfc = es_leaf - e_sfc;              // Pa
    rhum_leaf = 1. - vpd_sfc / es_leaf;     // 0 to 1.0
    y = rhum_leaf;

    return y;
}


// void PHOTOSYNTHESIS_AMPHI(double Iphoton,double *rstompt, double zzz,double cca,double tlk,
//                           double *leleaf, double *A_mgpt, double *resppt, double *cipnt,
//                           double *wjpnt, double *wcpnt)
std::tuple<double, double, double, double, double, double> PHOTOSYNTHESIS_AMPHI(
    double Iphoton, double delz, double zzz, double ht, double cca,double tlk, double leleaf,
    double vapor, double pstat273, double kballstr, double latent, double co2air, double co2bound_res,
    py::array_t<double, py::array::c_style> rhov_air_np
)
{

    /*

             This program solves a cubic equation to calculate
             leaf photosynthesis.  This cubic expression is derived from solving
             five simultaneous equations for A, PG, cs, CI and GS.
             Stomatal conductance is computed with the Ball-Berry model.
             The cubic derivation assumes that b', the intercept of the Ball-Berry
             stomatal conductance model, is non-zero.

    		 amphistomatous leaf is assumed, computed A on both sides of the leaf

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

            gs - stomatal conductance (mols m-2 s-1), typically 0.01-0.20
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


            Jan 14, 1999 Updated the cubic solutions for photosynthesis.  There are
            times when the restriction that R^2 < Q^3 is violated.  I therefore need
            alternative algorithms to solve for the correct root.

    ===============================================================
    */

    double rstompt, A_mgpt, resppt, cipnt, wjpnt, wcpnt;

    double tprime25, bc, ttemp, gammac;
    double jmax, vcmax,jmaxz, vcmaxz, cs, ci;
    double kct, ko, tau;
    double rd, rdz;
    double rb_mole,gb_mole,dd,b8_dd;
    double rh_leaf, k_rh, gb_k_rh,ci_guess;
    double j_photon,alpha_ps,bbeta,gamma;
    double denom,Pcube,Qcube,Rcube;
    double P2, P3, Q, R;
    double root1, root2;
    double root3, arg_U, ang_L;
    double aphoto, j_sucrose, wj;
    double gs_leaf_mole, gs_co2,gs_m_s;
    double ps_1,delta_1,Aquad1,Bquad1,Cquad1;
    double theta_ps, wc, B_ps, a_ps, E_ps, psguess;
    double sqrprod, product;
    double rt;

    double a, b, c, wp1, wp2, wp, aa, bb,cc, Aps1, Aps2, Aps;

    double rr,qqq, minroot, maxroot, midroot;

    rt = rugc * tlk;                // product of universal gas constant and abs temperature

    tprime25 = tlk - tk_25;       // temperature difference

    ttemp = exp((skin * tlk - hkin) / rt) + 1.0;  // denominator term

    // initialize min and max roots

    minroot= 1e10;
    maxroot=-1e10;
    midroot=0;
    root1=0;
    root2=0;
    root3=0;
    aphoto=0;


    // KC and KO are solely a function of the Arrhenius Eq.


    kct = TEMP_FUNC(kc25, ekc, tprime25, tk_25, tlk);
    ko = TEMP_FUNC(ko25, eko, tprime25, tk_25,tlk);
    tau = TEMP_FUNC(tau25, ektau, tprime25, tk_25,tlk);
    // printf("%5.4f %5.4f %5.4f %5.4f \n", kct, ko, tau, tlk);

    bc = kct * (1.0 + o2 / ko);

    // if(Iphoton < 1)
    if(Iphoton < 0) // TODO: Peishi
        Iphoton = 0;

    /*
            gammac is the CO2 compensation point due to photorespiration, umol mol-1
            Recalculate gammac with the new temperature dependent KO and KC
            coefficients

            gammac = .5 * O2*1000/TAU
    */

    gammac = 500.0 * o2 / tau;

    /*
            temperature corrections for Jmax and Vcmax

            Scale jmopt and VCOPT with a surrogate for leaf nitrogen
            specific leaf weight (Gutschick and Weigel).

            normalized leaf wt is 1 at top of canopy and is 0.35
            at forest floor.  Leaf weight scales linearly with height
            and so does jmopt and vcmax
            zoverh=0.65/HT=zh65

    */


    // The upper layer was in flower, so Ps should be nill..this is unique for alfalfa

    if (zzz < 0.9 * ht)
    {
        jmaxz = jmopt ;
        vcmaxz = vcopt ;
        //vcmaxz = leaf.Vcmax;
        //jmaxz = 29 + 1.64*vcmaxz;
    }
    else
    {
        jmaxz = 0.1 * jmopt ;
        vcmaxz = 0.1 * vcopt ;

        //vcmaxz = 0.1 * leaf.Vcmax;
        //jmaxz = 29 + 1.64*vcmaxz;
    }


    // values for ideal alfalfa and changes in leaf N, Vcmax and reflected NIR

    jmaxz = jmopt;
    vcmaxz = vcopt;
    //vcmaxz = leaf.Vcmax;
    //jmaxz = 29 + 1.64*vcmaxz;

    /*
            Scale rd with height via vcmax and apply temperature
            correction for dark respiration
    */

    rdz=vcmaxz * 0.004657;


    // reduce respiration by 40% in light according to Kok effect, as reported in Amthor


    if(Iphoton > 10)
        rdz *= 0.4;

    rd = TEMP_FUNC(rdz, erd, tprime25, tk_25, tlk);


    // Apply temperature correction to JMAX and vcmax


    jmax = TBOLTZ(jmaxz, ejm, toptjm, tlk);
    vcmax = TBOLTZ(vcmaxz, evc, toptvc, tlk);

    /*
            Compute the leaf boundary layer resistance

            gb_mole leaf boundary layer conductance for CO2 exchange,
            mol m-2 s-1

            RB has units of s/m, convert to mol-1 m2 s1 to be
            consistant with R.

            rb_mole = RBCO2 * .0224 * 1.01 * tlk / (met.pstat * 273.16)
    */

    // rb_mole = bound_layer_res.co2 * tlk * (pstat273);
    rb_mole = co2bound_res * tlk * (pstat273);

    gb_mole = 1. / rb_mole;

    dd = gammac;
    b8_dd = 8 * dd;


    /***************************************

            APHOTO = PG - rd, net photosynthesis is the difference
            between gross photosynthesis and dark respiration. Note
            photorespiration is already factored into PG.

    ****************************************

            coefficients for Ball-Berry stomatal conductance model

            Gs = k A rh/cs + b'

            rh is relative humidity, which comes from a coupled
            leaf energy balance model
    */

    rh_leaf  = SFC_VPD(delz, tlk, zzz, leleaf, latent, vapor, rhov_air_np);

    k_rh = rh_leaf * kballstr;  // combine product of rh and K ball-berry
    // printf("%5.4f %5.4f %5.4f %5.4f \n", leleaf, vapor, o2, tau);

    /*
            Gs from Ball-Berry is for water vapor.  It must be divided
            by the ratio of the molecular diffusivities to be valid
            for A
    */
    k_rh = k_rh / 1.6;      // adjust the coefficient for the diffusion of CO2 rather than H2O

    gb_k_rh = gb_mole * k_rh;

    ci_guess = cca * .7;    // initial guess of internal CO2 to estimate Wc and Wj


    // cubic coefficients that are only dependent on CO2 levels

    // factors of 2 are included

    alpha_ps = 1.0 + (bprime16 / (gb_mole)) - k_rh;
    bbeta = cca * (2*gb_k_rh - 3.0 * bprime16 - 2*gb_mole);
    gamma = cca * cca * gb_mole * bprime16 * 4;
    theta_ps = 2* gb_k_rh - 2 * bprime16;

    /*
            Test for the minimum of Wc and Wj.  Both have the form:

            W = (a ci - ad)/(e ci + b)

            after the minimum is chosen set a, b, e and d for the cubic solution.

            estimate of J according to Farquhar and von Cammerer (1981)


            J photon from Harley
    */


    if (jmax > 0)
        j_photon = qalpha * Iphoton / sqrt(1. +(qalpha2 * Iphoton * Iphoton / (jmax * jmax)));
    else
        j_photon = 0;


    wj = j_photon * (ci_guess - dd) / (4. * ci_guess + b8_dd);


    wc = vcmax * (ci_guess - dd) / (ci_guess + bc);







    if(wj < wc)
    {

        // for Harley and Farquhar type model for Wj

        psguess=wj;

        B_ps = b8_dd;
        a_ps = j_photon;
        E_ps = 4.0;
    }
    else
    {
        psguess=wc;

        B_ps = bc;
        a_ps = vcmax;
        E_ps = 1.0;
    }

    /*
            if wj or wc are less than rd then A would probably be less than zero.  This would yield a
            negative stomatal conductance.  In this case, assume gs equals the cuticular value. This
            assumptions yields a quadratic rather than cubic solution for A

    */




    if (wj <= rd)
        goto quad;

    if (wc <= rd)
        goto quad;

    /*
    cubic solution:

     A^3 + p A^2 + q A + r = 0
    */

    denom = E_ps * alpha_ps;

    Pcube = (E_ps * bbeta + B_ps * theta_ps - a_ps * alpha_ps + E_ps * rd * alpha_ps);
    Pcube /= denom;

    Qcube = (E_ps * gamma + (B_ps * gamma / cca) - a_ps * bbeta + a_ps * dd * theta_ps + E_ps * rd * bbeta + rd * B_ps * theta_ps);
    Qcube /= denom;

    Rcube = (-a_ps * gamma + a_ps * dd * (gamma / cca) + E_ps * rd * gamma + rd * B_ps * gamma / cca);
    Rcube /= denom;


    // Use solution from Numerical Recipes from Press


    P2 = Pcube * Pcube;
    P3 = P2 * Pcube;
    Q = (P2 - 3.0 * Qcube) / 9.0;
    R = (2.0 * P3 - 9.0 * Pcube * Qcube + 27.0 * Rcube) / 54.0;


    /*
            Test = Q ^ 3 - R ^ 2
            if test >= O then all roots are real
    */

    rr=R*R;
    qqq=Q*Q*Q;

    // real roots


    arg_U = R / sqrt(qqq);

    ang_L = acos(arg_U);

    root1 = -2.0 * sqrt(Q) * cos(ang_L / 3.0) - Pcube / 3.0;
    root2 = -2.0 * sqrt(Q) * cos((ang_L + PI2) / 3.0) - Pcube / 3.0;
    root3 = -2.0 * sqrt(Q) * cos((ang_L -PI2) / 3.0) - Pcube / 3.0;

    // rank roots #1,#2 and #3 according to the minimum, intermediate and maximum
    // value


    if(root1 <= root2 && root1 <= root3)
    {   minroot=root1;
        if (root2 <= root3)
        {   midroot=root2;
            maxroot=root3;
        }
        else
        {   midroot=root3;
            maxroot=root2;
        }
    }


    if(root2 <= root1 && root2 <= root3)
    {   minroot=root2;
        if (root1 <= root3)
        {   midroot=root1;
            maxroot=root3;
        }
        else
        {   midroot=root3;
            maxroot=root1;
        }
    }


    if(root3 <= root1 && root3 <= root2)
    {   minroot=root3;
        if (root1 < root2)
        {   midroot=root1;
            maxroot=root2;
        }
        else
        {   midroot=root2;
            maxroot=root1;
        }

    }  // end of the loop for real roots


    // find out where roots plop down relative to the x-y axis


    if (minroot > 0 && midroot > 0 && maxroot > 0)
        aphoto=minroot;


    if (minroot < 0 && midroot < 0 && maxroot > 0)
        aphoto=maxroot;


    if (minroot < 0 && midroot > 0 && maxroot > 0)
        aphoto=midroot;

    /*
             Here A = x - p / 3, allowing the cubic expression to be expressed
             as: x^3 + ax + b = 0
    */

    // aphoto=root3;  // back to original assumption

    /*
            also test for sucrose limitation of photosynthesis, as suggested by
            Collatz.  Js=Vmax/2
    */
    j_sucrose = vcmax / 2. - rd;

    if(j_sucrose < aphoto)
        aphoto = j_sucrose;

    cs = cca - aphoto / (2*gb_mole);

    if(cs > 1000)
        cs=co2air;

    /*
            Stomatal conductance for water vapor


    		alfalfa is amphistomatous...be careful on where the factor of two is applied
    		just did on LE on energy balance..dont want to double count

    		this version should be for an amphistomatous leaf since A is considered on both sides

    */

    gs_leaf_mole = (kballstr * rh_leaf * aphoto / cs) + bprime;


    // convert Gs from vapor to CO2 diffusion coefficient


    gs_co2 = gs_leaf_mole / 1.6;
    // printf("%5.4f %5.4f %5.4f %5.4f \n", aphoto, gs_co2, cs, Pcube);


    /*
            stomatal conductance is mol m-2 s-1
            convert back to resistance (s/m) for energy balance routine
    */

    gs_m_s = gs_leaf_mole * tlk * pstat273;

    // need point to pass rstom out of subroutine

    rstompt = 1.0 / gs_m_s;


    // to compute ci, Gs must be in terms for CO2 transfer


    ci = cs - aphoto / gs_co2;

    /*
             if A < 0 then gs should go to cuticular value and recalculate A
             using quadratic solution
    */


    // recompute wj and wc with ci


    wj = j_photon * (ci - dd) / (4. * ci + b8_dd);

    wc = vcmax * (ci - dd) / (ci + bc);

    /* Collatz uses a quadratic model to compute a dummy variable wp to allow
     for the transition between wj and wc, when there is colimitation.  this
     is important because if one looks at the light response curves of the
     current code one see jumps in A at certain Par values

      theta wp^2 - wp (wj + wc) + wj wc = 0
      a x^2 + b x + c = 0
      x = [-b +/- sqrt(b^2 - 4 a c)]/2a

    */





    a=0.98;
    b= -(wj +wc);
    c=wj*wc;

    wp1=(-b + sqrt(b*b - 4*a*c))/(2*a);
    wp2=(-b - sqrt(b*b - 4*a*c))/(2*a);

    // wp = min (wp1,wp2);

    if(wp1 < wp2)
        wp=wp1;
    else
        wp=wp2;



// beta A^2 - A (Jp+Js) + JpJs = 0

    aa = 0.95;
    bb= -(wp+ j_sucrose);
    cc = wp* j_sucrose;


    Aps1=(-bb + sqrt(bb*bb - 4*aa*cc))/(2*aa);
    Aps2=(-bb - sqrt(bb*bb - 4*aa*cc))/(2*aa);

    // Aps=min(Aps1,Aps2);

    if(Aps1 < Aps2)
        Aps=Aps1;
    else
        Aps = Aps2;

    if(Aps < aphoto && Aps > 0)
        aphoto=Aps - rd;




    if(aphoto <= 0.0)
        goto quad;

    goto OUTDAT;



    // if aphoto < 0  set stomatal conductance to cuticle value

quad:


    gs_leaf_mole = bprime;
    gs_co2 = gs_leaf_mole / 1.6;

    /*
            stomatal conductance is mol m-2 s-1
            convert back to resistance (s/m) for energy balance routine
    */

    gs_m_s = gs_leaf_mole * tlk * (pstat273);

    // need pointer to pass rstom out of subroutine as a pointer

    rstompt = 1.0 / gs_m_s;


    /*
            a quadratic solution of A is derived if gs=ax, but a cubic form occurs
            if gs =ax + b.  Use quadratic case when A is less than zero because gs will be
            negative, which is nonsense

    */

    ps_1 = cca * gb_mole * gs_co2;
    delta_1 = gs_co2 + gb_mole;
    denom = gb_mole * gs_co2;

    Aquad1 = delta_1 * E_ps;
    Bquad1 = -ps_1 * E_ps - a_ps * delta_1 + E_ps * rd * delta_1 - B_ps * denom;
    Cquad1 = a_ps * ps_1 - a_ps * dd * denom - E_ps * rd * ps_1 - rd * B_ps * denom;

    product=Bquad1 * Bquad1 - 4.0 * Aquad1 * Cquad1;

    if (product >= 0)
        sqrprod= sqrt(product);

    aphoto = (-Bquad1 - sqrprod) / (2.0 * Aquad1);
    /*
             Tests suggest that APHOTO2 is the correct photosynthetic root when
             light is zero because root 2, not root 1 yields the dark respiration
             value rd.
    */

    cs = cca - aphoto / gb_mole;
    ci = cs - aphoto / gs_co2;


OUTDAT:

    /*
            compute photosynthesis with units of mg m-2 s-1 and pass out as pointers

            A_mg = APHOTO * 44 / 1000
    */
    A_mgpt = aphoto * .044;

    resppt=rd;

    cipnt=ci;

    wcpnt=wc;

    wjpnt=wj;

    /*

         printf(" cs       ci      gs_leaf_mole      CA     ci/CA  APS  root1  root2  root3\n");
         printf(" %5.1f   %5.1f   %6.3f    %5.1f %6.3f  %6.3f %6.3f %6.3f  %6.3f\n", cs, ci, gs_leaf_mole, cca, ci / cca,aphoto,root1, root2, root3 );

    */

    return std::make_tuple(rstompt, A_mgpt, resppt, cipnt, wjpnt, wcpnt);
}


double UZ (double zzz, double ht, double wnd)
{
    double y,zh, zh2, zh3, y1, uh;
    /*
             U(Z) inside the canopy during the day is about 1.09 u*
             This simple parameterization is derived from turbulence
             data measured in the WBW forest by Baldocchi and Meyers, 1988.
    */

    zh=zzz/ht;

    // use Cionco exponential function

    // uh=input.wnd*log((0.55-0.33)/0.055)/log((2.8-.333)/0.055);
    uh=wnd*log((0.55-0.33)/0.055)/log((2.8-.333)/0.055);

    y=uh*exp(-2.5*(1-zh));

    return y;
}


std::tuple<double, double, double> BOUNDARY_RESISTANCE(
    double delz, double zzz, double ht, double TLF, double grasshof, 
    double press_kPa, double wnd, double pr33, double sc33, double scc33,
    py::array_t<double, py::array::c_style> tair_filter_np
)
{
    /*

    **************************************

    BOUNDARY_RESISTANCE

    This subroutine computes the leaf boundary layer
    resistances for heat, vapor and CO2 (s/m).

    Flat plate theory is used, as discussed in Schuepp (1993) and
    Grace and Wilson (1981).

    We consider the effects of turbulent boundary layers and sheltering.
    Schuepp's review shows a beta factor multiplier is necessary for SH in
    flows with high turbulence.  The concepts and theories used have been
    validated with our work on HNO3 transfer to the forest.


    Schuepp. 1993 New Phytologist 125: 477-507


    Diffusivities have been corrected using the temperature/Pressure algorithm in Massman (1998)


    */


    double Re,Re5,Re8;               // Reynolds numbers
    double Sh_heat,Sh_vapor,Sh_CO2;  // Sherwood numbers

    double graf, GR25;      // Grasshof numbers

    double deltlf;

    double Res_factor;

    double nnu_T_P, ddh_T_P, ddv_T_P, ddc_T_P, T_kelvin;

    double heat, vapor, co2;

    int JLAY;

    auto tair_filter = tair_filter_np.unchecked<1>();

    JLAY = (int)(zzz / delz);

    /*     TLF = solar.prob_beam[JLAY] * prof.sun_tleaf[JLAY] + solar.prob_sh[JLAY] * prof.shd_tleaf[JLAY];  */

    /*     'Difference between leaf and air temperature  */

    // deltlf = (TLF - prof.tair_filter[JLAY]);
    // T_kelvin=prof.tair_filter[JLAY] + 273.16;
    deltlf = (TLF - tair_filter(JLAY-1));
    T_kelvin=tair_filter(JLAY-1) + 273.16;

    if(deltlf > 0)
        // graf = non_dim.grasshof * deltlf / T_kelvin;
        graf = grasshof * deltlf / T_kelvin;
    else
        graf=0;


    // nnu_T_P=nnu*(101.3 /input.press_kPa)*pow((T_kelvin/273.16),1.81);
    nnu_T_P=nnu*(101.3 / press_kPa)*pow((T_kelvin/273.16),1.81);


    // Re = lleaf * UZ(zzz) / nnu_T_P;
    Re = lleaf * UZ(zzz, ht, wnd) / nnu_T_P;

    if (Re > 0.)
        Re5 = sqrt(Re);
    else
    {
        // printf("bad RE in RESHEAT\n");
        Re5=100.;
    }


    Re8 = pow(Re,.8);

    if( Re > 14000.)
    {

        Res_factor=0.036*Re8*betfact;

        /*
        turbulent boundary layer

        SH = .036 * Re8 * pr33*betfact;
        SHV = .036 * Re8 * sc33*betfact;
        SHCO2 = .036 * Re8 * scc33*betfact;

        */

        // Sh_heat = Res_factor * non_dim.pr33;
        // Sh_vapor = Res_factor * non_dim.sc33;
        // Sh_CO2 = Res_factor * non_dim.scc33;
        Sh_heat = Res_factor * pr33;
        Sh_vapor = Res_factor * sc33;
        Sh_CO2 = Res_factor * scc33;

    }
    else
    {

        Res_factor=0.66*Re5*betfact;

        /*
        laminar sublayer

        SH = .66 * Re5 * pr33*betfact;
        SHV = .66 * Re5 * sc33*betfact;
        SHCO2 = .66 * Re5 * scc33*betfact;
        */


        // Sh_heat = Res_factor * non_dim.pr33;
        // Sh_vapor = Res_factor * non_dim.sc33;
        // Sh_CO2 = Res_factor * non_dim.scc33;
        Sh_heat = Res_factor * pr33;
        Sh_vapor = Res_factor * sc33;
        Sh_CO2 = Res_factor * scc33;

    }


    //   If there is free convection

    if(graf / (Re * Re) > 1.)
    {

//     Compute Grashof number for free convection

        if (graf < 100000.)
            GR25 = .5 * pow(graf,.25);
        else
            GR25 = .13 * pow(graf,.33);


        // Sh_heat = non_dim.pr33 * GR25;
        // Sh_vapor = non_dim.sc33 * GR25;
        // Sh_CO2 = non_dim.scc33 * GR25;
        Sh_heat = pr33 * GR25;
        Sh_vapor = sc33 * GR25;
        Sh_CO2 = scc33 * GR25;
    }

    // lfddx=lleaf/ddx


    // Correct diffusivities for temperature and pressure

    // ddh_T_P=ddh*(101.3/input.press_kPa)*pow((T_kelvin/273.16),1.81);
    // ddv_T_P=ddv*(101.3/input.press_kPa)*pow((T_kelvin/273.16),1.81);
    // ddc_T_P=ddc*(101.3/input.press_kPa)*pow((T_kelvin/273.16),1.81);
    ddh_T_P=ddh*(101.3/press_kPa)*pow((T_kelvin/273.16),1.81);
    ddv_T_P=ddv*(101.3/press_kPa)*pow((T_kelvin/273.16),1.81);
    ddc_T_P=ddc*(101.3/press_kPa)*pow((T_kelvin/273.16),1.81);

    // bound_layer_res.heat = lleaf/(ddh_T_P * Sh_heat);
    // bound_layer_res.vapor = lleaf/(ddv_T_P * Sh_vapor);
    // bound_layer_res.co2 = lleaf / (ddc_T_P * Sh_CO2);
    heat = lleaf/(ddh_T_P * Sh_heat);
    vapor = lleaf/(ddv_T_P * Sh_vapor);
    co2 = lleaf / (ddc_T_P * Sh_CO2);

    // if (isinf(bound_layer_res.vapor) == 1)
    // if (isinf(vapor) == 1)
    if ((isinf(vapor) == 1) || (vapor > 9999)) // TODO: Peishi
    {
        // bound_layer_res.vapor = 9999;
        vapor = 9999;
    }

    return std::make_tuple(heat, vapor, co2);
    // return;
}


std::tuple<double, double> FRICTION_VELOCITY(
    double ustar, double H_old, double sensible_heat_flux,
    double air_density, double T_Kelvin
)
{
    // this subroutine updates ustar and stability corrections
    // based on the most recent H and z/L values

    double xzl, logprod, phim;
    double zl;

    // this subroutine is uncessary for CanAlfalfa since we measure and input ustar

    // met.ustar=input.ustar;
    // met.H_old= 0.85 *met.sensible_heat_flux+ 0.15*met.H_old;    // filter sensible heat flux to reduce run to run instability
    // met.zl = -(0.4*9.8*met.H_old*14.75)/(met.air_density*1005.*met.T_Kelvin*pow(met.ustar,3.)); // z/L

    H_old= 0.85*sensible_heat_flux+0.15*H_old;    // filter sensible heat flux to reduce run to run instability
    zl = -(0.4*9.8*H_old*14.75)/(air_density*1005.*T_Kelvin*pow(ustar,3.)); // z/L
    // printf("%5.4f %5.4f %5.4f %5.4f", H_old, air_density, T_Kelvin, ustar);

    return std::make_tuple(H_old, zl);
    // return;
}


std::tuple<double, double, double, double> ENERGY_BALANCE_AMPHI(
    double qrad, double taa, double rhovva, double rvsfc,
    double stomsfc, double air_density, double latent, double press_Pa, double heat
)
{
    /*
            ENERGY BALANCE COMPUTATION for Amphistomatous leaves

            A revised version of the quadratic solution to the leaf energy balance relationship is used.

            Paw U, KT. 1987. J. Thermal Biology. 3: 227-233


             H is sensible heat flux density on the basis of both sides of a leaf
             J m-2 s-1 (W m-2).  Note KC includes a factor of 2 here for heat flux
             because it occurs from both sides of a leaf.
    */

    // printf("%5.4f %5.4f %5.4f %5.4f %5.4f \n", qrad, taa, rhovva, latent, heat);
    // printf("%5.4f %5.4f %5.4f %5.4f %5.4f \n", qrad, taa, rhovva, rvsfc, stomsfc);
    // printf("Energy balance inputs: %5.4f %5.4f %5.4f %5.4f %5.4f %5.4f %5.4f %5.4f %5.4f \n", 
    //     qrad, taa, rhovva, rvsfc, stomsfc, air_density, latent, press_Pa, heat);

    double est, ea, tkta, le2;
    double tk2, tk3, tk4;
    double dest, d2est;
    double lecoef, hcoef, hcoef2, repeat, acoeff, acoef;
    double bcoef, ccoef, product;
    double atlf, btlf, ctlf,vpd_leaf,llout;
    double ke;

    double tsfckpt, lept, H_leafpt, lout_leafpt;

    double latent18 = latent * 18.;



    tkta=taa;   // taa is already in Kelvin

    est = ES(tkta);  //  es(T)  Pa


    // ea  = RHOA * TAA * 1000 / 2.165

    ea = 1000 * rhovva * tkta /2.1650;   // vapor pressure above leaf, Pa rhov is kg m-3



    // Vapor pressure deficit, Pa


    vpd_leaf = est - ea;

    if (vpd_leaf < 0.)
        vpd_leaf = 0;


    // Slope of the vapor pressure-temperature curve, Pa/C
    // evaluate as function of Tk


    dest = DESDT(tkta, latent18);


    // Second derivative of the vapor pressure-temperature curve, Pa/C
    // Evaluate as function of Tk


    d2est = DES2DT(tkta, latent18);


    // Compute products of air temperature, K

    tk2 = tkta * tkta;
    tk3 = tk2 * tkta;
    tk4 = tk3 * tkta;



    // Longwave emission at air temperature, W m-2


    llout = epsigma * tk4;

    /*

            Coefficient for latent heat flux

            Oaks evaporate from only one side. They are hypostomatous.
            Cuticle resistance is included in STOM.

    */

    // stomsfc is already for top and bottom from Photosynthesis_amphi

    // ke = 1./ (rvsfc + stomsfc);  // hypostomatous

    ke = 2/ (rvsfc + 2* stomsfc);  // amphistomatous..to add the Rb, need to assess rstop = rsbottom and add

    lecoef = air_density * .622 * latent * ke / press_Pa;


    // Coefficients for sensible heat flux


    hcoef = air_density*cp/heat;
    hcoef2 = 2 * hcoef;


    // The quadratic coefficients for the a LE^2 + b LE +c =0


    repeat = hcoef + epsigma4 * tk3;

    acoeff = lecoef * d2est / (2. * repeat);
    acoef = acoeff / 4.;

    bcoef = -(repeat) - lecoef * dest / 2. + acoeff * (-qrad / 2. + llout);

    ccoef = repeat * lecoef * vpd_leaf + lecoef * dest * (qrad / 2. - llout) + acoeff * ((qrad * qrad) / 4. + llout * llout - qrad * llout);


    // LE1 = (-BCOEF + (BCOEF ^ 2 - 4 * ACOEF * CCOEF) ^ .5) / (2 * ACOEF)

    product = bcoef * bcoef - 4. * acoef * ccoef;

    // LE2 = (-BCOEF - (BCOEF * BCOEF - 4 * acoef * CCOEF) ^ .5) / (2. * acoef)


    le2= (-bcoef - sqrt(product)) / (2. * acoef);
    lept=le2;  // need to pass pointer out of subroutine
    // *lept=le2;  // need to pass pointer out of subroutine


    // solve for Ts using quadratic solution


    // coefficients to the quadratic solution

    atlf = epsigma12 * tk2 + d2est * lecoef / 2.;

    btlf = epsigma8 * tk3 + hcoef2 + lecoef * dest;

    ctlf = -qrad + 2 * llout + lecoef * vpd_leaf;


    // IF (BTLF * BTLF - 4 * ATLF * CTLF) >= 0 THEN

    product = btlf * btlf - 4 * atlf * ctlf;


    // T_sfc_K = TAA + (-BTLF + SQR(BTLF * BTLF - 4 * ATLF * CTLF)) / (2 * ATLF)
    if (product >= 0)
        tsfckpt = tkta + (-btlf + sqrt(product)) / (2 * atlf);
        // *tsfckpt = tkta + (-btlf + sqrt(product)) / (2 * atlf);
    else
        tsfckpt=tkta;
        // *tsfckpt=tkta;

    // if(*tsfckpt < -230 || *tsfckpt > 325)
    //     *tsfckpt=tkta;
    if(tsfckpt < -230 || tsfckpt > 325)
        tsfckpt=tkta;

    // long wave emission of energy
    lout_leafpt =epsigma2*pow(tsfckpt,4);
    // *lout_leafpt =epsigma2*pow(*tsfckpt,4);

    // H is sensible heat flux
    H_leafpt = hcoef2 * (tsfckpt- tkta);
    // *H_leafpt = hcoef2 * (*tsfckpt- tkta);


    // printf("Energy balance: %5.4f %5.4f %5.4f %5.4f %5.4f \n", qrad, tsfckpt, lept, H_leafpt, lout_leafpt);
    // printf("%5.4f %5.4f %5.4f %5.4f %5.4f \n", qrad, taa, rhovva, latent, heat);
    return std::make_tuple(tsfckpt, lept, H_leafpt, lout_leafpt);
    // return;
}


void ENERGY_AND_CARBON_FLUXES(
    double jtot, double delz, double zzz, double ht, double grasshof, double press_kPa, 
    double co2air, double wnd, double pr33, double sc33, double scc33,
    double rhovva, double air_density,
    double press_Pa, double lai, double pai, double pstat273, double kballstr,
    // Input arrays
    py::array_t<double, py::array::c_style> tair_filter_np, py::array_t<double, py::array::c_style> zzz_ht_np,
    py::array_t<double, py::array::c_style> prob_beam_np, py::array_t<double, py::array::c_style> prob_sh_np,
    py::array_t<double, py::array::c_style> rnet_sun_np, py::array_t<double, py::array::c_style> rnet_sh_np,
    py::array_t<double, py::array::c_style> quantum_sun_np, py::array_t<double, py::array::c_style> quantum_sh_np,
    py::array_t<double, py::array::c_style> can_co2_air_np, py::array_t<double, py::array::c_style> rhov_air_np,
    py::array_t<double, py::array::c_style> rhov_filter_np, py::array_t<double, py::array::c_style> dLAIdz_np,
    // Output arrays
    py::array_t<double, py::array::c_style> sun_rs_np, py::array_t<double, py::array::c_style> shd_rs_np,
    py::array_t<double, py::array::c_style> sun_gs_np, py::array_t<double, py::array::c_style> shd_gs_np,
    py::array_t<double, py::array::c_style> sun_tleaf_np, py::array_t<double, py::array::c_style> shd_tleaf_np,
    py::array_t<double, py::array::c_style> sun_resp_np, py::array_t<double, py::array::c_style> shd_resp_np,
    py::array_t<double, py::array::c_style> sun_wj_np, py::array_t<double, py::array::c_style> shd_wj_np,
    py::array_t<double, py::array::c_style> sun_wc_np, py::array_t<double, py::array::c_style> shd_wc_np,
    py::array_t<double, py::array::c_style> sun_A_np, py::array_t<double, py::array::c_style> shd_A_np,
    py::array_t<double, py::array::c_style> sun_rbh_np, py::array_t<double, py::array::c_style> shd_rbh_np,
    py::array_t<double, py::array::c_style> sun_rbv_np, py::array_t<double, py::array::c_style> shd_rbv_np,
    py::array_t<double, py::array::c_style> sun_rbco2_np, py::array_t<double, py::array::c_style> shd_rbco2_np,
    py::array_t<double, py::array::c_style> sun_ci_np, py::array_t<double, py::array::c_style> shd_ci_np,
    py::array_t<double, py::array::c_style> sun_cica_np, py::array_t<double, py::array::c_style> shd_cica_np,
    py::array_t<double, py::array::c_style> dLEdz_np, py::array_t<double, py::array::c_style> dHdz_np,
    py::array_t<double, py::array::c_style> dRNdz_np, py::array_t<double, py::array::c_style> dPsdz_np,
    py::array_t<double, py::array::c_style> Ci_np, py::array_t<double, py::array::c_style> drbv_np,
    py::array_t<double, py::array::c_style> dRESPdz_np, py::array_t<double, py::array::c_style> dStomCondz_np
)
{
    /*

                    The ENERGY_AND_CARBON_FLUXES routine to computes coupled fluxes
            of energy, water and CO2 exchange, as well as leaf temperature.  Computataions
            are performed for each layer in the canopy and on the sunlit and shaded fractions.

                    Analytical solution for leaf energy balance and leaf temperature is used.  The program
                    is derived from work by Paw U (1986) and was corrected for errors with a re-derivation
            of the equations.  The quadratic form of the solution is used, rather than the quartic
            version that Paw U prefers.

            Estimates of leaf temperature are used to adjust kinetic coefficients for enzyme kinetics,
            respiration and photosynthesis, as well as the saturation vapor pressure at the leaf surface.

            The Analytical solution of the coupled set of equations for photosynthesis and stomatal
            conductance by Baldocchi (1994, Tree Physiology) is used.  This equation is a solution to
            a cubic form of the photosynthesis equation.  The photosynthesis algorithms are from the
            model of Farquhar.  Stomatal conductance is based on the algorithms of Ball-
            Berry and Collatz et al., which couple gs to A

                    Layer 1 is the soil, Layer 30 is top of the canopy

    */

    int JJ;

    double Tair_K_filtered;    //  temporary absolute air temperature
    double T_sfc_K,T_sfc_C;    // surface temperatures in Kelvin and Centigrade
    double H_sun,LE_sun,loutsun,Rn_sun,A_sun;  // energy fluxes on sun leaves
    double LE_shade,H_shade,loutsh,Rn_shade,A_shade;  // energy fluxes on shaded leaves
    double LE_leaf, H_leaf, lout_leaf, resp_shade, resp_sun;
    double rs_sun, rs_shade, A_mg, resp, internal_CO2;
    double wj_leaf, wc_leaf;

    double latent, latent18;
    double tsfckpt, lept, H_leafpt, lout_leafpt;
    double bound_layer_res_heat, bound_layer_res_vapor, bound_layer_res_co2;

    std::tuple <double, double, double> bound_layer_res;
    std::tuple <double, double, double, double> leaf_energy;
    std::tuple <double, double, double, double, double, double> leaf_photosynthesis;

    // Input arrays
    auto tair_filter = tair_filter_np.unchecked<1>();
    auto zzz_ht = zzz_ht_np.unchecked<1>();
    auto prob_beam = prob_beam_np.unchecked<1>();
    auto prob_sh = prob_sh_np.unchecked<1>();
    auto rnet_sun = rnet_sun_np.unchecked<1>();
    auto rnet_sh = rnet_sh_np.unchecked<1>();
    auto quantum_sun = quantum_sun_np.unchecked<1>();
    auto quantum_sh = quantum_sh_np.unchecked<1>();
    auto can_co2_air = can_co2_air_np.unchecked<1>();
    auto rhov_air = rhov_air_np.unchecked<1>();
    auto rhov_filter = rhov_filter_np.unchecked<1>();
    auto dLAIdz = dLAIdz_np.unchecked<1>();
    // Output arrays
    auto sun_rs = sun_rs_np.mutable_unchecked<1>();
    auto shd_rs = shd_rs_np.mutable_unchecked<1>();
    auto sun_gs = sun_gs_np.mutable_unchecked<1>();
    auto shd_gs = shd_gs_np.mutable_unchecked<1>();
    auto sun_tleaf = sun_tleaf_np.mutable_unchecked<1>();
    auto shd_tleaf = shd_tleaf_np.mutable_unchecked<1>();
    auto sun_resp = sun_resp_np.mutable_unchecked<1>();
    auto shd_resp = shd_resp_np.mutable_unchecked<1>();
    auto sun_wj = sun_wj_np.mutable_unchecked<1>();
    auto shd_wj = shd_wj_np.mutable_unchecked<1>();
    auto sun_wc = sun_wc_np.mutable_unchecked<1>();
    auto shd_wc = shd_wc_np.mutable_unchecked<1>();
    auto sun_A= sun_A_np.mutable_unchecked<1>();
    auto shd_A= shd_A_np.mutable_unchecked<1>();
    auto sun_rbh= sun_rbh_np.mutable_unchecked<1>();
    auto shd_rbh= shd_rbh_np.mutable_unchecked<1>();
    auto sun_rbv= sun_rbv_np.mutable_unchecked<1>();
    auto shd_rbv= shd_rbv_np.mutable_unchecked<1>();
    auto sun_rbco2= sun_rbco2_np.mutable_unchecked<1>();
    auto shd_rbco2= shd_rbco2_np.mutable_unchecked<1>();
    auto sun_ci= sun_ci_np.mutable_unchecked<1>();
    auto shd_ci= shd_ci_np.mutable_unchecked<1>();
    auto sun_cica= sun_cica_np.mutable_unchecked<1>();
    auto shd_cica= shd_cica_np.mutable_unchecked<1>();
    auto dLEdz= dLEdz_np.mutable_unchecked<1>();
    auto dHdz= dHdz_np.mutable_unchecked<1>();
    auto dRNdz= dRNdz_np.mutable_unchecked<1>();
    auto dPsdz= dPsdz_np.mutable_unchecked<1>();
    auto Ci= Ci_np.mutable_unchecked<1>();
    auto drbv= drbv_np.mutable_unchecked<1>();
    auto dRESPdz= dRESPdz_np.mutable_unchecked<1>();
    auto dStomCondz= dStomCondz_np.mutable_unchecked<1>();

    for (JJ=1; JJ <= jtot; JJ++)
    {

        // zero summing values

        H_sun=0;
        LE_sun=0;
        Rn_sun=0;
        loutsun=0;
        rs_sun=0;
        // prof.sun_gs[JJ]=0;
        sun_gs(JJ-1)=0;
        A_sun=0;
        A_mg=0;
        resp=0;
        resp_shade=0;
        resp_sun=0;
        wc_leaf=0;
        wj_leaf=0;
        internal_CO2=0;

        /*

            First compute energy balance of sunlit leaf, then
            repeat and compute algorithm for shaded leaves.

            The stomatal resistances on the sunlit and shaded leaves are pre-estimated
            as a function of PAR using STOMATA

            Remember layer is two-sided so include upper and lower streams
            are considered.

            KC is the convective transfer coeff. (W M-2 K-1). A factor
            of 2 is applied since convective heat transfer occurs
            on both sides of leaves.

            To calculate the total leaf resistance we must combine stomatal
            and the leaf boundary layer resistance.  Many crops are amphistomatous
            so KE must be multiplied by 2.  Deciduous forest, on the other hand
            has stomata on one side of the leaf.

        */

        Tair_K_filtered = tair_filter(JJ-1) + 273.16;  // absolute air temperature


        // Initialize surface temperature with air temperature

        T_sfc_K = Tair_K_filtered;



        //      Energy balance on sunlit leaves


        //  update latent heat with new temperature during each call of this routine

        latent = LAMBDA(T_sfc_K);
        latent18=latent*18.;
        // printf("%5.4f %5.4f \n", T_sfc_K, latent);

        if(prob_beam(JJ-1) > 0.0)
        {

            // initial stomatal resistance as a function of PAR. Be careful and use
            // the light dependent version only for first iteration
            rs_sun = sun_rs(JJ-1);
            // printf("rs_sun: %5.4f", rs_sun);


            //   Compute the resistances for heat and vapor transfer, rh and rv,
            //   for each layer, s/m
            // BOUNDARY_RESISTANCE(prof.ht[JJ],prof.sun_tleaf[JJ]);
            // bound_layer_res_heat, bound_layer_res_vapor, bound_layer_res_co2 = BOUNDARY_RESISTANCE(
            bound_layer_res = BOUNDARY_RESISTANCE(
                delz, zzz_ht(JJ-1), ht, sun_tleaf(JJ-1), grasshof, press_kPa, wnd, pr33, sc33, scc33, tair_filter_np
            );
            bound_layer_res_heat = std::get<0>(bound_layer_res);
            bound_layer_res_vapor = std::get<1>(bound_layer_res);
            bound_layer_res_co2 = std::get<2>(bound_layer_res);
            // printf("bound layer res: %5.4f %5.4f %5.4f %5.4f\n", press_Pa,bound_layer_res_heat, bound_layer_res_vapor, bound_layer_res_co2);

            // compute energy balance of sunlit leaves
            // ENERGY_BALANCE_AMPHI(solar.rnet_sun[JJ], &T_sfc_K, Tair_K_filtered, prof.rhov_filter[JJ], bound_layer_res.vapor, rs_sun, &LE_leaf, &H_leaf, &lout_leaf);
            // T_sfc_K, LE_leaf, H_leaf, lout_leaf = ENERGY_BALANCE_AMPHI(
            leaf_energy = ENERGY_BALANCE_AMPHI(
                rnet_sun(JJ-1), T_sfc_K, rhovva, rhov_filter(JJ-1), rs_sun, air_density, 
                latent, press_Pa, bound_layer_res_heat
            );
            T_sfc_K = std::get<0>(leaf_energy);
            LE_leaf = std::get<1>(leaf_energy);
            H_leaf = std::get<2>(leaf_energy);
            lout_leaf = std::get<3>(leaf_energy);
            // printf("%5.4f %5.4f %5.4f %5.4f %5.4f \n", rnet_sun(JJ-1), T_sfc_K, LE_leaf, H_leaf, lout_leaf);

            // compute photosynthesis of sunlit leaves if leaves have emerged
            if(lai > pai)
                // PHOTOSYNTHESIS_AMPHI(solar.quantum_sun[JJ], &rs_sun, prof.ht[JJ],
                //                      prof.co2_air[JJ], T_sfc_K, &LE_leaf, &A_mg, &resp, &internal_CO2,
                //                      &wj_leaf,&wc_leaf);
                // rs_sun, A_mg, resp, internal_CO2, wj_leaf, wc_leaf = PHOTOSYNTHESIS_AMPHI(
                leaf_photosynthesis = PHOTOSYNTHESIS_AMPHI(
                    quantum_sun(JJ-1), delz, zzz_ht(JJ-1), ht, can_co2_air(JJ-1), T_sfc_K, LE_leaf,
                    bound_layer_res_vapor, pstat273, kballstr, latent, co2air, bound_layer_res_co2, rhov_air_np
                );
                rs_sun = std::get<0>(leaf_photosynthesis);
                A_mg = std::get<1>(leaf_photosynthesis);
                resp = std::get<2>(leaf_photosynthesis);
                internal_CO2 = std::get<3>(leaf_photosynthesis);
                wj_leaf = std::get<4>(leaf_photosynthesis);
                wc_leaf = std::get<5>(leaf_photosynthesis);

            // Assign values of function to the LE and H source/sink strengths

            T_sfc_C=T_sfc_K-273.16;             // surface temperature, Centigrade
            H_sun = H_leaf;                                 // sensible heat flux
            LE_sun = LE_leaf;                                       // latent heat flux
            sun_tleaf(JJ-1) = T_sfc_C;
            loutsun = lout_leaf;                            // long wave out
            Rn_sun = rnet_sun(JJ-1) - lout_leaf;  // net radiation
            // printf("Output -- %5.4f %5.4f %5.4f %5.4f %5.4f \n", rnet_sun(JJ-1), T_sfc_C, LE_leaf, H_leaf, lout_leaf);

            A_sun = A_mg;                           // leaf photosynthesis, mg CO2 m-2 s-1
            resp_sun=resp;                          // stomatal resistance

            sun_resp(JJ-1)=resp;     // respiration on sun leaves
            sun_gs(JJ-1)=1./rs_sun;      // stomatal conductance
            sun_wj(JJ-1)=wj_leaf;
            sun_wc(JJ-1)=wc_leaf;
            sun_A(JJ-1) = A_sun*1000./mass_CO2;  // micromolC m-2 s-1
            sun_rs(JJ-1) = rs_sun;
            sun_rbh(JJ-1) = bound_layer_res_heat;
            sun_rbv(JJ-1) = bound_layer_res_vapor;
            sun_rbco2(JJ-1) = bound_layer_res_co2;
            sun_ci(JJ-1)=internal_CO2;
            // prof.sun_resp[JJ]=resp;     // respiration on sun leaves
            // prof.sun_gs[JJ]=1./rs_sun;      // stomatal conductance
            // prof.sun_wj[JJ]=wj_leaf;
            // prof.sun_wc[JJ]=wc_leaf;
            // prof.sun_A[JJ] = A_sun*1000./mass_CO2;  // micromolC m-2 s-1
            // prof.sun_rs[JJ] = rs_sun;
            // prof.sun_rbh[JJ] = bound_layer_res.heat;
            // prof.sun_rbv[JJ] = bound_layer_res.vapor;
            // prof.sun_rbco2[JJ] = bound_layer_res.co2;
            // prof.sun_ci[JJ]=internal_CO2;
        }

        //    Energy balance on shaded leaves

        // initial value of stomatal resistance based on light
        rs_shade = shd_rs(JJ-1);


        // boundary layer resistances on shaded leaves.  With different
        // surface temperature, the convective effect may differ from that
        // computed on sunlit leaves
        // BOUNDARY_RESISTANCE(prof.ht[JJ],prof.shd_tleaf[JJ]);
        // bound_layer_res_heat, bound_layer_res_vapor, bound_layer_res_co2 = BOUNDARY_RESISTANCE(
        bound_layer_res = BOUNDARY_RESISTANCE(
            delz, zzz_ht(JJ-1), ht, shd_tleaf(JJ-1), grasshof, press_kPa, wnd, pr33, sc33, scc33, tair_filter_np
        );
        bound_layer_res_heat = std::get<0>(bound_layer_res);
        bound_layer_res_vapor = std::get<1>(bound_layer_res);
        bound_layer_res_co2 = std::get<2>(bound_layer_res);
        // printf("bound layer res: %5.4f %5.4f %5.4f %5.4f\n", press_Pa,bound_layer_res_heat, bound_layer_res_vapor, bound_layer_res_co2);

        // Energy balance of shaded leaves
        // ENERGY_BALANCE_AMPHI(solar.rnet_sh[JJ], &T_sfc_K, Tair_K_filtered, prof.rhov_filter[JJ], bound_layer_res.vapor, rs_shade, &LE_leaf, &H_leaf, &lout_leaf);
        // T_sfc_K, LE_leaf, H_leaf, lout_leaf = ENERGY_BALANCE_AMPHI(
        leaf_energy = ENERGY_BALANCE_AMPHI(
            rnet_sh(JJ-1), T_sfc_K, rhovva, rhov_filter(JJ-1), rs_shade, air_density, 
            latent, press_Pa, bound_layer_res_heat
        );
        T_sfc_K = std::get<0>(leaf_energy);
        LE_leaf = std::get<1>(leaf_energy);
        H_leaf = std::get<2>(leaf_energy);
        lout_leaf = std::get<3>(leaf_energy);

        // compute photosynthesis and stomatal conductance of shaded leaves

        if(lai > pai)
            // PHOTOSYNTHESIS_AMPHI(solar.quantum_sh[JJ], &rs_shade,prof.ht[JJ], prof.co2_air[JJ],
            //                      T_sfc_K, &LE_leaf, &A_mg, &resp, &internal_CO2,&wj_leaf,&wc_leaf);
            // rs_sun, A_mg, resp, internal_CO2, wj_leaf, wc_leaf = PHOTOSYNTHESIS_AMPHI(
            leaf_photosynthesis = PHOTOSYNTHESIS_AMPHI(
                quantum_sh(JJ-1), delz, zzz_ht(JJ-1), ht, can_co2_air(JJ-1), T_sfc_K, LE_leaf,
                bound_layer_res_vapor, pstat273, kballstr, latent, co2air, bound_layer_res_co2, rhov_air_np
            );
            rs_shade = std::get<0>(leaf_photosynthesis);
            A_mg = std::get<1>(leaf_photosynthesis);
            resp = std::get<2>(leaf_photosynthesis);
            internal_CO2 = std::get<3>(leaf_photosynthesis);
            wj_leaf = std::get<4>(leaf_photosynthesis);
            wc_leaf = std::get<5>(leaf_photosynthesis);

        // re-assign variable names from functions output

        T_sfc_C=T_sfc_K-273.16;
        LE_shade = LE_leaf;
        H_shade = H_leaf;
        loutsh = lout_leaf;
        Rn_shade = rnet_sh(JJ-1) - lout_leaf;

        // prof.shd_wj[JJ]=wj_leaf;
        // prof.shd_wc[JJ]=wc_leaf;
        A_shade = A_mg;
        resp_shade=resp;

        shd_wj(JJ-1)=wj_leaf;
        shd_wc(JJ-1)=wc_leaf;
        shd_resp(JJ-1)=resp;
        shd_tleaf(JJ-1) = T_sfc_C;
        shd_gs(JJ-1)=1./rs_shade;
        shd_A(JJ-1) = A_shade*1000/mass_CO2;   // micromolC m-2 s-1
        shd_rs(JJ-1) = rs_shade;
        shd_rbh(JJ-1) = bound_layer_res_heat;
        shd_rbv(JJ-1) = bound_layer_res_vapor;
        shd_rbco2(JJ-1) = bound_layer_res_co2;
        shd_ci(JJ-1)= internal_CO2;
        // prof.shd_wj[JJ]=wj_leaf;
        // prof.shd_wc[JJ]=wc_leaf;
        // prof.shd_resp[JJ]=resp;
        // prof.shd_tleaf[JJ] = T_sfc_C;
        // prof.shd_gs[JJ]=1./rs_shade;
        // prof.shd_A[JJ] = A_shade*1000/mass_CO2;   // micromolC m-2 s-1
        // prof.shd_rs[JJ] = rs_shade;
        // prof.shd_rbh[JJ] = bound_layer_res.heat;
        // prof.shd_rbv[JJ] = bound_layer_res.vapor;
        // prof.shd_rbco2[JJ] = bound_layer_res.co2;
        // prof.shd_ci[JJ]= internal_CO2;


        // compute layer energy fluxes, weighted by leaf area and sun and shaded fractions
        // prof.dLEdz[JJ] = prof.dLAIdz[JJ] * (prob_beam[JJ] * LE_sun + solar.prob_sh[JJ] * LE_shade);
        // prof.dHdz[JJ] = prof.dLAIdz[JJ] * (prob_beam[JJ] * H_sun + solar.prob_sh[JJ] * H_shade);
        // prof.dRNdz[JJ] = prof.dLAIdz[JJ] * (prob_beam[JJ] * Rn_sun + solar.prob_sh[JJ] * Rn_shade);
        dLEdz(JJ-1) = dLAIdz(JJ-1) * (prob_beam(JJ-1) * LE_sun + prob_sh(JJ-1) * LE_shade);
        dHdz(JJ-1) = dLAIdz(JJ-1) * (prob_beam(JJ-1) * H_sun + prob_sh(JJ-1) * H_shade);
        dRNdz(JJ-1) = dLAIdz(JJ-1) * (prob_beam(JJ-1) * Rn_sun + prob_sh(JJ-1) * Rn_shade);


        // photosynthesis of the layer,  prof.dPsdz has units mg m-3 s-1
        // prof.dPsdz[JJ] = prof.dLAIdz[JJ] * (A_sun * prob_beam[JJ] + A_shade * solar.prob_sh[JJ]);
        // prof.Ci[JJ] = (prof.sun_ci[JJ] * prob_beam[JJ] + prof.shd_ci[JJ] * solar.prob_sh[JJ]);
        // prof.shd_cica[JJ]=prof.shd_ci[JJ]/prof.co2_air[JJ];
        // prof.sun_cica[JJ]=prof.sun_ci[JJ]/prof.co2_air[JJ];
        dPsdz(JJ-1) = dLAIdz(JJ-1) * (A_sun * prob_beam(JJ-1) + A_shade * prob_sh(JJ-1));
        Ci(JJ-1) = (sun_ci(JJ-1) * prob_beam(JJ-1) + shd_ci(JJ-1) * prob_sh(JJ-1));
        shd_cica(JJ-1)=shd_ci(JJ-1)/can_co2_air(JJ-1);
        sun_cica(JJ-1)=sun_ci(JJ-1)/can_co2_air(JJ-1);

        // scaling boundary layer conductance for vapor, 1/rbv
        // prof.drbv[JJ] = (prob_beam[JJ]/ prof.sun_rbv[JJ] + solar.prob_sh[JJ]/ prof.shd_rbv[JJ]);
        drbv(JJ-1) = (prob_beam(JJ-1)/ sun_rbv(JJ-1) + prob_sh(JJ-1)/ shd_rbv(JJ-1));


        // photosynthesis of layer, prof.dPsdz has units of micromoles m-2 s-1
        // prof.dPsdz[JJ] = prof.dLAIdz[JJ] * (prof.sun_A[JJ] * prob_beam[JJ] +
        //                                     prof.shd_A[JJ] * solar.prob_sh[JJ]);
        dPsdz(JJ-1) = dLAIdz(JJ-1) * (sun_A(JJ-1) * prob_beam(JJ-1) + shd_A(JJ-1) * prob_sh(JJ-1));


        //  respiration of the layer, micromol m-2 s-1
        // prof.dRESPdz[JJ] = prof.dLAIdz[JJ] * (resp_sun * prob_beam[JJ] + resp_shade * solar.prob_sh[JJ]);
        dRESPdz(JJ-1) = dLAIdz(JJ-1) * (resp_sun * prob_beam(JJ-1) + resp_shade * prob_sh(JJ-1));

        // prof.dStomCondz has units of: m s-1
        // prof.dStomCondz[JJ] = prof.dLAIdz[JJ] * (prob_beam[JJ]*prof.sun_gs[JJ] + solar.prob_sh[JJ]*prof.shd_gs[JJ]);
        dStomCondz(JJ-1) = dLAIdz(JJ-1) * (prob_beam(JJ-1)*sun_gs(JJ-1) + prob_sh(JJ-1)*shd_gs(JJ-1));

        /*

                printf(" HSUN      LESUN   AVAIL ENERGY    lout_leaf    day_local\n");
                printf(" %6.1f     %6.1f     %6.1f         %6.1f   %6i\n", H_sun, LE_sun, solar.rnet_sun[JJ], loutsun, jd);
                printf("\n");
                printf(" HSH      LESH     AVAIL ENERGY  lout_leaf   TIME \n");
                printf(" %6.2f    %6.2f      %6.2f       %6.2f   %6.2f\n", H_shade, LE_shade, solar.rnet_sh[JJ], loutsh, time_var.local_time);
                printf("\n");
                printf(" TL SUN   TL SH     TA      RHOV     CO2\n");
                printf("  %6.2f    %6.2f    %6.2f    %7.5f   %6.2f\n", prof.sun_tleaf[JJ], prof.shd_tleaf[JJ], prof.tair[JJ], prof.rhov_air[JJ], prof.co2_air[JJ]);
                printf("\n");
                printf(" TSOIL    soilevap    soilheat    SOILRAD    GSOIL\n");
                printf(" %6.3f    %5.2f       %5.2f      %5.2f       %5.2f\n",soil.sfc_temperature, soilevap, soilheat,rnsoil - loutsoil, soil.gsoil);
                printf("\n");
                printf("\n");
                printf("LAYER    LAI  solar.sin_beta \n");
                printf(" %3i   %6.2f   %6.3f \n", JJ, time_var.lai, solar.sin_beta);

          */

    }        // next JJ

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

    m.def("stomata", &STOMATA, "Subroutine to provide first guess of rstom to run the energy balance model.",
    py::arg("jtot"), py::arg("lai"), py::arg("pai"), py::arg("rcuticle"),
    py::arg("par_sun_np"), py::arg("par_shade_np"), py::arg("sun_rs_np"), py::arg("shd_rs_np")); 

    m.def("tboltz", &TBOLTZ, "Subroutine to calculate Boltzmann temperature distribution for photosynthesis.",
    py::arg("rate"), py::arg("eakin"), py::arg("topt"), py::arg("tl")); 

    m.def("temp_func", &TEMP_FUNC, "Subroutine to perform Arhennius temperature function.",
    py::arg("rate"), py::arg("eact"), py::arg("tprime"), py::arg("tref"), py::arg("t_lk")); 

    m.def("soil_respiration", &SOIL_RESPIRATION, "Subroutine to compute soil respiration.",
    py::arg("Ts"), py::arg("base_respiration")); 

    m.def("es", &ES, "Subroutine to compute saturation vapor pressure given temperature.", py::arg("tk")); 

    m.def("llambda", &LAMBDA, "Subroutine to compute latent heat of vaporization.", py::arg("tak")); 

    m.def("desdt", &DESDT, "Subroutine to compute the first derivative of es with respect to tk.", 
    py::arg("t"), py::arg("latent18")); 

    m.def("des2dt", &DES2DT, "Subroutine to compute the second derivative of the saturation vapor pressure temperature curve.", 
    py::arg("T"), py::arg("latent18")); 

    m.def("sfc_vpd", &SFC_VPD, "Subroutine to compute the relative humidity at the leaf surface for application in the Ball Berry Equation.",
    py::arg("delz"), py::arg("tlk"), py::arg("Z"), py::arg("leleafpt"),
    py::arg("latent"), py::arg("vapor"), py::arg("rhov_air_np")); 

    m.def("photosynthesis_amphi", &PHOTOSYNTHESIS_AMPHI, "Subroutine to compute photosynthesis on the amphistomatous leaves.",
    py::arg("Iphoton"), py::arg("delz"), py::arg("zzz"), py::arg("ht"), py::arg("cca"),
    py::arg("leleaf"), py::arg("tlk"), py::arg("vapor"), py::arg("pstat273"), py::arg("kballstr"),
    py::arg("latent"), py::arg("co2air"), py::arg("co2bound_res"), py::arg("rhov_air_np")); 

    m.def("uz", &UZ, "Subroutine to compute wind speed as a function of z.", 
    py::arg("zzz"), py::arg("ht"), py::arg("wnd")); 

    m.def("boundary_resistance", &BOUNDARY_RESISTANCE, "Subroutine to compute leaf boundary layer resistances for heat, water, CO2.",
    py::arg("delz"), py::arg("zzz"), py::arg("ht"), py::arg("TLF"), py::arg("grasshof"),
    py::arg("press_kPa"), py::arg("wnd"), py::arg("pr33"), py::arg("sc33"), py::arg("scc33"), 
    py::arg("tair_filter_np")); 

    m.def("friction_velocity", &FRICTION_VELOCITY, "Subroutine to update friction velocity with new z/L", 
    py::arg("ustar"), py::arg("H_old"), py::arg("sensible_heat_flux"), py::arg("air_density"), py::arg("T_Kelvin"));

    m.def("energy_balance_amphi", &ENERGY_BALANCE_AMPHI, "Subroutine to calcuate the energy balance on the amphistomatous leaves.", 
    py::arg("qrad"), py::arg("taa"), py::arg("rhovva"), py::arg("rvsfc"),
    py::arg("stomsfc"), py::arg("air_density"), py::arg("latent"), py::arg("press_Pa"), py::arg("heat"));

    m.def("energy_and_carbon_fluxes", &ENERGY_AND_CARBON_FLUXES, "Subroutine to compute coupled fluxes of energy, water and CO2 exchange, as well as leaf temperature.", 
    py::arg("jtot"), py::arg("delz"), py::arg("zzz"), py::arg("ht"), py::arg("grasshof"), py::arg("press_kPa"),
    py::arg("co2air"), py::arg("wnd"), py::arg("pr33"), py::arg("sc33"), py::arg("scc33"),
    py::arg("rhovva"), py::arg("air_density"),
    py::arg("press_Pa"), py::arg("lai"), py::arg("pai"), py::arg("pstat273"), py::arg("kballstr"),
    // Input arrays
    py::arg("tair_filter_np"), py::arg("zzz_ht_np"), py::arg("prob_beam_np"), py::arg("prob_sh_np"), py::arg("rnet_sun_np"), py::arg("rnet_sh_np"),
    py::arg("quantum_sun_np"), py::arg("quantum_sh_np"), py::arg("can_co2_air_np"), py::arg("rhov_air_np"), py::arg("rhov_filter_np"), py::arg("dLAIdz_np"),
    // Output arrays
    py::arg("sun_rs_np"), py::arg("shd_rs_np"), py::arg("sun_gs_np"), py::arg("shd_gs_np"), py::arg("sun_tleaf_np"), py::arg("shd_tleaf_np"),
    py::arg("sun_resp_np"), py::arg("shd_resp_np"), py::arg("sun_wj_np"), py::arg("shd_wj_np"), py::arg("sun_wc_np"), py::arg("shd_wc_np"),
    py::arg("sun_A_np"), py::arg("shd_A_np"), py::arg("sun_rbh_np"), py::arg("shd_rbh_np"),
    py::arg("sun_rbv_np"), py::arg("shd_rbv_np"), py::arg("sun_rbco2_np"), py::arg("shd_rbco2_np"), py::arg("sun_ci_np"), py::arg("shd_ci_np"),
    py::arg("sun_cica_np"), py::arg("shd_cica_np"), py::arg("dLEdz_np"), py::arg("dHdz_np"), py::arg("dRNdz_np"), py::arg("dPsdz_np"),
    py::arg("Ci_np"), py::arg("drbv_np"), py::arg("dRESPdz_np"), py::arg("dStomCondz_np"));

    m.def("conc", &CONC, "Subroutine to compute scalar concentrations from source estimates and the Lagrangian dispersion matrix",
    py::arg("cref"), py::arg("soilflux"), py::arg("factor"),
    py::arg("sze3"), py::arg("jtot"), py::arg("jtot3"), py::arg("met_zl"), py::arg("delz"),
    py::arg("izref"), py::arg("ustar_ref"), py::arg("ustar"), 
    py::arg("source"), py::arg("cncc"), py::arg("dispersoin_np"));
}