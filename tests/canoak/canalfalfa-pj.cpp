
// #include "stdafx.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
// #include <io.h>
#include <sstream>
#include <float.h>
#include <string.h>
#include <errno.h>
#include <algorithm>


#define PI 3.14159                              // pi
#define sze 32                                  // canopy layers is 30 with flex
#define sze3 152                                // 5 times canopy layers is 150 with flex
#define szeang 19                               // number of sky angle classes is 18 with flex
#define soilsze 12                              // number of soil layers is 10 with flex
#define PI180 0.017453292                                               // pi divided by 180, radians per degree
#define PI9 2.864788976
#define PI2 6.283185307     // 2 time pi





/*

----------------------------------------------------


        12-17-2019, CanAlfalfa.cpp

		run the model from days 180 to 194

		produce new input file
		produce new disp file

		run with NIRv

		Had set soil.nir reflect to zero for NIRv
		back to normal value for diffuse fraction




        DENNIS BALDOCCHI
        Ecosystem Science Division
        Department of Environmental Science, Policy and Management
                & Berkeley Atmospheric Science Center
        345 Hilgard Hall
        University of California, Berkeley
        Berkeley, CA 94720-3110
        baldocchi@berkeley.edu
        510-642-2874



-------------------------------------------------

 CANVEG:

         CANVEG is a coupled biophysical and ecophysiolgical model that
         computes fluxes of water, heat and CO2 exchange within vegetation
         canopies and between the canopy and the atmosphere. In doing so CANOAK
         computes the canopy microclimate (light, wind, temperature, humidity and
         CO2), which provides drivers for physiological processes such as photosynthesis,
         respiration, transpiration and stomatal conductance.  The canopy is divided into
         30 layers and at each layer the energy balance, photosynthesis, transpiration,
         stomatal conductance and respiration of sunlit and shaded leaves is computed.
         Stomatal conductance is computed using a model that is linked to photosynthesis.


        This version has been cleaned up from the original using modular subroutines and
        structures to declare shared variables. It is compiled on Microsoft C++ 6.0


         The mechanistic and biochemical photosynthesis model of Farquhar is used to scale
         leaf CO2 exchange to the canopy.

         Stomatal conductance is computed using the Ball-Berry routine, that
         is dependent on leaf photosynthesis, relative humidity and the
         CO2 concentration at the leaf's surface.

         Photosynthesis and stomatal conductance are computed using a
         cubic analytical solution I developed from the coupled models
         during my stay in Viterbo, 92/93 (Baldocchi, 1994).

         Photosynthetic parameters (Vcmax and Jmax) are scaled
         with height according to specific leaf weight.  SLW is a
         surrogate for the effect of leaf nitrogen on these parameters.
         Kinetic photosynthetic coefficients are derived from Peter
         Harley's 1992 field measurements at WBW.  For the seasonal runs Vcmax and
         Jmax are scaled to seasonal changes in leaf area index.

         This model calculates isoprene emissions from the forest canopy using
         algorithms by Guenther.

         Turbulent diffusion is based on Lagrangian theory. A dispersion matrix is
         calculated from the random walk algorithm of Thomson (from program MOVTHM.C).
         The dispersion matrix is scaled according to u*.  The dispersion matrix also
         scales with stability, since it is dependent on sigma w and sigma w depends on z/L.
         Dispersion matrices's functional dependence on stability have been derived by
         regression against 5 stability classes.

         Light profiles are computed on the basis of layers with constant
         DELTA Z to be compatible with the meteorological model.  Clumping of
         foliage is consider via use of the Markov model for the probability
         of beam penetration.  The penetration and scattering routines of Norman (1979) are
         used, these use a slab, one dimensional 'adding' method for scattering.
         The probability of sunlit leaf area is computed using the
		 Markov model which introduces a clumping factor to the
         Poisson eq.


         Feedbacks are considered between the source sink function,
         which is a function of C, and the dispersed concentration field,
         which is a function of sources and sinks.

         Solar elevation is computed with the algorithm of Welgraven

         Soil energy balance and heat fluxes are computed using a
     numerical solution to the Fourier heat transfer equation.
     after Campbell.

     Analytical (quadratic) solutions are used for the leaf and soil surface energy balances

         Overall the model requires simple environmental inputs. Solar
     radiation, wind speed, air temperature, humidity, CO2 and a
     deep 32 cm soil temperature, at least, are needed to calculate
     fluxes.  Of course parameters describing leaf area, height and
     physiological capacity of the vegetation are needed, but these
     are canopy specific.


     Information on the photosynthetic and stomatal conductance model are reported in:

     Baldocchi, D.D. 1994. An analytical solution for coupled leaf photosynthesis
     and stomatal conductance models. Tree Physiology 14: 1069-1079.


     Information on the leaf photosynthetic parameters can be found in:

     Harley, P.C. and Baldocchi, 1995.Scaling carbon dioxide and water vapor exchange
     from leaf to canopy in a deciduous forest:leaf level parameterization.
     Plant, Cell and Environment. 18: 1146-1156.

     Wilson, K.B., D.D. Baldocchi and P.J. Hanson. 2000. Spatial and seasonal variability of
     photosynthesis parameters and their relationship to leaf nitrogen in a deciduous forest.
     Tree Physiology. 20, 565-587.


     Tests of the model are reported in:

     Baldocchi, D.D. and P.C. Harley. 1995. Scaling carbon dioxide and water vapor
     exchange from leaf to canopy in a deciduous forest: model testing and application.
     Plant, Cell and Environment. 18: 1157-1173.


	 Baldocchi, D.D. 1997. Measuring and modeling carbon dioxide and water vapor
     exchange over a temperate broad-leaved forest during the 1995 summer drought.
     Plant, Cell and Environment. 20: 1108-1122

     Baldocchi, D.D and T.P. Meyers. 1998. On using eco-physiological, micrometeorological
     and biogeochemical theory to evaluate carbon dioxide, water vapor and gaseous deposition
     fluxes over vegetation. Agricultural and Forest Meteorology 90: 1-26.

     Baldocchi, D.D. Fuentes, J.D., Bowling, D.R, Turnipseed, A.A. Monson, R.K. 1999. Scaling
     isoprene fluxes from leaves to canopies: test cases over a boreal aspen and a mixed species temperate
     forest. J. Applied Meteorology. 38, 885-898.

     Baldocchi, D.D. and K.B.Wilson. 2001. Modeling CO2 and water vapor exchange of a
     temperate broadleaved forest across hourly to decadal time scales. Ecological Modeling
          142: 155-184

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        DEBUGGING NOTES

	12/17/2019

	Runs with NIRv for combination of Vcmax = f(N) and NIRrefl=f(N)

	from Ollinger assume NIRrefl% ~ 15 N% + 5

	from regression of Globnet Vcmax@25C ~ 26 + 15.8 N%

	needed to reset vopt and jopt with Vcmax.

	9/27/2019

		runs for NIRv..Vcmax 0.8 Vcmax

		runs for NIRv..Vcmax 0.6 Vcmax

		9/26/2019
		added APAR as output, put Vcmax back to normal

		Set soil reflectance to zero




		3-14-2019  running for the Smoke period of summer 2018 for a C3 vegetation of high LAI 3

		3-1-2018

		I converted the RAD subroutines to Matlab and find a few minor glitches in ADUM. Have not changed it yet

		3-14-16

		put vctop back to 298 K because it was hot and A seems a bit high.

		2-16-16

		fixed sonic Gill and night values of soil respiration are larger now, 8 ummol m-2 s-1, so changed that parameter

		tweak vctop to 303 from 311

		10-30-15

		used pedo transfer function to convert volumetric water content to matric potential
		then solved for relative humidity via psi = R Tk/Vw ln(rh)

		used this to drive soil evaporation

		removed isoprene subroutines...no need

		10-14-15


		Energy balance functions are a function of Q, Q = Rin - Rout + Lin

		alfalfa are amphistomatous, eg Gates Biophysical Ecology

		what about flowers and ET??  reduced Vcmax and Jmax in upper 10% of canopy due to flowers

		check code and derivations on where factors of 2 are placed
		so far just applied a two to energy balance, but need to consider other derivations??

		put new soil resistance algorithm, Kondo. Better for the drier soils

		make sure right soil properties
		appropriate leaf size
		assume spherical leaf distribution
		leaf Vcmax, LAI
		lat/long
		wind profiles
		bulk density, soil org C
		have Cp soil and K-thermal soil weighted for water, mineral, organic and air fractions
		need to vary LAI with time

		conc...Izref and Cref, not at 2.8 m...

		change to 30 layer canopy and 150 layers total

		found inconsistencies in ES with tc and Tk..and mb not Pa units.. These have been fixed

		Revised soil heat transfer model using more modern algorithms and parameterizations from Campbell and Norman, reflecting
		the Campbell et al paper 1994 Soil Sci

		inputs include friction velocity, soil moisture and soil temperature at 32 cm.  So vewer factors need to be computed

		added subroutines for AMPHI, Energy balance and  photosynthesis

		corrected pore fraction

		linearization of surface energy balance make soil LE a function of vpd and T of air over soil, not in the top soil layer
		fixed this added mistake

		6-3-14
		Incorporating the Ps rate limitation of Collatz.
		Collatz uses a quadratic model to compute a dummy variable wp to allow
		for the transition between wj and wc, when there is colimitation.  this
		is important because if one looks at the light response curves of the
		current code one see jumps in A at certain Par values

		5-19-14 Finding some glitches in the input. On some days sun angle is above 0, yet solar
		radiation inputs are zero. This is causing some numbers and computations to explode.

		Looks like I need to go through the input files and interpolate solar radiation between the
		first instance and 0 for periods with sun angles above zero. Then the computations work.

		Inserted new code for sun angles from matlab. It is cleaner. Need to remove old Walraven
		code from Fortran after I am confident of the new code. Am writing out sun angles now

		5-2-2014.. having problems reading the strings to rename the input file. Was working in the past and
		still does for the first run, but crashes on second with sprintf_s..the names of the sub components are ok
		have tried to make sure I have the right number of characters for inbuff. found problem. typo on fptr10
		and I was not closing fptr10.

		6-10-2010
		Converting this version to cpp for use with the new Microsoft v2010 C compiler

		This version includes fopen_s, scanf_s, strcpy_s

		debugged and working!

		2-10-2010

		run with vcopt and Jmax const with depth, 110 day start of growing season

  *.vcnst


	11-3-2003
  found error in computation of cosine(zenith). Luckily in Canoak it was not called

  10-24-2003

  Diffuse Direct assumes Solar constant is 1320 instead of 1366 W m-2. Needs to be fixed

  12-2-2002
  Make Dij more general for ustar.  dij5000O._C

  Ran dispersion matrix for different z/L and revise Dij(z/L) function in CONC

  Has Massman and Weil TL parameterization, exponential sigma w profile, giving turb length scale
  rather constant with height in canopy, consistent with mixing layer flows and canopy shear

   1 million particles were released and scale with u* equal 1 m/s.
   Much smoother matrix is formed!
   Before I released 20, 000.

  Also corrected minor errors in Dispersion matrix, working with Alex Knohl

        11-25-2002.

  Georg noticed error in root3. It should be

        root3 = -2.0 * sqrt(Q) * cos((ang_L -PI2) / 3.0) - Pcube / 3.0;

  It had (+ PI4) instead

                9-15-2002. Georg Wohlfarht noticed I was not using boundary_lay_res.vapor. I was using bound_lay_res.heat
        for both heat and vapor transfer.  Corrected rvsfc to boundary_lay_res.vapor

        7-15-2002. Need to consider diffusion of 13C across the boundary layer, explicitly.

    Found I am using constant diffusivities, irrespective of temperature and pressure, based on
    data from Monteith and Unsworth at 20 C.

    Modified code and adopt reference values of Massman (1998) and am correcting diffusivities for
    temperature and pressure

   11-20-2001. Found typo in Dij=f(z/L); should be: f=(a + x)/(b + c x)

   7-21-2001. Set soil had some errors due to typos in Campbell's book, found by Georg Wohlfahrt

   6-29-2001  We need to scale soil respiration with leaf area index if we are to apply this model to other
   sites.  Raich and Tufekcioglu (2000, Biogeochemistry) show that Soil Respiration (gC m-2 yr-1)
   is close to a 3 to one ratio with litterfall (gC m-2 y-1) for mature forests. In the future
   it may be worthwhile to add this scheme to the model.


   6-25-2001  Added the Daaman and Simmonds convection effects to soil resistances.
   Also keeping rh at soil surface too high. Using RH of air to deduce if the soil
   surface may be wet. Then vpdfact=1, else 0.1, like the Canpond simulations.

   Found unit error in the vpd at the soil. Plus it was using the ea at the top of the canopy
   rather than near the soil, as a reference value. Also we need to adjust the vpdsoil factor
   for wet and dry soils. I am assuming the soil is wet if RH is above some level, eg 95%.
   Then vpdfact is one. Otherwise the vpdfact is 0.1, like the test for the Pipo in Oregon.

   6-23-2001. Need to incorporate Daamond and Simmons routines for soil resistances, like
   Canpond. At present soil evaporation is too large during the winter. Thought I had put these
   routines in, but did not.


   fixed zeroing of daily sums, resp eg

   3/13/2001 Redoing the Dij using the algorithm of Massman and Weil. Recomputed
   Dij function for z/L

   Debugging ISOTOPE and CONC

   found that I needed to adjust the CO2 profile by the factor ma/mc rhoair_mole_m-3
   to convert from mole -3 to umol

   Need to make sure 13C and 12C or (12C + 13C) cancel.

   3/12/2001 Got Isotope model running


   2-23-2001 add info on wj and wc profiles

   1-7-2001

    found met.zl was set to zero in CONC for neutral run of dispersion matrix. need to
    turn it off here and in canoak!

        1-2-2001
        Be careful on reading data with C++ compiler. I was getting warning
        errors using floats, so I changed everything to double. When I did this
        the input structure and Dispersion matrices were read in as garbage.
        I had to change those variables back to float and then they read ok!


        12-28-2000

        converting global variables over to structures

       */

//*****************************************************************


// Declare Subroutines


//  Radiation transfer routines


void RNET();    // computes net radiation profile

void PAR();     // computes visible light profiles

void DIFFUSE_DIRECT_RADIATION();        // computes direct and diffuse radiation from radiation inputs

void NIR();     // computes near infrared radiation profiles

double SKY_IR(double a);  // computes the incoming longwave radiation flux density

void IRFLUX();  // computes longwave radiation flux density

void G_FUNC_DIFFUSE();   // computes the leaf orientation function for diffuse radiation

void GFUNC();   // computes the leaf orientation angle for the given sun angle and
// a spherical leaf inclination

//*****************************************************************

//  canopy structure routines

void ANGLE();                                           // computes sun angles

double GAMMAF(double a);                        // gamma function

void LAI_TIME();                    // updates LAI with time

void FREQ(double a);                            // leaf angle frequency distribution for G



//************************************************************************

// photosynthesis, stomatal conductance and respiration


void STOMATA();   // computes inital stomatal conductance profiles as f(PAR)


// computes leaf photosynthesis with Farquhar model

void PHOTOSYNTHESIS(double a,double *b,double c, double d,double e,double *f,
                    double *g, double *h, double *i, double *j, double *k);

void PHOTOSYNTHESIS_AMPHI(double a,double *b,double c, double d,double e,double *f,
                          double *g, double *h, double *i, double *j, double *k);

double TEMP_FUNC(double a,double b,double c,double d, double e);  // Arrenhius function for temperature

double TBOLTZ(double a,double b, double c, double d); // Boltzmann function for temperature

void SOIL_RESPIRATION();  // computes soil respiration


//********************************************************************************

// Turbulence and leaf boundary layer routines

double UZ(double a);  // wind speed computation as a function of z

void BOUNDARY_RESISTANCE(double a, double b); // leaf boundary layer resistances for heat, water, CO2

void FRICTION_VELOCITY();       // updates friction velocity with new z/L



//**********************************************

// Concentration calculation routines for q, T and CO2

void CONC(double *ap1, double *ap2, double c, double d, double e);


//**************************************************************

//   leaf energy balance subroutines and functions

void ENERGY_AND_CARBON_FLUXES();  // profiles of energy, stomatal conductance and photosynthesis

void ENERGY_BALANCE(double b, double *apc,double d,double e, double f,
                    double g, double *hpt, double *ipt, double *jpt);   // computes leaf energy balance

void ENERGY_BALANCE_AMPHI(double b, double *apc,double d,double e, double f,
                          double g, double *hpt, double *ipt, double *jpt);   // computes leaf energy balance

double SFC_VPD(double a, double b, double *c);  // humidity at leaf surface

double LAMBDA(double a); // latent heat of vaporization

double ES(double a);    // saturation vapor pressure function

double DESDT(double a); // first derivative of saturation vapor pressure function

double DES2DT(double a); // second derivative of saturation vapor pressure function



//*****************************************************

// soil energy balance functions and subroutines

void SET_SOIL();  // initialize soil parameters

void SET_SOIL_TEMP();  // initialize deep soil temperature

void SOIL_ENERGY_BALANCE();  // soil energy balance

double SOIL_SFC_RESISTANCE(double a);  // soil resistance to water transfer





// declare subroutine for inputting data

void INPUT_DATA();

//  house keeping routines


void FILENAMES();  // defines file names and opens files
void ZERO();       // zeros arrays


//*******************************************************************************
errno_t err;

// Declare Structures


// meteorological inputs

//      &dayy,&hhrr,&ta,&rglobal,&parin,&pardif,&ea,&wnd,&ppt,&co2air,&press_mb,&flag

struct input_variables {

    int dayy;                       // day
    float hhrr;                       // hour
    float ta;                       // air temperature, C
    float rglobal;          // global radiation, W m-2
    float parin;            // photosynthetically active radiation, micromole m-2 s-1
    float pardif;           // diffuse PAR, micromol m-2 s-1
    float ea;               // vapor pressure, kPa
    float wnd;              // wind speed, m s-1
    float co2air;           // CO2 concentration, ppm
    float press_kPa;        // air pressure, mb
    float ustar;
    float tsoil;            //  32 cm
    float soil_moisture;    // 10 cm

} input;


// structure for time variables

struct time_variables {

    double local_time;

    long int daytime;   // day+hour coding, e.g. 1230830
    int year;                       // year
    int days;                       // day
    int jdold;                      // previous day
    int fileyear;       // year for filenames
    int filestart;      // time for filestart
    int fileend;        // time for fileend
    int leafout;        // day of leaf out
    int fulleaf;        // date of full leaf
    int leafdrop;       // date of first fall frost and end of Ps
    int leafoff;        // date of no leaves in autumn

    int count;          // number of iterations

    int phenology;      // test if soil is warming

    double lai;         // lai as a function of time

} time_var;


// structure for meteorological variables

struct meteorology {

    double ustar;                   // friction velocity, m s-1
    double ustarnew;                // updated friction velocity with new H, m s-1
    double rhova_g;                 // absolute humidity, g m-3
    double rhova_kg;                // absolute humidity, kg m-3
    double sensible_heat_flux;      // sensible heat flux, W M-2
    double H_old;                   // old sensible heat flux, W m-2
    double air_density;             // air density, kg m-3
    double T_Kelvin;                // absolute air temperature, K
    double dispersion[sze3][sze];   // Lagrangian dispersion matrix, s m-1
    double zl;                      // z over L, height normalized by the Obukhov length
    double press_kpa;               // station pressure, kPa
    double press_bars;              // station pressure, bars
    double press_Pa;                // pressure, Pa
    double pstat273;                // gas constant computations
    double air_density_mole;        // air density, mole m-3
    double relative_humidity;       // relative humidity, ea/es(T)
    double vpd;                     // vapor pressure deficit
} met;



// structure for surface resistances and conductances

struct surface_resistances {

    double gcut;        // cuticle conductances, mol m-2 s-1
    double kballstr;    // drought factor to Ball Berry Coefficient
    double rcuticle;    // cuticle resistance, m2 s mol-1

} sfc_res;


// structure for plant and physical factors

struct factors {
    double latent;      // latent heat of vaporization, J kg-1
    double latent18;    // latent heat of vaporization times molecular mass of vapor, 18 g mol-1
    double heatcoef;    // factor for sensible heat flux density
    double a_filt;          // filter coefficients
    double b_filt;      // filter coefficients
    double co2;         // CO2 factor, ma/mc * rhoa (mole m-3)

} fact;

// structure for bole respiration and structure

struct bole_respiration_structure {

    double factor;              // base respiration, micromoles m-2 s-1, data from Edwards
    double respiration_mole;  // bole respiration, micromol m-2 s-1
    double respiration_mg;   // bole respiration, mg CO2 m-2 s-1
    double calc;            // calculation factor
    double layer[sze];   // bole pai per layer
} bole;



// structure for canopy architecture

struct canopy_architecture {
    double bdens[soilsze];          // probability density of leaf angle
} canopy;

// structure for non dimensional variables

struct non_dimensional_variables {

    // Prandtl Number

    double pr;
    double pr33;


    //  Schmidt number for vapor

    double sc;
    double sc33;

    //  Schmidt number for CO2

    double scc;
    double scc33;


    // Schmidt number for ozone

    double sco3;
    double sco333;

    // Grasshof number

    double grasshof;


    // multiplication factors with leaf length and diffusivity

    double lfddv;
    double lfddh;

}  non_dim;


// boundary layer resistances

struct boundary_layer_resistances {

    double vapor;                                   // resistance for water vapor, s/m
    double heat;                                    // resistance for heat, s/m
    double co2;                     // resistance for CO2, s/m

} bound_layer_res;


// radiation variables, visible, near infrared and infrared

struct solar_radiation_variables {

    // profiles of the probabilities of sun and shade leaves

    double prob_beam[sze];  // probability of beam or sunlit fraction
    double prob_sh[sze];    // probability of shade

    double ir_dn[sze];              // downward directed infrared radiation, W m-2
    double ir_up[sze];              // upward directed infrared radiation. W m-2


    // inputs of near infrared radiation and components, W m-2

    double nir_beam;                // incoming beam component near infrared radiation, W m-2
    double nir_diffuse;             // incoming diffuse component near infrared radiation, W m-2
    double nir_total;               // incoming total near infrared radiaion, W m-2

    // computed profiles of near infrared radiation, W m-2

    double nir_dn[sze];             // downward scattered near infrared radiation
    double nir_up[sze];             // upward scattered near infrared radiation
    double nir_sun[sze];    // near infrared radiation on sunlit fraction of layer
    double nir_sh[sze];             // near infrared radiation on shaded fraction of layer
    double beam_flux_nir[sze]; // flux density of direct near infrared radiation

    // leaf and soil optical properities of near infrared radiation

    double nir_reflect;             // leaf reflectance in the near infrared
    double nir_trans;               // leaf transmittance in the near infrared
    double nir_soil_refl;   // soil reflectance in the near infrared
    double nir_absorbed;    // leaf absorptance in the near infrared

    //  inputs of visible light, PAR, W m-2

    double par_diffuse;     // diffuse component of incoming PAR, parin
    double par_beam;                // beam component of incoming PAR, parin
    double par_total;

    // computed profiles of visible radiation, PAR, W m-2

    double par_shade[sze];  // PAR on shaded fraction of layer
    double par_sun[sze];    // PAR on sunlit fraction of layer, beam and diffuse
    double beam_flux_par[sze];  // PAR in the beam component
    double par_down[sze];   // downward scattered PAR
    double par_up[sze];     // upward scattered PAR

    // flux densities of visible quanta on sun and shade leaves for photosynthesis
    // calculations, micromoles m-2 s-1

    double quantum_sun[sze]; // par on sunlit leaves
    double quantum_sh[sze];  // par on shaded leaves

    // optical properties of leaves and soil for PAR

    double par_absorbed;            // PAR leaf absorptance
    double par_reflect;                     // PAR leaf reflectance
    double par_trans;                       // PAR leaf transmittance
    double par_soil_refl;           // PAR soil reflectance


    // Net radiation profiles, W m-2

    double rnet_sun[sze];           // net radiation flux density on sunlit fraction of layer
    double rnet_sh[sze];            // net radiation flux density on shade fraction of layer

    double Q_sun[sze];         // Rin - Rout + Lin, use as inputs for leaf and soil energy balance
    double Q_sh[sze];

    double exxpdir[sze];       // exponential transmittance of diffuse radiation through a layer
    double beta_rad;           // solar elevation angle, radians
    double sine_beta;          // sine of beta
    double beta_deg;           // solar elevation angle, degrees
    double ratrad;             // radiation ratio to detect cloud amount
    double ratradnoon;         // radiation ratio at noon for guestimating cloud amount at night
    double diff_tot_par;       // ratio par diffuse to total
    double apar;               // fraction of PAR absorbed by canopy
} solar;



// physical properties of soil and soil energy balance variables

struct soil_variables {

    // soil properties

    double z_soil[soilsze];       // depth increments of soil layers
    double bulk_density[soilsze]; // bulk density of soil
    double T_soil[soilsze];       // soil temperature
    double k_conductivity_soil[soilsze];   // thermal conductivity of soil
    double cp_soil[soilsze];      // specific heat of soil, f(texture, moisture)
    double T_Kelvin;              // soil surface temperature in Kelvin
    double T_air;                 // air temperature above soil, C
    double sfc_temperature;       // soil surface temperature in C
    double Temp_ref;                      // reference soil temperature, annual mean, C
    double Temp_amp;              // amplitude of soil temperature, C
    double resistance_h2o;        // soil resistance for water vapor
    double water_content_sfc;     // volumetric water content of soil surface
    double water_content_15cm;    // vol water content of 15 cm soil layer
    double water_content_litter;  // vol water content of litter
    double T_base;                // base soil temperature
    double T_15cm;                // soil temperature at 15 cm
    double Tave_15cm;             // daily average soil temperature at 15 cm
    double amplitude;             // amplitude of soil temperature cycle
    double clay_fraction;         // clay fraction
    double peat_fraction;         // organic matter, peat fraction
    double pore_fraction;
    double mineral_fraction;
    double air_fraction;


    // soil energy flux densities, W m-2

    double lout;                          // longwave efflux from soil
    double evap;                          // soil evaporation
    double heat;                          // soil sensible heat flux density
    double rnet;                          // net radiation budget of the soil
    double gsoil;                 // soil heat flux density


    // soil CO2 respiratory efflux

    double respiration_mole;         // soil respiration, micromol m-2 s-1
    double respiration_mg;       // soil respiration, mg CO2 m-2 s-1
    double base_respiration;     // base rate of soil respiration, micromol m-2 s-1
    double resp_13;              // respiration of 13C micromole m-2 s-1

    double dt;                 // time step, s
    long int mtime;            // number of time steps per hour
} soil;




// Structure for Profile information,fluxes and concentrations


struct profile {

    // microclimate profiles

    double tair[sze3];        // air temp (C)
    double tair_filter[sze3]; // numerical filter of Tair
    double u[sze3];           // wind speed (m/s)
    double rhov_air[sze3];    // water vapor density
    double rhov_filter[sze3]; // numerical filter of rhov_air
    double co2_air[sze3];     // co2 concentration (ppm)
    double Ci[sze3];          // Ci weighted by sun and shade fractions

    // canopy structure profiles

    double dLAIdz[sze];       // leaf area index of layer (m2/m2)
    double ht[sze];           // layer height (m)
    double dPAIdz[sze];       // plant area index of layer
    double Gfunc_solar[sze];      // leaf-sun direction cosine function
    double Gfunc_sky[sze][szeang]; // leaf-sky sector direction cosine function

    // variables for 13C isotopes

    double c13cnc[sze3];       // concentration of 13C
    double sour13co2[sze];    // source/sink strength 13C
    double d13C[sze3];         // del 13C
    double d13Cair[sze3];      // del 13C of the air
    double R13_12_air[sze3];   // 13C/12C ratio of the air
    double Rplant_sun[sze];    // ratio of discriminated 13C in sunlit leaves
    double Rplant_shd[sze];    // ratio of discriminated 13C in shaded leaves


    // source/sink strengths

    double source_co2[sze];    // source/sink strength of CO2

    // fluxes for total layer, with sun/shade fractions

    double dPsdz[sze];        // layer photosynthesis
    double dHdz[sze];         // layer sensible heat flux
    double dLEdz[sze];        // layer latent heat flux
    double dRNdz[sze];        // layer net radiation flux
    double dRESPdz[sze];      // layer respiration
    double dStomCondz[sze];   // layer stomatal conductance
    double drbv[sze];         // layer boundary layer conductance

    // sun leaf variables

    double sun_frac[sze];    // sun leaf fraction
    double sun_tleaf[sze];   // leaf temp (C)
    double sun_A[sze];       // layer A flux for sun only (micromol mn-2 s-1)
    double sun_gs[sze];      // stomatal conductance
    double sun_rs[sze];      // stomatal resistance to H2O (s/m)
    double sun_rbh[sze];     // boundary layer resistance to heat (s/m)
    double sun_rbv[sze];     // boundary layer resistance to H2O (s/m)
    double sun_rbco2[sze];   // boundary layer resistance to CO2 (s/m)
    double sun_ci[sze];      // Ci on sun leaves
    double sun_D13[sze];     // discrimination 13C
    double sun_cica[sze];    // Ci/Ca on sunlit leaves
    double sun_lai[sze];     // sunlit lai of layer
    double sun_T_filter[sze]; // filtered sunlit temperature
    double sun_wj[sze];       // electron transport rate of Ps for sun leaves
    double sun_wc[sze];       // carboxylatio velocity for sun leaves
    double sun_resp[sze];     // respiration

    // shade leaf variables

    double shd_frac[sze];    // shade leaf fraction
    double shd_tleaf[sze];   // temperature of shaded leaves
    double shd_A[sze];       // photosynthesis of shaded leaves
    double shd_gs[sze];              // stomatal conductance of shade leaves
    double shd_rs[sze];      // stomatal resistance of shaded leaves
    double shd_rbh[sze];     // boundary layer resistance for heat on shade leaves
    double shd_rbv[sze];     // boundary layer resistance for vapor on shade leaves
    double shd_rbco2[sze];   // boundary layer resistance for CO2 on shade leaves
    double shd_ci[sze];      // Ci on shade leaves
    double shd_D13[sze];     // del 13C on shaded leaves
    double shd_cica[sze];    // Ci/Ca ratio on shaded leaves
    double shd_lai[sze];     // shaded lai of layer
    double shd_T_filter[sze]; // previous temperature
    double shd_wc[sze];       // carboxylation rate for shade leaves
    double shd_wj[sze];       // electron transport rate for shade leaves
    double shd_resp[sze];     // respiration
} prof;


struct Leaf_variables {

    double N;
    double Vcmax;
} leaf;

// ------------------------------------------------------
//          DECLARE PARAMETER VALUES
// ------------------------------------------------------



// canopy structure variables

const double ht = 1;             //   0.55 Canopy height, m
const double pai = .0;            //    Plant area index

const double lai = 4;      //  1.65 Leaf area index data are from clip plots and correspond with broadband NDVI estimates

// Gaetan et al 2012 IEEE, vcmax 70, jmax 123

// Erice et al physiologia planatar, 170, 278, alfalfa A-Ci, Vcmax and Jmax

const double vcopt = 170.0 ;   // carboxylation rate at optimal temperature, umol m-2 s-1; from lit
const double jmopt = 278.0;  // electron transport rate at optimal temperature, umol m-2 s-1
const double rd25 = .22;     // dark respiration at 25 C, rd25= 0.34 umol m-2 s-1


int const jtot=30;            // number of canopy layers
int const jktot=31;          // jtot + 1
int const jtot3=150;        // number of layers in the domain, three times canopy height
int const izref=150;       // array value of reference ht at 2.8 m,  jtot/ht

const double pi4=12.5663706;

const double delz = 0.0175;            //  height of each layer, ht/jtot
const double zh65=0.3575;             //  0.65/ht

const double ustar_ref = 1.00;          // reference u* value for dispersion matrix, old value was 0.405


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



// Isotope ratio of PeeDee Belimdite standard (PDB)
// redefined as 13C/(12C+13C) for PDB, Tans et al 93

const double Rpdb_CO2=0.011115;

// Isotope ratio of PeeDee Belimdite standard (PDB)
// defined as 13C/12C, from Farquhar

const double Rpdb_12C=0.01124;


//     Declare file pointers for I/O


FILE *fptr1,*fptr4,*fptr6,*fptr7, *fptr8, *fptr9, *fptr10;

int main ()
{

    //   Declare internal variables


    int i_count,i, I, JJ;
    int j, ji, junk;


    long int daycnt;

    char delim[5];

    float dummy;

    double rnet_soil, netrad;

    double fc_mg, fc_mol, evaporation, wue, transpiration, transpiration_mole;

    double sumfc,sumevap,sumps, sumsens, sumlai;        // sums for daily averages and totals
    double sumpar,sumnet,sumta,sumbole, sumsoil, sumgs, sumNIRv, NIRv, sumAPAR;
    double sumh, sumle, sumrn, sumresp, sumksi, sumgsoil, sumisoprene, sumtsoil, sumCi, sumrv;

    double can_ps_mol, can_ps_mg, canresp;

    double isoprene_efflux;

    double testener;                        // test value for energy closure

    double tleaf_mean,tavg_sun,tavg_shade;  // integrated leaf temperatures

    double Kh,Kq,Kco2;   // eddy exchange coefficients

    double asinzero;



    // Constants for leaf boundary layers

    non_dim.lfddh=lleaf/ddh;

    // PRANDTL NUMBER

    // Prandtl Number

    non_dim.pr = nuvisc / dh;
    non_dim.pr33 = pow(non_dim.pr,.33);

    //  DIFFUSIVITY OF WATER VAPOR, m2 s-1

    non_dim.lfddv=lleaf/ddv;

    //  SCHMIDT NUMBER FOR VAPOR

    non_dim.sc = nuvisc / dv;
    non_dim.sc33 = pow(non_dim.sc,.33);

    //  SCHMIDT NUMBER FOR CO2

    non_dim.scc = nuvisc / dc;
    non_dim.scc33 = pow(non_dim.scc,.33);


    // Grasshof Number

    non_dim.grasshof=9.8*pow(lleaf,3)/pow(nnu,2);


    // assign heights to array

    for (i=1; i <= jtot; i++)
        prof.ht[i]= delz * (double) i;




    // Input year to name input files


    time_var.fileyear=2018;
    time_var.filestart=time_var.fileyear;

    time_var.fileend=2018;

    // Open Dispersion Matrix

    // err=fopen_s(&fptr4,"d:\\canalfalfa\\dij5000.csv","r");
    fptr4=fopen("./DIJ5000.csv","r");
    if(fptr4==0)
        printf("open dispbuff\n");



    //  Input data on Thomson dispersion matrix that was computed offline with
    //  MOVOAK.C, Dij (s m-1)


    for (j=1; j<=jtot; j++)
    {


        for(ji=1; ji <= jtot3; ji++)
        {

            // fscanf_s(fptr4,"%f, %i\n", &dummy, &junk);
            fscanf(fptr4,"%f, %i\n", &dummy, &junk);

            met.dispersion[ji][j] = dummy;


        } //  next ji
    } //  next j


    //	fclose(fptr4);



    /*************************************************************************
                                 MAIN PROGRAM
    **************************************************************************

           Describe canopy attributes


           Flow chart of the Main Program:

           1a) start run at beginning of year
           1b) establish new LAI
               1c) Compute continuous distrib of height at a function of lai
               1d) Set soil parameters

           2) input met values

           3) Compute solar elevation angle;

           4) Compute PAR and NIR profiles and their flux densities on
             the sunlit and shaded leaf fractions

           5) Compute sunlit and shaded leaf fractions

           6) Compute first estimate of stomatal conductance

           7) Compute first estimate of IRFLUX assuming leaf temperature
              equals air temperature.
           8) Compute first estimate of leaf energy balance and leaf
              temperature

           9) Compute photosynthesis, transpiration

           10) Compute soil energy balance

           11) update computation of friction velocity with new H and z/L

           12) Compute new scalar and source/sink profiles for CO2, T and e

           13) Iterate among 7 through 12 until convergence

           14) compute fluxes of isoprene, 13C isotopes  or whatever

           15) Thats all Folks!!!!

    ************************************************************************/



    // set up fileyear loop for one year or multiple year runs

    while(time_var.fileyear <= time_var.fileend)
    {




        //asinzero=asin(0);
        asinzero=0;



        // initialize date of leaf out and refine as we get more soil temperature information

        time_var.leafout=1;


        // initialize date of leaf drop, date of first fall freeze

        time_var.leafdrop=365;


        // assume 30 days till full leaf

        time_var.fulleaf=1;
        time_var.leafoff=365;

        // Define Filenames

        FILENAMES();




        //    SET_SOIL_TEMP();    // set deep soil temperature from the mean annual temperature

        time_var.year=time_var.fileyear;




        //  initialize some variables

        time_var.jdold=1;

        time_var.days=180;   // first day of expt

        /*

        while (time_var.days < 366)        // call for annual runs
        {

        */

        // zero summing variables for daily time integrals

        sumfc=0;
        sumevap=0;
        sumsens=0;
        sumps=0;
        sumpar=0;
        sumnet=0;
        sumbole=0;
        sumsoil=0;
        sumresp=0;
        sumta=0;
        daycnt=0;
        sumgs=0;
        sumgsoil=0;
        sumisoprene=0;
        sumtsoil =0;
        sumNIRv = 0;
        sumAPAR = 0;

        leaf.N = 3;  // leaf Nitrogen


        ZERO(); // re-zero arrays


        /* define parameters for soil energy balance model.  This version only
           computes heat transfer.  A water transfer module needs to be added.
           Preliminary tests show that the soil model can be sensitive to the depth
           of the litter layer
        */

        //   SET_SOIL();


        // set Ball-Berry stomatal factor

        sfc_res.kballstr = kball;

        LAI_TIME();            // define leaf area and canopy structure


        // loop through the input met file. There should be a line of data for
        // each hour of the year

        while(! feof(fptr1))
        {

            // input met data

            INPUT_DATA();

            SET_SOIL();   // update set soil with new soil moisture information


            // if new day then compute daily averages and update LAI

            if(time_var.days>time_var.jdold)
            {

                // compute daily averages

                sumfc /=  daycnt;
                sumevap /= daycnt;
                sumsens /=daycnt;
                sumpar /= daycnt;
                sumnet /= daycnt;
                sumps /=  daycnt;
                sumbole /=  daycnt;
                sumsoil /= daycnt;
                sumta /= daycnt;
                sumgs /= daycnt;
                sumresp /= daycnt;
                sumgsoil /=daycnt;
                sumtsoil /=daycnt;
                sumisoprene /= daycnt;
                sumNIRv /= daycnt;
                sumAPAR /= daycnt;

                soil.Tave_15cm=sumtsoil;

                // add info on Ci/Ca, Pbeam and Pshade, remove isoprene

                fprintf(fptr8,"%4i, %10.3f, %10.1f, %10.1f, %10.1f, %10.1f, %10.1f, %6.3f, %10.2f, %10.2f, %10.2f, %10.2f, %10.4f, %10.4f, %10.4f, %10.4f, %10.4f, %10.4f  \n",
                        time_var.jdold, sumfc, sumevap, sumsens, sumpar, sumnet, time_var.lai, sumps, sumresp, sumbole, sumsoil, sumta, sumgs,
                        sumgsoil, sumisoprene, sumtsoil,sumNIRv,sumAPAR);





                LAI_TIME();      // update LAI with new day



                //  Re-zero summing variables for daily averages

                sumfc=0;
                sumevap=0;
                sumsens=0;
                sumps=0;
                sumpar=0;
                sumnet=0;
                sumbole=0;
                sumsoil=0;
                sumta=0;
                daycnt=0;
                sumgs=0;
                sumresp=0;
                sumgsoil=0;
                sumtsoil=0;

            }


            // Compute solar elevation angle

            ANGLE();

            // make sure PAR is zero at night. Some data have negative offset, which causes numerical
            // problems

            if(solar.sine_beta <= 0.01)
                input.parin=0;


            //  Compute the fractions of beam and diffuse radiation from incoming measurements

            // Set the radiation factor to the day before for night calculations.  This way if the day
            // was cloudy, so will IR calculations for the night reflect this.

            if (solar.sine_beta > 0.05)
                DIFFUSE_DIRECT_RADIATION();
            else
                solar.ratrad = solar.ratradnoon;


            // computes leaf inclination angle distribution function, the mean direction cosine
            // between the sun zenith angle and the angle normal to the mean leaf

            // for CANOAK we use the leaf inclination angle data of Hutchison et al. 1983, J Ecology
            // for other canopies we assume the leaf angle distribution is spherical

            // Loop out if night time


            if(solar.sine_beta >= 0.01)
                GFUNC();

            // Compute PAR profiles

            PAR();

            //  Compute NIR profiles

            NIR();


            // Initialize humidity, prof.rhov_air, profile


            /*
                   Initialize IRFLUX and STOMATA with Tleaf equal to Tair
                   Tleaf is the leaf temperature weighted according to
                   the sunlit and shaded fractions
            */

            for(I=1; I<= jktot; I++)
            {
                prof.sun_tleaf[I] = input.ta;
                prof.shd_tleaf[I] = input.ta;
                prof.sun_T_filter[I] = input.ta;
                prof.shd_T_filter[I] = input.ta;
            }

            met.rhova_kg = met.rhova_g / 1000.;         /*  absolute humidity, kg m-3  */


            for(I=1; I<= jtot3; I++)
            {
                prof.tair[I] = input.ta;
                prof.tair_filter[I] = input.ta;
                prof.rhov_air[I] = met.rhova_kg;
                prof.rhov_filter[I] = met.rhova_kg;
                prof.co2_air[I] = input.co2air;

            }

            // initialize soil surface temperature with air temperature

            soil.sfc_temperature = input.ta;

            fact.heatcoef = met.air_density * cp;


            /*   Compute stomatal conductance for sunlit and
                 shaded leaf fractions as a function of light
                 on those leaves.  This is needed to make first
                 cut at leaf transpiration, as needed to compute
                 RH on the leaf surface for the Ball-Berry model
                 First loop Tleaf equals Tair
            */

            STOMATA();


            // any recursive looping would occur here, below TL initialization


            i_count = 0;
            time_var.count=i_count;

            met.ustarnew=met.ustar;
            met.H_old=0;

            IRFLUX();

            // iteration looping for energy fluxes and scalar fields
            // iterate until energy balance closure occurs or 75 iterations are
            // reached

            do
            {


                // compute net radiation balance on sunlit and shaded leaves

                RNET();


                // Compute leaf energy balance, leaf temperature, photosynthesis and stomatal conductance.

                ENERGY_AND_CARBON_FLUXES();

                // Soil energy balance

                SOIL_ENERGY_BALANCE();

                // update long wave radiation fluxes with new leaf and air temperatures

                IRFLUX();


                /*
                                Compute air temperature profiles from dispersion matrix


                                 Adjust dHdz[1] for soil heat flux

                                 sign convention used: fluxes from surface are positive
                                 those toward the surface are negative

                                 dHdz is net sensible heat exchange per unit leaf area

                                 Using filtered temperatures to minimize the system from being mathematically unstable.

                */

                // filter temperatures with each interation to minimize numerical instability

                if (i_count < 10)
                {   fact.a_filt=0.5;
                    fact.b_filt=0.5;
                }
                else
                {   fact.a_filt=0.85;
                    fact.b_filt=0.15;
                }


                // conc, for temperature profiles using source/sinks

                // inputs are source/sink[], scalar_profile[],
                //  ref_val, boundary_flux, unit conversions

                CONC(prof.dHdz,prof.tair, input.ta, soil.heat, fact.heatcoef);

                //  filter temperatures to remove numerical instabilities
                //  for each iteration

                for(I = 1; I<=jtot3; I++)
                {

                    if(prof.tair[I] < -10. || prof.tair[I] > 60.)
                        prof.tair[I]=input.ta;

                    prof.tair_filter[I] = prof.tair[I]*fact.a_filt+prof.tair_filter[I]*fact.b_filt;
                }


                // compute filtered sunlit and shaded temperatures
                // these are used to compute iterated longwave emissive
                // energy fluxes

                for (I=1; I<=jtot; I++)
                {
                    prof.sun_T_filter[I] = fact.a_filt * prof.sun_tleaf[I] + fact.b_filt * prof.sun_T_filter[I];
                    prof.shd_T_filter[I] = fact.a_filt * prof.shd_tleaf[I] + fact.b_filt * prof.shd_T_filter[I];
                }


                /*

                                Compute vapor density profiles from Dispersion matrix

                                The reference temperature level is at 36 m

                                sign convention used: fluxes from surface are positive
                                those toward the surface are negative

                                dLEdZ is net latent heat exchange per unit leaf area

                */

                // the arrays dLEdz and prof.rhov_air define pointers

                CONC(prof.dLEdz, prof.rhov_air,met.rhova_kg, soil.evap, fact.latent);


                // filter humidity computations

                for (I=1; I<= jtot3; I++)
                {
                    if(prof.rhov_air[I] < 0 || prof.rhov_air[I] > .030)
                        prof.rhov_air[I]=met.rhova_kg;

                    prof.rhov_filter[I] = fact.a_filt* prof.rhov_air[I] + fact.b_filt * prof.rhov_filter[I];
                }

                // compute soil respiration

                SOIL_RESPIRATION();


                //    Convert to umol m-2 s-1

                soil.respiration_mole = soil.respiration_mg * 1000. / mass_CO2;

                /*
                        sign convention used: photosynthetic uptake is positive
                        respiration is negative

                        prof.dPsdz is net photosynthesis per unit leaf area,
                        the profile was converted to units of mg m-2 s-1 to be
                        consistent with inputs to CONC
                */

                // change sign of dPsdz

                for(i=1; i<= jtot; i++)
                    prof.source_co2[i] = -prof.dPsdz[i];

                // compute bole respiration
                // prof.source_CO2 is adjusted by bole respiration

                //        BOLE_RESPIRATION();


                // revision is using micromol m-2 s-1

                // to convert umol m-3 to umol/mol we have to consider
                // Pc/Pa = [CO2]ppm = rhoc ma/ rhoa mc

                fact.co2=(mass_air/mass_CO2)*met.air_density_mole;

                CONC(prof.source_co2, prof.co2_air, input.co2air, soil.respiration_mole, fact.co2);


                // Integrate source-sink strengths to estimate canopy flux

                sumh = 0.0;   // sensible heat
                sumle = 0.0;  // latent heat
                sumrn = 0.0;  // net radiation
                can_ps_mol = 0.0;  // canopy photosynthesis
                sumlai = 0.0;    // leaf area
                canresp = 0.0;  // canopy respiration
                sumksi = 0.0;   // canopy stomatal conductance
                tleaf_mean=0;   // mean leaf temperature
                tavg_sun=0;     // avg sunlit temperature
                tavg_shade=0;   // avg shaded temperature
                sumrv=0;

                sumCi=0;

                for(JJ=1; JJ<= jtot; JJ++)
                {
                    sumh += prof.dHdz[JJ];
                    sumle += prof.dLEdz[JJ];
                    can_ps_mol += prof.dPsdz[JJ];  // computed in micromols now
                    canresp += prof.dRESPdz[JJ];
                    sumksi += prof.dStomCondz[JJ];
                    sumrn += prof.dRNdz[JJ];
                    sumlai += prof.dLAIdz[JJ];
                    tleaf_mean +=prof.sun_tleaf[JJ]*solar.prob_beam[JJ] + prof.shd_tleaf[JJ]*solar.prob_sh[JJ];

                    sumrv += prof.drbv[JJ];

                    sumCi += prof.Ci[JJ];


                    // need to weight by sun and shaded leaf areas then divide by LAI

                    tavg_sun +=prof.sun_tleaf[JJ]*prof.dLAIdz[JJ];
                    tavg_shade +=prof.shd_tleaf[JJ]*prof.dLAIdz[JJ];
                }


                // mean canopy leaf temperature

                tleaf_mean /=jtot;




                // leaf area weighted temperatures

                tavg_sun /=time_var.lai;
                tavg_shade /=time_var.lai;


                // Energy exchanges at the soil

                rnet_soil = soil.rnet - soil.lout;

                sumCi /= jtot;

                // canopy scale flux densities, vegetation plus soil

                sumh += soil.heat;
                sumle += soil.evap;
                sumrn += rnet_soil;


                met.sensible_heat_flux=sumh;

                // re-compute Monin Obuhkov scale length and new ustar values with iterated H

                FRICTION_VELOCITY();

                //  Net radiation balance at the top of the canopy

                netrad = (solar.beam_flux_par[jktot] + solar.par_down[jktot] - solar.par_up[jktot]) / 4.6 + solar.beam_flux_nir[jktot] + solar.nir_dn[jktot] - solar.nir_up[jktot] + solar.ir_dn[jktot] - solar.ir_up[jktot];

                NIRv = solar.nir_up[jktot] - solar.nir_up[1];  // NIR emitted by vegetation, total - soil



                // test for convergenece between the sum of the net radiation flux profile and the
                // net flux exiting the canopy

                testener=fabs((sumrn-netrad)/sumrn);

                i_count ++;
                time_var.count=i_count;

            } while(testener > .005 && i_count < 75); // end of while for integrated looping


            // compute isoprene flux density

            // isoprene_efflux=0;
            //
            // if(time_var.lai > pai)
            //  ISOPRENE_CANOPY_FLUX(&isoprene_efflux);
            // else
            //	 isoprene_efflux=0;


            //  check and convert units of all components to desireable values !!!!

            //    Convert to umol m-2 s-1

            can_ps_mg = can_ps_mol * mass_CO2/1000.;

            fc_mg = -(can_ps_mg - soil.respiration_mg );
            fc_mol=fc_mg * 1000. /mass_CO2;

            // compute transpiration and water use efficiency

            evaporation = sumle / LAMBDA(met.T_Kelvin);

            transpiration = 1000.*(evaporation - (soil.evap / LAMBDA(met.T_Kelvin)));

            transpiration_mole=1000.* transpiration/18.;   // mmol m-2 s-1

            wue = can_ps_mg / (transpiration);  /* mg co2 g h20  */


            //  Compute Eddy Exchange Coefficients at the flux measurement reference ht, 2.8 m

            I=izref;

            Kh=-sumh*10*delz/(fact.heatcoef*(prof.tair[I]-prof.tair[I-10]));
            Kq=-sumle*10*delz/(fact.latent*(prof.rhov_air[I]-prof.rhov_air[I-10]));
            Kco2=-fc_mol*10* delz/(prof.co2_air[I]-prof.co2_air[I-10]);



            solar.diff_tot_par=solar.par_diffuse/input.parin;


            solar.apar= (input.parin- solar.par_up[jktot] - solar.par_down[1]+ solar.par_up[1])/input.parin;

            if(input.parin < 5)
            {   solar.diff_tot_par=9999;
                solar.apar = 9999;
            }

            if(sumrv < 0)
            {
                sumrv = 9999;
            }


            if(sumrv > 10000)
            {
                sumrv = 9999;
            }
            //  Output values

            printf("  day  hour  icount  lai\n");
            printf(" %3i   %7.3f    %7i  %6.2f\n",time_var.days,time_var.local_time,i_count,time_var.lai);

            printf("par   ta       u    ustar   rhov  Tsoil_base  Tsoil15\n");
            printf("%5.0f %6.2f  %5.2f  %7.4f  %7.4f  %5.2f  %5.2f\n",input.parin,input.ta, input.wnd,
                   met.ustar, met.rhova_g, soil.T_base,soil.T_15cm);
            printf("\n");
            printf("   RN:       sumrn \n");
            printf("   %5.0f     %5.0f\n", netrad, sumrn);
            printf("\n");
            printf("INTEGRATED H:  LE:   GSOIL:  Soilevap  Soil heat \n");
            printf("  %5.2f       %5.2f    %5.2f  %5.2f  %5.2f\n",sumh, sumle, soil.gsoil, soil.evap, soil.heat);
            printf("\n");
            printf("CANOPY PHOTOSYNTHESIS\n");
            printf(" mg m-2 s-1   umol m-2 s-1\n");
            printf("   %6.3f        %6.3f\n", can_ps_mg, can_ps_mol);
            printf("\n");
            printf(" PLANT/SOIL/ROOT RESPIRATION \n");
            printf(" mg m-2 s-1    umol m-2 s-1\n");
            printf("  %6.3f        %6.3f\n", soil.respiration_mg, soil.respiration_mole);

            printf("tsoilref  %6.3f   tampt   %6.3f\n",soil.Temp_ref,soil.Temp_amp);


            // DISK OUTPUT

            fprintf(fptr6,"%ld,%6.1f,%6.1f,%6.1f,%6.1f,%6.2f,%6.2f,%6.2f,%6.2f,%6.1f,%7.4f,%7.3f,%7.3f,%7.3f,%7.3f,%7.3f,%7.3f,%7.4f,%7.4f,%7.4f,%7.4f,%7.4f,%7.3f,%7.3f, %7.3f \n ",
                    time_var.daytime, netrad, sumrn,sumh, sumle, transpiration_mole, can_ps_mol, canresp,
                    soil.respiration_mole,  soil.gsoil, sumksi, tleaf_mean, tavg_sun,
                    tavg_shade,Kh,Kq,Kco2, met.vpd, solar.diff_tot_par,
                    soil.T_15cm,solar.beta_deg, sumCi, sumrv, NIRv,solar.apar);

            fprintf(fptr7,"%ld, %6.1f, %6.1f, %6.1f, %6.1f, %6.2f \n",time_var.daytime,rnet_soil, soil.heat, soil.evap,soil.gsoil, soil.sfc_temperature);


            // compute profiles for only specified periods

            if (time_var.daytime > 1800000 && time_var.daytime < 1950000)
            {
                fprintf(fptr9, "%ld \n",time_var.daytime);

                for(I=1; I<=jtot3; I+=2)
                    fprintf(fptr9,"%i, %6.3f,%7.5f,%8.3f \n",
                            I,prof.tair[I],prof.rhov_air[I],prof.co2_air[I]);

                fprintf(fptr10, "%ld \n",time_var.daytime);
                for(I=1; I<=jtot; I+=2)
                    fprintf(fptr10,"%i, %6.3f, %6.3f,%6.3f, %6.3f, %6.3f, %6.3f, %6.3f, %6.3f, %6.3f, %6.3f , %6.3f, %6.3f\n",
                            I,prof.dHdz[I],prof.dLEdz[I],
                            prof.source_co2[I],prof.Ci[I], prof.sun_cica[I], prof.shd_cica[I],
                            prof.sun_wc[I],prof.shd_wc[I],prof.sun_wj[I],prof.shd_wj[I],solar.prob_beam[I], solar.prob_sh[I]);

            }





            // Daily sums of hourly data

            sumevap += sumle;
            sumsens +=sumh;
            sumfc += fc_mol;
            sumpar +=  input.parin;
            sumnet +=  netrad;
            sumbole += bole.respiration_mole;
            sumsoil += soil.respiration_mole;
            sumps+= can_ps_mol;
            sumresp += canresp;
            sumta += tleaf_mean;
            sumgs +=sumksi;
            sumgsoil += soil.heat;
            sumNIRv += NIRv;
            sumAPAR += solar.apar;
            //	 sumisoprene += isoprene_efflux;

            sumtsoil += soil.T_15cm;
            daycnt += 1;

            ZERO(); // re-zero sums



        }    // end of input of hourly met files   yrf.dat


        // fileyear loop

        fclose(fptr1);
        // fclose(fptr4);
        fclose(fptr6);
        fclose(fptr7);
        fclose(fptr8);
        fclose(fptr9);

        fclose(fptr10);



        // zero arrays and summing variables

        ZERO();


        sumfc=0;
        sumevap=0;
        sumsens=0;
        sumps=0;
        sumpar=0;
        sumnet=0;
        sumbole=0;
        sumsoil=0;
        sumta=0;
        daycnt=0;
        sumgs=0;
        sumresp=0;
        sumgsoil=0;
        sumisoprene=0;
        sumtsoil=0;

        //	time_var.phenology=0;

        // increment year if there is need to compute multiple years

        time_var.fileyear++;

        time_var.days=1;


    }   // external bracket of fileyear loop

} // end of main function



// ======================================================================

// Listing of Subroutines


void INPUT_DATA()
{

    double est;

    //  input data and check for bad data
    // note that the data were produced in single precision (float)
    // so I had to read them as single precision, otherwise I ingested
    // garbage

    //   fscanf(fptr1,"%i %i %g %g %g %g %g %g %g %g %g %ld\n",&input.dayy,&input.hhrr,&input.ta,&input.rglobal,&input.parin,&input.pardif,&input.ea,&input.wnd,&input.ppt,&input.co2air,&input.press_mb,&input.flag);

    // fscanf_s(fptr1,"%i,%i,%g,%g,%g,%g,%g,%g,%g,%g,%g,%ld\n",&input.dayy,&input.hhrr,&input.ta,&input.rglobal,&input.parin,&input.pardif,&input.ea,&input.wnd,&input.ppt,&input.co2air,&input.press_mb,&input.flag);

    //   fscanf_s(fptr1,"%i,%g,%g,%g,%g,%g,%g,%g,%g,%g, %g\n",&input.dayy,&input.hhrr,&input.ta,&input.rglobal,&input.ea,&input.wnd,&input.co2air,&input.press_kPa,&input.ustar,&input.tsoil,&input.soil_moisture);
    fscanf(fptr1,"%i,%g,%g,%g,%g,%g,%g,%g,%g,%g, %g\n",&input.dayy,&input.hhrr,&input.ta,&input.rglobal,&input.ea,&input.wnd,&input.co2air,&input.press_kPa,&input.ustar,&input.tsoil,&input.soil_moisture);

    time_var.daytime=input.dayy*10000 + floor(input.hhrr)*100 + (ceil(input.hhrr)-input.hhrr)*60;  // define daytime

    time_var.jdold=time_var.days;                      // identify previous day


    time_var.local_time=input.hhrr;

    time_var.days=input.dayy;

    // input.co2air =+ 100.;


    printf("input data\n");
    printf("%ld \n",time_var.daytime);

    // compute derived quantities for the model

    // day:night difference in u* based on our field data



    met.ustar=input.ustar;

    if(met.ustar < 0.07)
        met.ustar= input.wnd*0.095;  // found co2 blew up with case when u* was 0.05. plotted U vs u*

    met.T_Kelvin=input.ta+273.15;           // compute absolute air temperature

    met.rhova_g = input.ea * 2165/met.T_Kelvin;   // compute absolute humidity, g m-3

    est=ES(met.T_Kelvin);

    met.relative_humidity=input.ea*10./est;   // relative humidity

    met.vpd=est-input.ea*10.;                 // vapor pressure deficit, mb

    if(met.rhova_g<0)                         // check for bad data
        met.rhova_g=0;

    met.press_kpa=input.press_kPa;				// air pressure,  kPa
    met.press_bars=input.press_kPa/100.;        // air pressure, bars
    met.press_Pa=met.press_kpa*1000.;           // pressure, Pa

    // combining gas law constants

    met.pstat273 = .022624 / (273.16 * met.press_bars);

    // cuticular conductance adjusted for pressure and T, mol m-2 s-1

    sfc_res.gcut = bprime * met.T_Kelvin * met.pstat273;

    // cuticular resistance

    sfc_res.rcuticle = 1.0 / sfc_res.gcut;


    if(fabs(input.co2air) >=998.)               // check for bad CO2 data
        input.co2air=400.;

    if(input.rglobal < 0)                       // check for bad Rg
        input.rglobal=0;


    // check for bad par data
    input.parin=4.6*input.rglobal/2.;           // umol m-2 s-1


    if(input.parin < 0)                         // check for bad par data
        input.parin=0.;

    if(input.parin ==0)                         // check for night
    {
        solar.par_beam=0.;
        solar.par_diffuse=0.;
        solar.nir_beam=0.;
        solar.nir_diffuse=0.;
    }

//  set some limits on bad input data to minimize the model from blowing up

    if (solar.ratrad > 0.9 || solar.ratrad < 0.2)
        solar.ratrad=0.5;


    // if humidity is bad, estimate it as some reasonable value, e.g. annual average
    if(met.rhova_g > 30.)
        met.rhova_g=10.;

    // air density, kg m-3

    met.air_density = met.press_kpa * mass_air / (rugc * met.T_Kelvin);

    // air density, mole m-3

    met.air_density_mole=met.press_kpa/ (rugc * met.T_Kelvin) * 1000.;

    soil.Tave_15cm=input.tsoil;

    return;
}

void SOIL_RESPIRATION()
{


    // Computes soil respiration

    /*

    After Hanson et al. 1993. Tree Physiol. 13, 1-15

    reference soil respiration at 20 C, with value of about 5 umol m-2 s-1 from field studies
    */

    soil.base_respiration=8.0;  // at base temp of 22 c, night values, minus plant respiration

    // assume Q10 of 1.4 based on Mahecha et al Science 2010, Ea = 25169

    soil.respiration_mole = soil.base_respiration * exp((25169. / 8.314) * ((1. / 295.) - 1. / (soil.T_15cm + 273.16)));

    // soil wetness factor from the Hanson model, assuming constant and wet soils

    soil.respiration_mole *= 0.86;


    //  convert soilresp to mg m-2 s-1 from umol m-2 s-1

    soil.respiration_mg = soil.respiration_mole * .044;

    return;
}








void FILENAMES()
{

    // Constructs Filenames and Opens files


//       static const char disksubdir[] = "d:\\canalfalfa\\";
    static const char disksubdir[] = "./";

    static const char filesuffix[] =".csv";  // .csv

    // static const char dailyfile[] ="alfave";

    static const char dailyfile[] = "diffus";

    static const char hourlyfile[] ="season";

    static const char soilfile[] ="soil";

//       static const char infilesubdir[] = "d:\\canalfalfa\\";
    static const char infilesubdir[] = "./";

    static const char infilesuffix[] = "f.dat";

    char outbuff[30], Inbuff[21], soilbuff[30], avebuff[30], yrbuff[5];

    errno_t err;

//       _itoa_s(time_var.fileyear,yrbuff,10);
//       itoa(time_var.fileyear,yrbuff,10);
//       strcpy(yrbuff, std::to_string(time_var.fileyear));
//       yrbuff < std::to_string(time_var.fileyear);
    std::stringstream stream;
    stream << time_var.fileyear;
    stream >> yrbuff;


    // file name hourly output

    // sprintf_s(outbuff,30,"%s%s%s%s",disksubdir,hourlyfile,yrbuff,filesuffix);
    snprintf(outbuff,30,"%s%s%s%s",disksubdir,hourlyfile,yrbuff,filesuffix);

    printf("%s \n",outbuff);

    // err=fopen_s(&fptr6,outbuff,"w");
    fptr6=fopen(outbuff,"w");

    if(fptr6==0)
        printf("seasonYR.dat open fptr6");

    fprintf(fptr6,"%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,  %s, %s, %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n",
            "daytime","netrn","sumrn","sumh","sumle","transpiration", "canps","canresp",
            "soilresp","gsoil","ksi", "Tleaf", "Tlsun", "Tlshade", "Kh",
            "Kq", "Kco2", "vpd","diff_tot", "Tsoil15cm", "solar ang", "Ci_avg", "1/rv","NIRv","APAR");



    // file name soil

    // sprintf_s(soilbuff,30,"%s%s%s%s",disksubdir,soilfile,yrbuff,filesuffix);
    snprintf(soilbuff,30,"%s%s%s%s",disksubdir,soilfile,yrbuff,filesuffix);

    // err=fopen_s(&fptr7,soilbuff,"w");
    fptr7=fopen(soilbuff,"w");

    if(fptr7==0)
        printf(" open soilYR.dat fptr7\n");

    fprintf(fptr7,"%s, %s, %s, %s, %s, %s \n",
            "daytime","netrn","soilh","soille", "soilG", "Tsoil");


    // Name file for daily average output, alfave*.*

    // sprintf_s(avebuff,30,"%s%s%s%s",disksubdir,dailyfile,yrbuff,filesuffix);
    snprintf(avebuff,30,"%s%s%s%s",disksubdir,dailyfile,yrbuff,filesuffix);


    //  err=fopen_s(&fptr8,avebuff,"w");
    fptr8=fopen(avebuff,"w");

    if(fptr8==0)
        printf(" open fptr8\n");

    if(fptr8==NULL)
        printf("can't open fpr8\n");

    // fprintf_s(fptr8,"Day, Avg_FC, Avg_EVAP, AVG_H, Avg_PAR, Avg_RNET, lai,Avg_PS, Ave_Resp, Avg_BOLE,Avg_SOIL,Avg_TLeaf, Avg_Gs, Avg_Gsoil, Ave_isoprene, Ave_Tsoil_15cm\n ");
    fprintf(fptr8,"Day, Avg_FC, Avg_EVAP, AVG_H, Avg_PAR, Avg_RNET, lai,Avg_PS, Ave_Resp, Avg_BOLE,Avg_SOIL,Avg_TLeaf, Avg_Gs, Avg_Gsoil, Ave_isoprene, Ave_Tsoil_15cm\n ");

    // profiles

    // err=fopen_s(&fptr9,"d:\\canalfalfa\\profile_air.csv","w");
    fptr9=fopen("profile_air.csv","w");

    if(fptr9==0)
        printf(" open fptr9\n");

    if(fptr9==NULL)
        printf("can't open fptr9\n");

    // 	fprintf_s(fptr9,"%s, %s, %s, %s, %s, %s, %s \n",
    // "i"," tair"," qair"," co2", " Ci", " prob beam", " prob shade" );
    fprintf(fptr9,"%s, %s, %s, %s, %s, %s, %s \n",
            "i"," tair"," qair"," co2", " Ci", " prob beam", " prob shade" );



    // err=fopen_s(&fptr10,"d:\\canalfalfa\\profile_fluxes.csv","w");
    fptr10=fopen("./profile_fluxes.csv","w");

    if(fptr10==0)
        printf(" open fptr10\n");

    if(fptr10==NULL)
        printf("can't open fptr10\n");


    // fprintf_s(fptr10,"%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n",
    // "i"," dHdz", " dLEdz"," dfcdz"," Ci", " cicasun",
    //         " cicash", " wc sun", " wc sh", " wj sun", " wj sh", "prob beam", "prob diff");
    fprintf(fptr10,"%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n",
            "i"," dHdz", " dLEdz"," dfcdz"," Ci", " cicasun",
            " cicash", " wc sun", " wc sh", " wj sun", " wj sh", "prob beam", "prob diff");



    //  sprintf_s(Inbuff,30,"%s","d:\\canalfalfa\\alfmetinput.csv");
    //	strcpy (Inbuff,infilesubdir);
    //	strcat (Inbuff,yrbuff);
    //	strcat_s (Inbuff,21,infilesuffix);


    // printf("%s \n","d:\\canalfalfa\\alfmetinput.csv");

    //err=fopen_s(&fptr1,"d:\\canalfalfa\\alfmetinput.csv","r");


    // printf("%s \n", "D:\\Delta\\Smoke and Diffuse\\WP_table_2018gs.csv");
    // err = fopen_s(&fptr1, "D:\\Delta\\Smoke and Diffuse\\WP_table_2018gs.csv", "r");
    printf("%s \n", "AlfMetInput-nohead.csv");
    fptr1 = fopen("AlfMetInput-nohead.csv", "r");

    if(fptr1==0)
        printf(" open fptr1\n");

    if(fptr1==NULL)
        printf("can't open YRf.dat fptr1\n");

    return;
}



void G_FUNC_DIFFUSE()
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


    int IJ,IN, K, KK,I,II, IN1;

    double aden[18], TT[18], PGF[18], sin_TT[18], del_TT[18], del_sin[18];
    double PPP;
    double ang,dang, aang;
    double cos_A,cos_B,sin_A,sin_B,X,Y;
    double T0,TII,TT0,TT1;
    double R,S, PP,bang;
    double sin_TT1, sin_TT0, square;
    double llai;



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

            llai += prof.dLAIdz[II];


            // CALCULATE PROBABILITY FREQUENCY DISTRIBUTION, BDENS


            FREQ(llai);

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
                }                                                 /*  next IN } */

LOOPOUT:


//       Compute the integrated leaf orientation function, Gfun



                PP = 0.0;

                for(IN = 1; IN <= 16; IN++)
                    PP += (PGF[IN] * aden[IN]);


                PPP += (PP * canopy.bdens[I] * PI9);

            }  // next I

            prof.Gfunc_sky[II][KK] = PPP;

        }  // next IJ

        ang += dang;

    }  // NEXT KK

    return;
}


void GFUNC()
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

    int IJ,IN, K, I,II, IN1;

    double aden[19], TT[19], pgg[19];
    double sin_TT[19], del_TT[19],del_sin[19];
    double PPP, PP, aang;
    double cos_A,cos_B,sin_A,sin_B,X,Y, sin_TT0, sin_TT1;
    double T0,TII,TT0,TT1;
    double R,S,square;

    double llai = 0.0;



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

        llai += prof.dLAIdz[II];


        // Calculate the leaf angle probabilty distribution, bdens


        FREQ(llai);


        // Lemeur defines bdens as delta F/(PI/N), WHERE HERE N=9

        PPP = 0.0;

        for(I = 1; I <= 9; I++)
        {
            aang = ((I - 1.0) * 10.0 + 5.0) * PI180;

            cos_A = cos(aang);
            cos_B = cos(solar.beta_rad);
            sin_A = sin(aang);
            sin_B = sin(solar.beta_rad);

            X = cos_A * sin_B;
            Y = sin_A * cos_B;

            if((aang - solar.beta_rad) <= 0.0)
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


            PPP += (PP * canopy.bdens[I] * 9. / PI);

        } // next I

        prof.Gfunc_solar[II] = PPP;

        if(prof.Gfunc_solar[II] <= 0.0)
            prof.Gfunc_solar[II] = .01;
    }                           // next IJ


    return;
}

void ENERGY_AND_CARBON_FLUXES()
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

    double wj_leaf, wc_leaf;

    double rs_sun, rs_shade, A_mg, resp, internal_CO2;


    for (JJ=1; JJ <= jtot; JJ++)
    {

        // zero summing values

        H_sun=0;
        LE_sun=0;
        Rn_sun=0;
        loutsun=0;
        rs_sun=0;
        prof.sun_gs[JJ]=0;
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

        Tair_K_filtered = prof.tair_filter[JJ] + 273.16;  // absolute air temperature


        // Initialize surface temperature with air temperature

        T_sfc_K = Tair_K_filtered;



        //      Energy balance on sunlit leaves


        //  update latent heat with new temperature during each call of this routine

        fact.latent = LAMBDA(T_sfc_K);
        fact.latent18=fact.latent*18.;

        if(solar.prob_beam[JJ] > 0.0)
        {

            // initial stomatal resistance as a function of PAR. Be careful and use
            // the light dependent version only for first iteration

            rs_sun = prof.sun_rs[JJ];


            //   Compute the resistances for heat and vapor transfer, rh and rv,
            //   for each layer, s/m


            BOUNDARY_RESISTANCE(prof.ht[JJ],prof.sun_tleaf[JJ]);

            // compute energy balance of sunlit leaves

            ENERGY_BALANCE_AMPHI(solar.rnet_sun[JJ], &T_sfc_K, Tair_K_filtered, prof.rhov_filter[JJ], bound_layer_res.vapor, rs_sun, &LE_leaf, &H_leaf, &lout_leaf);


            // compute photosynthesis of sunlit leaves if leaves have emerged

            if(time_var.lai > pai)
                PHOTOSYNTHESIS_AMPHI(solar.quantum_sun[JJ], &rs_sun, prof.ht[JJ],
                                     prof.co2_air[JJ], T_sfc_K, &LE_leaf, &A_mg, &resp, &internal_CO2,
                                     &wj_leaf,&wc_leaf);


            // Assign values of function to the LE and H source/sink strengths

            T_sfc_C=T_sfc_K-273.16;             // surface temperature, Centigrade
            H_sun = H_leaf;                                 // sensible heat flux
            LE_sun = LE_leaf;                                       // latent heat flux
            prof.sun_tleaf[JJ] = T_sfc_C;
            loutsun = lout_leaf;                            // long wave out
            Rn_sun = solar.rnet_sun[JJ] - lout_leaf;  // net radiation

            A_sun = A_mg;                           // leaf photosynthesis, mg CO2 m-2 s-1
            resp_sun=resp;                          // stomatal resistance

            prof.sun_resp[JJ]=resp;     // respiration on sun leaves
            prof.sun_gs[JJ]=1./rs_sun;      // stomatal conductance
            prof.sun_wj[JJ]=wj_leaf;
            prof.sun_wc[JJ]=wc_leaf;

            prof.sun_A[JJ] = A_sun*1000./mass_CO2;  // micromolC m-2 s-1
            prof.sun_rs[JJ] = rs_sun;
            prof.sun_rbh[JJ] = bound_layer_res.heat;
            prof.sun_rbv[JJ] = bound_layer_res.vapor;
            prof.sun_rbco2[JJ] = bound_layer_res.co2;
            prof.sun_ci[JJ]=internal_CO2;

        }

        //    Energy balance on shaded leaves

        // initial value of stomatal resistance based on light

        rs_shade = prof.shd_rs[JJ];


        // boundary layer resistances on shaded leaves.  With different
        // surface temperature, the convective effect may differ from that
        // computed on sunlit leaves

        BOUNDARY_RESISTANCE(prof.ht[JJ],prof.shd_tleaf[JJ]);

        // Energy balance of shaded leaves

        ENERGY_BALANCE_AMPHI(solar.rnet_sh[JJ], &T_sfc_K, Tair_K_filtered, prof.rhov_filter[JJ], bound_layer_res.vapor, rs_shade, &LE_leaf, &H_leaf, &lout_leaf);

        // compute photosynthesis and stomatal conductance of shaded leaves

        if(time_var.lai > pai)
            PHOTOSYNTHESIS_AMPHI(solar.quantum_sh[JJ], &rs_shade,prof.ht[JJ], prof.co2_air[JJ],
                                 T_sfc_K, &LE_leaf, &A_mg, &resp, &internal_CO2,&wj_leaf,&wc_leaf);


        // re-assign variable names from functions output

        T_sfc_C=T_sfc_K-273.16;
        LE_shade = LE_leaf;
        H_shade = H_leaf;
        loutsh = lout_leaf;
        Rn_shade = solar.rnet_sh[JJ] - lout_leaf;

        prof.shd_wj[JJ]=wj_leaf;
        prof.shd_wc[JJ]=wc_leaf;

        A_shade = A_mg;

        resp_shade=resp;

        prof.shd_resp[JJ]=resp;
        prof.shd_tleaf[JJ] = T_sfc_C;
        prof.shd_gs[JJ]=1./rs_shade;
        prof.shd_A[JJ] = A_shade*1000/mass_CO2;   // micromolC m-2 s-1
        prof.shd_rs[JJ] = rs_shade;
        prof.shd_rbh[JJ] = bound_layer_res.heat;
        prof.shd_rbv[JJ] = bound_layer_res.vapor;
        prof.shd_rbco2[JJ] = bound_layer_res.co2;
        prof.shd_ci[JJ]= internal_CO2;


        // compute layer energy fluxes, weighted by leaf area and sun and shaded fractions

        prof.dLEdz[JJ] = prof.dLAIdz[JJ] * (solar.prob_beam[JJ] * LE_sun + solar.prob_sh[JJ] * LE_shade);
        prof.dHdz[JJ] = prof.dLAIdz[JJ] * (solar.prob_beam[JJ] * H_sun + solar.prob_sh[JJ] * H_shade);
        prof.dRNdz[JJ] = prof.dLAIdz[JJ] * (solar.prob_beam[JJ] * Rn_sun + solar.prob_sh[JJ] * Rn_shade);


        // photosynthesis of the layer,  prof.dPsdz has units mg m-3 s-1

        prof.dPsdz[JJ] = prof.dLAIdz[JJ] * (A_sun * solar.prob_beam[JJ] + A_shade * solar.prob_sh[JJ]);

        prof.Ci[JJ] = (prof.sun_ci[JJ] * solar.prob_beam[JJ] + prof.shd_ci[JJ] * solar.prob_sh[JJ]);

        prof.shd_cica[JJ]=prof.shd_ci[JJ]/prof.co2_air[JJ];

        prof.sun_cica[JJ]=prof.sun_ci[JJ]/prof.co2_air[JJ];

        // scaling boundary layer conductance for vapor, 1/rbv

        prof.drbv[JJ] = (solar.prob_beam[JJ]/ prof.sun_rbv[JJ] + solar.prob_sh[JJ]/ prof.shd_rbv[JJ]);


        // photosynthesis of layer, prof.dPsdz has units of micromoles m-2 s-1

        prof.dPsdz[JJ] = prof.dLAIdz[JJ] * (prof.sun_A[JJ] * solar.prob_beam[JJ] +
                                            prof.shd_A[JJ] * solar.prob_sh[JJ]);


        //  respiration of the layer, micromol m-2 s-1

        prof.dRESPdz[JJ] = prof.dLAIdz[JJ] * (resp_sun * solar.prob_beam[JJ] + resp_shade * solar.prob_sh[JJ]);

        // prof.dStomCondz has units of: m s-1

        prof.dStomCondz[JJ] = prof.dLAIdz[JJ] * (solar.prob_beam[JJ]*prof.sun_gs[JJ] + solar.prob_sh[JJ]*prof.shd_gs[JJ]);

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




double SKY_IR (double T)
{

    // Infrared radiation from sky, W m-2, using algorithm from Norman

    double y;

    y = sigma * pow(T,4.) * ((1. - .261 * exp(-.000777 * pow((273.16 - T), 2.))) * solar.ratrad + 1 - solar.ratrad);

    return y;
}

void ZERO()
{

    // re-zeros global arrays and variables

    int I;

    for(I=1; I<=jktot; I++)
    {
        solar.par_down[I]=0;
        solar.par_up[I]=0;
        prof.sun_lai[I]=0;
        prof.shd_lai[I]=0;
        prof.dStomCondz[I]=0;
        prof.sun_T_filter[I]=0;
        prof.shd_T_filter[I]=0;
        solar.nir_dn[I]=0;
        solar.nir_up[I]=0;
        solar.beam_flux_par[I]=0;
        solar.ir_dn[I]=0;
        solar.ir_up[I]=0;
        solar.prob_beam[I]=0;
        solar.prob_sh[I]=0;
        prof.dLEdz[I]=0;
        prof.dPsdz[I]=0;
        prof.dRNdz[I]=0;
        prof.dRESPdz[I]=0;
        solar.beam_flux_nir[I]=0;
        solar.rnet_sun[I]=0;
        solar.rnet_sh[I]=0;
        solar.nir_sun[I]=0;
        solar.nir_sh[I]=0;
        prof.dHdz[I]=0;
        solar.par_shade[I]=0;
        solar.par_sun[I]=0;
        prof.sun_tleaf[I]=0;
        prof.shd_tleaf[I]=0;
        solar.quantum_sun[I]=0;
        solar.quantum_sh[I]=0;
        bole.layer[I]=0;
        prof.sun_ci[I]=0;
        prof.shd_ci[I]=0;
        prof.drbv[I]=0;
    }

    for (I=1; I<= jtot3; I++)
    {
        prof.tair[I]=0;
        prof.rhov_air[I]=0;
        prof.rhov_filter[I]=0;
        prof.tair_filter[I]=0;
        prof.co2_air[I]=0;
        prof.d13C[I]=0;
        prof.d13Cair[I]=0;
        prof.c13cnc[I]=0;
    }

    for (I=1; I<= jktot; I++)
    {
        prof.sun_D13[I]=0;
        prof.shd_D13[I]=0;
        prof.shd_cica[I]=0;
        prof.sun_cica[I]=0;
        prof.source_co2[I]=0;
        prof.sun_wc[I]=0;
        prof.shd_wc[I]=0;
        prof.sun_wj[I]=0;
        prof.shd_wj[I]=0;

    }

    return;
}


void ENERGY_BALANCE (double qrad, double *tsfckpt, double taa, double rhovva, double rvsfc,
                     double stomsfc, double *lept, double *H_leafpt, double *lout_leafpt)
{


    /*
            ENERGY BALANCE COMPUTATION

            A revised version of the quadratic solution to the leaf energy balance relationship is used.

            Paw U, KT. 1987. J. Thermal Biology. 3: 227-233


             H is sensible heat flux density on the basis of both sides of a leaf
             J m-2 s-1 (W m-2).  Note KC includes a factor of 2 here for heat flux
             because it occurs from both sides of a leaf.
    */


    double est, ea, tkta, le2;
    double tk2, tk3, tk4;
    double dest, d2est;
    double lecoef, hcoef, hcoef2, repeat, acoeff, acoef;
    double bcoef, ccoef, product;
    double atlf, btlf, ctlf,vpd_leaf,llout;
    double ke;



    tkta=taa;   // taa is already in Kelvin

    est = ES(tkta);  //  es(T)  Pa


    // ea  = RHOA * TAA * 1000 / 2.165

    ea = 1000 * rhovva * tkta /2.170;   // vapor pressure above leaf, Pa rhov is kg m-3



    // Vapor pressure deficit, Pa


    vpd_leaf = est - ea;

    if (vpd_leaf < 0.)
        vpd_leaf = 0;


    // Slope of the vapor pressure-temperature curve, Pa/C
    // evaluate as function of Tk


    dest = DESDT(tkta);


    // Second derivative of the vapor pressure-temperature curve, Pa/C
    // Evaluate as function of Tk


    d2est = DES2DT(tkta);


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


    ke = 1./ (rvsfc + stomsfc);  // hypostomatous

    // ke = 2/ (rvsfc + stomsfc);  // amphistomatous

    lecoef = met.air_density * .622 * fact.latent * ke / met.press_Pa;


    // Coefficients for sensible heat flux


    hcoef = met.air_density*cp/bound_layer_res.heat;
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

    *lept=le2;  // need to pass pointer out of subroutine


    // solve for Ts using quadratic solution


    // coefficients to the quadratic solution

    atlf = epsigma12 * tk2 + d2est * lecoef / 2.;

    btlf = epsigma8 * tk3 + hcoef2 + lecoef * dest;

    ctlf = -qrad + 2 * llout + lecoef * vpd_leaf;


    // IF (BTLF * BTLF - 4 * ATLF * CTLF) >= 0 THEN

    product = btlf * btlf - 4 * atlf * ctlf;


    // T_sfc_K = TAA + (-BTLF + SQR(BTLF * BTLF - 4 * ATLF * CTLF)) / (2 * ATLF)

    if (product >= 0)
        *tsfckpt = tkta + (-btlf + sqrt(product)) / (2 * atlf);
    else
        *tsfckpt=tkta;


    if(*tsfckpt < -230 || *tsfckpt > 325)
        *tsfckpt=tkta;

    // long wave emission of energy

    *lout_leafpt =epsigma2*pow(*tsfckpt,4);

    // H is sensible heat flux

    *H_leafpt = hcoef2 * (*tsfckpt- tkta);


    return;
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

double DESDT (double t)
{

    // first derivative of es with respect to tk

    //  Pa

    double y;

    y = ES(t) * fact.latent18  / (rgc1000 * t * t);

    return y;
}


double DES2DT(double T)
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

    y = -2. * ES(T) * LAMBDA(T) * 18. / (rgc1000 * T * T * T) +  DESDT(T) * LAMBDA(T) * 18. / (rgc1000 * T * T);




    return y;
}


void PHOTOSYNTHESIS(double Iphoton,double *rstompt, double zzz,double cca,double tlk,
                    double *leleaf, double *A_mgpt, double *resppt, double *cipnt,
                    double *wjpnt, double *wcpnt)
{

    /*

             This program solves a cubic equation to calculate
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

    bc = kct * (1.0 + o2 / ko);

    if(Iphoton < 1)
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





    // growing season, full Ps capacity  (note newer data by Wilson et al shows more
    // dynamics


    jmaxz = jmopt ;
    vcmaxz = vcopt ;

    //vcmaxz = leaf.Vcmax;







    /*
            Scale rd with height via vcmax and apply temperature
            correction for dark respiration
    */

    rdz=vcmaxz * 0.004657;


    // reduce respiration by 40% in light according to Amthor


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

    rb_mole = bound_layer_res.co2 * tlk * (met.pstat273);

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

    rh_leaf  = SFC_VPD(tlk, zzz, leleaf);

    k_rh = rh_leaf * sfc_res.kballstr;  // combine product of rh and K ball-berry

    /*
            Gs from Ball-Berry is for water vapor.  It must be divided
            by the ratio of the molecular diffusivities to be valid
            for A
    */
    k_rh = k_rh / 1.6;      // adjust the coefficient for the diffusion of CO2 rather than H2O

    gb_k_rh = gb_mole * k_rh;

    ci_guess = cca * .7;    // initial guess of internal CO2 to estimate Wc and Wj


    // cubic coefficients that are only dependent on CO2 levels


    alpha_ps = 1.0 + (bprime16 / gb_mole) - k_rh;
    bbeta = cca * (gb_k_rh - 2.0 * bprime16 - gb_mole);
    gamma = cca * cca * gb_mole * bprime16;
    theta_ps = gb_k_rh - bprime16;

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


    // frost and end of leaf photosynthesis and respiration

    //   if(time_var.days > time_var.leafdrop)
    //   {
    //     wj = 0;
    //     j_photon = 0;
    //     wc=0;
    //     rd = 0;
    //   }

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

    cs = cca - aphoto / gb_mole;

    if(cs > 1000)
        cs=input.co2air;

    /*
            Stomatal conductance for water vapor


            forest are hypostomatous.
            Hence we don't divide the total resistance
            by 2 since transfer is going on only one side of a leaf.

    		alfalfa is amphistomatous...be careful on where the factor of two is applied
    		just did on LE on energy balance

    */

    gs_leaf_mole = (sfc_res.kballstr * rh_leaf * aphoto / cs) + bprime;


    // convert Gs from vapor to CO2 diffusion coefficient


    gs_co2 = gs_leaf_mole / 1.6;

    /*
            stomatal conductance is mol m-2 s-1
            convert back to resistance (s/m) for energy balance routine
    */

    gs_m_s = gs_leaf_mole * tlk * met.pstat273;

    // need point to pass rstom out of subroutine

    *rstompt = 1.0 / gs_m_s;


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

    gs_m_s = gs_leaf_mole * tlk * (met.pstat273);

    // need point to pass rstom out of subroutine as a pointer

    *rstompt = 1.0 / gs_m_s;


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
    *A_mgpt = aphoto * .044;

    *resppt=rd;

    *cipnt=ci;

    *wcpnt=wc;

    *wjpnt=wj;

    /*

         printf(" cs       ci      gs_leaf_mole      CA     ci/CA  APS  root1  root2  root3\n");
         printf(" %5.1f   %5.1f   %6.3f    %5.1f %6.3f  %6.3f %6.3f %6.3f  %6.3f\n", cs, ci, gs_leaf_mole, cca, ci / cca,aphoto,root1, root2, root3 );

    */

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


void SET_SOIL_TEMP()
{   float lagtsoil;

    soil.Temp_ref=20;   // long term mean for this experiment

    soil.T_base=input.tsoil;  // 32 cm soil temperature, measured

    return;
}

void ANGLE()
{

//       ANGLE computes solar elevation angles,

//       This subroutine is based on algorithms in Walraven. 1978. Solar Energy. 20: 393-397



    double theta_angle,G,EL,EPS,sin_el,A1,A2,RA;
    double delyr,leap_yr,T_local,time_1980,leaf_yr_4, delyr4;
    double day_savings_time, day_local;
    double S,HS,phi_lat_radians,value,declination_ang,ST,SSAS;
    double E_ang, zenith, elev_ang_deg, cos_zenith;

    double radd = .017453293;
    double twopi = 6.28318;

    // matlab code

    double lat_rad, long_rad, std_meridian, delta_long, delta_hours, declin;
    double cos_hour, sunrise, sunset, daylength, f, Et, Lc_deg, Lc_hr, T0, hour;
    double sin_beta, beta_deg, beta_rad, day, time_zone, lat_deg, long_deg;

    // Twitchell Island, CA

    double latitude = 38.1;     // latitude
    double longitude= 121.65;    // longitude

    // Eastern Standard TIME

    double zone = 8.0;          // Five hour delay from GMT


    delyr = time_var.year - 1980.0;
    delyr4=delyr/4.0;
    leap_yr=fmod(delyr4,4.0);
    day_savings_time=0.0;

    // Daylight Savings Time, Dasvtm =1
    // Standard time, Dasvtm= 0


    T_local = time_var.local_time;

    day_local = time_var.days;
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
    solar.beta_rad = elev_ang_deg * radd;
    solar.sine_beta = sin(solar.beta_rad);
    cos_zenith = cos(E_ang);
    solar.beta_deg = solar.beta_rad / PI180;

    // enter Matlab version

    time_zone=-8;
    lat_deg=38.1;
    long_deg=-121.65;

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

    sin_beta=sin(lat_rad)*sin(declin)+cos(lat_rad)*cos(declin)* cos(hour);

    // solar elevation, radians

    beta_rad=asin(sin_beta);

    // solar elevation, degrees

    beta_deg=beta_rad*180/PI;

    solar.beta_rad = beta_rad;
    solar.sine_beta = sin(solar.beta_rad);
    solar.beta_deg = solar.beta_rad / PI180;


    return;
}



void CONC(double *source, double *cncc, double cref, double soilflux, double factor)
{


    // Subroutine to compute scalar concentrations from source
    // estimates and the Lagrangian dispersion matrix


    double sumcc[sze3], cc[sze3];
    double disper, ustfact, disperzl, soilbnd;

    int i, j;


    // Compute concentration profiles from Dispersion matrix


    ustfact = ustar_ref / met.ustar;         // factor to adjust Dij with alternative u* values

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

            disper = ustfact * met.dispersion[i][j];                    // units s/m

            // scale dispersion matrix according to Z/L


            // if(met.zl < 0)
            // disperzl = disper * (0.679 (z/L) - 0.5455)/(z/L-0.5462);
            // else
            // disperzl=disper;

            // updated Dispersion matrix (Oct, 2015)..for alfalfa  runs for a variety of z?

            if(met.zl < 0)
                disperzl = disper * (0.97*-0.7182)/(met.zl -0.7182);
            else
                disperzl=disper * (-0.31 * met.zl + 1.00);

            sumcc[i] += delz * disperzl * source[j];


        } // next j


        // scale dispersion matrix according to Z/L


        disper = ustfact * met.dispersion[i][1];


        if(met.zl < 0)
            disperzl = disper * (0.97*-0.7182)/(met.zl -0.7182);
        else
            disperzl=disper * (-0.31 * met.zl + 1.00);


        // add soil flux to the lowest boundary condition

        soilbnd=soilflux*disperzl/factor;

        cc[i]=sumcc[i]/factor+soilbnd;


    } // next i


    //  Compute scalar profile below reference

    for(i=1; i<= jtot3; i++)
        cncc[i] = cc[i] + cref - cc[izref];


    return;
}



void IRFLUX()
{
    int J, JJ,JJP1,jktot1,JM1,K;
    double ir_in, abs_IR,reflc_lay_IR;
    double Tk_sun_filt,Tk_shade_filt, IR_source_sun,IR_source_shade,IR_source;
    double SDN[sze], SUP[sze];
    double emiss_IR_soil;

    ir_in = SKY_IR(met.T_Kelvin);


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
    solar.ir_dn[jktot] = ir_in;

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



        Tk_sun_filt = prof.sun_T_filter[JJ]+273.16;
        Tk_shade_filt = prof.shd_T_filter[JJ]+273.16;

        IR_source_sun = solar.prob_beam[JJ] *pow(Tk_sun_filt,4.);
        IR_source_shade = solar.prob_sh[JJ] * pow(Tk_shade_filt,4.);

        IR_source = epsigma * (IR_source_sun + IR_source_shade);

        /*
                ' Intercepted IR that is radiated up
        */

        SUP[JJP1] = IR_source * (1. - solar.exxpdir[JJ]);

        /*
                'Intercepted IR that is radiated downward
        */

        SDN[JJ] = IR_source * (1. - solar.exxpdir[JJ]);

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

        solar.ir_dn[JJ] = solar.exxpdir[JJ] * solar.ir_dn[JJP1] + SDN[JJ];

    } // next J

    emiss_IR_soil = epsigma * pow((soil.sfc_temperature + 273.16),4.);

    SUP[1] = solar.ir_dn[1] * (1. - epsoil);
    solar.ir_up[1] = emiss_IR_soil + SUP[1];

    for(J = 2; J<=jktot ; J++)
    {
        JM1 = J - 1;
        /*
                 '
                 ' REMEMBER THE IR UP IS FROM THE LAYER BELOW
        */

        solar.ir_up[J] = solar.exxpdir[JM1] * solar.ir_up[JM1] + SUP[J];

    } /* NEXT J  */



    for (K = 1; K<=2; K++)
    {

        for (J = 2; J<=jktot; J++)
        {
            JJ = jktot - J + 1;
            JJP1 = JJ + 1;

            reflc_lay_IR = (1 - solar.exxpdir[JJ]) * (epm1);
            solar.ir_dn[JJ] = solar.exxpdir[JJ] * solar.ir_dn[JJP1] + solar.ir_up[JJ] * reflc_lay_IR + SDN[JJ];
        }        // next J

        SUP[1] = solar.ir_dn[1] * (1 - epsoil);
        solar.ir_up[1] = emiss_IR_soil + SUP[1];

        for (J = 2; J<= jktot; J++)
        {
            JM1 = J - 1;
            reflc_lay_IR = (1 - solar.exxpdir[JM1]) * (epm1);
            solar.ir_up[J] = reflc_lay_IR * solar.ir_dn[J] + solar.ir_up[JM1] * solar.exxpdir[JM1] + SUP[J];
        }   // next J

    } // next K


    return;
}



void DIFFUSE_DIRECT_RADIATION()
{
    double fand, fir,fv;
    double rdir, rdvis,rsvis,wa;
    double ru, rsdir, rvt,rit, nirx;
    double xvalue,fvsb,fvd,fansb;
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
    fir = .54;
    fv = .46;

    ru = met.press_kpa / (101.3 * solar.sine_beta);

//         visible direct PAR


    rdvis = 624.0 * exp(-.185 * ru) * solar.sine_beta;


//      potential diffuse PAR

//        rsvis = .4 * (600.0 - rdvis) * solar.sine_beta;

// corrected version

    rsvis = 0.4 * (624. *solar.sine_beta -rdvis);



    /*
            solar constant was assumed to be: 1320 W m-2

            it is really 1373  W m-2

            water absorption in NIR for 10 mm precip water
    */

    wa = 1373.0 * .077 * pow((2. * ru),.3);

    /*

            direct beam NIR
    */
    rdir = (748.0 * exp(-.06 * ru) - wa) * solar.sine_beta;

    if(rdir < 0)
        rdir=0;

    /*
            potential diffuse NIR
    */

//       rsdir = .6 * (720 - wa - rdir) * solar.sine_beta;  // Eva asks if we should correct twice for angles?

// corrected version, Rdn=0.6(720 - RDN/cos(theta) -w) cos(theta).


    rsdir = 0.6* (748. -rdvis/solar.sine_beta-wa)*solar.sine_beta;

    if (rsdir < 0)
        rsdir=0;


    rvt = rdvis + rsvis;
    rit = rdir + rsdir;

    if(rit <= 0)
        rit = .1;

    if(rvt <= 0)
        rvt = .1;

    solar.ratrad = input.rglobal / (rvt + rit);


    if (time_var.local_time >= 12.00 && time_var.local_time <=13.00)
        solar.ratradnoon=solar.ratrad;
    /*
            ratio is the ratio between observed and potential radiation

            NIR flux density as a function of PAR

            since NIR is used in energy balance calculations
            convert it to W m-2: divide PAR by 4.6
    */


    nirx = input.rglobal - (input.parin / 4.6);


//        ratio = (PARIN / 4.6 + NIRX) / (rvt + rit)

    if (solar.ratrad >= .9)
        solar.ratrad = .89;

    if (solar.ratrad <= 0)
        solar.ratrad=0.22;


//     fraction PAR direct and diffuse

    xvalue=(0.9-solar.ratrad)/.70;

    fvsb = rdvis / rvt * (1. - pow(xvalue,.67));

    if(fvsb < 0)
        fvsb = 0.;

    if (fvsb > 1)
        fvsb=1.0;


    fvd = 1. - fvsb;


//      note PAR has been entered in units of uE m-2 s-1

    solar.par_beam = fvsb * input.parin;
    solar.par_diffuse = fvd * input.parin;

    if(solar.par_beam <= 0)
    {
        solar.par_beam = 0;
        solar.par_diffuse = input.parin;
    }

    if(input.parin == 0)
    {
        solar.par_beam=0.001;
        solar.par_diffuse=0.001;
    }

    xvalue=(0.9-solar.ratrad)/.68;
    fansb = rdir / rit * (1. - pow(xvalue,.67));

    if(fansb < 0)
        fansb = 0;

    if(fansb > 1)
        fansb=1.0;


    fand = 1. - fansb;


//      NIR beam and diffuse flux densities

    solar.nir_beam = fansb * nirx;
    solar.nir_diffuse = fand * nirx;

    if(solar.nir_beam <= 0)
    {
        solar.nir_beam = 0;
        solar.nir_diffuse = nirx;
    }

    if (nirx == 0)
    {
        solar.nir_beam=0.1;
        solar.nir_diffuse=0.1;
    }


    solar.nir_beam= nirx-solar.nir_diffuse;
    solar.par_beam = input.parin-solar.par_diffuse;


    return;
}




void FREQ (double lflai)
{
    int I;

    double STD, MEAN, CONS;
    double VAR,nuu,SUM,MU,FL1,MU1,nu1;
    double ANG,FL2,FL3;

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
        canopy.bdens[I] = CONS * FL1 * FL2 * FL3;
    }
    return;
}

double SFC_VPD (double tlk, double Z, double *leleafpt)
{

    // this function computes the relative humidity at the leaf surface for
    // application in the Ball Berry Equation

    //  latent heat flux, LE, is passed through the function, mol m-2 s-1
    //  and it solves for the humidity at leaf surface

    int J;
    double y, rhov_sfc,e_sfc,vpd_sfc,rhum_leaf;
    double es_leaf;

    es_leaf = ES(tlk);    // saturation vapor pressure at leaf temperature

    J = (int)(Z / delz);  // layer number

    rhov_sfc = (*leleafpt / (fact.latent)) * bound_layer_res.vapor + prof.rhov_air[J];  /* kg m-3 */

    e_sfc = 1000* rhov_sfc * tlk / 2.165;    // Pa
    vpd_sfc = es_leaf - e_sfc;              // Pa
    rhum_leaf = 1. - vpd_sfc / es_leaf;     // 0 to 1.0
    y = rhum_leaf;

    return y;
}




void LAI_TIME()
{


    // Evaluate how LAI and other canopy structural variables vary
    // with time

    long int J,I, II, JM1;
    double lai_z[sze];
    double TF,MU1,MU2,integr_beta;
    double dx,DX2,DX4,X,P_beta,Q_beta,F1,F2,F3;
    double beta_fnc[sze],ht_midpt[6],lai_freq[6];
    double cum_lai,sumlai,dff,XX;
    double cum_ht;
    double AA,DA,dff_Markov;
    double cos_AA,sin_AA,exp_diffuse;
    double lagtsoil;

    // lag is 100 days or 1.721 radians

    //  soil.T_base= 14.5 + 9. * sin((time_var.days * 6.283 / 365.) - 1.721);   */

    // compute seasonal trend of Tsoil Base at 85 cm, level 10 of the soil model




    // amplitude of the soil temperature at a reference depth

    // On average the mean annual temperature occurs around day 100 or 1.721 radians



    soil.T_base= input.tsoil; // seasonal variation in reference soil temperature at 32 cm


    // full leaf


    time_var.lai = lai;


    // optical properties PAR wave band
    // after Norman (1979) and NASA report

    solar.par_reflect = .0377;  // spectrometer and from alfalfa, NASA report 1139, Bowker
    solar.par_trans = .072;
    solar.par_soil_refl = 0;    // black soil .3;
    solar.par_absorbed = (1. - solar.par_reflect - solar.par_trans);


    // optical properties NIR wave band
    // after Norman (1979) and NASA report

    solar.nir_reflect = .60;  // Strub et al IEEE...spectrometer from Alfalfa
    solar.nir_trans = .26;
    solar.nir_soil_refl = 0;    //  black soils 0.6;  // updated


    // value for variable reflectance

    //solar.nir_reflect = (15 * leaf.N + 5)/100;

    //leaf.Vcmax = 26.87 + 15.8 * leaf.N;


    // Absorbed NIR

    solar.nir_absorbed = (1. - solar.nir_reflect - solar.nir_trans);


    // height of mid point of layer scaled to 0.55m tall alfalfa

    //ht_midpt[1] = 0.1;
    //ht_midpt[2]= 0.2;
    //ht_midpt[3]= 0.3;
    //ht_midpt[4] = 0.4;
    //ht_midpt[5] = 0.5;

    // 3 m tule
    ht_midpt[1] = 0.5;
    ht_midpt[2] = 1.0;
    ht_midpt[3] = 1.5;
    ht_midpt[4] = 2.0;
    ht_midpt[5] = 2.5;



    // lai of the layers at the midpoint of height, scaled to 1.65 LAI


    // lai_freq[1] = 0.05 * lai;
    // lai_freq[2] = 0.30 * lai;
    // lai_freq[3] = 0.30 * lai;
    //  lai_freq[4] = 0.30 * lai;
    //  lai_freq[5] = 0.05 * lai;

    lai_freq[1] = 0.6 * lai;
    lai_freq[2] = 0.60 * lai;
    lai_freq[3] = 0.60 * lai;
    lai_freq[4] = 0.60 * lai;
    lai_freq[5] = 0.6 * lai;






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


        ht_midpt[I] /= ht;   // was ht, but then it would divide by 24 or so


        // Total F in each layer. Should sum to LAI


        TF += lai_freq[I];


        // weighted mean lai

        MU1 += (ht_midpt[I] * lai_freq[I]);


        // weighted variance

        MU2 +=  (ht_midpt[I] * ht_midpt[I] * lai_freq[I]);
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

    lai_z[1] = beta_fnc[1] * time_var.lai / integr_beta;

    for(I = 2; I <= JM1; I++)
        lai_z[I] = beta_fnc[I] * time_var.lai / integr_beta;


    lai_z[jtot] = beta_fnc[jtot] * time_var.lai / integr_beta;

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


        prof.dLAIdz[I] = lai_z[I];
    } // next I



    G_FUNC_DIFFUSE();   // Direction cosine for the normal between the mean
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

        dff = prof.dLAIdz[J];  //  + prof.dPAIdz[J]
        sumlai += prof.dLAIdz[J];

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

            exp_diffuse = exp(-dff_Markov * prof.Gfunc_sky[J][II] / cos_AA);

            // for spherical distribution
            // exp_diffuse = exp(-DFF * prof.Gfunc_sky(J, II) / cos_AA)

            XX += (cos_AA * sin_AA * exp_diffuse);
            AA += DA;
        }  // next II

        /*
        'Itegrated probability of diffuse sky radiation penetration
        'for each layer
        */

        solar.exxpdir[J] = 2. * XX * DA;

        if(solar.exxpdir[J] > 1.)
            solar.exxpdir[J] = .9999;

    } // next J


    printf("lai  day  time_var.lai\n");
    printf("%5.2f  %4i  %5.2f\n", lai,time_var.days,time_var.lai);

    return;
}



void NIR()
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

    double nir_incoming,fraction_beam;
    double SUP[sze], SDN[sze], transmission_layer[sze], reflectance_layer[sze], beam[sze];
    double TBEAM[sze];
    double ADUM[sze];
    double exp_direct,sumlai, dff;
    double TLAY2,nir_normal, NSUNEN;
    double llai,NIRTT,DOWN, UP;

    nir_incoming = solar.nir_beam + solar.nir_diffuse;

    if(nir_incoming <= 1. || solar.sine_beta <=0)
        goto NIRNIGHT;

    fraction_beam = solar.nir_beam / (solar.nir_beam + solar.nir_diffuse);
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

        sumlai += prof.dLAIdz[JJ];  // + prof.dPAIdz[JJ]

        /*
             'Itegrated probability of diffuse sky radiation penetration
             'for each layer
             '
            EXPDIF[JJ] is computed in PAR and can be used in NIR and IRFLUX
        */

        reflectance_layer[JJ] = (1. - solar.exxpdir[JJ]) * solar.nir_reflect;


        //       DIFFUSE RADIATION TRANSMITTED THROUGH LAYER

        transmission_layer[JJ] = (1. - solar.exxpdir[JJ]) * solar.nir_trans + solar.exxpdir[JJ];
    } // next J


//       COMPUTE THE PROBABILITY OF beam PENETRATION


    sumlai = 0;

    for(J = 2; J <= jktot; J++)
    {
        JJ = jktot - J + 1;
        JJP1 = JJ + 1;

        // Probability of beam penetration.


        dff = prof.dLAIdz[JJ]; /* '+ prof.dPAIdz[JJ] */
        sumlai += prof.dLAIdz[JJ];


        exp_direct = exp(-dff * markov*prof.Gfunc_solar[JJ] / solar.sine_beta);

        // PEN1 = exp(-llai * prof.Gfunc_solar[JJ] / solar.sine_beta)
        // exp_direct = exp(-DFF * prof.Gfunc_solar[JJ] / solar.sine_beta)

        // Beam transmission

        beam[JJ] = beam[JJP1] * exp_direct;

        TBEAM[JJ] = beam[JJ];


        SUP[JJP1] = (TBEAM[JJP1] - TBEAM[JJ]) * solar.nir_reflect;

        SDN[JJ] = (TBEAM[JJP1] - TBEAM[JJ]) * solar.nir_trans;
    }  // next J

    /*
        initiate scattering using the technique of NORMAN (1979).
        scattering is computed using an iterative technique
    */

    SUP[1] = TBEAM[1] * solar.nir_soil_refl;
    solar.nir_dn[jktot] = 1. - fraction_beam;
    ADUM[1] = solar.nir_soil_refl;

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

        solar.nir_dn[JJ] = solar.nir_dn[JJP1] * transmission_layer[JJ] / (1. - ADUM[JJP1] * reflectance_layer[JJ]) + SDN[JJ];
        solar.nir_up[JJP1] = ADUM[JJP1] * solar.nir_dn[JJP1] + SUP[JJP1];
    }

    // lower boundary: upward radiation from soil


    solar.nir_up[1] = solar.nir_soil_refl * solar.nir_dn[1] + SUP[1];

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
            DOWN = transmission_layer[JJ] * solar.nir_dn[JJP1] + solar.nir_up[JJ] * reflectance_layer[JJ] + SDN[JJ];

            if ((fabs(DOWN - solar.nir_dn[JJ])) > .01)
                IREP = 1;

            solar.nir_dn[JJ] = DOWN;
        }

        // upward radiation at soil is reflected beam and downward diffuse

        solar.nir_up[1] = (solar.nir_dn[1] + TBEAM[1]) * solar.nir_soil_refl;

        for (JJ = 2; JJ <=jktot; JJ++)
        {
            JM1 = JJ - 1;

            UP = reflectance_layer[JM1] * solar.nir_dn[JJ] + solar.nir_up[JM1] * transmission_layer[JM1] + SUP[JJ];

            if ((fabs(UP - solar.nir_up[JJ])) > .01)
                IREP = 1;

            solar.nir_up[JJ] = UP;
        }

    }


    // Compute NIR flux densities


    solar.nir_total = solar.nir_beam + solar.nir_diffuse;
    llai = lai;

    for(J = 1; J<=jktot; J++)
    {
        llai -= prof.dLAIdz[J];   // decrement LAI


        // upward diffuse NIR flux density, on the horizontal

        solar.nir_up[J] *= solar.nir_total;

        if(solar.nir_up[J] <= 0.)
            solar.nir_up[J] = .1;


        // downward beam NIR flux density, incident on the horizontal


        solar.beam_flux_nir[J] = beam[J] * solar.nir_total;

        if(solar.beam_flux_nir[J] <= 0.)
            solar.beam_flux_nir[J] = .1;


        // downward diffuse radiaiton flux density on the horizontal

        solar.nir_dn[J] *= solar.nir_total;

        if(solar.nir_dn[J] <= 0.)
            solar.nir_dn[J] = .1;


        // total downward NIR, incident on the horizontal


        NIRTT = solar.beam_flux_nir[J] + solar.nir_dn[J];

    } // next J

    for(J = 1; J<=jtot; J++)
    {



        // normal radiation on sunlit leaves

        if(solar.sine_beta > 0.1)
            nir_normal = solar.nir_beam * prof.Gfunc_solar[J] / solar.sine_beta;
        else
            nir_normal=0;

        NSUNEN = nir_normal * solar.nir_absorbed;

        /*
                 ' Diffuse radiation received on top and bottom of leaves
                 ' drive photosynthesis and energy exchanges
        */

        solar.nir_sh[J] = (solar.nir_dn[J] + solar.nir_up[J]);


        // absorbed radiation, shaded

        solar.nir_sh[J] *= solar.nir_absorbed;


        // plus diffuse component


        solar.nir_sun[J] = NSUNEN + solar.nir_sh[J];  // absorbed NIR on sun leaves
    } // next J

NIRNIGHT:  // jump to here at night since fluxes are zero
    return;
}

void PAR()
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


    if(solar.sine_beta <= 0.1)
        goto par_night;

    fraction_beam = solar.par_beam / input.parin;



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

        dff = prof.dLAIdz[JJ];  /* + prof.dPAIdz[JJ] */
        sumlai += dff;

        reflectance_layer[JJ] = (1. - solar.exxpdir[JJ]) * solar.par_reflect;


        //   DIFFUSE RADIATION TRANSMITTED THROUGH LAYER

        transmission_layer[JJ] = (1. - solar.exxpdir[JJ]) * solar.par_trans + solar.exxpdir[JJ];

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
        dff = prof.dLAIdz[JJ];  /* + prof.dPAIdz[JJ] */

        sumlai += dff;

        exp_direct = exp(-dff*markov*prof.Gfunc_solar[JJ]/ solar.sine_beta);

        PEN2 = exp(-sumlai*markov*prof.Gfunc_solar[JJ]/ solar.sine_beta);


        /* lai Sunlit and shaded  */

        prof.sun_lai[JJ]=solar.sine_beta * (1 - PEN2)/ (markov*prof.Gfunc_solar[JJ]);

        prof.shd_lai[JJ]=sumlai - prof.sun_lai[JJ];


        /* note that the integration of the source term time solar.prob_beam with respect to
           leaf area will yield the sunlit leaf area, and with respect to solar.prob_sh the
           shaded leaf area.


        In terms of evaluating fluxes for each layer

        Fcanopy = sum {fsun psun + fshade pshade}  (see Leuning et al. Spitters et al.)

        psun is equal to exp(-lai G markov/sinbet)

        pshade = 1 - psun


        */

        solar.prob_beam[JJ] = markov*PEN2;

        if(solar.prob_beam == 0)
            PEN1 = 0;


        // probability of beam

        beam[JJ] = beam[JJP1] * exp_direct;

        QU = 1.0 - solar.prob_beam[JJ];

        if(QU > 1)
            QU=1;

        if (QU < 0)
            QU=0;


        // probability of umbra

        solar.prob_sh[JJ] = QU;

        TBEAM[JJ] = beam[JJ];


        // beam PAR that is reflected upward by a layer

        SUP[JJP1] = (TBEAM[JJP1] - TBEAM[JJ]) * solar.par_reflect;


        // beam PAR that is transmitted downward


        SDN[JJ] = (TBEAM[JJP1] - TBEAM[JJ]) * solar.par_trans;

    } // next J
    /*
         initiate scattering using the technique of NORMAN (1979).
         scattering is computed using an iterative technique.

         Here Adum is the ratio up/down diffuse radiation.

    */
    SUP[1] = TBEAM[1] * solar.par_soil_refl;

    solar.par_down[jktot] = 1.0 - fraction_beam;
    ADUM[1] = solar.par_soil_refl;

    for(J = 2,JM1=1; J<=jktot; J++,JM1++)
    {
        TLAY2 = transmission_layer[JM1] * transmission_layer[JM1];
        ADUM[J] = ADUM[JM1] * TLAY2 / (1. - ADUM[JM1] * reflectance_layer[JM1]) + reflectance_layer[JM1];
    } /* NEXT J */

    for(J = 1; J<= jtot; J++)
    {
        JJ = jtot - J + 1;
        JJP1 = JJ + 1;
        solar.par_down[JJ] = solar.par_down[JJP1] * transmission_layer[JJ] / (1. - ADUM[JJP1] * reflectance_layer[JJ]) + SDN[JJ];
        solar.par_up[JJP1] = ADUM[JJP1] * solar.par_down[JJP1] + SUP[JJP1];
    } // next J


    // lower boundary: upward radiation from soil

    solar.par_up[1] = solar.par_soil_refl * solar.par_down[1] + SUP[1];

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

    while (IREP==1)
    {
        IREP = 0;

        ITER += 1;

        for(J = 2; J <= jktot; J++)
        {
            JJ = jktot - J + 1;
            JJP1 = JJ + 1;
            DOWN = transmission_layer[JJ] * solar.par_down[JJP1] + solar.par_up[JJ] * reflectance_layer[JJ] + SDN[JJ];

            if((fabs(DOWN - solar.par_down[JJ])) > .01)
                IREP = 1;

            solar.par_down[JJ] = DOWN;
        }  // next J

        //  upward radiation at soil is reflected beam and downward diffuse  */

        solar.par_up[1] = (solar.par_down[1] + TBEAM[1]) * solar.par_soil_refl;

        for(JJ = 2; JJ <=jktot; JJ++)
        {
            JM1 = JJ - 1;
            UP = reflectance_layer[JM1] * solar.par_down[JJ] + solar.par_up[JM1] * transmission_layer[JM1] + SUP[JJ];

            if((fabs(UP - solar.par_up[JJ])) > .01)
                IREP = 1;

            solar.par_up[JJ] = UP;
        }  // next JJ

    }


    // Compute flux density of PAR

    llai = 0;

    for(J = 1; J<=jktot; J++)
    {
        llai += prof.dLAIdz[J];

        // upward diffuse PAR flux density, on the horizontal

        solar.par_up[J] *= input.parin;

        if(solar.par_up[J] <= 0)
            solar.par_up[J] = .001;

        // downward beam PAR flux density, incident on the horizontal

        solar.beam_flux_par[J] = beam[J] * input.parin;

        if(solar.beam_flux_par[J] <= 0)
            solar.beam_flux_par[J] = .001;


        // Downward diffuse radiatIon flux density on the horizontal

        solar.par_down[J] *= input.parin;

        if(solar.par_down[J] <= 0)
            solar.par_down[J] = .001;

        //   Total downward PAR, incident on the horizontal

        par_total = (int)solar.beam_flux_par[J] +(int)solar.par_down[J];


    } // next J

    if(solar.par_beam <= 0)
        solar.par_beam = .001;


    //  PSUN is the radiation incident on the mean leaf normal

    for(JJ = 1; JJ<=jtot; JJ++)
    {


        if (solar.sine_beta > 0.1)
            par_normal_quanta = solar.par_beam * prof.Gfunc_solar[JJ] / (solar.sine_beta);
        else
            par_normal_quanta=0;

        // amount of energy absorbed by sunlit leaf */

        par_normal_abs_energy = par_normal_quanta*solar.par_absorbed / 4.6;   // W m-2

        par_normal_abs_quanta = par_normal_quanta * solar.par_absorbed;      // umol m-2 s-1

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
        solar.quantum_sh[JJ] = (solar.par_down[JJ] + solar.par_up[JJ])*solar.par_absorbed;     /* umol m-2 s-1  */

        solar.quantum_sun[JJ] = solar.quantum_sh[JJ] + par_normal_abs_quanta;


        // calculate absorbed par


        solar.par_shade[JJ] = solar.quantum_sh[JJ] / 4.6;    /* W m-2 */

        /*
                 solar.par_sun is the total absorbed radiation on a sunlit leaf,
                 which consists of direct and diffuse radiation
        */

        solar.par_sun[JJ] = par_normal_abs_energy + solar.par_shade[JJ];


    } // next JJ

par_night:

    if(solar.sine_beta <= 0.1)
    {

        for(I = 1; I<=jtot; I++)
        {
            solar.prob_sh[I] = 1.;
            solar.prob_beam[I] = 0.;
        }
    }
    return;
}


double SOIL_SFC_RESISTANCE(double wg)
{   double y, wg0;

    // Camillo and Gurney model for soil resistance
    // Rsoil= 4104 (ws-wg)-805, ws=.395, wg=0
    // ws= 0.395

    // wg is at 10 cm, use a linear interpolation to the surface, top cm, mean between 0 and 2 cm

    wg0 = 1 * wg/10;

    //    y=4104.* (0.395-wg0)-805.;

    // model of Kondo et al 1990, JAM for a 2 cm thick soil

    y = 3e10 * pow((0.395-wg0), 16.6);

    return y;
}


void FRICTION_VELOCITY()
{


    // this subroutine updates ustar and stability corrections
    // based on the most recent H and z/L values

    double xzl, logprod, phim;

    // this subroutine is uncessary for CanAlfalfa since we measure and input ustar

    met.ustar=input.ustar;

    met.H_old= 0.85 *met.sensible_heat_flux+ 0.15*met.H_old;    // filter sensible heat flux to reduce run to run instability

    met.zl = -(0.4*9.8*met.H_old*14.75)/(met.air_density*1005.*met.T_Kelvin*pow(met.ustar,3.)); // z/L

    /*
            // restrict z/L within reasonable bounds to minimize numerical errors

            if (met.zl > .25)
            met.zl=.25;

            if (met.zl < -3.0)
            met.zl = -3.0;


            if(met.zl<0)

            {xzl=pow((1.-met.zl*16.),.25);

            logprod=((1.+xzl*xzl)/2.)*pow((1.+xzl)/2.,2.);


             if (logprod > 0 || logprod < 100000)
             logprod=log(logprod);
             else
             logprod=-1.;


            phim=logprod-2.*atan(xzl)+3.1415/2.;
            }
             else
            {
             phim = -5.*met.zl;
            }


            if ((1.77-phim)>0.)
            met.ustarnew=0.4*input.wnd/pow((1.77-phim),2.);
            else
            met.ustarnew=met.ustar;


            if (met.ustarnew<0.02)
            met.ustarnew=0.02;

            met.ustar= 0.85 *met.ustar+ 0.15 *met.ustarnew;

            if(met.ustar > 2.)
    {

            if(input.parin > 10)
            met.ustar=0.126*input.wnd;
            else
            met.ustar=0.055*input.wnd;

    }

            if (met.ustar < 0.025)
            met.ustar=0.025;

    		*/

    return;
}



void BOUNDARY_RESISTANCE(double zzz, double TLF)
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

    int JLAY;

    JLAY = (int)(zzz / delz);

    /*     TLF = solar.prob_beam[JLAY] * prof.sun_tleaf[JLAY] + solar.prob_sh[JLAY] * prof.shd_tleaf[JLAY];  */

    /*     'Difference between leaf and air temperature  */

    deltlf = (TLF - prof.tair_filter[JLAY]);

    T_kelvin=prof.tair_filter[JLAY] + 273.16;

    if(deltlf > 0)
        graf = non_dim.grasshof * deltlf / T_kelvin;
    else
        graf=0;


    nnu_T_P=nnu*(101.3 /input.press_kPa)*pow((T_kelvin/273.16),1.81);


    Re = lleaf * UZ(zzz) / nnu_T_P;

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

        Sh_heat = Res_factor * non_dim.pr33;
        Sh_vapor = Res_factor * non_dim.sc33;
        Sh_CO2 = Res_factor * non_dim.scc33;

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


        Sh_heat = Res_factor * non_dim.pr33;
        Sh_vapor = Res_factor * non_dim.sc33;
        Sh_CO2 = Res_factor * non_dim.scc33;

    }


    //   If there is free convection

    if(graf / (Re * Re) > 1.)
    {

//     Compute Grashof number for free convection

        if (graf < 100000.)
            GR25 = .5 * pow(graf,.25);
        else
            GR25 = .13 * pow(graf,.33);


        Sh_heat = non_dim.pr33 * GR25;
        Sh_vapor = non_dim.sc33 * GR25;
        Sh_CO2 = non_dim.scc33 * GR25;
    }

    // lfddx=lleaf/ddx


    // Correct diffusivities for temperature and pressure

    ddh_T_P=ddh*(101.3/input.press_kPa)*pow((T_kelvin/273.16),1.81);
    ddv_T_P=ddv*(101.3/input.press_kPa)*pow((T_kelvin/273.16),1.81);
    ddc_T_P=ddc*(101.3/input.press_kPa)*pow((T_kelvin/273.16),1.81);

    bound_layer_res.heat = lleaf/(ddh_T_P * Sh_heat);
    bound_layer_res.vapor = lleaf/(ddv_T_P * Sh_vapor);
    bound_layer_res.co2 = lleaf / (ddc_T_P * Sh_CO2);

    if (isinf(bound_layer_res.vapor) == 1)
    {
        bound_layer_res.vapor = 9999;
    }

    return;
}


void RNET()
{
    double ir_shade,q_sun,qsh;
    int JJ, JJP1;
    /*

     Radiation layers go from jtot+1 to 1.  jtot+1 is the top of the
     canopy and level 1 is at the soil surface.

     Energy balance and photosynthesis are performed for vegetation
     between levels and based on the energy incident to that level

    */
    for(JJ = 1; JJ<= jtot; JJ++)
    {

        JJP1=JJ+1;

        //Infrared radiation on leaves


        ir_shade = solar.ir_dn[JJ] + solar.ir_up[JJ];

        ir_shade = ir_shade * ep;  //  this is the absorbed IR
        /*

                 Available energy on leaves for evaporation.
                 Values are average of top and bottom levels of a layer.
                 The index refers to the layer.  So layer 3 is the average
                 of fluxes at level 3 and 4.  Level 1 is soil and level
                 j+1 is the top of the canopy. layer jtot is the top layer

                 Sunlit, shaded values
        */

        q_sun = solar.par_sun[JJ] + solar.nir_sun[JJ] + ir_shade;  // These are the Q values of Rin - Rout + Lin
        qsh = solar.par_shade[JJ] + solar.nir_sh[JJ] + ir_shade;
        solar.rnet_sun[JJ] = q_sun;
        solar.rnet_sh[JJ] = qsh;

    } // next JJ

    return;
}

void SET_SOIL()
{


    /*

      Routines, algorithms and parameters for soil moisture were from

      Campbell, G.S. 1985. Soil physics with basic. Elsevier

      updated to algorithms in Campbell and Norman and derived from Campbell et al 1994 Soil Science


      Need to adjust for clay and organic fractions.. Need to adjust heat capacity and conductivity for peat

    */

    int I;
    int IP1,IM1;
    int n_soil = 9;                 // number of soil layers

    double C1,C2,C3,C4;
    double z_litter = .00;   // depth of litter layer

    double Cp_water,Cp_air, Cp_org, Cp_mineral, Cp_soil, Cp_soil_num;  // heat capacity
    double K_water, K_air,K_org,K_mineral, K_soil, K_soil_num;		// thermal conductivigty
    double wt_water,wt_air,wt_org,wt_mineral;     // weighting factors, Campbell and Norman
    double fw,k_fluid;
    double desdt;

    /*
            Soil water content
    */

    soil.water_content_15cm = input.soil_moisture;     //  measured at 10 cmwater content of soil m3 m-3


    // Water content of litter. Values ranged between 0.02 and 0.126

    soil.water_content_litter = .0;   // assumed constant but needs to vary


    // soil content

    soil.clay_fraction = .3;      //  Clay fraction
    soil.peat_fraction = 0.129;    //  SOM = a C; C = 7.5%, a = 1.72
    soil.pore_fraction = 0.687;    // from alfalfa, 1 minus ratio bulk density 0.83 g cm-3/2.65 g cm-3, density of solids
    soil.mineral_fraction= 0.558;  // from bulk density asssuming density of solids is 2.65

    soil.air_fraction = soil.pore_fraction - soil.water_content_15cm;

    Cp_water= 4180;   // J kg-1 K-1, heat capacity
    Cp_air =  1065;
    Cp_org = 1920;
    Cp_mineral = 870;

    K_mineral= 2.5;  // W m-1 K-1, thermal conductivity
    K_org= 0.8;
    K_water= 0.25;


    // thermal conductivity code from Campbell and Norman

    fw=1./(1+pow((soil.water_content_15cm/0.15),-4));  // terms for Stefan flow as water evaporates in the pores

    desdt=DESDT(input.ta+273.15);
    K_air= 0.024 + 44100*2.42e-5*fw*met.air_density_mole*desdt/met.press_Pa;

    k_fluid=K_air + fw *(K_water-K_air);

    wt_air=2/(3*(1+.2*(K_air/k_fluid-1))) + 1/(3*(1+(1-2*.2)*(K_air/k_fluid -1)));
    wt_water=2/(3*(1+.2*(K_water/k_fluid-1))) + 1/(3*(1+(1-2*.2)*(K_water/k_fluid -1)));
    wt_mineral=2/(3*(1+.2*(K_mineral/k_fluid-1))) + 1/(3*(1+(1-2*.2)*(K_mineral/k_fluid -1)));
    wt_org=2/(3*(1+.2*(K_org/k_fluid-1))) + 1/(3*(1+(1-2*.2)*(K_org/k_fluid -1)));

    Cp_soil_num= ( met.air_density * Cp_air * soil.air_fraction + 1000.000 * Cp_water * soil.water_content_15cm +
                   1300.000 *Cp_org * soil.peat_fraction + 2650.000 * Cp_mineral * soil.mineral_fraction);

    Cp_soil=Cp_soil_num/( met.air_density *  soil.air_fraction + 1000.000 *  soil.water_content_15cm +
                          1300.000  * soil.peat_fraction + 2650.000 * soil.mineral_fraction);

    K_soil_num=soil.mineral_fraction * wt_mineral*K_mineral + soil.air_fraction * wt_air*K_air +
               soil.water_content_15cm * wt_water*K_water + soil.peat_fraction * wt_org*K_mineral;

    K_soil=K_soil_num/(soil.mineral_fraction * wt_mineral + soil.air_fraction * wt_air +
                       soil.water_content_15cm * wt_water + soil.peat_fraction * wt_org);

    soil.dt = 20.;          // Time step in seconds

    soil.mtime = (long int) (3600 /soil.dt);   // time steps per hour


    //  Assign soil layers and initial temperatures
    //  and compute layer heat capacities and conductivities


    for(I=0,IP1=1; I <= n_soil; I++,IP1++)
    {
        IM1=I-1;
        soil.z_soil[IP1] = soil.z_soil[I] + .005 * pow(1.5, (double)IM1);
        soil.T_soil[I] = soil.T_base;

        // assign bulk densities for litter and soil

        if (soil.z_soil[IP1] < z_litter)
            soil.bulk_density[IP1] = .074 ;  // litter
        else
            soil.bulk_density[IP1] = 0.83;   // soil  bulk density for the alfalfa, g cm-3

    }  //  next I

    for(I=1; I<=10; I++)
    {
        // Heat capacity and conductivity.


        // assuming in the numeric code it is bulk density times Cp as values in code have
        // Campbell and Norman have rhos Cs = rhoa Cpa + .. correctly in the book, pg 119.

        // use weighted Cp check units kg m-3 times J kg-1 K-1

        // 2 is if dt =t(i+1)-t(i-1)

        soil.cp_soil[I] = Cp_soil * (soil.z_soil[I + 1] - soil.z_soil[I - 1]) / (2. * soil.dt);


        // adopt new equations from Campbell and Norman and Campbell 1994, after the basic book was written

        //      C1 = .65 - .78 * soil.bulk_density[I] + .6 * soil.bulk_density[I] * soil.bulk_density[I];
        //      C2 = 1.06 * soil.bulk_density[I];  // corrected according to Campbell notes
        //     C3= 1. + 2.6 / sqrt(soil.mineral_fraction);
        //      C4 = .03 + .1 * soil.bulk_density[I] * soil.bulk_density[I];

        // soil conductivity needs debugging ?? or are units for bulk of soil kg m-3
        //soil.k_conductivity_soil[I] = (C1 + C2 * soil.water_content_15cm - (C1 - C4) * exp(-pow((C3 * soil.water_content_15cm), 4.))) / (soil.z_soil[I + 1] - soil.z_soil[I]);

        soil.k_conductivity_soil[I] = K_soil / (soil.z_soil[I + 1] - soil.z_soil[I]);


    }  // NEXT I
    return;
}

void SOIL_ENERGY_BALANCE()
{


    /*

     The soil energy balance model of Campbell has been adapted to
     compute soil energy fluxes and temperature profiles at the soil
     surface.  The model has been converted from BASIC to C.  We
     also use an analytical version of the soil surface energy
     balance to solve for LE, H and G.

     10/30/15 converted soil moisture to psi and then to rh and vpd soil.  yields better soil LE

     10/15 updated the Cp and Kthermal using more recent theory by Campbell and Norman

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

     */

    int J, mm1, I, IP1, IM1;

    double Fst,Gst, le2;

    double soil_par,soil_nir;

    double u_soil,Rh_soil,Rv_soil, kv_soil, kcsoil;

    double T_new_soil[soilsze];

    double a_soil[soilsze], b_soil[soilsze], c_soil[soilsze], d_soil[soilsze];
    double est,dest,d2est,tk2,tk3,tk4,llout,lecoef;
    double acoef,acoeff,bcoef,ccoef;
    double repeat, product;
    double vpdsoil,att,btt,ctt;
    double storage;
    double facstab, stabdel, vpdfact, ea, psi, psiPa, rhsoil;


    int n_soil = 9;         // number of soil layers
    int n_soil_1 = 10;      // m +1


    // kv_soil is the water vapor transfer coef for the soil

    soil.water_content_sfc=0;   // at the soil surface


    // soil surface resistance to water vapor transfer

    // updated and revisited the soil resistance model

    soil.resistance_h2o=SOIL_SFC_RESISTANCE(soil.water_content_15cm);


    //  Compute soilevap as a function of energy balance at the soil
    //  surface. Net incoming short and longwave energy


    // radiation balance at soil in PAR band, W m-2

    soil_par = (solar.beam_flux_par[1] + solar.par_down[1] - solar.par_up[1]) / 4.6;


    // radiation balance at soil in NIR band, W m-2

    soil_nir = solar.beam_flux_nir[1] + solar.nir_dn[1] - solar.nir_up[1];

    // incoming radiation balance at soil, solar and terrestrial, W m-2

    soil.rnet = soil_par + soil_nir + solar.ir_dn[1]*epsoil;  // the Q value of Rin - Rout + Lin

    // initialize T profile


    soil.T_base = input.tsoil;     // 32 cm


    // initialize the soil temperature profile at the lower bound.

    for (I=0; I<= n_soil_1; I++)
        soil.T_soil[I]=soil.T_base;


    // set air temperature over soil with lowest air layer, filtered

    soil.T_air = prof.tair_filter[1];




    // Compute Rh_soil and rv_soil from wind log profile for lowest layer


    u_soil = UZ(delz);              // wind speed one layer above soil


    // Rh_soil = 32.6 / u_soil; // bound. layer rest for heat above soil
    // Rv_soil = 31.7 / u_soil; // bound. layer rest for vapor above soil



// Stability factor from Daamen and Simmonds

    stabdel=5.*9.8*(delz)*(soil.sfc_temperature-soil.T_air)/((soil.T_air+273.)*u_soil*u_soil);

    if (stabdel > 0)
        facstab=pow(1.+stabdel,-0.75);
    else
        facstab=pow(1.+stabdel,-2.);

    if(time_var.count <= 1.)
        facstab=1.;


    if (facstab < .1)
        facstab=.1;

    if (facstab > 5)
        facstab=5;


    Rh_soil = 98.*facstab / u_soil;

    if (Rh_soil > 5000.)
        Rh_soil=5000.;

    if(Rh_soil < 5.)
        Rh_soil=5.;

    Rv_soil=Rh_soil;


    //  kcsoil is the convective transfer coeff for the soil. (W m-2 K-1)

    kcsoil = (cp * met.air_density) / Rh_soil;

    // soil surface conductance to water vapor transfer

    kv_soil = 1. / (Rv_soil + soil.resistance_h2o);


    // Boundary layer conductance at soil surface, W m-2 K-1

    soil.k_conductivity_soil[0] = kcsoil;

    T_new_soil[0] = soil.T_air;
    soil.T_soil[0] = soil.T_air;           // was soil.T_air  but re-setting boundary
    soil.T_soil[n_soil_1] = soil.T_base;
    T_new_soil[n_soil_1] = soil.T_soil[n_soil_1];


    // initialize absolute temperature

    soil.T_Kelvin=soil.T_air+273.16;


    // evaluate latent heat of evaporation at T_soil

    fact.latent = LAMBDA(soil.T_Kelvin);

    // evaluate saturation vapor pressure in the energy balance equation it is est of air layer

    est = ES(soil.T_Kelvin);  //  es(T) f Pa

    ea = prof.rhov_filter[1] * soil.T_Kelvin * 461.89;

    // Vapor pressure deficit, Pa

    if(met.relative_humidity > 0.75)
        vpdfact=1.0;
    else
        vpdfact=1.00;

    // should not use atmospheric humidity. Plus there is a unit problem
    // with est and ea.




    //  vpdsoil = est - ea;      // Pa

    //  if (vpdsoil < 0.)
    //  vpdsoil = 0;


    // Redo this using a pedo transfer function to convert volumetric water content to matric potential
    // then solve for RH;  psi = R Tk/ Mw ln(RH)

    // Slope of the vapor pressure-temperature curve, Pa/C
    //  evaluate as function of Tk

    // fit psi vs grav water content for Sherman Island

    // abs(psi) = -12.56 + -12.49 * log (grav content)

    // bulk density at Sherman Island is 1.11

    psi = -12.56 -12.49 * log(soil.water_content_15cm/soil.bulk_density[1]);  // - MPa

    psiPa = -psi *1000000;  // Pa

    rhsoil= exp(psiPa*1.805e-5/(8.314*soil.T_Kelvin));  // relative humidity

    vpdsoil = (1-rhsoil)*est;   // vpd of the soil

    dest = DESDT(soil.T_Kelvin);


    // Second derivative of the vapor pressure-temperature curve, Pa/C
    // Evaluate as function of Tk


    d2est = DES2DT(soil.T_Kelvin);


    //   Compute products of absolute air temperature

    tk2 = soil.T_Kelvin * soil.T_Kelvin;
    tk3 = tk2 * soil.T_Kelvin;
    tk4 = tk3 * soil.T_Kelvin;

    // Longwave emission at air temperature, W m-2

    llout = epsoil*sigma * tk4;


    // coefficients for latent heat flux density

    lecoef = met.air_density * .622 * fact.latent * kv_soil / met.press_Pa;

    // Weighting factors for solving diff eq.

    Fst = 0.6;
    Gst = 1. - Fst;


    //  solve by looping through the d[]/dt term of the Fourier
    //  heat transfer equation

    for(J = 1; J<= soil.mtime; J++)
    {

        for(I = 1; I<=n_soil; I++) // define coef for each soil layer
        {
            IM1=I-1;
            IP1=I+1;

            c_soil[I] = -soil.k_conductivity_soil[I] * Fst;
            a_soil[IP1] = c_soil[I];
            b_soil[I] = Fst * (soil.k_conductivity_soil[I] + soil.k_conductivity_soil[IM1]) + soil.cp_soil[I];
            d_soil[I] = Gst * soil.k_conductivity_soil[IM1] * soil.T_soil[IM1] + (soil.cp_soil[I] - Gst * (soil.k_conductivity_soil[I] + soil.k_conductivity_soil[IM1])) * soil.T_soil[I] + Gst * soil.k_conductivity_soil[I] * soil.T_soil[IP1];
        }

        d_soil[1] = d_soil[1] + soil.k_conductivity_soil[0] * T_new_soil[0] * Fst + soil.rnet - soil.lout - soil.evap;
        d_soil[n_soil] = d_soil[n_soil] + soil.k_conductivity_soil[n_soil] * Fst * T_new_soil[n_soil_1];

        mm1=n_soil-1;
        for(I = 1; I<= mm1; I++)
        {
            IP1=I+1;
            c_soil[I] = c_soil[I] / b_soil[I];
            d_soil[I] = d_soil[I] / b_soil[I];
            b_soil[IP1] = b_soil[IP1] - a_soil[IP1] * c_soil[I];
            d_soil[IP1] = d_soil[IP1] - a_soil[IP1] * d_soil[I];
        }

        T_new_soil[n_soil] = d_soil[n_soil] / b_soil[n_soil];

        for(I = mm1; I>= 1; I--)
            T_new_soil[I] = d_soil[I] - c_soil[I] * T_new_soil[I + 1];


        // soil temperature at 15 cm

        soil.T_15cm=T_new_soil[7];


        // compute soil conductive heat flux density, W m-2

        soil.gsoil = soil.k_conductivity_soil[1] * (T_new_soil[1] - T_new_soil[2]);
        storage = soil.cp_soil[1] * (T_new_soil[1] - soil.T_soil[1]);
        soil.gsoil += storage;


        // test if gsoil is in bounds??

        if(soil.gsoil < -500. || soil.gsoil > 500.)
            soil.gsoil=0;



        // The quadratic coefficients for the solution to

        //   a LE^2 + b LE +c =0


        // should be a function of Q, not Rnet

        repeat = kcsoil + 4. * epsoil*sigma * tk3;

        acoeff = lecoef * d2est / (2. * repeat);
        acoef = acoeff;

        bcoef = -(repeat) - lecoef * dest + acoeff * (-2.*soil.rnet +2* llout+2*soil.gsoil);

        ccoef = (repeat) * lecoef * vpdsoil + lecoef * dest * (soil.rnet - llout-soil.gsoil) +
                acoeff * ((soil.rnet * soil.rnet) + llout * llout + soil.gsoil*soil.gsoil -
                          2.*soil.rnet * llout - 2.*soil.rnet*soil.gsoil+2*soil.gsoil*llout);


// LE1 = (-BCOEF + (BCOEF ^ 2 - 4 * ACOEF * CCOEF) ^ .5) / (2 * ACOEF)

        product = bcoef * bcoef - 4 * acoef * ccoef;

// LE2 = (-BCOEF - (BCOEF * BCOEF - 4 * acoef * CCOEF) ^ .5) / (2 * acoef)

        if (product >= 0)
            le2= (-bcoef - pow(product,.5)) / (2 * acoef);
        else
            le2 = 0;

        // latent energy flux density over soil, W m-2

        soil.evap=le2;

        // solve for Ts using quadratic solution

        att = 6 * epsoil*sigma * tk2 + d2est * lecoef / 2;

        btt = 4 * epsoil*sigma * tk3 + kcsoil + lecoef * dest;

        ctt = -soil.rnet + llout+soil.gsoil + lecoef * vpdsoil;


        // IF (BTLF * BTLF - 4 * ATLF * CTLF) >= 0 THEN /

        product = btt * btt - 4. * att * ctt;


        // T_sfc_K = TAA +
        //     (-BTLF + SQR(BTLF * BTLF - 4 * ATLF * CTLF)) / (2 * ATLF)

        if (product >= 0.)
            soil.sfc_temperature = soil.T_air + (-btt + sqrt(product)) / (2. * att);
        else
            soil.sfc_temperature=soil.T_air;

        // Soil surface temperature, K

        soil.T_Kelvin=soil.sfc_temperature+273.16;

        // IR emissive flux density from soil, W m-2

        soil.lout = epsoil*sigma*pow(soil.T_Kelvin,4);

        // Sensible heat flux density over soil, W m-2

        soil.heat = kcsoil * (soil.T_soil[1]- soil.T_air);



        /*

          printf(" \n");
          printf("GSOIL  HSOIL  LESOIL  solar.rnsoil  lout  TSOIL\n");
          printf("%5.1f  %5.1f  %5.1f  %5.1f\n", soil.gsoil, soil.heat, soil.evap, solar.rnsoil - soil.lout, loutsoil, soil.sfc_temperature);
          printf("\n");


          printf("  Z   TEMP\n");

        */



        // compute new soil temperature profile

        for(I=0; I<=n_soil_1; I++)
        {

//   printf("%5.2f    %5.2f\n",z_soil[I],T_new_soil[I]);

            // check for extremes??

            if (T_new_soil[I] < -10. || T_new_soil[I] > 70.)
                T_new_soil[I]=input.ta;

            soil.T_soil[I]=T_new_soil[I];

        }  // next i


    }             // next J
    return;
}



void STOMATA()
{

    /* ----------------------------------------------

            SUBROUTINE STOMATA

            First guess of rstom to run the energy balance model.
                    It is later updated with the Ball-Berry model.
    -------------------------------------------------
    */

    int JJ;

    double rsfact;

    rsfact=brs*rsm;

    for(JJ = 1; JJ <=jtot; JJ++)
    {


        // compute stomatal conductance
        // based on radiation on sunlit and shaded leaves
        //  m/s.

        //  PAR in units of W m-2

        if(time_var.lai == pai)
        {
            prof.sun_rs[JJ] = sfc_res.rcuticle;
            prof.shd_rs[JJ] = sfc_res.rcuticle;
        }
        else
        {
            if(solar.par_sun[JJ] > 5.0)
                prof.sun_rs[JJ] = rsm + (rsfact) / solar.par_sun[JJ];
            else
                prof.sun_rs[JJ] = sfc_res.rcuticle;

            if(solar.par_shade[JJ] > 5.0)
                prof.shd_rs[JJ] = rsm + (rsfact) / solar.par_shade[JJ];
            else
                prof.shd_rs[JJ] = sfc_res.rcuticle;
        }
    }
    return;
}





double UZ (double zzz)
{
    double y,zh, zh2, zh3, y1, uh;
    /*
             U(Z) inside the canopy during the day is about 1.09 u*
             This simple parameterization is derived from turbulence
             data measured in the WBW forest by Baldocchi and Meyers, 1988.
    */

    zh=zzz/ht;

    // use Cionco exponential function

    uh=input.wnd*log((0.55-0.33)/0.055)/log((2.8-.333)/0.055);

    y=uh*exp(-2.5*(1-zh));

    return y;
}


void ENERGY_BALANCE_AMPHI (double qrad, double *tsfckpt, double taa, double rhovva, double rvsfc,
                           double stomsfc, double *lept, double *H_leafpt, double *lout_leafpt)
{


    /*
            ENERGY BALANCE COMPUTATION for Amphistomatous leaves

            A revised version of the quadratic solution to the leaf energy balance relationship is used.

            Paw U, KT. 1987. J. Thermal Biology. 3: 227-233


             H is sensible heat flux density on the basis of both sides of a leaf
             J m-2 s-1 (W m-2).  Note KC includes a factor of 2 here for heat flux
             because it occurs from both sides of a leaf.
    */


    double est, ea, tkta, le2;
    double tk2, tk3, tk4;
    double dest, d2est;
    double lecoef, hcoef, hcoef2, repeat, acoeff, acoef;
    double bcoef, ccoef, product;
    double atlf, btlf, ctlf,vpd_leaf,llout;
    double ke;



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


    dest = DESDT(tkta);


    // Second derivative of the vapor pressure-temperature curve, Pa/C
    // Evaluate as function of Tk


    d2est = DES2DT(tkta);


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

    lecoef = met.air_density * .622 * fact.latent * ke / met.press_Pa;


    // Coefficients for sensible heat flux


    hcoef = met.air_density*cp/bound_layer_res.heat;
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

    *lept=le2;  // need to pass pointer out of subroutine


    // solve for Ts using quadratic solution


    // coefficients to the quadratic solution

    atlf = epsigma12 * tk2 + d2est * lecoef / 2.;

    btlf = epsigma8 * tk3 + hcoef2 + lecoef * dest;

    ctlf = -qrad + 2 * llout + lecoef * vpd_leaf;


    // IF (BTLF * BTLF - 4 * ATLF * CTLF) >= 0 THEN

    product = btlf * btlf - 4 * atlf * ctlf;


    // T_sfc_K = TAA + (-BTLF + SQR(BTLF * BTLF - 4 * ATLF * CTLF)) / (2 * ATLF)

    if (product >= 0)
        *tsfckpt = tkta + (-btlf + sqrt(product)) / (2 * atlf);
    else
        *tsfckpt=tkta;


    if(*tsfckpt < -230 || *tsfckpt > 325)
        *tsfckpt=tkta;

    // long wave emission of energy

    *lout_leafpt =epsigma2*pow(*tsfckpt,4);

    // H is sensible heat flux

    *H_leafpt = hcoef2 * (*tsfckpt- tkta);


    return;
}


void PHOTOSYNTHESIS_AMPHI(double Iphoton,double *rstompt, double zzz,double cca,double tlk,
                          double *leleaf, double *A_mgpt, double *resppt, double *cipnt,
                          double *wjpnt, double *wcpnt)
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

    bc = kct * (1.0 + o2 / ko);

    if(Iphoton < 1)
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

    rb_mole = bound_layer_res.co2 * tlk * (met.pstat273);

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

    rh_leaf  = SFC_VPD(tlk, zzz, leleaf);

    k_rh = rh_leaf * sfc_res.kballstr;  // combine product of rh and K ball-berry

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
        cs=input.co2air;

    /*
            Stomatal conductance for water vapor


    		alfalfa is amphistomatous...be careful on where the factor of two is applied
    		just did on LE on energy balance..dont want to double count

    		this version should be for an amphistomatous leaf since A is considered on both sides

    */

    gs_leaf_mole = (sfc_res.kballstr * rh_leaf * aphoto / cs) + bprime;


    // convert Gs from vapor to CO2 diffusion coefficient


    gs_co2 = gs_leaf_mole / 1.6;

    /*
            stomatal conductance is mol m-2 s-1
            convert back to resistance (s/m) for energy balance routine
    */

    gs_m_s = gs_leaf_mole * tlk * met.pstat273;

    // need point to pass rstom out of subroutine

    *rstompt = 1.0 / gs_m_s;


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

    gs_m_s = gs_leaf_mole * tlk * (met.pstat273);

    // need pointer to pass rstom out of subroutine as a pointer

    *rstompt = 1.0 / gs_m_s;


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
    *A_mgpt = aphoto * .044;

    *resppt=rd;

    *cipnt=ci;

    *wcpnt=wc;

    *wjpnt=wj;

    /*

         printf(" cs       ci      gs_leaf_mole      CA     ci/CA  APS  root1  root2  root3\n");
         printf(" %5.1f   %5.1f   %6.3f    %5.1f %6.3f  %6.3f %6.3f %6.3f  %6.3f\n", cs, ci, gs_leaf_mole, cca, ci / cca,aphoto,root1, root2, root3 );

    */

    return;
}

