#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
// #include <io.h>
#include <stdlib.h>
#include <time.h>

#define PI  3.14159        // Pi
// #define sze 41             // number of canopy layers plus one
// #define sze3 121           // number of atmospheric layers plus one

// constants for the Random Number Generator
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876

namespace py = pybind11;

/* =================================================================

4-21-2004

	Dennis Baldocchi
	Ecosystem Science Division
	Department of Environmental Science, Policy and Management
	151 Hilgard Hall
	University of California, Berkeley
	Berkeley, CA
	Baldocchi@nature.berkeley.edu

-------------------------------------------
         PROGRAM     DispersionMatrix_V2_oak.c

       This version is Compiled on Microsoft C++  

       This program computes the dispersion matrix, according to
       Raupach (1989).  This model is coupled later to CANOAK
       the oak photosynthesis model, to compute CO2 profiles.

       A number of particles is released at 40 levels
       in a horizontally, homogeneous canopy.

       The vertical velocity w is computed with a Markov sequence.
       The algorithm after the approach of Thomson (1987)
       is used.

              dw(t) = a(w,z) dt + b(w,z) du

       where du is a random number with random increment with
       mean of zero, a variance equal one and a Gaussian probability distribution.

       The random number is drawn from the rejection
       technique described in Spanier and Gelbard (1969), Monte
       Carlo Principles and Neutron Transport.

       Tests show that this technique gives a mean of zero
       and variance of one--an improvement over earlier
       versions of this model.  Tests also show that it is
       faster than the central tendency method.  But we need to integrate between +/-
       5 standard deviations to get best results, rather than 3, as in V1.

       A uniform number of particles is released from each layer
       to assess the dispersion matrix.  Then linked to CANOAK
       the actual source-sink strength is computed.  We must also adjust the concentrations
       of the dispersion matrix, scaled to height defined at variable SZE, to the reference height in
       the respective field study.

       This model computes the random flight field for a 1-dimensional
       canopy and continuous release.

       Dispersion is only in the vertical and the canopy is
       assumed to be horizontally homogeneous.

      
       The system studied here is analogous to a
       volume averaged point source, stacked one atop another.
       Concentrations are computed on the principle of superposition.

       Since the canopy is horizontally homogeneous we can assume
       an infinite extent in the y direction, so the concentration
       does not vary in y and the source can be expressed in terms of
       unit y.

      
       In this program the particles released for a prescribe
       time duration--TIMEMAX [s]. Tests in the past show that TIMEMAX is different
       for tall forests and short crops.

       The simulation volume extends up to 3 * h, the crop height.
       This altitude is divided in 40 layers (DZ ihigh).

	 For atmospheric stability effects, different Dispersion matrices are produced for different z/L values    

       


***********************
	 Debug Notes

   7-23-2004. Mean w is about 0.3.  Is this correct??? Need to double check

    4-23-2004. Sustituted random number generator using one from Numerical Recipes
	It does note repeat until 2^31 calls.

    12-12-2002. Alex Knohl's turbulencen data shows that both sigmaw_h and sigmaw_z0 vary with z/L. At present
	I only vary sigmaw_h with z/L

	12-3-2002
	To avoid confusion simply define sigmah as the mult factor of ustar, eg 1.25 and multiply
	by ustar where appropriate

  Running with 1 million particles. Alex Knohl shows smoother profiles with such high counts.
	
	 12-2-2002
  Run with corrected Tl and sigma w profiles. Reference ustar is now 1 m/s. 100,000 particles 
  to get smoother dispersion matrix

  11/2002

  Using Massman and Weil parameterizations for Tl in the canopy. Working with Alex Knohl, found minor errors in
  Tl with how sigw/u* is applied. Needed to make some changes.


5/21/2001.

	Changed the random number range to plus minus five. The random number is now
	computed with a mean of zero and a variance of 1.00. Before the variance was about
	0.96.

	Working with old code on lap top. Need to bring back linear profiles of sigma w and new
	Massman algorithm of Tl. Dave Bowling was finding bizarre kinks in conc profiles with the
	non-linear sigma w profiles. Converting the code to a structure based style.



7/28/99

	Got note for Cesscati finding error in Do loop for random number that needs fixing. We find that
	we were missing a DO.  Nancy also recommends replacing RNDDIV with RAND_MAX, so the random
	number generator is not compiler specific.

	seeding srand with a clock call.

	Having problems with non-linear sigma w parameterization. Substituting the linear terms
	from movpond for test.

	Now using sigw=a exp(b z/h) function

*/




// Declare subroutines


	double SIGMA (double Z, double HH, double ustar);  // vertical profile of standard deviation of w
	double DW2DZ (double Z, double HH, double ustar);  // vertical profile of variance of w
	double TL (double Z, double HH, double DD, double ustar);     // vertical profile of Lagrangian Time scale
	void FILEOUT (double HH, double DD, double ustar);          // output information and data
	double RANDGEN ();        // random number generator
	double MIN(double x, double y);  // minimum of two numbers
	int Time_sec();					 // time routine
	double RAN0(long *a);         // random number generator, numerical recipes


// Declare structures

		struct random_numbers
		{
			double low;     // lower limit of random number generator
			double high;    // upper limit of random number generator
			double delta;   // difference between high and low
			// double random;  // random number
			double random;  // random number
			long seed;
		} random_int;
		// } random;

		struct domain_description
        {
			long nlevel;			// number of canopy layers
			long upper_boundary;    // upper boundary, number of canopy heights
			// long nlev[sze];
			
			double delta_z;        // distance between layers, m
		

        }domain;


		struct parcel_movement
        {
			double w;      // vertical velocity, w
			double term1;  // first term of Thomson Eq
			double term2;  // second term of Thomson Eq
			double term3;  // third term of Thomson Eq
			double z;      // vertical position, z
			double std_w;  // std deviation w
			double var_w;  // variance of w 
			double delta_t; // time step
			double sum_w;   // sum vertical velocities for mean
			double sum_var_rand;  // sum variance of random number
			double sum_random;    // sum random numbers
	
			// double wwmean[sze3];
    		// double consum[sze3];
			// double conc[sze3];
			// double DIJ[sze3][sze];
	

			long move;     // number of movement steps
			// long partadd[sze3];
			long sumn;
		

		}parcel;


		struct turbulence_statistics
		{
			double sigma_h;   // sigma w at canopy height and above
			double sigma_zo;   // sigma w at the ground
			double del_sigma;  // derivation of sigma w with height
			double laglen;        // Lagrangian length scale
			double Tlh;        // turbluent time scale
			double sigma_sur;
		}turb;

// Declare integers and floats 


long IZF; 
time_t ltime, start, finish;


// Declare Constants
// const double npart = 5000.;             // Number of parcels 1000000
// const double timemax = 5000.0;           // time of parcel run
// const double ustar = 1.00;                //  friction velocity (m s-1) old u* was 0.405
// const double HH = 24.;                    //  canopy height (m) 
// const double DD = 18.;					// zero plane displacement	

// File pointers for opening files


        FILE *fptr1;
// main()
void DISPERSION_MATRIX(
    int sze, int sze3, double npart, double timemax, double ustar, double HH, double DD, const char dij_fname[],
    py::array_t<double, py::array::c_style> DIJ_np
)
{

	// declare variables

	int I, part, ihigh, ilevel, nn, ihigh1;

	long IT;

	double timescale;
	double xmod,zmod;

    double nlev[sze];
    double wwmean[sze3];
    double consum[sze3];
    double conc[sze3];
    // double DIJ[sze3][sze];
    long partadd[sze3];

    auto DIJ = DIJ_np.mutable_unchecked<2>();

        //   Thomson dispersion matrix  
        fptr1=fopen(dij_fname,"w");
	    //    fptr1=fopen("c:\\wbw_mod\\DIJ5000O.oak","w");  //neutral
	    //    fptr1=fopen("./DIJ2.txt","w");  //neutral
            //  fptr1=fopen("c:\\wbw_mod\\DIJ5000O._C","w");  //neutral
			// fptr1=fopen("c:\\wbw_mod\\DIJ5000O._1","w");  //neutral u*=0.1 m s-1
			// fptr1=fopen("c:\\wbw_mod\\DIJ5000O.250","w"); // z/L = -0.25
			// fptr1=fopen("c:\\wbw_mod\\DIJ5000O.500","w"); // z/L = -0.50
			// fptr1=fopen("c:\\wbw_mod\\DIJ5000O.100","w"); // z/L = -1.00
		    // fptr1=fopen("c:\\wbw_mod\\DIJ5000O.200","w"); // z/L = -2.00
	        // fptr1=fopen("c:\\wbw_mod\\DIJ5000O.300","w"); // z/L = -3.00
        
		random_int.low=-5.;
		random_int.high=5.;
		random_int.delta=random_int.high-random_int.low;

		// domain.nlevel=40; 
		domain.nlevel=sze-1; 



		// identify variables 

		domain.delta_z = HH / domain.nlevel;     // distance between vertical layers 

		// domain.upper_boundary = 3;           // upper bound,three times canopy height, 72 m 
		// ihigh = domain.upper_boundary * domain.nlevel;    // number of vertical layers 
        ihigh = sze3-1;


		turb.sigma_h = 1.25;       //  sigma w > HH  */

		// Kaimal and Finnigan

		// sigma_w/u* = 1.25(1+ 3 abs(z/L))^.333  for z/L < 0

		// sigma_w/u* = 1.25 (1 + 0.2 z/L) for z/L 0 to 1

	//  turb.sigma_h= 1.506 ;            // z/L = -0.25
	//  turb.sigma_h = 1.696 ;          // z/L = -0.50
	//  turb.sigma_h = 1.984 ;         // z/L = -1.00
	//   turb.sigma_h = 2.391 ;        // z/L = -2.00
	//  turb.sigma_h = 2.692 ;       // z/L = -3.00

		// turb.sigma_h = 1.35;      // z/L = 0.5
		// turb.sigma_h= 1.5;        // z/L =1.0 


		turb.sigma_zo = 0.1496;         // sigmaw over u* at z equal zero for exponential profile

		turb.sigma_sur= turb.sigma_zo * turb.sigma_h * ustar;  // sigma w at zero for linear profile


		turb.del_sigma = (turb.sigma_h * ustar - turb.sigma_sur) / HH; // difference in sigma w


	



// seed random number with time 

   srand(Time_sec());
   random_int.seed=(long)rand();

/*
***************************************************************
       Time step length (* TL) and Lagrangian Length scale
****************************************************************
*/



parcel.delta_t = .1 * TL(HH, HH, DD, ustar);                         // time step */
turb.laglen = turb.sigma_h * ustar * TL(HH, HH, DD, ustar);          // Lagrangian length scale */


parcel.sumn = 0;

for (I = 1; I <= domain.nlevel; I++)
{
        /*
        ******************************************************
        number of particles per layer, redistribute npart #
        with height according to the relative source strength
        *****************************************************
        */

        // domain.nlev[I] =(int) npart / domain.nlevel;
        // parcel.sumn += domain.nlev[I];
        nlev[I] =(int) npart / domain.nlevel;
        parcel.sumn += nlev[I];
}

/*
'       nn is the number of time increments that the particle travels
'       TIMEMAX = t [s]
'       Actual time of particle trave is TIMEMAX*TL.
'       Number of travel steps is t/dt.
*/

        nn = (int)(timemax / parcel.delta_t);

/*
'    ****************************************************************
'       Start the release of particles:
'    ****************************************************************
*/
        parcel.sum_random = 0;
        parcel.sum_w = 0.;
        parcel.sum_var_rand = 0;
        parcel.move = 0;

        IT = 1;                           /* particle counter */


       /*
        assume values of a Gaussian distribution
        random numbers with a mean of zero and a variance of one.
        */

            
          ihigh1 =ihigh + 1;


      

                timescale = TL(HH, HH, DD, ustar);

/*
        Release of particles is carried out separately for each layer,
        starting with the lowest level.
*/

  for (ilevel = 1; ilevel <= domain.nlevel;ilevel++)
  {


/*

    ****************************************************************
       1-D case. Particles are released from each level z.  We have
       a continuous source and a horizontally homogeneous canopy.
    ****************************************************************
*/


    for(I = 1; I <=ihigh; I++)
    consum[I] = 0;
    // parcel.consum[I] = 0;



/*
****************************************************************
      at each level NLEV(LEVEL) particles are released
****************************************************************
*/
        // for (part = 1; part <= domain.nlev[ilevel]; part++)
        for (part = 1; part <= nlev[ilevel]; part++)
        {

        IT++;
        
            
	xmod=(double) IT;
	zmod=fmod(xmod,100);

	parcel.z = (double)ilevel * domain.delta_z;   // initial height at the level


if (zmod==0)
{
printf(" ilevel %6i Particle  %7li  height %f time steps %i \n",ilevel, IT, parcel.z, I) ;
}

// the initial vertical velocity

       

        random_int.random=RANDGEN();


//       vertical velocity, WW


        parcel.w = SIGMA(parcel.z, HH, ustar) * random_int.random;

        /*
        number of particle movements
        */

        parcel.move += 1;

        /* compute mean and variance of w and random number */

        parcel.sum_random += random_int.random;
        parcel.sum_w += parcel.w;
        // parcel.wwmean[ilevel] += parcel.w;
        wwmean[ilevel] += parcel.w;
        parcel.sum_var_rand += random_int.random*random_int.random;



   // The particle starts its run




/*         for (I=1; I <= nn;I++) */

        IZF =(int) MIN((int)(parcel.z / domain.delta_z) + 1, ihigh1);

        I=0;

        do
         {

       // Compute the vertical position and reflect z if it is zero
	   // Need to reflect also at top in future, but need deeper domain
	   
          parcel.z += parcel.w * parcel.delta_t;

          if(parcel.z <= 0)  // reflect particle if at ground 
          {
          parcel.w = -parcel.w;
          parcel.z = -parcel.z;
          }

          IZF = (int)MIN((int)(parcel.z / domain.delta_z) + 1, ihigh1);

/*

    Compute the concentration of material in the controlled volume.

    Here we use the algorithm of Raupach (1989).  The ensemble average
    concentration considers the fact that we have an extensive,
    horizontally homogeneous and continuous source.  Information from
    every step from t=0 to t=T is used. It is identical to releasing a plane
    source with length x or ut, as does Wilson et al.
*/
       
        // parcel.consum[IZF] += parcel.delta_t;
        consum[IZF] += parcel.delta_t;

/*       Compute the new vertical velocity.
       Introduce the bias velocity for the case of inhomogeneous
       turbulence.  This is needed to prevent the false accumulation
       of material near the ground that would otherwise occur as
       fast air is brought downward, yet slower air at lower levels
       is less apt to leave. (see Wilson et al. Thompson etc.)
*/

                random_int.random= RANDGEN();

                /*
                wnew = -wold dt/Tl) + 1/2 dvarw/dz (1+w^2/varw)+
                 (2 varw dt/Tl)du
                */

                timescale = TL(parcel.z, HH, DD, ustar);
                parcel.std_w = SIGMA(parcel.z, HH, ustar);
                parcel.var_w = parcel.std_w * parcel.std_w;
                parcel.term1 = -parcel.w * parcel.delta_t / timescale;
                parcel.term2 = .5 * DW2DZ(parcel.z, HH, ustar) * (1. + (parcel.w * parcel.w) / parcel.var_w) * parcel.delta_t;
                parcel.term3 = pow((2. * parcel.var_w * parcel.delta_t / timescale),.5) * random_int.random;

				         			
                parcel.w += parcel.term1 + parcel.term2 + parcel.term3;


/*    ****************************************************************
'       STATISTICAL CHECK OF RANDOM NUMBER AND MEAN VERTICAL VELOCITY
'    ****************************************************************
*/
                /*
                number of occurences at height IZF and its
                mean vertical velocity
                */

                // parcel.wwmean[IZF] += parcel.w;
                wwmean[IZF] += parcel.w;

                parcel.move += 1;
                parcel.sum_random += random_int.random;
                parcel.sum_w += parcel.w;
                parcel.sum_var_rand += random_int.random*random_int.random;

                I++;

               
                } while (I <=nn && IZF <= ihigh); /*  NEXT I  Particle-position  and end of while */

            //    parcel.partadd[IZF] += 1;
               partadd[IZF] += 1;

        }  //  next particle 


    /*
    Introduce computation of concentration at each level.
    Use super-position principle to compute canopy
    concentration profile.
    */

        for (I = 1; I <= ihigh; I++)
        // parcel.conc[I] = parcel.consum[I];
        conc[I] = consum[I];

       // Compute the dispersion matrix then reset concentrations 
		
		for (I=1; I<= ihigh;I++)
        // parcel.DIJ[I][ilevel] = (parcel.conc[I] - parcel.conc[ihigh]) / (domain.delta_z * domain.nlev[ilevel]);
        // DIJ[I][ilevel] = (conc[I] - conc[ihigh]) / (domain.delta_z * nlev[ilevel]);
        DIJ(I,ilevel) = (conc[I] - conc[ihigh]) / (domain.delta_z * nlev[ilevel]);
	


		for (I = 1; I<=ihigh ;I++)
		{
		// printf(" %6.2f   %6i\n", parcel.DIJ[I][ilevel], I);
		// fprintf(fptr1,"%7.2f %6i \n", parcel.DIJ[I][ilevel],I);
		// printf(" %6.2f   %6i\n", DIJ[I][ilevel], I);
		// fprintf(fptr1,"%7.2f %6i \n", DIJ[I][ilevel],I);
		// printf(" %6.2f   %6i\n", DIJ(I,ilevel), I);
		fprintf(fptr1,"%7.2f %6i \n", DIJ(I,ilevel),I);
		}


}	// Next ilevel


/*
'    ****************************************************************
'       Statistical check of random number: VARR=variance (should = 1),
'       SUMN = mean (should = 0), sumw# = mean vertical velocity
'       (should = 0)
'    ****************************************************************
*/

	

		// FILEOUT(HH, DD, ustar);

} // end of main 


double DW2DZ (double Z, double HH, double ustar)
{

	double y, dsigw2dz;
/*
    ****************************************************************
       COMPUTES ds2/dz FOR GIVEN s(z)
    ****************************************************************
*/

	if (Z < HH)
{

	  //y = 2.*Z*turb.del_sigma*turb.del_sigma*ustar*ustar+2.*turb.sigma_zo*turb.del_sigma*ustar; 
	 
	  // linear model */


		// first compute derivative of sigw^2/u*^2, need to convert to d(s2) /dz 

		// sigw=turb.sigma_zo*exp(2.132 *Z/HH);

		// sigw2=s(0)^2 * exp(4.264*Z/HH);

       // dsigw2dz=(s(0)^2/HH) * 4.264 * exp(4.264*Z/HH);
       
		// y=sigw2*ustar*ustar;

	 // first compute derivative of sigw^2/u*^2

       dsigw2dz=(turb.sigma_zo*turb.sigma_zo*4.264/HH)*exp(4.264*Z/HH);

	// need to convert to ds2/dz so multiplication by u* times u* is needed

        y=dsigw2dz*ustar*ustar;


	}
	else
	y = 0.0;
	return y;
}


void FILEOUT(double HH, double DD, double ustar)
{
	int I;

		parcel.sum_var_rand = (parcel.sum_var_rand - (pow(parcel.sum_random,2.0) / parcel.move)) / (parcel.move - 1.);

	
		parcel.sum_w = parcel.sum_w / (float) parcel.move;
		parcel.sum_random = parcel.sum_random / (float)parcel.move;
		
		
		fprintf(fptr1,"var random mean random \n");
		fprintf(fptr1,"%f , %ld \n",parcel.sum_var_rand,parcel.sumn);

		fprintf(fptr1, " z/h  Tl  sig w \n");

		for(I=1;I <=40; I++)
		fprintf(fptr1,"%5i,  %7.4f , %7.4f \n",I,TL((float)I, HH, DD, ustar),SIGMA((float)I, HH, ustar));
		
		printf(" \n");
		printf(" %6li \n", parcel.move);
		printf("mean r:  %f \n ", parcel.sum_random);
		printf("var. r:  %f \n ", parcel.sum_var_rand);
		printf("mean w:  %f \n ", parcel.sum_w);

return;
}

double RANDGEN ()
{
double y, f_fnc, oper, random_1, random_2, random_sqrd;

		/*
		Produces a Random number with a mean of zero and variance of one and
		a range of data between -5 and 5.. 

		This program uses the Rejection technique.
		It can be adapted to compute random numbers for skewed and kurtotic
		distributions with the Gram Charlier distribution. 
		
		This was originally done, but it was subsequently found that using skewed
		distributions is ill posed and produces no new information. Plus our
		more recent analysis of du/dt of turbulence data, that is non Gaussian, produces
		a Gaussian distribution of du/dt. Hence, non Gaussian distributions seem not
		needed in normal practice for canopy turbulence.
		
        The Rejection Methods is based on Spanier and Gelbard (1969)
        Monte Carlo Principles and Neutron Transport Theory
		
      
        For the Gaussian Routine 
       
        Call two random numbers.  If RAND #2 is less than
		PDF=F(RAND #1) then accept RAND #1. else repeat

        note: in C rand() returns number between 0 and
        32767.  The routine from Basic relied on a random
        number between 0 and 1

        Replaced rand() with flat number generator from 0 to 1
        */

          do
        {
        //random_1 = (double)rand()/RAND_MAX;   // RAND_MAX is defined in stdlib
        //random_2 = (double)rand()/RAND_MAX;

			random_1=RAN0(&random_int.seed);
			random_2=RAN0(&random_int.seed);

        // Value of x between high and low limits
		

        y = random_int.low + random_1 * random_int.delta;

        random_sqrd = y * y;

        /*
        PDF OF RAND

        Compute function with a Gausian distribution with mean of zero 
		and std of 1. But note the function is adjustable
		
        FFNC=EXP(-(RAND-MEAN)^2/(2*SIGRAN*SIGRAN))
        */

        oper = -random_sqrd / 2.0;
        f_fnc = exp(oper);
        }while (random_2 >= f_fnc);
        
        return y;
        }

double SIGMA (double Z, double HH, double ustar)
{
double y, sigw;
/*
'    ****************************************************************
'       This function gives the s(w) value for height z
'       Use linear decrease in sigma with z/h, as Wilson et al
'       show for a corn canopy.
'    ****************************************************************
'
'   DELSIG=(SIGMAH-SIGMASUR)/HH
'
*/

if (Z < HH)
{
 
	// y = turb.sigma_zo+Z*turb.del_sigma; //linear model */


// exponential model, computed as function of sigw/u* and z/h
// need to convert to sigma w so final multiplication by u* is needed

     
		sigw=turb.sigma_zo*exp(2.132 *Z/HH);

        y=sigw*ustar;  // multiply by ustar to compute sigma w


	}
	else
		y = turb.sigma_h * ustar; 

	return y;
}

double TL (double Z, double HH, double DD, double ustar)
{
double y, A1;

/*
    ****************************************************************
       This function gives the variation of T(L) with height
    ****************************************************************


        Adopt scaling values of Massman and Weil that adjust Tl profiles for different
        canopy strcutures

  u* Tl/h = A1 * (z-d)/h;  z > h

  u* Tl/h = A1 (1-d/h) gamma_3/ (sigmaw(z)/u*); z <= h

  A1 = A2 (sigmaw(z)/u* gamma_3)^1/2 (1-d/h)^-1/2

  gamma_3 = sigmaw(h)/u*

*/

                // factor of 2 comes from (1 - d/h)^-1/2; (1-0.75)^-1/2  
                


            A1=0.6*sqrt(SIGMA(Z, HH, ustar)/ustar*turb.sigma_h)* 2.;

		
        if (Z <= HH)
                {

        
        // The factor 0.25 = 1 - d/h = 1 - 0.75
        
                y = A1* HH*0.25*turb.sigma_h/SIGMA(Z, HH, ustar);
        }
                else
                {
                
				 // u* Tl/h = A1 * (z-d)/h;  z > h	

                y = A1 * (Z-DD)/ustar;
        }
            
        return y;
        }

double MIN (double z, double x)
{
		double y;

		if(z < x)
		y = z;
		else
		y = x;
		return y;
}

int Time_sec() 
{
		time_t t;

		t = time(NULL);
		// printf("Time_sec: %d\n", t); 
		return (*gmtime(&t)).tm_sec;
}

double RAN0(long *idum)
{

	// definitions for random number generator, Numerical Recipes



	// random number generator from Numerical Recipes in C
	// Press et al

	// flat distribution between 0 and 1
	// the period of repeat is ~2^31




	long k;
	double ans;

	*idum ^= MASK;
	k=(*idum)/IQ;
	*idum=IA*(*idum-k*IQ)-IR*k;
	if (*idum < 0) *idum += IM;
	ans=AM*(*idum);
	*idum ^= MASK;
	return ans;
}
 

PYBIND11_MODULE(dispersion, m) {
    m.doc() = "pybind11 plugin CONC"; // optional module docstring
    m.def("disp_mx", &DISPERSION_MATRIX, "Subroutine to calculate the dispersion_matrix",
    py::arg("sze"), py::arg("sze3"), py::arg("npart"), py::arg("timemax"), 
    py::arg("ustar"), py::arg("HH"), py::arg("DD"), py::arg("dij_fname"), py::arg("dij_np"));
}