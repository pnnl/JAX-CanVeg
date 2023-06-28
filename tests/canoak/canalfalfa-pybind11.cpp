#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
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


namespace py = pybind11;

       

// int add(int i, int j) {
//     return i + j;
// }

// PYBIND11_MODULE(example, m) {
//     m.doc() = "pybind11 example plugin"; // optional module docstring

//     m.def("add", &add, "A function that adds two numbers");
// }

void CONC(
        double cref, double soilflux, double factor,
        int sze3, int jtot, int jtot3, double met_zl, double delz, int izref,
        double ustar_ref, double ustar, 
        py::array_t<double, py::array::c_style> source_np,
        py::array_t<double, py::array::c_style> cncc_np,
        py::array_t<double, py::array::c_style> dispersion_np
        // double *source, double *cncc, 
        // double ustar_ref, double ustar, double dispersion[jtot3][jtot]  
        // double ustar_ref, double ustar, double dispersion[10][10]  
        // double ustar_ref, double ustar, double **dispersion
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
    m.doc() = "pybind11 plugin CONC"; // optional module docstring

    m.def("conc", &CONC, "Subroutine to compute scalar concentrations from source estimates and the Lagrangian dispersion matrix",
    py::arg("cref"), py::arg("soilflux"), py::arg("factor"),
    py::arg("sze3"), py::arg("jtot"), py::arg("jtot3"), py::arg("met_zl"), py::arg("delz"),
    py::arg("izref"), py::arg("ustar_ref"), py::arg("ustar"), 
    py::arg("source"), py::arg("cncc"), py::arg("dispersoin_np"));
}