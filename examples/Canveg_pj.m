% CanVeg

% 5/2/2022

% Canveg for Matlab

% Simpler version and cleaner code than CanVeg/Canoak in C

% the architecture is different from C as instead of using nested do loops for
% each hour, I am reading arrays that are dimensioned by A(hours, layers)
% so matrices are produced for each hour computed and for all the layers in
% the canopy, for fluxes and physiological variables and the surface layer,
% for meterological variables, Tair, eair, CO2.

%
%
% *************************************************************************
%                              MAIN PROGRAM
% **************************************************************************
%
%        Describe canopy attributes
%
%        Flow chart of the Main Program:
%
%        1) Set parameters for site, vegetation and soils and zero needed arrays
%
%        2) Compute (or Read) Dispersion Matrix
%
%        3) Input meteorological variables that drive the model
%
%        4) Compute solar elevation angles and Direct and Diffuse Fractions
%        of sunlight
%
%        5) Assess Leaf Inclination Angles
%
%        6) Compute PAR and NIR profiles and their flux densities on
%          the sunlit and shaded leaf fractions
%
%        7) Compute sunlit and shaded leaf fractions
%
%        8) Compute first estimate of stomatal conductance
%
%        9) Compute first estimate of IRFLUX assuming leaf temperature
%           equals air temperature.
%
%        10) Compute first estimate of leaf energy balance and leaf
%           temperature
%
%        11) Compute photosynthesis, transpiration, stomatal conductance
%
%        12) Compute soil energy balance
%
%        13) update computation of Dij, dispersion matrix, with new H and z/L
%
%        14) Compute new scalar and source/sink profiles for CO2, T and e
%        using Dij
%
%        15) Iterate among 9 through 14 until convergence
%
%        16) compute fluxes of isoprene, 13C isotopes, SIF  or whatever
%
%        17) Plot, Visualize, Analyze
%
%        18) Thats all Folks!!!!

% current version is set up for
% either alfalfa or tules, or generic
% Amphistomatous or Hypostomatous

% Plans are to add C4 photosynthesis and corn

% one needs to rest dispersion matrix for the new site and swap out the
% read of the site, veg and soil parameters

% parameter_tule or parameter_alfalfa

% fSetSoilTule or fSetSoilAlfalfa

% newest version uses switch and information on Vegetation is determined
% with the variable VegType


% debug..check spectral transfer function for wq and apply it to the model
% test. may be the factor..


clear all;
close all;


% cd 'c:\users\baldocchi\documents\matlab\canveg'

%VegType='Alfalfa';
%Site = 'US-Bi1';
%Site = 'US-Hn1';
%Site = 'US-Me2';
Site = 'US-Whs';


    tic;

    % set parameters, constants, and knowns

     % Call Lagrangian Dispersion Matrix, Dij

    % In this version I only call DispCanveg at the beginning and
    % scale Dij with u*

    % this will be run for n canopy layers to compute the trajectory of
    % 200,000 parcels, or more

    % for first set of runs will assume z/L is zero and near neutral
    % if we are to change Dij with z/L we will need to do some engineering
    % runs for the assumed canopy structure and compute a set of Dij with
    % different z/L classes and fit a simple correction term, like with
    % Canoak


%  Need to re run Dispersion every time LAI changes because the canopy
%  is divided into #N = LAI/0.1 layers of equal LAI increments which then
%  have heights of height/#N and the remaining layers from canopy height to
%  measurement height have increments of 0.1 m.


    %*********************************

    [prm]=parameter_alfalfa();  % model parameters
    prm.hrs=48;
     % Call Dispersion Matrix if this is new run
     % otherwise comment it out and just read in data
        %    [DIJ]=DispCanveg_v2a(prm);   % uncomment these if new runs...1000000 parcels
        %    Dij=DIJ.DIJ;

          %   csvwrite('D:\Canalfalfa\Dij_AlfalfaL5.csv',DIJ.DIJ) ;  % run with case LAI 5, ht 0.80 m
            %   csvwrite('D:\Canalfalfa\Dij_zlm025.csv',DIJ.DIJ) ;  % turb.sigma_h= 1.506 ;   // z/L = -0.25
            %   csvwrite('D:\Canalfalfa\Dij_zlm050.csv',DIJ.DIJ) ; % turb.sigma_h = 1.696 ;   z/L = -0.50
            %   csvwrite('D:\Canalfalfa\Dij_zlm100.csv',DIJ.DIJ) ; % turb.sigma_h = 1.984 ;   z/L = -1.00
            %    csvwrite('D:\Canalfalfa\Dij_zlm300.csv',DIJ.DIJ) ; % turb.sigma_h = 2.692 ;   z/L = -3.00
            %     csvwrite('D:\Canalfalfa\Dij_zlp050.csv',DIJ.DIJ) ; % turb.sigma_h = 1.35;      // z/L = 0.5
            %            csvwrite('D:\Canalfalfa\Dij_zlp100.csv',DIJ.DIJ) ; %turb.sigma_h= 1.5;        // z/L =1.0
%           csvwrite('D:\Canalfalfa\Dij_Alfalfa.csv',DIJ.DIJ) ;  % run with case LAI 3, ht 0.78 m
%           csvwrite('D:\Canalfalfa\Dij_AlfalfaTwitchell.csv',DIJ.DIJ) ;  % run with case LAI 3, ht 0.78 m
   %        csvwrite('D:\Canalfalfa\Dij_Alfalfa.csv',DIJ.DIJ) ;  % run with case LAI 3, ht 0.78 m

   switch Site
       case 'US-Bi1'
           prm.Dij = 'Dij_US-Bi1_50L.csv'
       case 'US-Hn1'
           prm.Dij = 'Dij_US-Hn1_50L.csv'
       case 'US-Me2'
           prm.Dij = 'Dij_US-Me2_50L.csv'
       case 'US-Whs'
           prm.Dij = 'Dij_US-Whs_50L.csv'
       otherwise
   end

   Dij=csvread(prm.Dij);


    % visualize Dispersion Matrix

%    figure(100)
%    clf
%    plot(Dij,prm.zht(1:prm.nlayers_atmos))
%    xlabel('Dij s/m')
%     ylabel('height')

  % input met files with day, hour and met conditions to run the model

  % the philosophy has been to run the model on simple inputs from a
  % meterological station

  % at present we estimate diffuse radiation from measured and potential
  % radiation, but if there is an input of diffuse radiation that can be
  % used.

  % this version inputs u* from the flux tower, but in other situations u*
  % is not available and it will need to be computed.  To do later

  switch Site
      case 'US-Bi1'
          prm.filename = 'US-Bi1-forcings.txt';
      case 'US-Hn1'
          % prm.filename = 'US-Hn1-forcings-v2.csv';
          prm.filename = 'US-Hn1-forcings.txt';
      case 'US-Me2'
          prm.filename = 'US-Me2-forcings.txt';
      case 'US-Whs'
          prm.filename = 'US-Whs-forcings.txt';
      otherwise
  end


  inmet=csvread(prm.filename);

  met.year=inmet(:,1);             % year
  met.day=inmet(:,2);             % day of year
  met.hhour=inmet(:,3);           % hour
  met.T_air_K=inmet(:,4)+273.15;  % air temperature, K
  met.rglobal=inmet(:,5);         % global shortwave radiation, W m-2
  met.eair=inmet(:,6);            % vapor pressure, kPa
  met.wind=inmet(:,7);            % wind velocity, m/s
  met.CO2=inmet(:,8);             % CO2, ppm
  met.P_kPa=inmet(:,9);           % atmospheric pressure, kPa
  met.ustar=inmet(:,10);           % friction velocity, m/s
  met.Tsoil=inmet(:,11);          % soil temperature, C...16 cm
  met.soilmoisture=inmet(:,12);   % soil moisture, fraction
  met.zcanopy=inmet(:,13);        % aerodynamic canopy height
  met.LAI=inmet(:,14);        % aerodynamic canopy height
  % met.Tsoil_2=inmet(:,13);          % soil temperature, C...2 cm
  % met.Tsoil_4=inmet(:,14);          % soil temperature, C...4 cm
  % met.Tsoil_8=inmet(:,15);          % soil temperature, C...8 cm
  % met.LAI=inmet(:,16);            % LAI computed by NDVI and fit with LI2200
  % met.LongIRout=inmet(:,17);      % Longwave radiated by the canopy

    % code tends to blow up with really light winds on hot days

    met.wind(met.wind < 0.75)=0.75;
    met.ustar(met.ustar < 0.1)=0.1;

    met.rglobal(met.rglobal<=0)=0;
    met.parin= 4.6*met.rglobal/2;   % visible, or photosynthetic photon flux density, umol m-2 s-1

    met.eair_Pa=met.eair*1000;      % vapor pressure, Pa
    met.P_Pa=met.P_kPa*1000;        % pressure, Pa

    % 	Compute gas and model coefficients that are used repeatedly

    [met.es]=fES(met.T_air_K);       % saturation vapor pressure, Pa
    met.vpd_Pa=met.es-met.eair_Pa;   % atmospheric vapor pressure deficit, Pa

    met.air_density = met.P_kPa .* prm.Mair ./ (prm.rugc .* met.T_air_K);  % air density, kg m-3
    met.air_density_mole=1000*met.air_density/prm.Mair;   % air density, moles m-3

	[met.dest]=fdESdT(met.T_air_K);    % slope saturation vapor pressure Temperature Pa K-1
	[met.d2est]=fd2ESdT(met.T_air_K);  % second derivative es(T)
    [met.llambda]=fLambda(met.T_air_K); % latent heat of vaporization, J kg-1


    prm.nn=length(met.day);   % number of 30 minute runs
    prm.ndays=prm.nn/48;      % number of days


    met.zL=zeros(prm.nn,1);  % z/L initial value at 0

    % call soil information.  Need data on soil moisture to apply these
    % subroutines as they compute heat capacity and thermal conductivity

    [soil]=fSetSoilAlfalfa(met,prm);    % coefficients for soils.


      % zero the arrays that will be used to pre-allocate memory to speed
      % up computations

      [quantum,nir,ir,Qin,rnet,Sun,Shade,Veg]=fZeroArrays(prm);

      % Tsfc initialized as ones, and assign value to air temperature for
      % first iteration

      Sun.Tsfc=Sun.Tsfc.*met.T_air_K;

      Shade.Tsfc=Shade.Tsfc.*met.T_air_K;

    % compute sun angles based on time and lat long

    % disp(prm.time_zone,prm.lat_deg,prm.long_deg, met.day, met.hhour)
    sunang= fSunAngle(prm.time_zone,prm.lat_deg,prm.long_deg, met.day, met.hhour);

    sunang.sine_beta=sin(sunang.beta_rad);  % sine of solar elevation angle, beta

    % Given incoming solar radiation and PAR compute the fractions of
    % direct and diffuse for NIR and PAR wavebands

    % compute direct and diffuse radiation from Weiss and Norman

    [sunrad]=fDiffuse_Direct_Radiation(met.rglobal,met.parin,met.P_kPa, sunang.sine_beta);

    % inputs of photon flux density of visible light, or quantum flux,
    % umol m-2 s-1. quantum.xxx is visible light used to compute photosynthesis

    quantum.inbeam=sunrad.par_beam;
    quantum.indiffuse=sunrad.par_diffuse;
    quantum.incoming = quantum.inbeam+ quantum.indiffuse;


    % inputs of near infrared radiation, W m-2

    nir.inbeam=sunrad.nir_beam;
    nir.indiffuse=sunrad.nir_diffuse;
    nir.incoming = nir.inbeam+ nir.indiffuse;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Alternatively, Diffuse Fraction from Oliphant and Stoy. 2018 JGR
    % Biogeoscience. They tested 4 models with Fluxnet data


%     Kt=met.rglobal./sunang.extraterr;
%     [diffuse_fraction.Oliphant]=fPARdf(Kt);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% call LeafAngle to compute leaf angle distribution and G functions
% we assume spherical leaf distribution for alfalfa, erectophile for tule
% and corn

% options include planophile, erectophile, uniform, plagiophile, extremophile...

  [leafang]=LeafAngle(sunang, prm);


%  plot(leafang.thetaSky,leafang.Gfunc_Sky,'.')
%  xlabel('Zenith Angle')
%  ylabel('Gfunc = f( \theta)')

 % compute radiative transfer through the canopy for the input dataset
 % Code has been changed to process the matrices of radiation by time and
 % layers


 % initialize IR fluxes with air temperature and IR flux. Possible to use
 % data from pyrgeometer as input if available.

     ir.in = fSKY_IR(met.T_air_K,sunrad,prm);

     % Initialize profiles of scalars and sources and sinks

      [prof]=initial_profile_Matrix(met,prm);


    % compute wind velocity with height in the canopy
    % wind speed in the canopy is used to compute boundary layer
    % resistances

    % [prof.wind]=fUZ_Matrix(met,prm);


 % compute radiation fields for visible quantum flux, umol m-2, NIR and IR, W m-2


% visible, quantum flux, umol m-2 s-1, for photosynthesis

    waveband='par';
%     [quantum]=fRadTranCanopy_Matrix(sunang,leafang,quantum,waveband,prm);

      [quantum]=fRadTranCanopy_MatrixV2(sunang,leafang,quantum,waveband,prm);

% Near Infrared, W m-2

    waveband='nir';
   % [nir]=fRadTranCanopy_Matrix(sunang,leafang,nir,waveband,prm);
     [nir]=fRadTranCanopy_MatrixV2(sunang,leafang,nir,waveband,prm);

    nir.albedo=nir.up_flux(:,prm.jktot)./nir.incoming;

    figure(155)
    clf
    histogram(nir.albedo(nir.incoming > 30))

    xlabel('NIR albedo')

% Compute IR, iterate, compute Tsfc -> IR -> Rnet -> Energy Balance -> Tsfc
% loop again and apply updated Tsfc info until convergence

% test shows looping 10 to 15 times is good enough to get stable results for the
% canopy model

for j=1:15
%for j=1:1

%disp('T soil');
%disp(soil.T_soil(11,:));
disp('T sfc');
disp(soil.sfc_temperature(11));

 ir.flag=j;

  % update canopy wind profile with iteration of z/L and use in boundary
  % layer resistance computations

  % #TODO: Why is calculation of uz in the iteration?
  [prof.wind]=fUZ_Matrix(met,prm);



% Compute IR fluxes with Bonan's algorithms of Norman model
[ir]= fIR_RadTranCanopy_MatrixV2(leafang,ir,quantum,soil,Sun,Shade, prm);
%disp('ir');
%disp(ir.ir_dn(11,:));

% Incoming Short and Longwave radiation
[Qin]=fQin_Matrix(quantum,nir,ir,prm);

% Compute Energy Fluxes for H, LE, gs, A on Sun and Shade Leaves
% compute new boundary layer conductances based on new leaf energy balance
% and delta T, in case convection occurs

% different coefficients will be assigned if amphistomatous or
% hypostomatous

[Sun,Shade]=fEnergy_Carbon_Fluxes_Matrix(Sun, Shade, Qin, quantum, met, prof, prm);


% compute soil fluxes


[soil]=fSoilEnergyBalanceMatrix(quantum, nir,ir, met, prof, prm, soil,j);

 nanmean(soil.rnet-soil.evap-soil.heat-soil.gsoil)

% compute Profiles of C's, zero layer jtot +1 as that is not a dF/dz or
% Source/Sink level

% Compute Profiles of T, e and CO2

% compute source sink strength for each layer

% S = dF/dz
% make sure profile is not upside down..soil 1, top nlayer

   % dz=prof.delz(1,1);


    prof.Ps=(quantum.prob_beam(:,1:prm.nlayers) .* Sun.Ps +...
        quantum.prob_shade(:,1:prm.nlayers) .*Shade.Ps) .*  prm.adens;

    prof.LE=(quantum.prob_beam(:,1:prm.nlayers) .* Sun.LE +...
        quantum.prob_shade(:,1:prm.nlayers) .*Shade.LE) .* prm.adens;

    prof.H=(quantum.prob_beam(:,1:prm.nlayers) .* Sun.H +...
        quantum.prob_shade(:,1:prm.nlayers) .*Shade.H) .* prm.adens;

    prof.Rnet=(quantum.prob_beam(:,1:prm.nlayers) .* Sun.Rnet +...
        quantum.prob_shade(:,1:prm.nlayers) .*Shade.Rnet) .* prm.adens;


    % What Does C code do

    % prof.dRNdz[JJ] = prof.dLAIdz[JJ] * (solar.prob_beam[JJ] * Rn_sun + solar.prob_sh[JJ] * Rn_shade);


    prof.Tsfc=(quantum.prob_beam(:,1:prm.nlayers) .* Sun.Tsfc +...
        quantum.prob_shade(:,1:prm.nlayers) .*Shade.Tsfc) ;



    % Compute Scalar Profiles with fConcMatrix
    % it needs information on source/sink, Dij, soil boundary flux and factor for units

        % Temperature
        fact.heatcoef = met.air_density * prm.Cp;

        soilflux=soil.heat;  % assume soil heat flux is 20 W m-2 until soil sub is working


       [prof.Tair_K]=fConcMatrix(prof.H,soilflux, prof.delz, Dij,met,met.T_air_K, prm, fact.heatcoef);


       % Depending of Dij I was getting Tair to blow up or go off bounds

%        prof.Tair_K=real(prof.Tair_K);
%        testdel=abs(prof.Tair_K - met.T_air_K) ./met.T_air_K;
%        use= testdel > 0.2;   % keep Tair profile within 20% of Tair
%        test=sum(use,2);
%        u2=test>0;
%
%        prof.Tair_K(u2) = met.T_air_K(u2);

       % with larger Dij value I need to filter new T profiles

       prof.Tair_K=0.25 .*prof.Tair_K + 0.75 * prof.Told_K;
       prof.Told_K=prof.Tair_K;



% Compute vapor pressure profiles

soilflux=soil.evap;  % W m-2....

% in fConcMatrix fact.lecoef is in the denominator insteat of multiplier

% if we divide W m -2 = J m-2 s-1 by Lambda we have g m-2 s-1

% need to convert g to Pa

% eair =rhovair R Tk/mv  Jones..

        fact.lecoef= fLambda(prof.Tair_K(:,prm.jktot))*18.01 ./(1000* 8.314 .*prof.Tair_K(:,prm.jktot)) ;

        % Below is added by Peishi
        prof.eair_old_Pa=prof.eair_Pa;
        [prof.eair_Pa]=fConcMatrix(prof.LE,soil.evap,prof.delz, Dij,met,met.eair_Pa, prm, fact.lecoef);


         prof.eair_Pa=0.25 .*prof.eair_Pa + 0.75 * prof.eair_old_Pa;

       prof.eair_old_Pa=prof.eair_Pa;


        % Compute CO2 profiles
%
%         // compute soil respiration

%
%                 //    Convert to umol m-2 s-1
%
%
%         // revision is using micromol m-2 s-1
%
%                 // to convert umol m-3 to umol/mol we have to consider
%                 // Pc/Pa = [CO2]ppm = rhoc ma/ rhoa mc
%
                 fact.co2=(28.97/44)*met.air_density_mole;

                 Rsoil = SoilRespiration(Veg.Ps,soil.T_soil(:,10),met.soilmoisture,met.zcanopy,Veg.Rd,prm);
               %%  if alfalfa
                %switch VegType
                 %case 'Alfalfa'

                 %Rsoil = SoilRespiration(Veg.Ps,soil.T_soil(:,10),met.soilmoisture,met.zcanopy,Veg.Rd,prm);
                    %case 'Tule'

                    %Rsoil = SoilRespiration(Veg.Ps,soil.T_soil(:,10),met.WaterTable,met.zcanopy,Veg.Rd,prm);

                    %case 'DeciduousForest'

                       %% met.zcanopy=ones(length(met.soilmoisture),1) *30;

                       %% Rsoil = SoilRespiration(Veg.Ps,soil.T_soil(:,10),met.soilmoisture,met.zcanopy,Veg.Rd,prm);

                        %Rsoil.Respiration=ones(length(met.soilmoisture),1) *6;

                    %otherwise
                %end

                 % if tule

                 %%%% switch


            %    soilflux=10*ones(prm.nn,1);  % estimate unil I add soil respiration module

                 % For Concentrations for CO2 need to change sign of
                 % prof.Ps, hence negative (-) sign applied

                 [prof.co2]=fConcMatrix(-prof.Ps,soilflux, prof.delz, Dij,met,met.CO2, prm, fact.co2);

                 soilflux=Rsoil.Respiration;
                 %disp("Veg.Ps");
                 %disp(Veg.Ps(1:10));
                 %disp("Veg.Rd");
                 %disp(Veg.Rd(1:10));
                 %disp("soilflux");
                 %disp(soilflux(1:10));
                 %disp("zl");
                 %disp(met.zL(1:10));
                 %disp("metco2");
                 %disp(met.CO2(1:10));
                 %disp("co2");
                 %disp(prof.co2(1:10,1:5));

                 % TODO: let's turn that off for now --PJ
                 %prof.co2=real(prof.co2);

                   %use=(prof.co2 < 385) ;

                   %test=sum(use,2);
                   %u2=test>0;

                   %% find rows with at least one logical value 1
                   %prof.co2(u2)=met.CO2(u2);



% compute met.zL from HH and met.ustar

 HH=sum((quantum.prob_beam(:,1:prm.nlayers) .* Sun.H +...
        quantum.prob_shade(:,1:prm.nlayers) .*Shade.H) .* prm.dff(1:prm.nlayers)',2);

met.zL = -(0.4*9.8* HH*prm.meas_ht) ./(met.air_density*1005.* met.T_air_K .*power(met.ustar,3.));

met.zL(met.zL>0.25)=0.25;
met.zL(met.zL <-3)=-3;

% test for achieving steady state on iterations amoung T, Rnet and fluxes
% results show 5 loops is enough

avgTsfctst(j)=nanmean(nanmean(prof.Tsfc)) ;

    Veg.Ps=sum((quantum.prob_beam(:,1:prm.nlayers) .* Sun.Ps +...
        quantum.prob_shade(:,1:prm.nlayers) .*Shade.Ps) .* prm.dff(1:prm.nlayers)',2);

    Veg.Rd=sum((quantum.prob_beam(:,1:prm.nlayers) .* Sun.Resp +...
        quantum.prob_shade(:,1:prm.nlayers) .*Shade.Resp).* prm.dff(1:prm.nlayers)',2);

 end

figure(1111)
clf
plot(avgTsfctst)
xlabel('number of iterations')
ylabel('Average Tsfc')

   % compute canopy integrated fluxes  int(psun * Fsun(l) + pshade Fshade(i)) dl

    % Net photosynthesis, A - Rd


    % ??? should I consider Markov on prm.dff??



    Veg.Ps=sum((quantum.prob_beam(:,1:prm.nlayers) .* Sun.Ps +...
        quantum.prob_shade(:,1:prm.nlayers) .*Shade.Ps) .* prm.dff(1:prm.nlayers)',2);

    Veg.Rd=sum((quantum.prob_beam(:,1:prm.nlayers) .* Sun.Resp +...
        quantum.prob_shade(:,1:prm.nlayers) .*Shade.Resp) .* prm.dff(1:prm.nlayers)',2);

    Veg.Gpp=Veg.Ps+Veg.Rd;

    Veg.LE=sum((quantum.prob_beam(:,1:prm.nlayers) .* Sun.LE +...
        quantum.prob_shade(:,1:prm.nlayers) .*Shade.LE).* prm.dff(1:prm.nlayers)',2);

    Veg.H=sum((quantum.prob_beam(:,1:prm.nlayers) .* Sun.H +...
        quantum.prob_shade(:,1:prm.nlayers) .*Shade.H).* prm.dff(1:prm.nlayers)',2);

    Veg.gs=sum((quantum.prob_beam(:,1:prm.nlayers) .* Sun.gs +...
        quantum.prob_shade(:,1:prm.nlayers) .*Shade.gs) .* prm.dff(1:prm.nlayers)',2);

    Veg.Rnet=sum((quantum.prob_beam(:,1:prm.nlayers) .* Sun.Rnet +...
        quantum.prob_shade(:,1:prm.nlayers) .* Shade.Rnet) .* prm.dff(1:prm.nlayers)',2);

    % compute profile of Rnet
      prof.Rnet=quantum.prob_beam(:,1:prm.nlayers) .* Sun.Rnet +...
        quantum.prob_shade(:,1:prm.nlayers) .* Shade.Rnet;

    % trying to debug and find why Rnet ~= sumRn in Matlab, but does in C


    % matlab is computing
%     rad.prob_beam=prm.markov' .* rad.P0(:,1:prm.jtot);
%     exp_direct = exp(-(prm.dff .* prm.markov)' .* (leafang.Gfunc ./ sunang.sine_beta));
%     rad.P0=cumprod(exp_direct,2,'reverse');

    % C code uses dff
    % prof.dRNdz[JJ] = prof.dLAIdz[JJ] * (solar.prob_beam[JJ] * Rn_sun + solar.prob_sh[JJ] * Rn_shade);
    % solar.prob_beam[JJ] = markov*PEN2;
    % PEN2 = exp(-sumlai*markov*prof.Gfunc_solar[JJ]/ solar.sine_beta);

    Veg.Tsfc=sum((quantum.prob_beam(:,1:prm.nlayers) .* Sun.Tsfc(:,1:prm.nlayers) +...
        quantum.prob_shade(:,1:prm.nlayers) .*Shade.Tsfc(:,1:prm.nlayers)) .* prm.dff(1:prm.nlayers)',2);

    Veg.Tsfc=Veg.Tsfc/prm.LAI;

   Veg.vpd=sum((quantum.prob_beam(:,1:prm.nlayers) .* Sun.vpd_Pa(:,1:prm.nlayers) +...
        quantum.prob_shade(:,1:prm.nlayers) .*Shade.vpd_Pa(:,1:prm.nlayers)).* prm.dff(1:prm.nlayers)',2);

    Veg.vpd=Veg.vpd/prm.LAI;

    % net radiation budget at top of the canopy


    % Rnet = PARin-PARout +NIRin - NIR out + IRin - (1-ep)IRin - IRout

   % Can.Rnet_calc=quantum.incoming/4.6 -quantum.up_flux(:,prm.jktot)/4.6+nir.incoming -nir.up_flux(:,prm.jktot)+ prm.ep* ir.in-ir.ir_up(:,prm.jktot);

    Can.Rnet_calc=quantum.beam_flux(:,prm.jktot)/4.6 + quantum.dn_flux(:,prm.jktot)/4.6 -quantum.up_flux(:,prm.jktot)/4.6 +...
                  nir.beam_flux(:,prm.jktot) + nir.dn_flux(:,prm.jktot) - nir.up_flux(:,prm.jktot) +...
                  ir.ir_dn(:,prm.jktot)+ -ir.ir_up(:,prm.jktot);

    % NetRad, C,,,NetRad==sumrn
    %  netrad = (solar.beam_flux_par[jktot] + solar.par_down[jktot] - solar.par_up[jktot]) / 4.6 + ...
    %  solar.beam_flux_nir[jktot] + solar.nir_dn[jktot] - solar.nir_up[jktot] + solar.ir_dn[jktot] - solar.ir_up[jktot];



    Can.LE=Veg.LE + soil.evap;
    Can.H=Veg.H + soil.heat;
    Can.Rnet=Veg.Rnet+soil.rnet;
    Can.NEE=Rsoil.Respiration-Veg.Gpp;

      % compute canopy conductance

    Can.Avail=Can.Rnet_calc-soil.gsoil;
    Can.Gsoil=soil.gsoil;

    Can.albedo_calc=(quantum.up_flux(:,prm.jktot)/4.6 + nir.up_flux(:,prm.jktot))./(quantum.incoming/4.6+nir.incoming);
    Can.NIRalbedo=( + nir.up_flux(:,prm.jktot))./(+nir.incoming);

    Can.NIR_refl=nir.up_flux(:,prm.jktot)-nir.up_flux(:,1);

    % canopy radiative temperature

    Can.Trad = power(ir.ir_up(:,prm.jktot)./prm.epsigma,0.25);


    % H = rho Cp (Taero-Tair)/Rah

    % Rah ~ sum(Dij) from source/sink to reference height


   for i=1: prm.jtot

    sumDij(i)=sum(Dij(i:prm.nlayers_atmos,i));  % sum Dij from Source/Sink to reference

    wtDijH(:,i)=sumDij(i) .* prof.H(:,i) .* prof.delz(i);  % how connected is dH/dz at i related to z..s/m  J m-3 s-1

    end

   dT=(wtDijH) ./(met.air_density * 1005);    % J m-3 / g m-3 * J/g C

    % need to add soil and veg to estimate Canopy


    % remember Ci-Cref = sum(Dij S(i) dz)
    % so the above code computes an array of Ci-Cref and we average for the
    % canopy aerodynamic temperature

     Can.Taero=nanmean(dT,2)+ met.T_air_K;



      toc;

      %prm.Veg=VegType;

     %Dirtsfc=reshape(Veg.Tsfc,prm.hrs,prm.ndays);
     %Dirtair=reshape(met.T_air_K,prm.hrs,prm.ndays);
     %Dirsoiltsfc=reshape(soil.sfc_temperature,prm.hrs,prm.ndays);

    %figure(3638)
    %clf
    %plot(1:prm.hrs,nanmean(Dirtsfc,2))
    %hold on
    %plot(1:prm.hrs,nanmean(Dirtair,2))
    %hold on
    %plot(1:prm.hrs,nanmean(Dirsoiltsfc,2))
    %xlabel('hour')
    %ylabel ('T C')
    %title(prm.title);
    %legend('veg-tsfc','tair', 'soil-tsfc')

      %%met.Trad=((met.LongIRout-(1-prm.ep)*ir.in)/prm.epsigma).^ 0.25;

      %%OutputPlots(prm,Can,soil,met,prof,Veg);
      %figure(3636)
     %clf
      %plot(nanmean(prof.Tair_K,1),prof.zht(1:prm.nlayers_atmos));
      %xlabel('Tair, K')
      %ylabel('ht m')

      %figure(3635)
     %clf
      %plot(nanmean(prof.eair_Pa,1),prof.zht(1:prm.nlayers_atmos));
      %xlabel('eair, Pa')
      %ylabel('ht m')

      %figure(3634)
     %clf
      %plot(nanmean(prof.co2,1),prof.zht(1:prm.nlayers_atmos));
      %xlabel('CO2, ppm')
      %ylabel('ht m')

    %DiH=reshape(Can.H,prm.hrs,prm.ndays);
    %DiLE=reshape(Can.LE,prm.hrs,prm.ndays);
     %DiLEsoil=reshape(soil.evap,prm.hrs,prm.ndays);
        %DiHsoil=reshape(soil.heat,prm.hrs,prm.ndays);
           %DiGsoil=reshape(soil.gsoil,prm.hrs,prm.ndays);
              %DiRnsoil=reshape(soil.rnet,prm.hrs,prm.ndays);
     %Dirnet=reshape(Can.Rnet,prm.hrs,prm.ndays);
      %figure(3637)
    %clf
    %plot(1:prm.hrs,nanmean(Dirnet,2),'LineWidth', 1.25);
    %hold on
    %plot(1:prm.hrs,nanmean(DiLE,2),'LineWidth', 1.25)
    %hold on
    %plot(1:prm.hrs,nanmean(DiH,2),'LineWidth', 1.25)
    %hold on
    %plot(1:prm.hrs,nanmean(DiGsoil,2),'LineWidth', 1.25)
    %xlabel('hour')
    %ylabel ('Energy Flux Density, W m-2')
    %title(prm.title);
    %legend('Rn','LE','H','Gsoil')

       %Taero=Can.Taero;
       %%Trad=met.Trad;

      %%save('d:\CanAlfalfa\TaeroTrad_L5.mat','Taero','Trad');
      %%save('d:\CanAlfalfa\TaeroTrad_L4.mat','Taero','Trad');

      %% Does Rnet decay in the canopy with exponential function, eg Beer's
      %% Law

     %x=reshape(quantum.P0,48,17,51);
     %xx=nanmean(nanmean(x(20:32,:,1:50),2));


     %y=reshape(prof.Rnet,48,17,50);
     %yy=nanmean(nanmean(y(14:36,:,:),2));

     %yyy=reshape(yy,1,50);
     %xxx=reshape(xx,1,50);

     %figure(58)
     %clf
     %plot(xxx,yyy,'.')
     %xlabel('P_0')
     %ylabel('Rnet, W m-2')
     %title('CanVeg, Alfalfa, 0700-1700')



     %x=reshape(quantum.P0,48,17,51);
     %xx=x(:,:,1:50);

     %use=find(xx>0);

     %y=reshape(prof.Rnet,48,17,50);

     %figure(588)
     %clf
     %plot(xx(use),y(use),'.')
     %xlabel('P_0')
     %ylabel('Rnet, W m-2')
      %title('CanVeg, Alfalfa, P0>0')




       %%  profile of Tsun and Tshade

     %figure(41)
     %clf
     %plot(nanmean(Sun.Tsfc(:,1:prm.jtot),1),prm.sumlai,'+-','LineWidth', 1)
     %ax=gca;
      %set(ax, 'ydir','reverse');
     %hold on
     %plot(nanmean(Shade.Tsfc(:,1:prm.jtot),1),prm.sumlai,'+-','LineWidth', 1)
     %ax=gca;
      %set(ax, 'ydir','reverse');
       %xlabel('Surface Temperature')
      %ylabel('Canopy Depth')
      %legend('sun','shade');

