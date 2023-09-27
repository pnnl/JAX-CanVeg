
% Forced Diffusion Bouldin Island

% Oct 5, 2022

% Dennis Baldocchi
% biomet lab
% UC Berkeley

% new version of data has had CO2 probes re calibrated and with offsets
% considered

% later there is a 2nd sensor

clear all;
close all;

cd 'd:\Bouldin Island\Alfalfa';
readtable('BouldinAlfalfaForcedDiffusion2018_2021.csv');


% DateVector=datevec(ans.DateTime)
% 
%  year=DateVector(:,1);
%  month=DateVector(:,2);
%  day=DateVector(:,3);
%  hour=DateVector(:,4);
%  minute=DateVector(:,5);
%decday=ans.DecDay;

year=ans.year;
doy=ans.doy;
hhmm=ans.hhmm;

hour=floor(hhmm/100);
minute=hhmm-hour*100;

hrmn=hour+minute/60;


Efflux_1=ans.BA_metF_FD_Flux;
soil_ppm_1=ans.BA_metF_FD_CO2_1;
atm_ppm_1=ans.BA_metF_FD_CO2_2;


histogram(atm_ppm_1)
histogram(soil_ppm_1)



atm_ppm_1(atm_ppm_1 < 360)=NaN;
atm_ppm_1(atm_ppm_1 > 2000)=NaN;

soil_ppm_1(atm_ppm_1 < 360)=NaN;
soil_ppm_1(atm_ppm_1 > 5000)=NaN;

delta_ppm=(soil_ppm_1-atm_ppm_1);

Efflux_1(isnan(delta_ppm))=NaN;

Refl810=ans.BA_metF_NDVI_810out./ans.BA_metF_NDVI_810in;
Refl650=ans.BA_metF_NDVI_650out./ans.BA_metF_NDVI_650in;

NDVI=(Refl810-Refl650)./(Refl810+Refl650);

NDVI(NDVI<0.1)=NaN;
NDVI(NDVI > 0.9)=NaN;

NDVId=reshape(NDVI,48,[]);
Efflux_1d=reshape(Efflux_1,48,[]);


% lots of Nans so sums are biased. try means

Fc_day_mean=nanmean(Efflux_1d,1);  % umol m-2 s-1

fnum=0;
fnum=fnum+1;
figure(fnum)
plot(nanmean(NDVId(12:38,:)),Fc_day_mean,'.')
xlabel('NDVI')
ylabel('R_{soil} \mu mole m^{-2} s^{-1}')




fnum=fnum+1;
figure(fnum)
clf;
plot(Efflux_1,'.')


fnum=fnum+1;
figure(fnum)
clf;
histogram(delta_ppm)


xlim([0 1000])
xlabel('CO2_{soil} - CO2_{air}')

% use=(soil_ppm_1-atm_ppm_1) > 400;
% Efflux_1(use)=NaN;


Efflux_2=ans.BA_metF_FD_B_Flux;
soil_ppm_2=ans.BA_metF_FD_B_CO2_1;
atm_ppm_2=ans.BA_metF_FD_B_CO2_2;

fnum=fnum+1;
figure(fnum)
clf;
histogram(soil_ppm_2-atm_ppm_2)

use2=(soil_ppm_2-atm_ppm_2) > 400;

Efflux_2(use2)=NaN;


fnum=fnum+1;
figure(fnum)
clf;

plot(year+doy/365, Efflux_1, '.')
ylabel('F_{CO_2} \mu mol m^{-2} s^{-1}')
xlabel ('day-hour')
ylim([0 15])
title('Forced Diffusion Chamber')


fnum=fnum+1;
figure(fnum)
clf;

plot(year+doy/365, Efflux_2, '.')
ylabel('F_{CO_2} \mu mol m^{-2} s^{-1}')
xlabel ('day-hour')
ylim([0 10])
title('Forced Diffusion Chamber #2')

decday=doy+hrmn/24;;

fnum=fnum+1;
figure(fnum)
clf;
plot(decday, soil_ppm_1, '.')

fnum=fnum+1;
figure(fnum)
clf;
plot(decday, atm_ppm_1, '.-')

fnum=fnum+1;
figure(fnum)
clf;
plot(decday, soil_ppm_1-atm_ppm_1, '.')

fnum=fnum+1;
figure(fnum)
clf;
plot( soil_ppm_1-atm_ppm_1, '.')


fnum=fnum+1;
figure(fnum)
clf;
plot(Efflux_1,Efflux_2,'.')

% mean diel


fnum=fnum+1;
figure(fnum)
clf;
plot(1:48,nanmean(Efflux_1d,2),'.-', 'linewidth',2')




