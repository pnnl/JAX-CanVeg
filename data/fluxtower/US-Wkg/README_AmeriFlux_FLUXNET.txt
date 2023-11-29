AmeriFlux FLUXNET data product README (last changed on 20211116)

This README file describes the AmeriFlux FLUXNET data product available for download at https://ameriflux.lbl.gov/.

AmeriFlux FLUXNET data product is shared under the CC-BY-4.0 data usage license (Creative Commons by Attribution 4.0 International). Read details at https://ameriflux.lbl.gov/data/data-policy/#cc-by-4.

Please direct questions or suggestions about the FLUXNET data product to ameriflux-support@lbl.gov.




1) OVERVIEW OF THE DATA FILE CONTENTS

The FLUXNET data products are generated from the AmeriFlux BASE data products by the AmeriFlux Management Project (AMP) using the ONEFlux processing codes (Learn about data pipeline at https://ameriflux.lbl.gov/data/data-processing-pipelines/). ONEFlux processing codes are jointly developed by a global collaboration that includes AMP. Key computation steps include friction velocity threshold estimation and filtering, gap-filling of micrometeorological and flux variables, partitioning of CO2 fluxes into ecosystem respiration and gross primary production, uncertainty estimates, and more (https://fluxnet.org/data/fluxnet2015-dataset/data-processing/). The FLUXNET data product is compatible with the FLUXNET2015 dataset. Read a general overview of the processing codes in the following paper.

	Pastorello, G., Trotta, C., Canfora, E. et al. The FLUXNET2015 dataset and the ONEFlux processing pipeline for eddy covariance data. Sci Data 7, 225 (2020). https://doi.org/10.1038/s41597-020-0534-3 




2) DATA FILE NAME

The FLUXNET data file names have the following format:

	[PUBLISHER]_[SITE_ID]_FLUXNET_[GROUPING]_[RESOLUTION]_[FIRST_YEAR]-[LAST_YEAR]_[SITE_VERSION]-[CODE_VERSION].[EXT]

For instance:

	AMF_US-Ha1_FLUXNET_FULLSET_1992-2020_3-5.zip
	- AMF_US-Ha1_FLUXNET_FULLSET_HR_1992-2020_3-5.csv
	- AMF_US-Ha1_FLUXNET_FULLSET_DD_1992-2020_3-5.csv
	- AMF_US-Ha1_FLUXNET_FULLSET_WW_1992-2020_3-5.csv
	- AMF_US-Ha1_FLUXNET_FULLSET_MM_1992-2020_3-5.csv
	- AMF_US-Ha1_FLUXNET_FULLSET_YY_1992-2020_3-5.csv
	- AMF_US-Ha1_FLUXNET_ERA5_HR_1992-2020_3-5.csv
	- AMF_US-Ha1_FLUXNET_ERA5_DD_1992-2020_3-5.csv
	- AMF_US-Ha1_FLUXNET_ERA5_WW_1992-2020_3-5.csv
	- AMF_US-Ha1_FLUXNET_ERA5_MM_1992-2020_3-5.csv
	- AMF_US-Ha1_FLUXNET_ERA5_YY_1992-2020_3-5.csv
	- AMF_US-Ha1_FLUXNET_AUXMETEO_1992-2020_3-5.csv
	- AMF_US-Ha1_FLUXNET_AUXNEE_1992-2020_3-5.csv

These example files are published by the AmeriFlux network (AMF), containing data collected at the "Harvard Forest EMS Tower (HFR1)" site (US-Ha1) from 1992 to 2020. 

GROUPING denotes grouping of variables from release included in a file, including:
* FULLSET: All flux, meteorological and soil variables, and associated quality and uncertainty information, and key variables from intermediate processing steps
* SUBSET: Core set of flux, meteorological and soil variables with quality and uncertainty information needed for general uses of the data. SUBSET data are a subset of FULLSET data.
* AUXMETEO: Auxiliary variables related to meteorological downscaling
* AUXNEE: Auxiliary variables related to Net Ecosystem Exchange (NEE), Ecosystem respiration (RECO), and gross primary production (GPP) processing
* ERA5: Full record (1981–most recent year) of European Centre for Medium-Range Weather Forecasts Reanalysis v5 (ERA5) downscaled meteorological variables for the site location

RESOLUTION indicates the temporal resolution of the files, including half-hourly (HH), hourly (HR), daily (DD), weekly (WW), monthly (MM), and yearly (YY) time steps. Note that there are two possible finest-grained temporal resolutions for FLUXNET data products -- HR or HH -- depending on the temporal resolution of the original AmeriFlux BASE data product. 

SITE_VERSION is an integer indicating the version of the original dataset for the site used; CODE_VERSION indicates the version of the code of the data processing pipeline used to process the dataset for the site.

EXT: File extension, including csv: Comma-separated values in a text file (ASCII), and zip: Archive file with all temporal resolutions for the same site and data product.




3) DATA FILE CONTENTS

The complete output from the ONEFlux data processing pipeline includes over 200 variables: measured and derived data, quality flags, uncertainty quantification variables, and results from intermediate data processing steps. The variable names follow the naming conventions of <VARIABLE_ROOT>_<QUALIFIER>, where VARIABLE_ROOT describes the physical quantities (e.g., TA, NEE) and QUALIFIER describes the information of processing methods (e.g., VUT, CUT), uncertainties (e.g., RANDUNC), and quality flags (e.g., QC). See the VARIABLE LIST section below for details.

Timestamps in the data and metadata files use the format YYYYMMDDHHMM, truncated at the appropriate resolution (e.g., YYYYMMDD in daily or YYYYMM in monthly files). Two formats of time associated with a record are used: (1) single timestamp (i.e., TIMESTAMP in daily, monthly, yearly files), where the meaning of the timestamp is unambiguous, and (2) a pair of timestamps (i.e., TIMESTAMP_START, TIMESTAMP_END in (half-)hourly, weekly files), indicating the start and end of the period the timestamps represent. All timestamps are reported in local standard time (i.e., without daylight saving time). The time zone information with respect to UTC time is reported in the site metadata.

The number and order of columns in these files are not guaranteed to be uniform, except that timestamps are always located in the first or first-two column(s). The first row provides the variable labels (see VARIABLE LIST below).

These data files are provided as plain ASCII text using comma-separated values (CSV) formatting. The -9999 value indicates missing data records.




4) VARIABLE LIST (Selected)

Below is a list of the variable root names, descriptions, and units appearing in the FULLSET, SUBSET, and ERA5 files. Separate units are listed if different units are used in different temporal resolutions. See https://fluxnet.org/data/fluxnet2015-dataset/fullset-data-product/ and https://fluxnet.org/data/fluxnet2015-dataset/subset-data-product/ for a complete list of variable names and units in the FULLSET and SUBSET, respectively.

VARIABLE_ROOT	Description					Units
TA		Air temperature      				deg C
SW_IN_POT	Potential shortwave incoming radiation		W m−2
SW_IN 		Shortwave incoming radiation 			W m−2
LW_IN		Longwave incoming radiation 			W m−2
VPD		Vapor pressure saturation deficit		hPa
PA 		Atmospheric pressure 				kPa
P 		Precipitation 					mm (HH/HR) mm d−1 (DD-MM) mm y−1 (YY)
WS 		Wind speed 					m s −1
WD 		Wind direction 					Decimal degrees
RH 		Relative humidity			 	%
USTAR 		Friction velocity 				m s−1
NETRAD 		Net radiation 					W m−2
PPFD_IN		Incoming photosynthetic photon flux density	µmolPhoton m−2  s−1
PPFD_DIF	Diffuse PPFD_IN					µmolPhoton m−2  s−1
PPFD_OUT	Outgoing photosynthetic photon flux density	µmolPhoton m−2  s−1
SW_DIF		Diffuse SW_IN					W m−2
SW_OUT		Shortwave outgoing radiation			W m−2
LW_OUT		Longwave outgoing radiation			W m−2
CO2 		CO2 mole fraction 				µmolCO2  mol−1
TS 		Soil temperature 				deg C
SWC 		Soil water content 				%
G 		Soil heat flux 					W m−2
LE 		Latent heat flux 				W m−2
H 		Sensible heat flux 				W m−2
NEE 		Net Ecosystem Exchange 				µmolCO2  m−2  s−1 (HH/HR) gC m−2  d−1 (DD-MM) gC m−2 y−1 (YY)
RECO 		Ecosystem Respiration				µmolCO2  m−2  s−1 (HH/HR) gC m−2  d−1 (DD-MM) gC m−2 y−1 (YY)
GPP 		Gross Primary Production			µmolCO2  m−2  s−1 (HH/HR) gC m−2  d−1 (DD-MM) gC m−2 y−1 (YY)

A list of the most commonly seen qualifiers are provided here:

QUALIFIER	Description
_#		Layer qualifier, numeric index “#” increases with the depth, 1 is shallowest
_F		Gap-filled variable 
_QC		Quality flag; See USAGE NOTE for details
_NIGHT		Variable aggregated using only nighttime data
_DAY		Variable aggregated using only daytime data
_SD		Standard deviation
_SE		Standard Error
_MDS	 	Marginal Distribution Sampling gap-filling method
_ERA		Data filled by using ERA5 downscaling
_JSB		Longwave radiation calculated using the JSBACH algorithm (Sonke Zaehle)
_CORR		Energy fluxes corrected by energy balance closure correction factor (EBC_CF); See USAGE NOTE for details.
_CORR_25	Energy fluxes corrected by EBC_CF, 25th percentile; See USAGE NOTE for details.
_CORR_75	Energy fluxes corrected by EBC_CF, 75th percentile; See USAGE NOTE for details.
_METHOD		Method used to estimate uncertainty or energy balance closure correction
_RANDUNC	Random uncertainty
_CORR_JOINTUNC	Joint uncertainty combining from EBC_CF and random uncertainty
_VUT		Variable USTAR threshold for each year
_CUT		Constant USTAR threshold for each year
_REF		Most representative NEE after filtering using multiple USTAR thresholds; See USAGE NOTE for details.
_MEAN		Average NEE after filtering using multiple USTAR thresholds; See USAGE NOTE for details.
_USTAR50	NEE filtering by using the median value of the USTAR thresholds distribution; See USAGE NOTE for details.
_REF_JOINTUNC	Joint uncertainty combining from multiple USTAR thresholds and random uncertainty
_DT		Partitioning NEE using the daytime flux method, Lasslop et al. (2010)
_NT		Partitioning NEE using the nighttime flux method, Reichstein et al. (2005)
_SR		Partitioning NEE using the van Gorsel et al. (2009) method




5) USAGE NOTE

Detailed documentation on how to use and interpret FLUXNET data product is available online at https://fluxnet.org/data/fluxnet2015-dataset/. A quick start guide (https://fluxnet.org/data/fluxnet2015-dataset/variables-quick-start-guide/) also guides non-expert users to get started selecting variables quickly. Some main points about data usage are presented here.

Risks in the application of standard procedures:  When standardized methods are applied across different sites, the possible differences owing to data treatment are avoided or minimized. This is one of the main goals of FLUXNET data products and ONEFlux processing codes. However, there is also the possibility that the standard methods don’t work properly or as expected at specific sites and under certain conditions. This is particularly true for the CO2 flux partitioning, which as with all models, is based on assumptions that could not always be not valid. 

Using the QC flags: There are quality-control flag variables (i.e., _QC) in the dataset to help users filter and interpret variables, especially for gap-filled and process knowledge-based variables. For instance, TA_F_QC is the quality flag for the gap-filled air temperature variable TA_F. At the half-hourly or hourly resolution, the _QC variable indicates if the corresponding record is a measured value (_QC = 0​) or the quality level of the gap-filling that was used for that record (_QC = 1​ better, _QC = 3​ worse quality). At coarser temporal resolutions, i.e., daily (DD) through yearly (YY), the quality flag indicates the percentage of measured (_QC = 0​) or good quality gap-filled (_QC = 1​) records aggregated from finer temporal resolutions. 

Percentile variants for fluxes and reference values: For most flux variables, there are reference values and percentile versions of the variables to describe their uncertainty. For NEE, RECO, and GPP, the percentiles are generated from the bootstrapping of the USTAR threshold estimation step bootstrapping. Three different reference values are provided (i.e., _MEAN, _USTAR50, and _REF) to cover different user needs. The _REF version should be the most representative, particularly if related to the percentiles. For the energy-balance-corrected H and LE variables, the percentiles (e.g., _CORR, _CORR_25, _CORR_75) indicate the variability due to the uncertainty in the correction factor applied.
