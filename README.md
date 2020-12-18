# Dynamics of liquids in microchannels using high resolution spectroscopy
This project studies time-resolved fluctuations in speckle patterns using X-ray Photon Correlation Spectroscopy (XPCS), a scattering technique to study the dynamics of disordered condensed matter systems. The aim of this project is to study the behavior of complex fluid flows with nanoparticle spheres in particular, flow velocity profiles from the X-ray data.

The important files in this project are:
* **XPCS_Setup.py** establishes the regions of interest from the meta data of scattering experiments
* **XPCS_flow_analysis.py** contains code for model functions and fitting algorithms towards assessment of flow parameters
* **Fourier_Modes.py** contains functions for generating flow profiles based on obtaining best parameters at each generation  

# Step-1 - Data aquisition from images taken at different time intervals
XPCS experiments were performed in a small angle X-ray scattering
(SAXS) geometry at the Coherent Hard X-ray beamline(11-ID) of the Brookhaven
National Laboratory (BNL), Upton, USA. A double crystal monochromator si(111)
was used to select 9.6 keV X-rays with a relative bandwidth. The sample
was placed at a distance of 10.18 m from the detector. The parasitic scattering
from beam was prevented by position of guard slits, final coherent flux in the sample was 10^11 ph/s. The scattering was recorded with a 2 dimensional (2D) sensor -Eiger4M detector. As the name indicates, sensor can record images of size 2000 x 2000 pixels. The experiments were carried at 750 frames/sec at an exposure time of 1.34 milli-sec.
![xpcs](https://user-images.githubusercontent.com/63168221/102576496-a8386280-40c3-11eb-8e24-ce4d84af2bb4.png)
# Step-2 - Experimental intensity autocorrelations with model function
I obtained the intensity autocorrelation functions, which is basically a measure of disorderness of the system stored as a Unique Indentification Number in the 2D detector after the experiment via autocorrelator algorithm. This was followed by data fitting using poiseuille model in the least squares sense to obtain experimental fitting parameters such as flow velocity and diffusion coefficient governed by Stokes-Einstein relationship.
![Experiment and model](https://user-images.githubusercontent.com/63168221/102647223-66470500-4133-11eb-96c5-55c822ed98b2.png)
# Step-3 - Solving inverse problem of obtaining flow profiles from XPCS data
 I used Fourier Mode decomposition method to decompose flow profiles into discrete sets of fourier modes and computed intensity autocorrelation functions numerically. This was followed by calculating mean squared error between numerically obtained autocorrelations with experimental data for the entire phase space of grid points. In our case, the solution space consists of 161050 nodes and the flow profile with lowest error is obtained from this phase space through this developed technique.
 
![Fourier modes](https://user-images.githubusercontent.com/63168221/102646760-9e017d00-4132-11eb-97c4-99b2e747db81.png)
