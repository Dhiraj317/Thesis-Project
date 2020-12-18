# Dynamics of liquids in microchannels using high resolution spectroscopy
This project studies time-resolved changes using X-ray Photon Correlation Spectroscopy (XPCS), a scattering technique to study the dynamics of condensed matter systems. The aim of this project is to study the behavior of complex fluid flows with nanoparticle spheres in particular, flow velocity profiles from the X-ray data from 2D detector.
![xpcs](https://user-images.githubusercontent.com/63168221/102576496-a8386280-40c3-11eb-8e24-ce4d84af2bb4.png)

Step-1

First we obtain the intensity autocorrelation functions stored as a Unique Indentification Number in the 2D detector after the experiment.This is followed by data fitting using well established poiseuille model to obtain experimental fitting parameters such as flow velocity and diffusion coefficient governed by Stokes-Einstein relationship.
![Experiment and model](https://user-images.githubusercontent.com/63168221/102575936-5f33de80-40c2-11eb-93d6-dfca52fd9f24.png)

Step-2

Solving inverse problem of obtaining flow profiles from XPCS data. I used Fourier Mode decomposition method to decompose flow profiles into discrete sets of fourier modes and computing intensity autocorrelation functions numerically. This followed by calculating mean squared error between numerically obtained autocorrelations with experimental data for the entire phase space of grid points. In our case, the solution space consists of 161050 nodes and the flow profile with lowest error is obtained from this phase space through this developed technique.  
![Fourier modes](https://user-images.githubusercontent.com/63168221/102646760-9e017d00-4132-11eb-97c4-99b2e747db81.png)
