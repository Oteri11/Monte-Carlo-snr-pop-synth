# Galactic SNR Population Synthesis & Selection Effects

This repository contains the simulation code and radio-emission pipeline supporting the paper:

> **Monte Carlo Population Synthesis of Galactic Supernova Remnants: Selection Effects and the Missing-Remnant Problem**  
> Oghenekevwe Oteri and Oluwole Emmanuel Oyewande (*Department of Physics, University of Ibadan*)

---

## Overview

The Milky Way hosts only ~310 confirmed radio supernova remnants (SNRs), despite population arguments predicting thousands. This repository provides a forward Monte Carlo population-synthesis framework to investigate how observational selection criteria (surface-brightness thresholds and interferometric angular-scale limits) bias the observed Galactic SNR census.

### Simulation Workflow
1. **Stochastic Remnant Birth:** Samples a Poisson-distributed intrinsic population ($\sim 2{,}000$ remnants over $10^5\text{ yr}$ at $R_{\rm SN} = 0.02\text{ yr}^{-1}$) partitioned into Core-Collapse (80%) and Type Ia (20%).
2. **Galactic Distribution:** Positions remnants using a Lorimer et al. radial profile and an 80-pc exponential vertical scale height, with log-normal ambient ISM densities.
3. **Hydrodynamic Evolution & Radio Emission:** Calculates forward-shock radii and velocities via the Truelove & McKee (1999) framework and computes 1.4-GHz synchrotron luminosities using the corrected Sarbadhicary et al. (2017, 2019) semi-analytic prescription.
4. **Synthetic Observational Filter:** Projects remnants to heliocentric distances and applies an illustrative radio selection mask:
   - Beam surface brightness: $S_{\nu,\text{beam}} > 5\text{ mJy beam}^{-1}$
   - Angular diameter limits: $1' < \theta < 15'$

---

## Repository Structure

```text
├── snr_population_synthesis_simulation.py   # Main 500-iteration Monte Carlo driver script
├── s17lc.py                                # Shock dynamics & corrected Sarbadhicary synchrotron module
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
