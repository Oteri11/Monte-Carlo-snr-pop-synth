#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#
# Supplementary Python Program to calculate radio light curves
# of supernova remnants from ejecta-dominated through Sedov-Taylor
# stages.
#
# Paper: Sarbadhicary et al (2017), MNRAS, 464, 2326
# Erratum to Paper: https://doi.org/10.1093/mnras/stz1490
#
# Creators: S. Sarbadhicary, L. Chomiuk (MSU)
# Modifiers: O. Oteri (UoI) - Integrated erratum constants.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

import numpy as np

#<><><><><><><><><> F U N C T I O N S <><><><><><><><><><><><>#

def radius_velocity(t, n0, mej, e51, sntype='cc'):
    """
    Returns the shock radius and velocity for a given density,
    energy and ejecta mass using the characteristic scales and 
    Sedov-Taylor solution described by Truelove and McKee (1999).

    Parameters:
    ----------
    t : [float, list, ndarray] age of SNR (in years)
    n0 : [float, list, ndarray] density (atoms/cc)
    e51 : [float, list, ndarray] energy (in units of 10^51 ergs)
    mej : [float, list, ndarray] ejecta mass (in units of Msun)
    sntype : ['ia' or 'cc']
    """

    t = np.asarray(t)
    n0 = np.asarray(n0)
    mej = np.asarray(mej)
    e51 = np.asarray(e51)

    # Characteristic variables - Truelove & McKee (1999)
    tch = 423*(e51**(-0.5))*(mej**(5.0/6.0))*(n0**(-1.0/3.0)) # years
    rch = 3.07*(mej**(1.0/3.0))*(n0**(-1.0/3.0)) # pcs
    vch = 7090*(e51**0.5)*(mej**(-0.5)) # km/s
    
    # Check whether to apply Ia or CC physics
    if (sntype=='ia'):
        tstar_st = 0.481
        t_st0 = tstar_st*tch
        tstar = t/tch
        if np.isscalar(tstar):
            vstar = 0.805*tstar**(-(3.0/10.0)) if t<t_st0 else 0.569*((1.42*tstar - 0.2935)**(-3.0/5.0))
            rstar = 1.15*tstar**(7.0/10.0) if t<t_st0 else (1.42*tstar - 0.2935)**(2.0/5.0)
        else:
            w = np.where(t>=t_st0)
            vstar = 0.805*tstar**(-(3.0/10.0))
            vstar[w] = 0.569*((1.42*tstar[w] - 0.2935)**(-3.0/5.0))
            rstar = 1.15*tstar**(7.0/10.0)
            rstar[w] = (1.42*tstar[w] - 0.2935)**(2.0/5.0)
        vs = vstar*vch  # in km/s
        rs = rstar*rch  # in pc
    
    elif(sntype=='cc'):
        tstar_st = 0.424
        t_st0 = tstar_st*tch
        tstar = t/tch
        if np.isscalar(tstar):
            vstar = 0.906*tstar**(-(1.0/4.0)) if t<t_st0 else 0.569*((1.42*tstar - 0.28)**(-3.0/5.0))
            rstar = 1.20906*tstar**(3.0/4.0) if t<t_st0 else (1.42*tstar - 0.28)**(2.0/5.0)
        else:
            w = np.where(t>=t_st0)
            vstar = 0.906*tstar**(-(1.0/4.0))
            vstar[w] = 0.569*((1.42*tstar[w] - 0.28)**(-3.0/5.0))
            rstar = 1.20906*tstar**(3.0/4.0)
            rstar[w] = (1.42*tstar[w] - 0.28)**(2.0/5.0)
        vs = vstar*vch  # in km/s
        rs = rstar*rch  # in pc

    return (rs, vs)

def luminosity(rs, vs, n0, epse, p, nu, b0ref=9.0):
    """
    Calculates a simplified radio-luminosity scaling motivated by the shock-energy 
    processing rate discussed by Sarbadhicary et al. (2017), utilizing the corrected 
    upstream magnetic-field constants published in their 2019 erratum.

    Parameters:
    ----------
    rs: [float, list, ndarray] shock radius (in pc)
    vs: [float, list, ndarray] shock velocity (in km/s)
    n0: [float, list, ndarray] ISM density (in atoms/cc)
    epse: [float, list, ndarray] electron acceleration efficiency 
    p: [float, list, ndarray] electron spectral index
    nu: [float, list, ndarray] observing frequency (in Hz)
    b0ref: [float, list, ndarray] magnetic field at density n0=1 (in uG)
    """

    rs = np.asarray(rs)
    vs = np.asarray(vs)
    n0 = np.asarray(n0)
    epse = np.asarray(epse)
    p = np.asarray(p)
    nu = np.asarray(nu)
    b0ref = np.asarray(b0ref)

    #~~~~~~ Erratum Constants ~~~~~~~~~~~~~~~#
    ef = 0.38  # Emission filling factor
    c1 = 6.27e18 # Pacholyzyk (1980) constants. 
    c5 = 9.68e-24 
    c6 = 8.1e-41
    mu = 1.4 
    mp = 1.67e-24 # g
    rho0 = mu*mp*n0 # density in g/cm^3
    me =  9.11e-28  # in grams
    c = 3.0e10 # cm/s
    eta = 4. # compression factor
    Em = me*c**2 # Rest energy of electrons
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #~~~~~~ CGS units for r,v  ~~~~~~~~~~~~~~~#
    rs_cgs = rs*3.086e18 # cm
    vs_cgs = vs*1.0e5 # cm/s

    #~~~~~~ Electron Acceleration ~~~~~~~~~~~~~~~#
    N_0 = (p-2.)*epse*rho0*(vs_cgs**2)*(Em**(p-2))

    #~~~~~~ Magnetic Field Processing (Eq A4, A7, A3, A8) ~~~~~~~~#
    B0 = (b0ref*1.0e-6)*n0**0.47 
    vA = B0/np.sqrt(4.*3.14*rho0) # Alfven Velocity
    MA = vs_cgs/vA # Alfven Mach Number
    
    epsCR = 0.01*(0.15*MA + 6.)  # Cosmic ray acceleration efficiency
    epsCR = np.clip(epsCR,None,0.1) 
    epsub = (epsCR/2.)*((vs_cgs/c) + (1./MA)) 
    Bu = np.sqrt(8.*3.14*epsub*rho0*vs_cgs**2) 
    B = Bu*np.sqrt((1. + 2.*(eta**2))/3.) 
    B = np.clip(B,4.*B0,None) # SNR magnetic field must be at least a simple compression of the ISM field (B0)

    #~~~~~~ Luminosity Calculation ~~~~~~~~~~~~~~~~~~~#
    s = ef*(4./3.)*(rs_cgs) 
    nu1 = 2.*c1*((s*c6*N_0)**(2./(p+4)))*(B**((p+2)/(p+4))) 
    
    L1 = 4.0*(3.14**2)*(rs_cgs**2)*(B**(-0.5))*(c5/c6) 
    L2 = 1 - np.exp(-(nu/nu1)**(-(p+4)/2.))

    # Avoid numerical errors for nu >> nu1 later in SNR age
    if np.isscalar(L2):
        if nu>20.*nu1:
            L2 = ((nu/nu1)**(-1.0*(p+4)/2.))
    else:
        L2[nu>20.*nu1] = ((nu/nu1)**(-1.0*(p+4)/2.))[nu>20.*nu1]
    L3 = (nu/(2.*c1))**(5./2.)

    return L1*L2*L3 

#<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>#