import numpy as np
import matplotlib.pyplot as plt
import sarbadhicary_scaling_module as s17lc

# ==========================================
# 1. Constants and Parameters
# ==========================================

RATE_SN = 0.02       # Galactic SN rate (events per year)
MAX_AGE = 100000     # Maximum theoretical observable age (years)
Z_SCALE = 80         # Vertical scale height (pc)
R_SUN = 8.5          # Solar Galactocentric distance (kpc)

N_ITERATIONS = 500000   # Number of Monte Carlo iterations
survivor_counts = [] # List to store the number of survivors from each run

# Legacy VLA Telescope Observational Limits (Reference Mask)
MIN_FLUX_mJY = 5.0          # 5 mJy/beam thermal noise floor
MIN_SIZE_ARCMIN = 1.0       # Minimum angular resolution (1 arcmin)
MAX_SIZE_ARCMIN = 15.0      # Maximum Recoverable Scale (MRS) (15 arcmin)
BEAM_SIZE_ARCMIN = 1.0      # Standardized beam resolution

# Lorimer 2006 rejection sampling function 
# We use "Rejection Sampling" to generate radii that follow the Lorimer 2006 probability curve.
# NOTE: This is adopted as a phenomenological proxy for the spatial distribution 
# of massive star progenitors, NOT an exact empirical mapping of SNRs.
def sample_lorimer_radius(n_samples):
    B, C = 1.9, 5.1
    R_peak = (B / C) * R_SUN
    max_pdf = (R_peak / R_SUN)**B * np.exp(-C * (R_peak - R_SUN) / R_SUN)
    
    samples = []
    while len(samples) < n_samples:
        # Guess random radii between 0 and 25 kpc
        R_guess = np.random.uniform(0, 25, n_samples) 
        y_guess = np.random.uniform(0, max_pdf, n_samples)
        
        # Lorimer PDF formula
        pdf_val = (R_guess / R_SUN)**B * np.exp(-C * (R_guess - R_SUN) / R_SUN)
        
        # Keep only the guesses that fall under the probability curve
        valid = R_guess[y_guess < pdf_val]
        samples.extend(valid)
        
    return np.array(samples[:n_samples])

# ==========================================
# 2. MONTE CARLO ITERATION LOOP
# ==========================================

print(f"Running Monte Carlo ensemble for {N_ITERATIONS} iterations...")

for i in range(N_ITERATIONS):
    # PHASE 1: STOCHASTIC BIRTH
    # Drawing the total number of physical SNRs from a Poisson distribution 
    # with a mean of 2,000 (0.02 * 100000) to accurately reflect statistical fluctuations.
    N_snrs = np.random.poisson(RATE_SN * MAX_AGE)
    
    # Ages are sampled uniformly over the 100,000 year interval
    ages_yr = np.random.uniform(0, MAX_AGE, N_snrs)
    
    # Phenomenological log-normal distribution for ISM density.
    # The 'mean' (-1.33) and 'sigma' (0.49) apply to the underlying normal 
    # distribution of ln(n_0), yielding the physical number density n_0 in cm^-3.
    densities_cm3 = np.random.lognormal(mean=-1.33, sigma=0.49, size=N_snrs)
    
    # Vertical Scale Height Assumption:
    # For simplicity, an 80-pc exponential scale height is adopted for the ENTIRE 
    # synthetic population. This is a simplifying assumption, as Type-Ia progenitors 
    # typically have a broader vertical distribution due to older stellar ages.
    z_heights_pc = np.random.exponential(scale=Z_SCALE, size=N_snrs) 
    z_heights_pc *= np.random.choice([1, -1], size=N_snrs)
    
    radii_kpc = sample_lorimer_radius(N_snrs)
    
    theta_rad = np.random.uniform(0, 2 * np.pi, N_snrs)
    x_kpc = radii_kpc * np.cos(theta_rad)
    y_kpc = radii_kpc * np.sin(theta_rad)
    z_kpc = z_heights_pc / 1000.0
    
    # Supernova Type Partitioning (80% Core-Collapse, 20% Type Ia)
    is_core_collapse = np.random.choice([True, False], p=[0.8, 0.2], size=N_snrs)
    
    cc_age = ages_yr[is_core_collapse]
    cc_rho = densities_cm3[is_core_collapse]
    ia_age = ages_yr[~is_core_collapse]
    ia_rho = densities_cm3[~is_core_collapse]
    
    m_ej_solar_masses = np.where(is_core_collapse, np.random.normal(loc=5.0, scale=1.5, size=N_snrs), 1.4)
    m_ej_cc = m_ej_solar_masses[is_core_collapse]
    m_ej_1a = m_ej_solar_masses[~is_core_collapse]

    # PHASE 2: HYDRODYNAMIC EVOLUTION & RADIO EMISSION
    e51 = 1.0      # Energy in units of 10^51 ergs
    p = 2.5        # Electron energy distribution index
    nu = 1.4e9     # Target frequency (1.4 GHz)
    epse = 0.0042  # Electron acceleration efficiency

    # Calculate Radius and Velocity using Truelove & McKee Characteristic Scales
    cc_rs, cc_vs = s17lc.radius_velocity(cc_age, cc_rho, m_ej_cc, e51, sntype='cc')
    ia_rs, ia_vs = s17lc.radius_velocity(ia_age, ia_rho, m_ej_1a, e51, sntype='ia')
    
    # Calculate approximate Radio Luminosity based on Sarbadhicary scaling
    cc_lum = s17lc.luminosity(cc_rs, cc_vs, cc_rho, epse, p, nu)
    ia_lum = s17lc.luminosity(ia_rs, ia_vs, ia_rho, epse, p, nu)  

    # Recombine arrays into single population
    rs = np.zeros(N_snrs)
    rs[is_core_collapse] = cc_rs
    rs[~is_core_collapse] = ia_rs
    
    lum = np.zeros(N_snrs)
    lum[is_core_collapse] = cc_lum
    lum[~is_core_collapse] = ia_lum
    
    rad_pc = rs
    rad_kpc = rs / 1000.0

    # PHASE 3: GEOMETRIC PROJECTION AND OBSERVATIONAL FILTERING
    d_kpc = np.sqrt((x_kpc - R_SUN)**2 + y_kpc**2 + z_kpc**2)
    
    # Apparent Angular Diameter (in arcminutes)
    ang_d_arcmin = ((rad_kpc * 2.0) / d_kpc) * 3437.75

    # Inverse Square Law Flux Calculation
    distances_m = d_kpc * 1000.0 * 3.086e16
    flux_W_m2_Hz = lum / (4.0 * np.pi * distances_m**2)
    flux_mJy = (flux_W_m2_Hz / 1e-26) * 1000.0

    # Beam Surface Brightness Correction
    # NOTE: Because a detailed shell brightness profile and convolution with the telescope 
    # beam are beyond the scope of this undergraduate model, the mean flux per beam 
    # is used as an approximate surface-brightness measure.
    n_beams = np.maximum(1.0, (ang_d_arcmin / BEAM_SIZE_ARCMIN)**2)
    flux_per_beam_mJy = flux_mJy / n_beams

    # Synthetic Legacy Telescope Mask (VLA Simulation)
    detected_mask = (
        (flux_per_beam_mJy > MIN_FLUX_mJY) & 
        (ang_d_arcmin > MIN_SIZE_ARCMIN) & 
        (ang_d_arcmin < MAX_SIZE_ARCMIN)
    )
    
    # Store the total number of survivors for this run
    survivor_counts.append(np.sum(detected_mask))

print(f"Simulation complete! Mean recovered remnants: {np.mean(survivor_counts):.1f} $\\pm$ {np.std(survivor_counts, ddof=1):.1f}. Plotting results...")

# ==========================================
# 3. PLOT IMAGES
# ==========================================

plt.figure(figsize=(9, 6))
plt.hist(survivor_counts, bins=12, color='royalblue', edgecolor='black', alpha=0.8)
plt.axvline(310, color='red', linestyle='--', linewidth=2, label="Dave Green's Catalog (~310)")
plt.axvline(np.mean(survivor_counts), color='darkorange', linestyle='-', linewidth=2, label=f"Mean Simulation Count ({np.mean(survivor_counts):.1f})")

plt.title("Simulation Results: Detected Supernova Remnants", fontsize=14)
plt.xlabel("Number of Detectable Remnants", fontsize=12)
plt.ylabel("Frequency (Number of Iterations)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()