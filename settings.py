from scipy.stats import lognorm, uniform, norm
import markov
############
# SETTINGS #
############
CROP_SIZE = 4
"""DEFINE MAP ESTIMATE BEHAVIOUR"""
MAP_ESTIMATE_ITERATIONS = 100 # amount of random walk iterations to find an initial estimate for a
MAP_REFINE_ITERATIONS = 100 # amount of iterations performed to refine the initial estimate for a.
MAP_REFINE_FRAMES_USED = 3 # the number of on-state frames used in the map refinement. More frmaes = better result, but longer computation.
MAP_PROPOSAL_XY_SIGMA = 0.3
MAP_PROPOSAL_RADIUS_SIGMA = 0.1
MAP_PROPOSAL_INTENSITY_SIGMA = 50.0

MAP_REFINE_XY_SIGMA = 0.05
MAP_REFINE_RADIUS_SIGMA = 0.2
MAP_REFINE_INTENSITY_SIGMA = 50.0
NUMERICAL_DERIVATIVE_DELTA = [0.01, 0.01, 0.01, 0.01]
MARKOV_SAMPLES = 20
############
### DATA ###
############
"""PSF parameters"""
PSF_INTENSITY_MU = 1000
PSF_INTENSITY_LOGNORM_S = 1.0
PSF_RADIUS_MU = 1.0 # mean of the gaussian psf's standard deviation in units of pixels.
PSF_RADIUS_LOGNORM_S = 0.1
"""Noise model"""
NOISE_MEAN = 0
NOISE_SIGMA = 2.0
PDF_NOISE = norm(loc = NOISE_MEAN, scale = NOISE_SIGMA)
##############
### PRIORS ###
##############
"""DEFINE PRIOR PROBABILITIES FOR X, Y, SIGMA, and INTENSITY."""
PRIOR_PDF_X = 1 / (2 * CROP_SIZE)
PRIOR_PDF_Y = 1 / (2 * CROP_SIZE)
PRIOR_PDF_SIGMA = lognorm(s = PSF_RADIUS_LOGNORM_S, scale = PSF_RADIUS_MU)
PRIOR_PDF_INTENSITY = lognorm(s = PSF_INTENSITY_LOGNORM_S, scale = PSF_INTENSITY_MU)
"""DEFINE MARKOV MODEL TRANSITION PROBABILITIES"""
P_ON = 0.3
P_OFF = 0.2
P_BLEACH = 0.2
P_FALSE_EMISSION = 0.0
HMM = markov.GetHMM(P_ON,P_OFF,P_BLEACH,P_FALSE_EMISSION)
"""GLOBALS - values depend on data"""
MODEL_WIDTH = 0
MODEL_HEIGHT = 0
MODEL_DEPTH = 0