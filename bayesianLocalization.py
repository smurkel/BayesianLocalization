from util import *

import simulator
import matplotlib.pyplot as plt
from scipy.stats import lognorm, uniform, norm

np.seterr(divide = 'ignore')

"""SET IMAGE AND MICROSCOPE PARAMETERS"""
PSF_INTENSITY_MU = 1000
PSF_INTENSITY_LOGNORM_S = 0.7
PSF_RADIUS_MU = 2.0 # mean of the gaussian psf's standard deviation in units of pixels.
PSF_RADIUS_LOGNORM_S = 0.1

NOISE_MEAN = 0
NOISE_SIGMA = 2.0

IMG_SIZE = 4
"""DEFINE MAP ESTIMATE BEHAVIOUR"""
MAP_ESTIMATE_ITERATIONS = 150
MAP_REFINE_ITERATIONS = 40
MAP_PROPOSAL_XY_SIGMA = 0.3
MAP_PROPOSAL_RADIUS_SIGMA = 0.1
MAP_PROPOSAL_INTENSITY_SIGMA = 50.0
MAP_REFINE_XY_SIGMA = 0.01
"""HESSIAN STUFF"""
HESSIAN_DELTA = [0.01, 0.01, 0.01, 0.01]
"""DEFINE THE NOISE MODEL - IN PRACTICE, THE NOISE MODEL WILL BE DERIVED FROM IMAGES; IT IS A GAUSSIAN PDF WITH MEAN AND STD TAKEN FROM BACKGROUND PIXEL INTENSITY."""
PDF_NOISE = norm(loc = NOISE_MEAN, scale = NOISE_SIGMA)
"""DEFINE PRIOR PROBABILITIES FOR X, Y, SIGMA, and INTENSITY."""
PDF_X = uniform(loc = -IMG_SIZE, scale = 2 * IMG_SIZE)
PDF_Y = uniform(loc = -IMG_SIZE, scale = 2 * IMG_SIZE)
PDF_SIGMA = lognorm(s = PSF_RADIUS_LOGNORM_S, scale = PSF_RADIUS_MU)
PDF_INTENSITY = lognorm(s = PSF_INTENSITY_LOGNORM_S, scale = PSF_INTENSITY_MU)
#################

def prior_x(val):
    # x prior probability is: flat probability, but it is somewhere in the image.
    return PDF_X.pdf(val)

def prior_y(val):
    # y prior probability is: flat probability, but it is somewhere in the image.
    return PDF_Y.pdf(val)

def prior_sigma(val):
    # sigma prior probability is: log normal distribution
    return PDF_SIGMA.pdf(val)

def prior_intensity(val):
    # intensity prior probability is lognormal.
    return PDF_INTENSITY.pdf(val)

def estimate_map(data, initial_suggestion = None):
    """This can be improved by starting the random walk not at a random sample from the prior distributions, but instead near the mode of the likelihood. Also: the data should be cropped such that the mode of the likelihood
    corresponds to the mean of the x and y prio dsitributions; i.e. crop with probable particle position in center.
    Useful lecture: https://www.youtube.com/watch?v=pHsuIaPbNbY&ab_channel=MLSSIceland2014"""
    parameter_samples = list()
    if initial_suggestion:
        x = initial_suggestion
    else:
        x = (PDF_X.rvs(1), PDF_Y.rvs(1), PDF_SIGMA.rvs(1), PDF_INTENSITY.rvs(1))
    p_x = likelihood(data, x) + prior_probability(*x)
    MAP = x
    max_log = p_x
    for i in range(MAP_ESTIMATE_ITERATIONS):
        print("ESTIMATING MAP iteration {}/{}".format(i, MAP_ESTIMATE_ITERATIONS))
        if p_x > max_log:
            MAP = x
            max_log = p_x
        # Step
        dx0 = np.random.normal(loc=0, scale=MAP_PROPOSAL_XY_SIGMA)
        dx1 = np.random.normal(loc=0, scale=MAP_PROPOSAL_XY_SIGMA)
        dx2 = np.random.normal(loc=0, scale=MAP_PROPOSAL_RADIUS_SIGMA)
        dx3 = np.random.normal(loc=0, scale=MAP_PROPOSAL_INTENSITY_SIGMA)

        # Perform step
        y = (x[0]+dx0, x[1]+dx1, x[2]+dx2, x[3]+dx3)

        # Calculate new score
        p_y = likelihood(data, y) + prior_probability(*y)
        R = p_y - p_x

        if R > 0:
            parameter_samples.append([y[0], y[1], y[2], y[3], p_y])
            x = y
            p_x = p_y
        else:
            if np.random.uniform(0, 1, 1) < np.exp(R):
               parameter_samples.append([y[0], y[1], y[2], y[3], p_y])
               x = y
               p_x = p_y
            else:
                parameter_samples.append([x[0], x[1], x[2], x[3], p_x])

    return MAP, parameter_samples

def refine_map(data, map):
    x = map
    Px = likelihood(data, x)  # prior_probability not checked, because only xpos and ypos change which are uniformly distributed.
    for i in range(MAP_REFINE_ITERATIONS):
        dpos = np.random.normal(loc=0, scale=MAP_REFINE_XY_SIGMA)
        if (i % 2) == 0:
            y = (x[0] + dpos, x[1], x[2], x[3])
        else:
            y = (x[0], x[1] + dpos, x[2], x[3])
        Py = likelihood(data, y) # prior_probability not checked, because only xpos and ypos change which are uniformly distributed.
        if Py > Px:
            x = y
            Px = Py
    return x


def prior_probability(x, y, sigma, intensity):
    return np.log(prior_x(x)) + np.log(prior_y(y)) + np.log(prior_sigma(sigma)) + np.log(prior_intensity(intensity))

def likelihood(data, parameters):
    # Data is an image with a gaussian spot and noise.
    # Parameters are of a model that describes a Gaussian spot.
    # Here:
    # 1 - 'simulate' the data according to parameters.
    model = simulator.Particle(*parameters).getImage(IMG_SIZE)
    # 2 - subtract model from data. what remains is noise.
    noise = data - model
    # 3 - evaluate the probability of observing this noise given the noise model
    log_probability = eval_noise(noise)
    return log_probability

def eval_noise(noise):
    N = noise.shape[0]
    log_probability = 0.0
    for i in range(N):
        for j in range(N):
                log_probability += np.log(PDF_NOISE.pdf(noise[i,j]))
    return log_probability


def hessian(data, params):
    def f(x):
        return likelihood(data, (x[0],x[1],x[2],x[3])) + prior_probability(x[0],x[1],x[2],x[3])
    Hessian = np.zeros((4,4))
    # Second derivatives first.
    x = np.zeros(4)
    x[0] = params[0]
    x[1] = params[1]
    x[2] = params[2]
    x[3] = params[3]

    for i in range(4):
        dxi = np.zeros(4)
        dxi[i] = HESSIAN_DELTA[i]
        for j in range(4):
            dxj = np.zeros(4)
            dxj[j] = HESSIAN_DELTA[j]
            temp = (f(x + dxi + dxj) - f(x + dxj - dxi) - f(x - dxj + dxi) + f(x + dxj - dxi) ) / (2 * HESSIAN_DELTA[i] * HESSIAN_DELTA[j])
            Hessian[i, j] = temp
    return Hessian

def laplace_approximation_value(data, MAP):
    MAP_logprob = likelihood(data, MAP) + prior_probability(*MAP)
    H = hessian(data, MAP)
    print(H)
    log_value = MAP_logprob + np.log(2 * np.pi ** (4/2)) - np.log(np.sqrt((-np.linalg.det(H))))
    return log_value

