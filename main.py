from util import *

import simulator
import matplotlib.pyplot as plt
from scipy.stats import lognorm, uniform, norm

np.seterr(divide = 'ignore')

img = Load("spot_x1_y2_s2_i2000_n8.tiff")

"""SET IMAGE AND MICROSCOPE PARAMETERS"""
PSF_INTENSITY_MU = 1000
PSF_INTENSITY_LOGNORM_S = 0.7
PSF_RADIUS_MU = 1.7 # mean of the gaussian psf's standard deviation in units of pixels.
PSF_RADIUS_LOGNORM_S = 0.4

NOISE_MEAN = 0
NOISE_SIGMA = 2.0

IMG_SIZE = 5
"""DEFINE MCMC BEHAVIOUR"""
MCMC_ITERATIONS = 1000
MCMC_BURN = 500
MCMC_XY_SIGMA = 0.5
MCMC_RADIUS_SIGMA = 0.5
MCMC_INTENSITY_SIGMA = 150.0
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
    if val > 0:
        return PDF_SIGMA.pdf(val)
    else:
        return 0

def prior_intensity(val):
    # intensity prior probability is lognormal.
    if val > 0:
        return PDF_INTENSITY.pdf(val)
    else:
        return 0

def mcmc_posterior_pdf(data):

    parameter_samples = list()
    x = (PDF_X.rvs(1), PDF_Y.rvs(1), PDF_SIGMA.rvs(1), PDF_INTENSITY.rvs(1))
    p_x = likelihood(data, x) + prior_probability(*x)
    for i in range(MCMC_ITERATIONS):
        print("Iteration #{}".format(i))
        # Step
        dx0 = np.random.normal(loc=0, scale=MCMC_XY_SIGMA)
        dx1 = np.random.normal(loc=0, scale=MCMC_XY_SIGMA)
        dx2 = np.random.normal(loc=0, scale=MCMC_RADIUS_SIGMA)
        dx3 = np.random.normal(loc=0, scale=MCMC_INTENSITY_SIGMA)

        # Perform step
        y = (x[0]+dx0, x[1]+dx1, x[2]+dx2, x[3]+dx3)

        # Calculate new score
        p_y = likelihood(data, y) + prior_probability(*y)
        R = p_y - p_x
        if R > 0:
            parameter_samples.append(y)
            x = y
            p_x = p_y
        else:
            if np.random.uniform(0, 1, 1) > np.exp(R):
                parameter_samples.append(y)
                x = y
                p_x = p_y
            else:
                parameter_samples.append(x)
    return parameter_samples[MCMC_BURN:]

def prior_probability(x, y, sigma, intensity):
    return np.log(prior_x(x)) + np.log(prior_y(y)) + np.log(prior_sigma(sigma)) + np.log(prior_intensity(intensity))

def likelihood(data, parameters):
    # Data is an image with a gaussian spot and noise.
    # Parameters are of a model that describes a Gaussian spot.
    # Here:
    # 1 - 'simulate' the data according to parameters.
    model = simulator.particle(IMG_SIZE, *parameters)
    # 2 - subtract model from data. what remains is noise.
    noise = data - model
    # 3 - evaluate the probability of observing this noise given the noise model
    log_probability = eval_noise(noise)
    return False


def eval_noise(noise):
    N = noise.shape[0]
    log_probability = 0.0
    for i in range(N):
        for j in range(N):
                log_probability += np.log(PDF_NOISE.pdf(noise[i,j]))
    return log_probability

data = simulator.particle(IMG_SIZE, 0, 0, 2, 5000.0) + simulator.noise(IMG_SIZE, 0, 0.2)

prms = mcmc_posterior_pdf(data)
prms = np.asarray(prms)


n_bins = 30
x = prms[:, 0] + IMG_SIZE
y = prms[:, 1] + IMG_SIZE
s = prms[:, 2]
i = prms[:, 3]
plt.subplot(4,1,1)
plt.hist(x, bins = n_bins)
plt.title("X")
plt.subplot(4,1,2)
plt.hist(y, bins = n_bins)
plt.title("Y")
plt.subplot(4,1,3)
plt.hist(s, bins = n_bins)
plt.title("S")
plt.subplot(4,1,4)
plt.hist(i, bins = n_bins)
plt.title("I")

plt.figure(2)
n = np.shape(x)[0]
c = np.zeros((n, 4))
c[:, 0] = np.linspace(0, 1, n)
c[:, 1] = np.linspace(1, 0, n)
c[:, 3] = np.zeros(n) + 0.5
plt.imshow(data)
plt.plot(x, y, linewidth = 1, color = (0.0, 0.0, 0.0))
for j in range(n):
    plt.plot(x[j], y[j], markerfacecolor = c[j], marker = 'o', markeredgecolor = (1.0, 1.0, 1.0), markersize = 5)

plt.figure(3)
plt.subplot(1,3,1)
plt.imshow(data)
plt.subplot(1,3,2)
final_model = simulator.particle(IMG_SIZE, x[-1], y[-1], s[-1], i[-1])
plt.imshow(final_model)
plt.title("FInal model")
plt.subplot(1,3,3)
plt.imshow(simulator.particle(IMG_SIZE,x[0],y[0],s[0],i[0]))
plt.title("First non burn model")

plt.figure(4)
plt.subplot(1,4,1)
plt.plot(x)
plt.subplot(1,4,2)
plt.plot(y)
plt.subplot(1,4,3)
plt.plot(s)
plt.subplot(1,4,4)
plt.plot(i)
plt.show()