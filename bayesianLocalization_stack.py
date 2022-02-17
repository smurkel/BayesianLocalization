
import settings
import numpy as np
import matplotlib.pyplot as plt
from util import *
np.seterr(divide='ignore')

def logprior_a(particle):
    logp_a0 = settings.PRIOR_PDF_X
    logp_a1 = settings.PRIOR_PDF_X
    logp_a2 = settings.PRIOR_PDF_SIGMA.logpdf(particle.a[2])
    logp_a3 = settings.PRIOR_PDF_INTENSITY.logpdf(particle.a[3])
    return logp_a0 + logp_a1 + logp_a2 + logp_a3

def frame_loglikelihood(data_crop_frame, model_crop_frame, particle, particle_state_frame):
    residual_noise = data_crop_frame - model_crop_frame
    if particle_state_frame:
        residual_noise -= particle.img
    return noise_frame_loglikelihood(residual_noise)

def loglikelihood(data_crop, model_crop, particle):
    nx, ny, nf = data_crop.shape
    loglikelihood = 0.0
    for f in range(nf):
        loglikelihood += frame_loglikelihood(data_crop[:, :,f], model_crop[:, :, f], particle, particle.b[f])
    return loglikelihood

def noise_frame_loglikelihood(noise_frame):
    nx, ny = noise_frame.shape
    loglikelihood_frame = 0.0
    for i in range(nx):
        for j in range(ny):
            loglikelihood_frame += settings.PDF_NOISE.logpdf(noise_frame[i, j])
    return loglikelihood_frame

def noise_loglikelihood(noise):
    nx, ny, nf = noise.shape
    loglikelihood = 0.0
    for frame in range(nf):
        loglikelihood += noise_frame_loglikelihood(noise[:, :, frame])
    return loglikelihood


def maximum_a_posteriori_estimate(model, particle):
    pass