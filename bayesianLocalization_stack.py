
import settings
import numpy as np
import matplotlib.pyplot as plt
from util import *
np.seterr(divide='ignore')
from copy import deepcopy

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
    print("START")
    print(particle)
    def gen_residual_noise_image():
        roi = particle.update_roi()
        data = model.data_zsum[roi[0]:roi[2],roi[1]:roi[3]]
        model_background = model.stack_zsum[roi[0]:roi[2],roi[1]:roi[3]]
        particle_img = particle.update_img() * model.depth # multiplied by model.depth to pretend it is ON in every frame.
        return data - model_background - particle_img

    ## STAGE 1 - estimating x and y with intensity fixed
    # Calculate probabilities
    residual_noise = gen_residual_noise_image()
    current_params = particle.a
    current_score = noise_frame_loglikelihood(residual_noise)
    # Note that in a typical MCMC algorithm, the 'score' would be the loglikelihood plus the prior probability of the
    # the parameters. Since the brightness parameter in our model is per-frame, i.e., it corresponds to the
    # observed nr. of photons in a single ON-state frame, and we start out by looking at the summed intensity
    # of the entire dataset/model stack, we initially leave out the prior term for efficiency. Note that the prior
    # probablitity distribution of x and y (a[0] and a[1]) is a constant - it would make no difference in the random
    # walk anyway.

    it = 0
    while it < settings.MAP_ESTIMATE_ITERATIONS:
        p0 = np.random.normal(loc=current_params[0], scale=settings.MAP_PROPOSAL_XY_SIGMA) # ap: proposed new parameter values a
        p1 = np.random.normal(loc=current_params[1], scale=settings.MAP_PROPOSAL_XY_SIGMA)
        p2 = np.random.normal(loc=current_params[2], scale=settings.MAP_PROPOSAL_RADIUS_SIGMA)
        p3 = current_params[3]
        proposed_params = [p0, p1, p2, p3]

        particle.a = proposed_params
        residual_noise = gen_residual_noise_image()
        proposal_score = noise_frame_loglikelihood(residual_noise)

        if proposal_score > current_score:
            current_score = proposal_score
            current_params = proposed_params
        else:
            particle.a = current_params

        it+=1

    print("INITIAL ESTIMATE")
    print(particle)
    # FIND SOME INITIAL MARKOV CHAIN BASED ON THE CURRENT PARAMS

    b = list()
    roi = particle.update_roi()
    particle_img = particle.update_img()

    for f in range(model.depth):
        data = model.data[roi[0]:roi[2], roi[1]:roi[3], f]
        model_background = model.stack[roi[0]:roi[2], roi[1]:roi[3], f]
        residual_particle_img = data - model_background
        off_state_residual = noise_frame_loglikelihood(residual_particle_img)
        on_state_residual = noise_frame_loglikelihood(residual_particle_img - particle_img)
        b.append(on_state_residual > off_state_residual)

    particle.b = b
    print("STATE SEQUENCE")
    print(particle.b)

    # FOR A SUBSET OF THE ON-STATE FRAMES, REFINE THE ESTIMATE FOR a BY RANDOM WALK
    # a) grab a subset of frames
    on_state_frames = np.where(particle.b)[0]
    print("SELECTED FRAMES")
    print(on_state_frames)
    selected_frames = np.random.choice(on_state_frames, size=min([len(on_state_frames), settings.MAP_REFINE_FRAMES_USED]), replace=False)
    # b) compute current score with that subset

    def score_particle():
        score = 0
        roi = particle.update_roi()
        particle_img = particle.update_img()
        for f in selected_frames:
            data = model.data[roi[0]:roi[2], roi[1]:roi[3], f]
            model_background = model.stack[roi[0]:roi[2], roi[1]:roi[3], f]
            residual = data - model_background - particle_img
            score += noise_frame_loglikelihood(residual) + logprior_a(particle)
        return score

    current_params = particle.a
    current_score = score_particle()
    print("Starting value:", current_params)
    # c) do the refinement
    it = 0
    PROPOSAL_SIGMA = [settings.MAP_REFINE_XY_SIGMA, settings.MAP_REFINE_XY_SIGMA, settings.MAP_REFINE_RADIUS_SIGMA, settings.MAP_REFINE_INTENSITY_SIGMA]
    while it < settings.MAP_REFINE_ITERATIONS:
        for i in range(4):
            proposed_params = deepcopy(current_params)
            proposed_params[i] += np.random.normal(loc = 0, scale = PROPOSAL_SIGMA[i])
            particle.a = proposed_params
            proposal_score = score_particle()
            if proposal_score > current_score:
                print("Found better params:", particle.a)
                current_score = proposal_score
                current_params = proposed_params
            else:
                particle.a = current_params
        it += 1