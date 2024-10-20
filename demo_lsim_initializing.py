from LSIM import lsim
import numpy as np
import pandas as pd

# %% This section indicates how to initialize LSIM parameters by prioir determinstic values

C = 2
channel_obs_dim = [1, 2]
channel_state_num = [3, 2]
num_gmm_component = [1, 1]

# %%
lsim_init_determinstic = lsim(C, channel_state_num, channel_obs_dim, num_gmm_component)

lsim_params = lsim_init_determinstic.parameters

C = lsim_params.C

ch_names = lsim_params.channels_name_unique
state_names = lsim_params.states_name_unique
gmm_names = lsim_params.gmm_names_unique

# it is better coupling column be normalized to sum to one, but if not we normalize each column in next line
lsim_params.coupling_theta_IM = np.array([[3,1],[3,2]])
lsim_params.coupling_theta_IM = lsim_params.coupling_theta_IM / lsim_params.coupling_theta_IM.sum(axis=0, keepdims=1)

#######################################################################################
# Init first LSIM channel with 3 states
zee = 0

# Init first state of first LSIM channel
i = 0; k = 0
lsim_params.gmm_para_mu.loc[ch_names[zee], (state_names[i], gmm_names[k])] = np.array([-1],dtype=np.float64)
lsim_params.gmm_para_sigma.loc[ch_names[zee], (state_names[i], gmm_names[k])] = np.array([0.1],dtype=np.float64)
lsim_params.gmm_para_P.loc[ch_names[zee], (state_names[i], gmm_names[k])] = 1

# Init second state of first LSIM channel
i = 1; k = 0
lsim_params.gmm_para_mu.loc[ch_names[zee], (state_names[i], gmm_names[k])] = np.array([1],dtype=np.float64)
lsim_params.gmm_para_sigma.loc[ch_names[zee], (state_names[i], gmm_names[k])] = np.array([0.2],dtype=np.float64)
lsim_params.gmm_para_P.loc[ch_names[zee], (state_names[i], gmm_names[k])] = 1

# Init third state of first LSIM channel
i = 2; k = 0
lsim_params.gmm_para_mu.loc[ch_names[zee], (state_names[i], gmm_names[k])] = np.array([3],dtype=np.float64)
lsim_params.gmm_para_sigma.loc[ch_names[zee], (state_names[i], gmm_names[k])] = np.array([0.15],dtype=np.float64)
lsim_params.gmm_para_P.loc[ch_names[zee], (state_names[i], gmm_names[k])] = 1

# Init init-state probabilities
lsim_params.pi_0.loc[ch_names[zee]] = np.array([[0.25],[0.25],[0.5]],dtype=np.float64)

# Init C transition matrices for the first channel
lsim_params.transition_matrices.loc[ch_names[0], ch_names[zee]] = np.array([[0.98,0.01,0.01],[0.01,0.98,0.01],[0.1,0.1,0.8]],dtype=np.float64)
lsim_params.transition_matrices.loc[ch_names[1], ch_names[zee]] = np.array([[0.98,0.01,0.01],[0.01,0.98,0.01]],dtype=np.float64)

#######################################################################################
# Init second LSIM channel with 2 states with 2-dim observations
zee = 1

# Init first state of second LSIM channel
i = 0; k = 0
lsim_params.gmm_para_mu.loc[ch_names[zee], (state_names[i], gmm_names[k])] = np.array([[-1],[-1]],dtype=np.float64)
lsim_params.gmm_para_sigma.loc[ch_names[zee], (state_names[i], gmm_names[k])] = np.array([[0.1],[0.1]],dtype=np.float64)
lsim_params.gmm_para_P.loc[ch_names[zee], (state_names[i], gmm_names[k])] = 1

# Init second state of second LSIM channel
i = 1; k = 0
lsim_params.gmm_para_mu.loc[ch_names[zee], (state_names[i], gmm_names[k])] = np.array([[1],[1]],dtype=np.float64)
lsim_params.gmm_para_sigma.loc[ch_names[zee], (state_names[i], gmm_names[k])] = np.array([[0.2],[0.2]],dtype=np.float64)
lsim_params.gmm_para_P.loc[ch_names[zee], (state_names[i], gmm_names[k])] = 1


# Init init-state probabilities
lsim_params.pi_0.loc[ch_names[zee]] = np.array([[0.25],[0.75]],dtype=np.float64)

# Init C transition matrices for the second channel
lsim_params.transition_matrices.loc[ch_names[0], ch_names[zee]] =  np.array([[0.97,0.03],[0.05,0.95],[0.01,0.99]],dtype=np.float64)
lsim_params.transition_matrices.loc[ch_names[1], ch_names[zee]] = np.array([[0.98,0.02],[0.01,0.99]],dtype=np.float64)


# set LSIM parameters with new values
lsim_init_determinstic.parameters = lsim_params

# generate observation sequences from the model
obs, latent_states = lsim_init_determinstic.generate_time_series(250) # generate 250 samples

# performing Viterbi decoding
P_star_model, X_star = lsim_init_determinstic.viterbi_lsim(obs)

lsim_init_determinstic.plot_chmm_timeseries(obs, 1)
lsim_init_determinstic.plot_chmm_timeseries(obs, 2)

#plot Viterbi path for the first channel
lsim_init_determinstic.plot_chmm_timeseries(X_star, 1)
