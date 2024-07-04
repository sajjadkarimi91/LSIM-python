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


max_state_num = np.max(lsim_params.channel_state_num)
max_gmm_num = np.max(lsim_params.num_gmm_component)
dimension_numbers_reshape = np.concatenate([[0], np.cumsum(lsim_params.channel_obs_dim)])
dimension_numbers_index = np.concatenate([[-1], np.cumsum(lsim_params.channel_obs_dim) - 1])
state_numbers_index = np.concatenate([[-1], np.cumsum(lsim_params.channel_state_num) - 1])
state_names = lsim_params.states_name_unique.copy()
sum_states = state_numbers_index[-1] + 1
ch_names = lsim_params.channels_name_unique
state_names = lsim_params.states_name_unique
gmm_names = lsim_params.gmm_names_unique


lsim_params.coupling_theta_IM = np.array([[3,2],[1,8]]) # it is better coupling colomn be normalized to sum to one, but if not we normalize each colomn in next line
lsim_params.coupling_theta_IM = lsim_params.coupling_theta_IM / lsim_params.coupling_theta_IM.sum(axis=0, keepdims=1)

# loop on all LSIM channels
for zee in range(C):
    for i in range(lsim_params.channel_state_num[zee]): #loop on each state per channel

        temp_sigma = 1
        temp_mu = 1
        P = 1

        for k in range(1, lsim_params.num_gmm_component[zee]):  #loop on each GMM per hidden state
            lsim_params.gmm_para_mu.loc[ch_names[zee], (state_names[i], gmm_names[k])] = temp_mu[i, :]
            lsim_params.gmm_para_sigma.loc[ch_names[zee], (state_names[i], gmm_names[k])] = temp_sigma[i, :]
            lsim_params.gmm_para_P.loc[ch_names[zee], (state_names[i], gmm_names[k])] = P[i, :]

        lsim_params.gmm_para_P.loc[ch_names[zee], state_names[i]] = 1
        lsim_params.transition_matrices.loc[:, ch_names[zee]] = 1#  np.array([0,1;1,0])


# set LSIM parameters
lsim_init_determinstic.parameters = lsim_params

# generate observation sequences from the model
lsim_init_determinstic.parameters = lsim_params