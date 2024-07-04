from LSIM import lsim
import numpy as np
import pandas as pd

# %% A sample code for initializing LSIM


C = 3
channel_obs_dim = [4, 1, 3]
channel_state_num = [3, 4, 5]
num_gmm_component = [1, 2, 3]

T = 1000
from_lsim = True


test_lsim = lsim(C, channel_state_num, channel_obs_dim, num_gmm_component)

# test_lsim.reinit_random_para()


# %% A s data generation


obs1, latent_states = test_lsim.generate_time_series(T)
obs1.columns = pd.MultiIndex.from_product([['obs:0001'],np.arange(obs1.columns.shape[0])])

obs2, latent_states = test_lsim.generate_time_series(T+10)
obs2.columns = pd.MultiIndex.from_product([['obs:0002'],np.arange(obs2.columns.shape[0])])

obs3, latent_states = test_lsim.generate_time_series(T-15)
obs3.columns = pd.MultiIndex.from_product([['obs:0003'],np.arange(obs3.columns.shape[0])])

# %% EM algorithm for fitting LSIM model on observations
# data and finally forward-backward inference from observation

frames = [obs1,obs2,obs3]
obs = pd.concat(frames,axis=1)

max_itration = 10
extra_options = {'plot': True, 'check_convergence': True, 'time_series':True}
test_lsim.em_lsim(obs, max_itration, extra_options)

# %% data and finally forward-backward inference for trained LSIM model on first observation

P_O_model, alpha, beta, alpha_T, b_c_ot_nc, P_observ_cond_to_state, P_observ_cond_to_state_comp = test_lsim.forward_backward_lsim(obs1)

# %% Equivalent HMM  for given LSIM model
eq_hmm_test = test_lsim.chmm_cart_prod()



