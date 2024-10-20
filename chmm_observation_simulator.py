from LSIM import lsim
import numpy as np
import pandas as pd

# %%

C = 4
channel_obs_dim = [4, 5, 2, 3]
channel_state_num = [3, 4, 5, 3]
num_gmm_component = [4, 1, 3, 2]

C = 3
channel_obs_dim = [4, 1, 3]
channel_state_num = [3, 4, 5]
num_gmm_component = [1, 2, 3]

T = 100
from_lsim = False

# %%
current_lsim = lsim(C, channel_state_num, channel_obs_dim, num_gmm_component)
current_lsim.reinit_random_para()
obs, latent_states = current_lsim.generate_time_series(from_lsim, T)
current_lsim.compute_joint_state_observation_prob(obs, latent_states)
current_lsim._gmm_pdf_fast(obs)

P_O_model, alpha, beta, alpha_T, b_c_ot_nc, P_observ_cond_to_state, P_observ_cond_to_state_comp = current_lsim.forward_backward_lsim(obs)

#%%

obs1, latent_states = current_lsim.generate_time_series(T)
obs1.columns = pd.MultiIndex.from_product([['obs:0001'],np.arange(obs1.columns.shape[0])])

obs2, latent_states = current_lsim.generate_time_series(T+10)
obs2.columns = pd.MultiIndex.from_product([['obs:0002'],np.arange(obs2.columns.shape[0])])

obs3, latent_states = current_lsim.generate_time_series(T-15)
obs3.columns = pd.MultiIndex.from_product([['obs:0003'],np.arange(obs3.columns.shape[0])])

frames = [obs1,obs2,obs3]
obs = pd.concat(frames,axis=1)

max_itration = 100
extra_options = {'plot': True, 'check_convergence': True, 'time_series':True}
current_lsim.em_lsim(obs, max_itration, extra_options)

# %%

current_lsim.plot_chmm_timeseries(obs, 1)
current_lsim.plot_chmm_timeseries(obs, 2)
current_lsim.plot_chmm_timeseries(obs, 3)

# %%

C = 4
channel_obs_dim = [4, 5, 2, 3]
channel_state_num = [3, 4, 5, 4]
num_gmm_component = [1, 1, 1, 1]
T = 100
from_lsim = False

current_lsim_test = lsim(C, channel_state_num, channel_obs_dim, num_gmm_component)
eq_hmm_test = current_lsim_test.chmm_cart_prod()

# %%

# C = 3
# channel_obs_dim = [4, 5, 2]
# channel_state_num = [3, 3, 3]
# T = 100
#
#
# temp_1 = np.random.randn(channel_obs_dim[0], T)
# temp_2 = 1 + np.random.randn(channel_obs_dim[1], T)
# temp_3 = 3 + np.random.randn(channel_obs_dim[2], T)
#
#
# channels_state_name = current_lsim.parameters.channels_state_name
# dimension_of_channels = current_lsim.parameters.dimension_of_channels
# channels_name_unique = current_lsim.parameters.channels_name_unique
# channels_dim_name = current_lsim.parameters.channels_dim_name
#
# time_series = pd.DataFrame(np.concatenate([temp_1, np.concatenate([temp_2, temp_3])]),
#                            index=[channels_dim_name, dimension_of_channels])

# %%

# aa = time_series.transpose().cov()  # np.dot(time_series, time_series.transpose())
# aa_exp = pd.DataFrame(np.exp(aa), index=[channels_dim_name, dimension_of_channels])
# aa_hat = np.log(aa_exp)
# # bb_hat = math.log1p(aa_exp) error
# diff_0 = np.subtract(aa, aa_hat)


# %%
