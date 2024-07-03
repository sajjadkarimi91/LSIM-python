from LSIM import lsim
import numpy as np
import pandas as pd

# %%


C = 3
channel_obs_dim = [4, 1, 3]
channel_state_num = [3, 4, 5]
num_gmm_component = [1, 2, 3]

T = 1000
from_lsim = True

# %%
test_lsim = lsim(C, channel_state_num, channel_obs_dim, num_gmm_component)

# test_lsim.reinit_random_para()


#%%

obs1, latent_states = test_lsim.generate_time_series(T)
obs1.columns = pd.MultiIndex.from_product([['obs:0001'],np.arange(obs1.columns.shape[0])])

obs2, latent_states = test_lsim.generate_time_series(T+10)
obs2.columns = pd.MultiIndex.from_product([['obs:0002'],np.arange(obs2.columns.shape[0])])

obs3, latent_states = test_lsim.generate_time_series(T-15)
obs3.columns = pd.MultiIndex.from_product([['obs:0003'],np.arange(obs3.columns.shape[0])])

frames = [obs1,obs2,obs3]
obs = pd.concat(frames,axis=1)

max_itration = 10
extra_options = {'plot': True, 'check_convergence': True, 'time_series':True}
test_lsim.em_lsim(obs, max_itration, extra_options)

eq_hmm_test = test_lsim.chmm_cart_prod()

P_O_model, alpha, beta, alpha_T, b_c_ot_nc, P_observ_cond_to_state, P_observ_cond_to_state_comp = test_lsim.forward_backward_lsim(obs1)



C = 2
channel_obs_dim = [2, 3]
channel_state_num = [3, 2]
num_gmm_component = [1, 2]

T = 1000
from_lsim = True

# %%
test_lsim = lsim(C, channel_state_num, channel_obs_dim, num_gmm_component)

lsim_params = test_lsim.parameters

C = lsim_params.C
lsim_para = lsim_params

max_state_num = np.max(lsim_para.channel_state_num)
max_gmm_num = np.max(lsim_para.num_gmm_component)
dimension_numbers_reshape = np.concatenate([[0], np.cumsum(lsim_para.channel_obs_dim)])
dimension_numbers_index = np.concatenate([[-1], np.cumsum(lsim_para.channel_obs_dim) - 1])
state_numbers_index = np.concatenate([[-1], np.cumsum(lsim_para.channel_state_num) - 1])
state_names = lsim_params.states_name_unique.copy()
sum_states = state_numbers_index[-1] + 1
ch_names = lsim_para.channels_name_unique
state_names = lsim_para.states_name_unique
gmm_names = lsim_para.gmm_names_unique

# C = lsim_para.C
flag_EM = True

# GMM initialization
for zee in range(C):

    # temp_sigma = clf.covariances_.copy()
    # temp_mu = clf.means_.copy()
    # P = clf.weights_.copy()

    temp_sigma = 1
    temp_mu = 1
    P = 1

    for i in range(lsim_para.channel_state_num[zee]):

        temp_mu_gmm = temp_mu[0, :]
        lsim_para.gmm_para_mu.loc[ch_names[zee], (state_names[i], gmm_names[0])] = temp_mu[0, :]
        lsim_para.gmm_para_sigma.loc[ch_names[zee], (state_names[i], gmm_names[0])] = temp_sigma[0, :]
        lsim_para.gmm_para_P.loc[ch_names[zee], (state_names[i], gmm_names[0])] = P[0]

        temp_mu = np.delete(temp_mu, 0, 0)
        temp_sigma = np.delete(temp_sigma, 0, 0)
        P = np.delete(P, 0, 0)

        for k in range(1, lsim_para.num_gmm_component[zee]):
            temp_distance = np.sum(np.power(np.abs(temp_mu_gmm - temp_mu), 2), axis=1)
            ind_min = np.argmin(temp_distance)
            lsim_para.gmm_para_mu.loc[ch_names[zee], (state_names[i], gmm_names[k])] = temp_mu[ind_min, :]
            lsim_para.gmm_para_sigma.loc[ch_names[zee], (state_names[i], gmm_names[k])] = temp_sigma[ind_min, :]
            lsim_para.gmm_para_P.loc[ch_names[zee], (state_names[i], gmm_names[k])] = P[ind_min]

            temp_mu = np.delete(temp_mu, ind_min, 0)
            temp_sigma = np.delete(temp_sigma, ind_min, 0)
            P = np.delete(P, ind_min, 0)

        lsim_para.gmm_para_P.loc[ch_names[zee], state_names[i]] = 1
        lsim_para.transition_matrices.loc[:, ch_names[zee]] = 1#  np.array([0,1;1,0])

lsim_para.coupling_theta_IM = np.ones((C, C)) / C
