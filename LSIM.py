# https://realpython.com/python-first-steps/
import warnings
import numpy as np
import pandas as pd
from sklearn import mixture
from sklearn import exceptions as skit_exceptions


class lsim_para:

    # Initializer / Instance Attributes
    def __init__(self, C_in, channel_state_num_in, channel_obs_dim_in, num_gmm_component_in):

        # Suppress only ConvergenceWarning
        warnings.filterwarnings("ignore", category=skit_exceptions.ConvergenceWarning)

        self.C = C_in
        self.channel_obs_dim = channel_obs_dim_in
        self.channel_state_num = channel_state_num_in
        self.num_gmm_component = num_gmm_component_in

        self.channels_name_unique, self.states_name_unique, self.gmm_names_unique, self.dim_names_unique, self.states_of_channels, self.channel_names, self.channels_dim_name, self.dimension_of_channels, \
        self.gmm_states_of_channels, self.gmm_of_channels, self.channels_states_gmm_joint_1, self.channels_states_gmm_joint_2, self.channels_states_gmm_joint_3 = self.make_channels_title(
            C_in, channel_state_num_in,
            channel_obs_dim_in,
            num_gmm_component_in)

        self.dict_states2num = pd.DataFrame(list(range(1, 1 + len(self.states_name_unique))),
                                            index=self.states_name_unique)

        # coupling, pi_0 & transition matrix def
        temp = np.random.rand(C_in, C_in)
        temp = temp / temp.sum(axis=0, keepdims=1)
        self.coupling_theta_IM = temp.copy()

        self.pi_0 = pd.DataFrame([], index=[self.channel_names, self.states_of_channels], columns=[0],
                                 dtype=float)
        self.pi_steady = pd.DataFrame([], index=[self.channel_names, self.states_of_channels], columns=[0],
                                      dtype=float)
        self.transition_matrices = pd.DataFrame([], index=[self.channel_names, self.states_of_channels],
                                                columns=[self.channel_names, self.states_of_channels],
                                                dtype=float)

        # GMM para def
        self.gmm_para_mu = pd.DataFrame([], index=[self.channels_dim_name, self.dimension_of_channels],
                                        columns=[self.gmm_states_of_channels, self.gmm_of_channels], dtype=float)
        self.gmm_para_sigma = pd.DataFrame([], index=[self.channels_dim_name, self.dimension_of_channels],
                                           columns=[self.gmm_states_of_channels, self.gmm_of_channels], dtype=float)
        self.gmm_para_P = pd.DataFrame([], index=self.channels_name_unique,
                                       columns=[self.gmm_states_of_channels, self.gmm_of_channels], dtype=float)
        # init pi_0
        for c in range(C_in):
            temp = np.random.rand(channel_state_num_in[c])
            temp = temp / np.sum(temp)
            self.pi_0.loc[self.channels_name_unique[c], 0] = temp.copy()
            self.pi_steady.loc[self.channels_name_unique[c], 0] = temp.copy()

        # init transition matrix
        for zee in range(C_in):
            for c in range(C_in):
                temp = np.random.rand(channel_state_num_in[c], channel_state_num_in[zee])
                temp = np.exp(-3 * self.coupling_theta_IM[c, zee]) + temp / temp.sum(axis=1, keepdims=1)
                temp = temp / temp.sum(axis=1, keepdims=1)
                self.transition_matrices.loc[self.channels_name_unique[c], self.channels_name_unique[zee]] = temp.copy()

        # init GMM_para including P, MU & Sigma
        for zee in range(C_in):
            for s in range(channel_state_num_in[zee]):
                temp = 1 + s + np.random.randn(channel_obs_dim_in[zee], num_gmm_component_in[zee])
                self.gmm_para_mu.loc[self.channels_name_unique[zee], (self.states_name_unique[s],
                                                                      slice(self.gmm_names_unique[0],
                                                                            self.gmm_names_unique[
                                                                                num_gmm_component_in[zee] - 1]))] \
                    = temp.copy()

                temp = 1 + 2 * np.random.rand(channel_obs_dim_in[zee], num_gmm_component_in[zee])
                self.gmm_para_sigma.loc[self.channels_name_unique[zee], (self.states_name_unique[s],
                                                                         slice(self.gmm_names_unique[0],
                                                                               self.gmm_names_unique[
                                                                                   num_gmm_component_in[zee] - 1]))] \
                    = temp.copy()

                temp = np.random.rand(1, num_gmm_component_in[zee])
                temp = temp / np.sum(temp)
                self.gmm_para_P.loc[self.channels_name_unique[zee], (self.states_name_unique[s],
                                                                     slice(self.gmm_names_unique[0],
                                                                           self.gmm_names_unique[
                                                                               num_gmm_component_in[zee] - 1]))] \
                    = temp[0].copy()

    def make_channels_title(self, C_in, channel_state_num_in, channel_obs_dim_in, num_gmm_component_in):

        channels_name_unique_res = []
        states_name_unique_res = []
        gmm_names_unique_res = []
        dim_names_unique_res = []

        channels_dim_name_res = []
        dimension_of_channels_res = []

        states_of_channels_res = []
        channels_state_name_res = []

        gmm_of_channels_res = []
        gmm_states_of_channels_res = []

        channels_states_gmm_joint_1 = []
        channels_states_gmm_joint_2 = []
        channels_states_gmm_joint_3 = []

        for s in range(1, np.max(channel_state_num_in) + 1):
            states_name_unique_res.append('state:' + "%03d" % (s,))

        for d in range(1, np.max(channel_obs_dim_in) + 1):
            dim_names_unique_res.append('dim:' + "%03d" % (d,))

        for g in range(1, np.max(num_gmm_component_in) + 1):
            gmm_names_unique_res.append('gmm:' + "%03d" % (g,))  # g.__str__()

        for c in range(1, C_in + 1):

            channels_name_unique_res.append('channel:' + "%03d" % (c,))

            for d in range(1, channel_obs_dim_in[c - 1] + 1):
                channels_dim_name_res.append('channel:' + "%03d" % (c,))
                dimension_of_channels_res.append('dim:' + "%03d" % (d,))
            for s in range(1, channel_state_num_in[c - 1] + 1):
                channels_state_name_res.append('channel:' + "%03d" % (c,))
                states_of_channels_res.append('state:' + "%03d" % (s,))

            for s in range(1, channel_state_num_in[c - 1] + 1):

                for g in range(1, np.max(num_gmm_component_in) + 1):
                    channels_states_gmm_joint_1.append('channel:' + "%03d" % (c,))
                    channels_states_gmm_joint_2.append('state:' + "%03d" % (s,))
                    channels_states_gmm_joint_3.append('gmm:' + "%03d" % (g,))

        for s in range(1, np.max(channel_state_num_in) + 1):

            for g in range(1, np.max(num_gmm_component_in) + 1):
                gmm_states_of_channels_res.append('state:' + "%03d" % (s,))
                gmm_of_channels_res.append('gmm:' + "%03d" % (g,))

        return channels_name_unique_res, states_name_unique_res, gmm_names_unique_res, dim_names_unique_res, states_of_channels_res, channels_state_name_res, channels_dim_name_res, dimension_of_channels_res, \
               gmm_states_of_channels_res, gmm_of_channels_res, channels_states_gmm_joint_1, channels_states_gmm_joint_2, channels_states_gmm_joint_3


class general_para(lsim_para):

    def __init__(self, lsim_para, from_lsim:True):

        if from_lsim:
            temp_channel_state_num = lsim_para.channel_state_num.copy()
            self.pi_0 = lsim_para.pi_0.copy()
            self.gmm_para_mu = lsim_para.gmm_para_mu.copy()
            self.gmm_para_sigma = lsim_para.gmm_para_sigma.copy()
            self.gmm_para_P = lsim_para.gmm_para_P.copy()
            self.C = lsim_para.C

            index_chmm_colomn = []
            for zee in range(lsim_para.C):
                # print(zee)
                temp_index = temp_channel_state_num
                temp_index = temp_index[zee + 1:]
                temp_raw = np.kron(range(1, temp_channel_state_num[zee] + 1), np.ones((1, int(np.prod(temp_index)))))

                temp_index = temp_channel_state_num
                temp_index = temp_index[:zee]
                index_chmm_colomn.append(np.kron(np.ones((1, int(np.prod(temp_index)))), temp_raw)[0].tolist())

            # reversed list [::-1]
            temp = temp_channel_state_num[::-1]
            temp.pop(len(temp) - 1)
            weight_state_column = np.cumprod(temp)
            weight_state_column = np.insert(weight_state_column, 0, 1)
            weight_state_column = weight_state_column[::-1]

            for zee in range(lsim_para.C):
                list(map(str, index_chmm_colomn[zee]))

            index_chmm_colomn_np = np.array(index_chmm_colomn, dtype=np.int64)
            self.index_chmm_colomn_states = index_chmm_colomn_np

            self.transition_matrices = pd.DataFrame([], index=index_chmm_colomn,
                                                    columns=[lsim_para.channel_names,
                                                             lsim_para.states_of_channels])
            temp_var = np.zeros((lsim_para.C, 1))
            temp_index = list(range(1, lsim_para.C + 1))

            for zee in range(lsim_para.C):

                for s in range(temp_channel_state_num[zee]):

                    for j in range(index_chmm_colomn[0].__len__()):

                        for c in range(lsim_para.C):
                            temp_index[c] = [(index_chmm_colomn_np[c, j])]
                            temp_var[c] = lsim_para.transition_matrices.loc[(lsim_para.channels_name_unique[c],
                                                                             lsim_para.states_name_unique[
                                                                                 int(index_chmm_colomn_np[c, j]) - 1]),
                                                                            (lsim_para.channels_name_unique[zee],
                                                                             lsim_para.states_name_unique[s])]
                        self.transition_matrices.loc[tuple(temp_index), (
                            lsim_para.channels_name_unique[zee], lsim_para.states_name_unique[s])] = np.dot(
                            lsim_para.coupling_theta_IM[:, zee], temp_var)
        else:

            temp_channel_state_num = lsim_para.channel_state_num.copy()
            self.pi_0 = lsim_para.pi_0.copy()
            self.gmm_para_mu = lsim_para.gmm_para_mu.copy()
            self.gmm_para_sigma = lsim_para.gmm_para_sigma.copy()
            self.gmm_para_P = lsim_para.gmm_para_P.copy()
            self.C = lsim_para.C

            index_chmm_colomn = []

            for zee in range(lsim_para.C):
                # print(zee)
                temp_index = temp_channel_state_num
                temp_index = temp_index[zee + 1:]
                temp_raw = np.kron(range(1, temp_channel_state_num[zee] + 1),
                                   np.ones((1, int(np.prod(temp_index)))))

                temp_index = temp_channel_state_num
                temp_index = temp_index[:zee]
                index_chmm_colomn.append(np.kron(np.ones((1, int(np.prod(temp_index)))), temp_raw)[0].tolist())

            for zee in range(lsim_para.C):
                list(map(str, index_chmm_colomn[zee]))

            index_chmm_colomn_np = np.array(index_chmm_colomn, dtype=np.int64)
            self.index_chmm_colomn_states = index_chmm_colomn_np

            self.transition_matrices = pd.DataFrame([], index=index_chmm_colomn,
                                                    columns=[lsim_para.channel_names,
                                                             lsim_para.states_of_channels])
            for zee in range(lsim_para.C):
                for j in range(index_chmm_colomn[0].__len__()):
                    temp_index = index_chmm_colomn_np[:, j]
                    temp = np.random.rand(temp_channel_state_num[zee])
                    temp = temp / np.sum(temp)
                    self.transition_matrices.loc[tuple(temp_index), lsim_para.channels_name_unique[zee]] = temp.copy()

    pass


class lsim():
    # Class Attribute
    main_idea = "multi-channel time-series modelling & classification"

    # Initializer / Instance Attributes
    def __init__(self, C_in, channel_state_num_in, channel_obs_dim_in, num_gmm_component_in):
        self.parameters = lsim_para(C_in, channel_state_num_in, channel_obs_dim_in, num_gmm_component_in)

    def reinit_random_para(self):
        self.parameters = lsim_para(self.parameters.C, self.parameters.channel_state_num,
                                    self.parameters.channel_obs_dim, self.parameters.num_gmm_component)

    def get_general_para(self):
        self.general_chmm_para = general_para(self.parameters, True)

    def generate_time_series(self, T):

        from_lsim = True
        self.general_chmm_para = general_para(self.parameters, from_lsim)

        channels_observation = pd.DataFrame([], index=[self.parameters.channels_dim_name,
                                                       self.parameters.dimension_of_channels],
                                            columns=np.arange(T), dtype=float)
        channels_hidden_states = pd.DataFrame([], index=self.parameters.channels_name_unique,
                                              columns=np.arange(T))

        for c in range(self.parameters.C):
            pi_0 = self.general_chmm_para.pi_0.loc[self.parameters.channels_name_unique[c]]
            temp = self._select_P(pi_0)
            # print(temp)
            channels_hidden_states.loc[self.parameters.channels_name_unique[c], 0] = temp.loc[0]
            channels_observation.loc[self.parameters.channels_name_unique[c], 0] \
                = self._gmm_gen(temp.loc[0],self.parameters.channels_name_unique[c])

        dict_states2num = self.parameters.dict_states2num

        for t in range(1, T):
            current_states = dict_states2num.loc[channels_hidden_states.loc[:, t - 1]]
            for c in range(self.parameters.C):
                current_pi = self.general_chmm_para.pi_0.loc[self.parameters.channels_name_unique[c]]
                values_to_set  = self.general_chmm_para.transition_matrices.loc[ tuple(current_states.values), self.parameters.channels_name_unique[c]].values[0]
                values_to_set = values_to_set.astype(np.float64)
                #values_to_set = np.array(values_to_set, dtype=np.float64)
                current_pi.loc[:, 0] = values_to_set
                temp = self._select_P(current_pi)
                channels_hidden_states.loc[self.parameters.channels_name_unique[c], t] = temp.loc[0]
                channels_observation.loc[self.parameters.channels_name_unique[c], t] = self._gmm_gen(temp.loc[0],self.parameters.channels_name_unique[c])

        return channels_observation, channels_hidden_states

    def _gmm_pdf_fast(self, obs):

        P_temp = self.parameters.gmm_para_P
        mu_temp = self.parameters.gmm_para_mu
        sig_temp = self.parameters.gmm_para_sigma

        T = obs.shape[1]
        max_state_num = np.max(self.parameters.channel_state_num)
        max_gmm_num = np.max(self.parameters.num_gmm_component)
        dimension_numbers_reshape = np.concatenate([[0], np.cumsum(self.parameters.channel_obs_dim)])

        np_obs = np.array(obs, dtype=np.float64)
        np_mu = np.array(mu_temp, dtype=np.float64)
        np_sig = np.array(sig_temp, dtype=np.float64)
        np_P = np.array(P_temp, dtype=np.float64)

        np_mu = np.reshape(np_mu, (dimension_numbers_reshape[-1], max_state_num, max_gmm_num))
        np_sig = np.reshape(np_sig, (dimension_numbers_reshape[-1], max_state_num, max_gmm_num))
        np_P = np.reshape(np_P, (self.parameters.C, max_state_num, max_gmm_num))

        np_mu[np.isnan(np_mu)] = 0
        np_sig[np.isnan(np_sig)] = 1
        np_P[np.isnan(np_P)] = 0

        np_obs = np.repeat(np_obs[:, :, np.newaxis], max_state_num, axis=2)
        np_obs = np.repeat(np_obs[:, :, :, np.newaxis], max_gmm_num, axis=3)

        np_mu = np.repeat(np_mu[:, np.newaxis, :, :], T, axis=1)
        np_sig = np.repeat(np_sig[:, np.newaxis, :, :], T, axis=1)
        np_P = np.repeat(np_P[:, np.newaxis, :, :], T, axis=1)

        ind_nan_obs = np.isnan(np_obs)  # np.where(np.isnan(np_obs))
        np_obs[ind_nan_obs] = np_mu[ind_nan_obs]

        quad_form = np.divide(np.power(np_obs - np_mu, 2), np_sig)
        logSqrtDetSigma = 0.5 * np.log(np_sig)
        logSqrtDetSigma[ind_nan_obs] = 0

        if np.sum(self.parameters.channel_obs_dim == 1) == self.parameters.C:
            PDFs = np.multiply(np_P, np.exp(-0.5 * quad_form - logSqrtDetSigma - np.log(2 * np.pi) / 2))
        else:
            D = np.log(2 * np.pi) / 2
            D = np.ones(quad_form.shape) * D
            D[ind_nan_obs] = 0
            logValue = np.cumsum(-0.5 * quad_form - logSqrtDetSigma - D, axis=0)
            logValue = np.concatenate([np.zeros((1, T, max_state_num, max_gmm_num)), logValue])

            PDFs = np.multiply(np_P, np.exp(
                logValue[dimension_numbers_reshape[1:], :, :, :] - logValue[dimension_numbers_reshape[:-1], :, :, :]))
        PDFs = np.moveaxis(PDFs, [2, 0, 3, 1], [0, 1, 2, 3])
        P_observ_cond_to_state_comp = np.reshape(PDFs, (PDFs.shape[0] * PDFs.shape[1], PDFs.shape[2], T), 'F')

        from_state_num = np.arange(len(self.parameters.channel_state_num)) * max_state_num
        to_state_num = from_state_num + self.parameters.channel_state_num
        ind_valid = np.zeros(len(self.parameters.channel_state_num) * max_state_num, dtype=bool)

        for a, b in zip(from_state_num, to_state_num):
            ind_valid[a:b] = True

        P_observ_cond_to_state_comp = P_observ_cond_to_state_comp[ind_valid, :, :]
        P_observ_cond_to_state = np.sum(P_observ_cond_to_state_comp, axis=1)

        return P_observ_cond_to_state, P_observ_cond_to_state_comp

    def compute_joint_state_observation_prob(self, obs, latent_states):

        T = obs.shape[1]

        self.get_general_para()
        channel_names = self.parameters.channels_name_unique.copy()
        P_observ_cond_to_state, P_observ_cond_to_state_comp = self._gmm_pdf_fast(obs)
        pd_P_observ_cond_to_state = pd.DataFrame([], index=[self.parameters.channel_names,
                                                            self.parameters.states_of_channels],
                                                 columns=list(range(T)), dtype=float)
        pd_P_observ_cond_to_state.loc[:] = P_observ_cond_to_state

        log_joint_prob = 0
        t = 0

        for zee in range(self.general_chmm_para.C):
            log_joint_prob += np.log(
                self.general_chmm_para.pi_0.loc[(channel_names[zee], latent_states.loc[channel_names[zee], t]), t]) + \
                              np.log(pd_P_observ_cond_to_state.loc[
                                         (channel_names[zee], latent_states.loc[channel_names[zee], t]), t])

        latent_states_num = latent_states.map(lambda x: int(x[-3:]))
        for t in range(1, T):
            for zee in range(self.general_chmm_para.C):
                temp_index = latent_states_num.loc[:, t].values
                log_joint_prob += np.log(self.general_chmm_para.transition_matrices.loc[tuple(temp_index), (
                    channel_names[zee], latent_states.loc[channel_names[zee], t])]) + \
                                  np.log(pd_P_observ_cond_to_state.loc[
                                             (channel_names[zee], latent_states.loc[channel_names[zee], t]), t])

        return log_joint_prob

    def forward_backward_lsim(self, obs, flag_EM=False):

        T = obs.shape[1]
        C = self.parameters.C
        lsim_para = self.parameters
        max_gmm_num = np.max(self.parameters.num_gmm_component)
        transition_matrices = np.array(lsim_para.transition_matrices)
        state_numbers_index = np.concatenate([[-1], np.cumsum(lsim_para.channel_state_num) - 1])

        sum_states = state_numbers_index[-1] + 1
        P_ot_c_cond_past = np.zeros([C, T])
        b_c_ot_nc = np.zeros([sum_states, T])

        alpha = np.zeros([sum_states, T])
        alpha_T = np.zeros([sum_states, T])
        beta = np.zeros([sum_states, T])
        ## beta2 = np.zeros([sum_states, T])
        a_alpha_b = np.zeros([sum_states, C, T - 1])
        P_observ_cond_to_state, P_observ_cond_to_state_comp = self._gmm_pdf_fast(obs)
        P_vz_tm1_vw_t_O = np.zeros([sum_states, sum_states, T])
        P_vz_tm1_vw_t_O_cond = np.zeros([sum_states, sum_states, T])

        mat_mult = np.zeros([sum_states, C])
        cuopling_repmat = np.zeros([sum_states, C])
        cuopling_repmat_beta = np.zeros([sum_states, C])
        weights_subsystems = np.zeros([sum_states, C])
        weights_subsystems_opt = np.zeros([sum_states, C])

        coupling_transmat = np.zeros([sum_states, sum_states])
        diag_zero = np.ones([sum_states, sum_states])

        mat_mult_one = np.zeros([sum_states, C])
        vec_index_x = []
        vec_index_y = []

        for zee in range(C):
            zee_index = np.arange(state_numbers_index[zee] + 1, state_numbers_index[zee + 1] + 1)
            vec_index_x.extend(np.repeat(int(zee), lsim_para.channel_state_num[zee], axis=0))
            vec_index_y.extend(np.arange(state_numbers_index[zee] + 1, state_numbers_index[zee + 1] + 1))

            zee_coupling = lsim_para.coupling_theta_IM[:, zee]
            cuopling_repmat[zee_index, :] = np.repeat(zee_coupling[:, np.newaxis], lsim_para.channel_state_num[zee],
                                                      axis=1).transpose()

            diag_zero[zee_index[0]:zee_index[-1] + 1, zee_index[0]:zee_index[-1] + 1] = 0

            zee_coupling = lsim_para.coupling_theta_IM[zee, :]
            cuopling_repmat_beta[zee_index, :] = np.repeat(zee_coupling[:, np.newaxis],
                                                           lsim_para.channel_state_num[zee], axis=1).transpose()
            weights_subsystems_opt[zee_index, :] = np.repeat(zee_coupling[:, np.newaxis],
                                                             lsim_para.channel_state_num[zee], axis=1).transpose()

        for zee in range(C):
            zee_index = np.arange(state_numbers_index[zee] + 1, state_numbers_index[zee + 1] + 1)
            coupling_transmat[:, zee_index] = np.multiply(cuopling_repmat_beta[:, zee][:, np.newaxis],
                                                          transition_matrices[:, zee_index])

        vec_mat_mult = (vec_index_y, vec_index_x)
        mat_mult_one[vec_mat_mult] = 1

        t = 0
        alpha[:, t] = lsim_para.pi_0[:].values[:, 0]
        P_ot_c_cond_past[:, t] = np.dot(mat_mult_one.transpose(),
                                        np.multiply(alpha[:, t], P_observ_cond_to_state[:, t]))
        b_c_ot_nc[:, t] = np.divide(P_observ_cond_to_state[:, t], np.dot(mat_mult_one, P_ot_c_cond_past[:, t]))

        for t in range(1, T):

            alpha_tm1_tm1 = np.multiply(alpha[:, t - 1], b_c_ot_nc[:, t - 1])
            mat_mult[vec_mat_mult] = alpha_tm1_tm1

            a_alpha_b[:, :, t - 1] = np.dot(transition_matrices.transpose(), mat_mult)
            a_alpha_b_coupled = np.multiply(a_alpha_b[:, :, t - 1], cuopling_repmat)
            alpha[:, t] = np.sum(a_alpha_b_coupled, axis=1)

            if np.abs(np.sum(alpha[:, t]) - C) > 0.0001:  # fordebug
                yu = 0

            P_ot_c_cond_past[:, t] = np.dot(mat_mult_one.transpose(),
                                            np.multiply(alpha[:, t], P_observ_cond_to_state[:, t]))
            b_c_ot_nc[:, t] = np.divide(P_observ_cond_to_state[:, t], np.dot(mat_mult_one, P_ot_c_cond_past[:, t]))

            P_vz_tm1_vw_t_O_cond[:, :, t] = coupling_transmat + np.dot(np.multiply(alpha_tm1_tm1, diag_zero),
                                                                       coupling_transmat)
            P_vz_tm1_vw_t_O[:, :, t] = np.multiply(alpha_tm1_tm1, P_vz_tm1_vw_t_O_cond[:, :, t])

        beta[:, -1] = b_c_ot_nc[:, -1]
        alpha_T[:, -1] = np.multiply(alpha[:, -1], beta[:, -1])

        flag_nan = np.sum(np.sum(lsim_para.coupling_theta_IM, axis=1) < 0.0000000001) > 0;
        channel_numbers_index = np.concatenate([[-1], np.arange(C, C * C + 1, C) - 1])

        for t in range(T - 2, -1, -1):

            mat_mult[vec_mat_mult] = beta[:, t + 1]
            # beta_C2 = transitions_matrices * mat_mult;# compute from t=1 to t = T - 1

            beta_C = np.dot(P_vz_tm1_vw_t_O_cond[:, :, t + 1], mat_mult)

            if C > 1:
                temp = P_vz_tm1_vw_t_O[:, :, t + 1]
                temp_2d_f = np.divide(np.power(temp, 2), alpha[:, t + 1])
                temp_aplha = np.multiply(alpha[:, t], b_c_ot_nc[:, t])

                h_zee_temp = np.cumsum(np.power(temp_aplha, 2))
                # h_zee_all = h_zee_temp[state_numbers_index[1:]] - np.concatenate([[0], h_zee_temp[state_numbers_index[1:C]]])
                h_zee_all = h_zee_temp[state_numbers_index[1:]]  # instead of concatenation
                h_zee_all[1:] = h_zee_all[1:] - h_zee_temp[state_numbers_index[1:C]]  # instead of concatenation
                h_zee_all = np.repeat(h_zee_all, C)

                f_zee_temp1 = np.cumsum(temp_2d_f, axis=0)
                f_zee_temp = f_zee_temp1[state_numbers_index[1:], :]  # instead of concatenation
                f_zee_temp[1:, :] = f_zee_temp[1:, :] - f_zee_temp1[state_numbers_index[1:C],
                                                        :]  # instead of concatenation
                f_zee_temp = np.cumsum(f_zee_temp, axis=1)
                f_zee_all = f_zee_temp[:, state_numbers_index[1:]]
                f_zee_all[:, 1:] = f_zee_all[:, 1:] - f_zee_temp[:, state_numbers_index[1:C]]
                f_zee_all = f_zee_all.flatten('F') + 0.000001

                nume_temp1 = np.cumsum(np.divide(f_zee_all, (f_zee_all - h_zee_all)), axis=0)
                nume_temp = nume_temp1[channel_numbers_index[1:]]  # instead of concatenation
                nume_temp[1:] = nume_temp[1:] - nume_temp1[channel_numbers_index[1:C]]  # instead of concatenation
                nume_temp = np.multiply(np.repeat(nume_temp, C), h_zee_all)

                denume_temp2 = np.cumsum(np.divide(1, (f_zee_all - h_zee_all)), axis=0)
                denume_temp = denume_temp2[channel_numbers_index[1:]]  # instead of concatenation
                denume_temp[1:] = denume_temp[1:] - denume_temp2[channel_numbers_index[1:C]]  # instead of concatenation
                denume_temp = 1 + np.multiply(np.repeat(denume_temp, C), h_zee_all)

                g_zee_all = np.divide(nume_temp, denume_temp)
                d_hat = np.divide((f_zee_all - g_zee_all), (f_zee_all - h_zee_all))

                temp_delta = np.reshape(d_hat, (C, C))
                mat_mult[vec_mat_mult] = 1
                weights_subsystems = np.dot(mat_mult, temp_delta)

            else:
                mat_mult[vec_mat_mult] = 1
                weights_subsystems = mat_mult

            temp_sum = np.sum(np.multiply(weights_subsystems, beta_C), axis=1)
            temp_sum[temp_sum < 0.0000001] = 0.0000001
            beta[:, t] = np.multiply(b_c_ot_nc[:, t], temp_sum)

            if flag_nan:
                ind_nan = np.isnan(beta[:, t])
                beta[ind_nan, t] = b_c_ot_nc[ind_nan, t]

            alpha_T[:, t] = np.multiply(beta[:, t], alpha[:, t])

            temp_sum = np.cumsum(alpha_T[:, t])
            normalizer = temp_sum[state_numbers_index[1:]]  # instead of concatenation
            normalizer[1:] = normalizer[1:] - temp_sum[state_numbers_index[1:C]]  # instead of concatenation
            normalizer_vec = np.dot(mat_mult_one, normalizer)

            alpha_T[:, t] = np.divide(alpha_T[:, t], normalizer_vec)
            beta[:, t] = np.divide(beta[:, t], normalizer_vec)

        P_O_model = np.sum(np.log(P_ot_c_cond_past.flatten()))

        # return P_O_model ,alpha_out , beta_out , alpha_T_out , b_c_ot_nc_out , P_observ_cond_to_state_out , P_observ_cond_to_state_comp_out
        if flag_EM:
            return P_O_model, alpha, beta, alpha_T, b_c_ot_nc, P_observ_cond_to_state, P_observ_cond_to_state_comp
        else:
            alpha_out = pd.DataFrame(alpha, index=[lsim_para.channel_names, lsim_para.states_of_channels],
                                     columns=[np.arange(T)], dtype=float)
            beta_out = pd.DataFrame(beta, index=[lsim_para.channel_names, lsim_para.states_of_channels],
                                    columns=[np.arange(T)], dtype=float)
            alpha_T_out = pd.DataFrame(alpha_T, index=[lsim_para.channel_names, lsim_para.states_of_channels],
                                       columns=[np.arange(T)], dtype=float)
            b_c_ot_nc_out = pd.DataFrame(b_c_ot_nc, index=[lsim_para.channel_names, lsim_para.states_of_channels],
                                         columns=[np.arange(T)], dtype=float)
            P_observ_cond_to_state_out = pd.DataFrame(P_observ_cond_to_state,
                                                      index=[lsim_para.channel_names, lsim_para.states_of_channels],
                                                      columns=[np.arange(T)], dtype=float)

            P_observ_cond_to_state_comp_out = pd.DataFrame(
                P_observ_cond_to_state_comp.reshape((sum_states * max_gmm_num, T)),
                index=[lsim_para.channels_states_gmm_joint_1, lsim_para.channels_states_gmm_joint_2,
                       lsim_para.channels_states_gmm_joint_3],
                columns=[np.arange(T)], dtype=float)

            return P_O_model, alpha_out, beta_out, alpha_T_out, b_c_ot_nc_out, P_observ_cond_to_state_out, P_observ_cond_to_state_comp_out

    def em_lsim(self, obs, max_itration, extra_options):

        try:
            if len(obs.columns.levels) != 2:
                obs.columns = pd.MultiIndex.from_product([['obs:0001'], np.arange(obs.columns.shape[0])])
        except:
            obs.columns = pd.MultiIndex.from_product([['obs:0001'], np.arange(obs.columns.shape[0])])

        num_trials = len(obs.columns.levels[0])
        name_of_trails = obs.columns.levels[0]
        length_observation = np.ones(len(name_of_trails), dtype=int)
        for tr in range(num_trials):
            length_observation[tr] = np.int64(obs.loc[:, name_of_trails[tr]].shape[1])

        all_observation = np.array(obs, dtype=np.float64)
        T_all = np.int64(np.sum(length_observation))
        C = self.parameters.C
        lsim_para = self.parameters

        max_state_num = np.max(lsim_para.channel_state_num)
        max_gmm_num = np.max(lsim_para.num_gmm_component)
        dimension_numbers_reshape = np.concatenate([[0], np.cumsum(lsim_para.channel_obs_dim)])
        dimension_numbers_index = np.concatenate([[-1], np.cumsum(lsim_para.channel_obs_dim) - 1])
        state_numbers_index = np.concatenate([[-1], np.cumsum(lsim_para.channel_state_num) - 1])
        trial_time_index = np.concatenate([[-1], np.cumsum(length_observation) - 1])
        state_names = self.parameters.states_name_unique.copy()
        sum_states = state_numbers_index[-1] + 1
        ch_names = lsim_para.channels_name_unique
        state_names = lsim_para.states_name_unique
        gmm_names = lsim_para.gmm_names_unique

        # C = lsim_para.C
        flag_EM = True

        # GMM initialization
        for zee in range(C):

            Y = obs.loc[ch_names[zee], :]
            Y = np.array(Y)

            # fit a Gaussian Mixture Model with N components
            init_gmm_numbers = lsim_para.channel_state_num[zee] * lsim_para.num_gmm_component[zee]
            clf = mixture.GaussianMixture(n_components=init_gmm_numbers, covariance_type='diag', max_iter=10)
            clf.fit(Y.transpose())

            temp_sigma = clf.covariances_.copy()
            temp_mu = clf.means_.copy()
            P = clf.weights_.copy()

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

                lsim_para.gmm_para_P.loc[ch_names[zee], state_names[i]] = (lsim_para.gmm_para_P.loc[ch_names[zee], state_names[i]].values /
                                                                           np.nansum(lsim_para.gmm_para_P.loc[ch_names[zee], state_names[i]].values))
                lsim_para.pi_0.loc[ch_names[zee]] = 1 / lsim_para.channel_state_num[zee]
                lsim_para.transition_matrices.loc[:, ch_names[zee]] = 1 / lsim_para.channel_state_num[zee]

        lsim_para.coupling_theta_IM = np.ones((C, C)) / C

        alpha_T_trials = np.zeros((sum_states, T_all))
        alpha_trials = np.zeros((sum_states, T_all))
        beta_trials = np.zeros((sum_states, T_all))

        b_c_ot_nc_trials = np.zeros((sum_states, T_all))
        P_observ_cond_to_state_trials = np.zeros((sum_states, T_all))
        P_observ_cond_to_state_comp_trials = np.zeros((sum_states, max_gmm_num, T_all))

        gamma_trail_time_index = 1 + trial_time_index[0:-1]
        L_T_1 = T_all - num_trials
        P_O_model = np.zeros((num_trials))

        coupling_transition_all = np.zeros((sum_states, sum_states))

        mat_mult_one = np.zeros([sum_states, C])
        vec_index_x = []
        vec_index_y = []
        for zee in range(C):
            vec_index_x.extend(np.repeat(int(zee), lsim_para.channel_state_num[zee], axis=0))
            vec_index_y.extend(np.arange(state_numbers_index[zee] + 1, state_numbers_index[zee + 1] + 1))
        vec_mat_mult = (vec_index_y, vec_index_x)
        mat_mult_one[vec_mat_mult] = 1

        log_likelyhood = np.nan * np.ones((max_itration))

        for itr in range(max_itration):

            for zee in range(C):
                zee_index = np.arange(state_numbers_index[zee] + 1, state_numbers_index[zee + 1] + 1)
                coupling_transition_all[:, zee_index] = np.multiply(lsim_para.transition_matrices.loc[:, ch_names[zee]],
                                                                    np.dot(mat_mult_one,
                                                                           lsim_para.coupling_theta_IM[:, zee])[:,
                                                                    np.newaxis])

            sum3_zeta = 0

            for tr in np.arange(num_trials, dtype=int):
                P_O_model[tr], alpha_trials[:, trial_time_index[tr] + 1:trial_time_index[tr + 1] + 1], beta_trials[:,trial_time_index[tr] + 1: trial_time_index[ tr + 1] + 1], \
                alpha_T_trials[:,trial_time_index[tr] + 1: trial_time_index[tr + 1] + 1], b_c_ot_nc_trials[ :, trial_time_index[tr] + 1:trial_time_index[tr + 1] + 1], \
                P_observ_cond_to_state_trials[:, trial_time_index[tr] + 1:trial_time_index[tr + 1] + 1], P_observ_cond_to_state_comp_trials[:, :, trial_time_index[ tr] + 1:trial_time_index[ tr + 1] + 1] \
                    = self.forward_backward_lsim(obs.loc[:, name_of_trails[tr]], flag_EM)

                zeta_part1 = np.repeat(np.multiply(alpha_trials[:, trial_time_index[tr] + 1:trial_time_index[tr + 1]],
                                                   b_c_ot_nc_trials[:,
                                                   trial_time_index[tr] + 1:trial_time_index[tr + 1]])[:, np.newaxis,
                                       :], state_numbers_index[-1] + 1, axis=1)
                zeta_part2 = np.repeat(
                    beta_trials[:, trial_time_index[tr] + 2:trial_time_index[tr + 1] + 1][np.newaxis, :, :],
                    state_numbers_index[-1] + 1, axis=0)
                sum3_zeta += np.sum(np.multiply(zeta_part1, zeta_part2), axis=2)

            sum3_zeta = np.multiply(sum3_zeta, coupling_transition_all)

            gamma_P0 = alpha_T_trials[:, gamma_trail_time_index]
            lsim_para.pi_0.loc[:] = np.mean(gamma_P0, axis=1)[:, np.newaxis]
            temp_coupling = np.zeros(sum_states + 1)

            for zee in range(C):
                zee_index = np.arange(state_numbers_index[zee] + 1, state_numbers_index[zee + 1] + 1)

                sum3_Zeta_zee = sum3_zeta[:, zee_index]
                sum32_Zeta_zee = np.sum(sum3_Zeta_zee, axis=1)
                temp_Trans_zee = np.divide(sum3_Zeta_zee, sum32_Zeta_zee[:, np.newaxis])
                temp_Trans_zee[np.isnan(temp_Trans_zee)] = 1 / lsim_para.channel_state_num[zee]
                lsim_para.transition_matrices.loc[:, ch_names[zee]] = temp_Trans_zee

                temp_coupling[1:] = np.cumsum(sum32_Zeta_zee)
                lsim_para.coupling_theta_IM[:, zee] = (temp_coupling[state_numbers_index[1:] + 1] - temp_coupling[state_numbers_index[:-1] + 1]) / L_T_1

                for st in range(lsim_para.channel_state_num[zee]):
                    temp_gamma_obs = np.zeros((lsim_para.num_gmm_component[zee], T_all))

                    for k in range(lsim_para.num_gmm_component[zee]):
                        temp_gamma_obs[k, :] = np.divide(np.multiply(alpha_T_trials[zee_index[st], :],
                                                                     P_observ_cond_to_state_comp_trials[zee_index[st],
                                                                     k, :]),
                                                         P_observ_cond_to_state_trials[zee_index[st], :])
                        temp_gamma_obs[k, np.isnan(temp_gamma_obs[k, :])] = 0

                        Y_update = np.array(obs.loc[ch_names[zee]])
                        Y_update[np.isnan(Y_update)] = 0
                        weight_update = temp_gamma_obs[k, :]
                        if np.sum(weight_update > 0) < 2:
                            weight_update += 1

                        lsim_para.gmm_para_mu.loc[ch_names[zee], (state_names[st], gmm_names[k])] = np.divide(
                            np.sum(np.multiply(Y_update, weight_update), axis=1), np.sum(weight_update))

                        Recentering = np.subtract(Y_update, lsim_para.gmm_para_mu.loc[ch_names[zee], (state_names[st], gmm_names[k])].to_numpy()[:,np.newaxis])
                        lsim_para.gmm_para_sigma.loc[ch_names[zee], (state_names[st], gmm_names[k])] \
                            = 0.001 * np.nanvar(Recentering,axis=1) + np.diagonal(np.divide(
                                np.dot(np.multiply(Recentering, weight_update), Recentering.transpose()), np.sum(weight_update)))

                    lsim_para.gmm_para_P.loc[
                        ch_names[zee], (state_names[st], slice(gmm_names[0], gmm_names[k]))] = np.divide(
                        np.sum(temp_gamma_obs, axis=1), np.sum(temp_gamma_obs))

            lsim_para.pi_steady = np.sum(alpha_T_trials, axis=1) / T_all
            log_likelyhood[itr] = np.sum(P_O_model)

            if extra_options['plot'] and np.remainder(itr, 10) == 0:
                k = 0

        if extra_options['plot']:
            pd.options.plotting.backend = "plotly"

            df_loglik = pd.DataFrame(log_likelyhood, dtype=float)
            df_loglik.plot().show()

        # transition_matrices = np.array(lsim_para.transition_matrices)
        # pi_0 = np.array(lsim_para.pi_0)
        # coupling_tetha_IM = np.array(lsim_para.coupling_theta_IM)
        self.parameters = lsim_para
        return lsim_para

    def _select_P(self, pi_0):

        temp_rand = np.random.rand()
        return np.cumsum(pi_0).gt(temp_rand).loc[:].idxmax()

    def _gmm_gen(self, state, channel):

        P = self.parameters.gmm_para_P.loc[channel, state]
        P = P.dropna()

        if P.__len__() > 1:

            # P.index = P.index.droplevel()
            current_gmm = self._select_P(P)

        else:
            current_gmm = P.index[0]

        mu_temp = self.parameters.gmm_para_mu.loc[channel, (state, current_gmm)].to_numpy()
        sig_temp = self.parameters.gmm_para_sigma.loc[channel, (state, current_gmm)].to_numpy()
        # mu_temp = self.parameters.gmm_para_mu[ state, current_gmm].loc[channel]
        # sig_temp = self.parameters.gmm_para_sigma[ state, current_gmm].loc[channel]

        # sig_temp.columns = sig_temp.columns.droplevel()
        return np.sqrt(sig_temp) * np.random.randn(sig_temp.__len__()) + mu_temp

    def plot_chmm_timeseries(self, obs, channel_number):

        pd.options.plotting.backend = "plotly"

        df = pd.DataFrame(obs.loc[self.parameters.channels_name_unique[channel_number - 1]].values.transpose(),
                          columns=self.parameters.dim_names_unique[
                                  0:self.parameters.channel_obs_dim[channel_number - 1]], dtype=float)
        df.plot().show()

    def chmm_cart_prod(self):
        self.get_general_para()
        old_lsim_para = self.parameters
        C = 1
        channel_state_num = np.array([np.prod(old_lsim_para.channel_state_num)])
        channel_obs_dim = np.array([sum(old_lsim_para.channel_obs_dim)])
        num_gmm_component = np.array([1])
        eq_hmm = lsim(C, channel_state_num, channel_obs_dim, num_gmm_component)

        temp = np.ones(1)
        for c in range(old_lsim_para.C):
            temp = np.kron(temp, old_lsim_para.pi_0.loc[old_lsim_para.channels_name_unique[c]])

        eq_hmm.parameters.pi_0 = temp

        index_chmm_colomn = []
        for zee in range(old_lsim_para.C):
            temp_index = old_lsim_para.channel_state_num
            temp_index = temp_index[zee + 1:]
            temp_raw = np.kron(range(1, old_lsim_para.channel_state_num[zee] + 1),
                               np.ones((1, int(np.prod(temp_index)))))

            temp_index = old_lsim_para.channel_state_num
            temp_index = temp_index[:zee]
            index_chmm_colomn.append(np.kron(np.ones((1, int(np.prod(temp_index)))), temp_raw)[0].tolist())

        index_chmm_colomn_np = np.array(index_chmm_colomn, dtype=np.int64)

        for s in range(channel_state_num[0]):

            temp_index = index_chmm_colomn_np[:, s]

            temp = np.ones(1)
            for c in range(old_lsim_para.C):
                temp = np.kron(temp, self.general_chmm_para.transition_matrices.loc[
                    tuple(temp_index), old_lsim_para.channels_name_unique[c]])

            eq_hmm.parameters.transition_matrices.loc[
                (eq_hmm.parameters.channels_name_unique[0], eq_hmm.parameters.states_name_unique[s])] = temp.copy()

        dimension_numbers_index = np.concatenate([[-1], np.cumsum(old_lsim_para.channel_obs_dim) - 1])

        channels_name_unique = eq_hmm.parameters.channels_name_unique
        dim_names_unique = eq_hmm.parameters.dim_names_unique
        states_name_unique = eq_hmm.parameters.states_name_unique

        for c in range(old_lsim_para.C):
            for s in range(old_lsim_para.channel_state_num[c]):
                row_num = np.where(index_chmm_colomn_np[c, :] == s + 1)[0]

                mu_temp = old_lsim_para.gmm_para_mu.loc[
                    old_lsim_para.channels_name_unique[c], tuple([states_name_unique[s], 'gmm:001'])]
                bb = np.tile(mu_temp, [row_num.shape[0], 1]).transpose()
                eq_hmm.parameters.gmm_para_mu.loc[(channels_name_unique[0],
                                                   slice(dim_names_unique[dimension_numbers_index[c] + 1],
                                                         dim_names_unique[dimension_numbers_index[c + 1]])), [
                                                      states_name_unique[i] for i in row_num]] = bb.copy()

                sig_temp = old_lsim_para.gmm_para_sigma.loc[
                    old_lsim_para.channels_name_unique[c], tuple([states_name_unique[s], 'gmm:001'])]
                bb = np.tile(sig_temp, [row_num.shape[0], 1]).transpose()
                eq_hmm.parameters.gmm_para_sigma.loc[(channels_name_unique[0],
                                                      slice(dim_names_unique[dimension_numbers_index[c] + 1],
                                                            dim_names_unique[dimension_numbers_index[c + 1]])), [
                                                         states_name_unique[i] for i in row_num]] = bb.copy()

        return eq_hmm


class chmm():
    # Class Attribute
    main_idea = "multi-channel time-series modelling & classification"

    # Initializer / Instance Attributes
    def __init__(self, C_in, channel_state_num_in, channel_obs_dim_in, num_gmm_component_in):
        self.parameters = lsim_para(C_in, channel_state_num_in, channel_obs_dim_in, num_gmm_component_in)

    def reinit_random_para(self):
        self.parameters = lsim_para(self.parameters.C, self.parameters.channel_state_num,
                                    self.parameters.channel_obs_dim, self.parameters.num_gmm_component)

    def get_general_para(self):
        self.general_chmm_para = general_para(self.parameters, True)

    def generate_time_series(self, from_lsim, T):

        if from_lsim:
            self.general_chmm_para = general_para(self.parameters, from_lsim)
        else:
            self.reinit_random_para()
            self.general_chmm_para = general_para(self.parameters, from_lsim)

        channels_observation = pd.DataFrame([], index=[self.parameters.channels_dim_name,
                                                       self.parameters.dimension_of_channels],
                                            columns=np.arange(T), dtype=float)
        channels_hidden_states = pd.DataFrame([], index=self.parameters.channels_name_unique,
                                              columns=np.arange(T))

        for c in range(self.parameters.C):
            pi_0 = self.general_chmm_para.pi_0.loc[self.parameters.channels_name_unique[c]]
            temp = self._select_P(pi_0)
            # print(temp)
            channels_hidden_states.loc[self.parameters.channels_name_unique[c], 0] = temp.loc[0]
            channels_observation.loc[self.parameters.channels_name_unique[c], 0] \
                = self._gmm_gen(temp.loc[0],self.parameters.channels_name_unique[c])

        dict_states2num = self.parameters.dict_states2num

        for t in range(1, T):
            current_states = dict_states2num.loc[channels_hidden_states.loc[:, t - 1]]
            for c in range(self.parameters.C):
                current_pi = self.general_chmm_para.pi_0.loc[self.parameters.channels_name_unique[c]]
                values_to_set  = self.general_chmm_para.transition_matrices.loc[ tuple(current_states.values), self.parameters.channels_name_unique[c]].values[0]
                values_to_set = values_to_set.astype(np.float64)
                #values_to_set = np.array(values_to_set, dtype=np.float64)
                current_pi.loc[:, 0] = values_to_set
                temp = self._select_P(current_pi)
                channels_hidden_states.loc[self.parameters.channels_name_unique[c], t] = temp.loc[0]
                channels_observation.loc[self.parameters.channels_name_unique[c], t] = self._gmm_gen(temp.loc[0],self.parameters.channels_name_unique[c])

        return channels_observation, channels_hidden_states

    def _gmm_pdf_fast(self, obs):

        P_temp = self.parameters.gmm_para_P
        mu_temp = self.parameters.gmm_para_mu
        sig_temp = self.parameters.gmm_para_sigma

        T = obs.shape[1]
        max_state_num = np.max(self.parameters.channel_state_num)
        max_gmm_num = np.max(self.parameters.num_gmm_component)
        dimension_numbers_reshape = np.concatenate([[0], np.cumsum(self.parameters.channel_obs_dim)])

        np_obs = np.array(obs, dtype=np.float64)
        np_mu = np.array(mu_temp, dtype=np.float64)
        np_sig = np.array(sig_temp, dtype=np.float64)
        np_P = np.array(P_temp, dtype=np.float64)

        np_mu = np.reshape(np_mu, (dimension_numbers_reshape[-1], max_state_num, max_gmm_num))
        np_sig = np.reshape(np_sig, (dimension_numbers_reshape[-1], max_state_num, max_gmm_num))
        np_P = np.reshape(np_P, (self.parameters.C, max_state_num, max_gmm_num))

        np_mu[np.isnan(np_mu)] = 0
        np_sig[np.isnan(np_sig)] = 1
        np_P[np.isnan(np_P)] = 0

        np_obs = np.repeat(np_obs[:, :, np.newaxis], max_state_num, axis=2)
        np_obs = np.repeat(np_obs[:, :, :, np.newaxis], max_gmm_num, axis=3)

        np_mu = np.repeat(np_mu[:, np.newaxis, :, :], T, axis=1)
        np_sig = np.repeat(np_sig[:, np.newaxis, :, :], T, axis=1)
        np_P = np.repeat(np_P[:, np.newaxis, :, :], T, axis=1)

        ind_nan_obs = np.isnan(np_obs)  # np.where(np.isnan(np_obs))
        np_obs[ind_nan_obs] = np_mu[ind_nan_obs]

        quad_form = np.divide(np.power(np_obs - np_mu, 2), np_sig)
        logSqrtDetSigma = 0.5 * np.log(np_sig)
        logSqrtDetSigma[ind_nan_obs] = 0

        if np.sum(self.parameters.channel_obs_dim == 1) == self.parameters.C:
            PDFs = np.multiply(np_P, np.exp(-0.5 * quad_form - logSqrtDetSigma - np.log(2 * np.pi) / 2))
        else:
            D = np.log(2 * np.pi) / 2
            D = np.ones(quad_form.shape) * D
            D[ind_nan_obs] = 0
            logValue = np.cumsum(-0.5 * quad_form - logSqrtDetSigma - D, axis=0)
            logValue = np.concatenate([np.zeros((1, T, max_state_num, max_gmm_num)), logValue])

            PDFs = np.multiply(np_P, np.exp(
                logValue[dimension_numbers_reshape[1:], :, :, :] - logValue[dimension_numbers_reshape[:-1], :, :, :]))
        PDFs = np.moveaxis(PDFs, [2, 0, 3, 1], [0, 1, 2, 3])
        P_observ_cond_to_state_comp = np.reshape(PDFs, (PDFs.shape[0] * PDFs.shape[1], PDFs.shape[2], T), 'F')

        from_state_num = np.arange(len(self.parameters.channel_state_num)) * max_state_num
        to_state_num = from_state_num + self.parameters.channel_state_num
        ind_valid = np.zeros(len(self.parameters.channel_state_num) * max_state_num, dtype=bool)

        for a, b in zip(from_state_num, to_state_num):
            ind_valid[a:b] = True

        P_observ_cond_to_state_comp = P_observ_cond_to_state_comp[ind_valid, :, :]
        P_observ_cond_to_state = np.sum(P_observ_cond_to_state_comp, axis=1)

        return P_observ_cond_to_state, P_observ_cond_to_state_comp

    def compute_joint_state_observation_prob(self, obs, latent_states):

        T = obs.shape[1]

        self.get_general_para()
        channel_names = self.parameters.channels_name_unique.copy()
        P_observ_cond_to_state, P_observ_cond_to_state_comp = self._gmm_pdf_fast(obs)
        pd_P_observ_cond_to_state = pd.DataFrame([], index=[self.parameters.channel_names,
                                                            self.parameters.states_of_channels],
                                                 columns=list(range(T)), dtype=float)
        pd_P_observ_cond_to_state.loc[:] = P_observ_cond_to_state

        log_joint_prob = 0
        t = 0

        for zee in range(self.general_chmm_para.C):
            log_joint_prob += np.log(
                self.general_chmm_para.pi_0.loc[(channel_names[zee], latent_states.loc[channel_names[zee], t]), t]) + \
                              np.log(pd_P_observ_cond_to_state.loc[
                                         (channel_names[zee], latent_states.loc[channel_names[zee], t]), t])

        latent_states_num = latent_states.map(lambda x: int(x[-3:]))
        for t in range(1, T):
            for zee in range(self.general_chmm_para.C):
                temp_index = latent_states_num.loc[:, t].values
                log_joint_prob += np.log(self.general_chmm_para.transition_matrices.loc[tuple(temp_index), (
                    channel_names[zee], latent_states.loc[channel_names[zee], t])]) + \
                                  np.log(pd_P_observ_cond_to_state.loc[
                                             (channel_names[zee], latent_states.loc[channel_names[zee], t]), t])

        return log_joint_prob

    def forward_backward_lsim(self, obs, flag_EM=False):

        T = obs.shape[1]
        C = self.parameters.C
        lsim_para = self.parameters
        max_gmm_num = np.max(self.parameters.num_gmm_component)
        transition_matrices = np.array(lsim_para.transition_matrices)
        state_numbers_index = np.concatenate([[-1], np.cumsum(lsim_para.channel_state_num) - 1])

        sum_states = state_numbers_index[-1] + 1
        P_ot_c_cond_past = np.zeros([C, T])
        b_c_ot_nc = np.zeros([sum_states, T])

        alpha = np.zeros([sum_states, T])
        alpha_T = np.zeros([sum_states, T])
        beta = np.zeros([sum_states, T])
        ## beta2 = np.zeros([sum_states, T])
        a_alpha_b = np.zeros([sum_states, C, T - 1])
        P_observ_cond_to_state, P_observ_cond_to_state_comp = self._gmm_pdf_fast(obs)
        P_vz_tm1_vw_t_O = np.zeros([sum_states, sum_states, T])
        P_vz_tm1_vw_t_O_cond = np.zeros([sum_states, sum_states, T])

        mat_mult = np.zeros([sum_states, C])
        cuopling_repmat = np.zeros([sum_states, C])
        cuopling_repmat_beta = np.zeros([sum_states, C])
        weights_subsystems = np.zeros([sum_states, C])
        weights_subsystems_opt = np.zeros([sum_states, C])

        coupling_transmat = np.zeros([sum_states, sum_states])
        diag_zero = np.ones([sum_states, sum_states])

        mat_mult_one = np.zeros([sum_states, C])
        vec_index_x = []
        vec_index_y = []

        for zee in range(C):
            zee_index = np.arange(state_numbers_index[zee] + 1, state_numbers_index[zee + 1] + 1)
            vec_index_x.extend(np.repeat(int(zee), lsim_para.channel_state_num[zee], axis=0))
            vec_index_y.extend(np.arange(state_numbers_index[zee] + 1, state_numbers_index[zee + 1] + 1))

            zee_coupling = lsim_para.coupling_theta_IM[:, zee]
            cuopling_repmat[zee_index, :] = np.repeat(zee_coupling[:, np.newaxis], lsim_para.channel_state_num[zee],
                                                      axis=1).transpose()

            diag_zero[zee_index[0]:zee_index[-1] + 1, zee_index[0]:zee_index[-1] + 1] = 0

            zee_coupling = lsim_para.coupling_theta_IM[zee, :]
            cuopling_repmat_beta[zee_index, :] = np.repeat(zee_coupling[:, np.newaxis],
                                                           lsim_para.channel_state_num[zee], axis=1).transpose()
            weights_subsystems_opt[zee_index, :] = np.repeat(zee_coupling[:, np.newaxis],
                                                             lsim_para.channel_state_num[zee], axis=1).transpose()

        for zee in range(C):
            zee_index = np.arange(state_numbers_index[zee] + 1, state_numbers_index[zee + 1] + 1)
            coupling_transmat[:, zee_index] = np.multiply(cuopling_repmat_beta[:, zee][:, np.newaxis],
                                                          transition_matrices[:, zee_index])

        vec_mat_mult = (vec_index_y, vec_index_x)
        mat_mult_one[vec_mat_mult] = 1

        t = 0
        alpha[:, t] = lsim_para.pi_0[:].values[:, 0]
        P_ot_c_cond_past[:, t] = np.dot(mat_mult_one.transpose(),
                                        np.multiply(alpha[:, t], P_observ_cond_to_state[:, t]))
        b_c_ot_nc[:, t] = np.divide(P_observ_cond_to_state[:, t], np.dot(mat_mult_one, P_ot_c_cond_past[:, t]))

        for t in range(1, T):

            alpha_tm1_tm1 = np.multiply(alpha[:, t - 1], b_c_ot_nc[:, t - 1])
            mat_mult[vec_mat_mult] = alpha_tm1_tm1

            a_alpha_b[:, :, t - 1] = np.dot(transition_matrices.transpose(), mat_mult)
            a_alpha_b_coupled = np.multiply(a_alpha_b[:, :, t - 1], cuopling_repmat)
            alpha[:, t] = np.sum(a_alpha_b_coupled, axis=1)

            if np.abs(np.sum(alpha[:, t]) - C) > 0.0001:  # fordebug
                yu = 0

            P_ot_c_cond_past[:, t] = np.dot(mat_mult_one.transpose(),
                                            np.multiply(alpha[:, t], P_observ_cond_to_state[:, t]))
            b_c_ot_nc[:, t] = np.divide(P_observ_cond_to_state[:, t], np.dot(mat_mult_one, P_ot_c_cond_past[:, t]))

            P_vz_tm1_vw_t_O_cond[:, :, t] = coupling_transmat + np.dot(np.multiply(alpha_tm1_tm1, diag_zero),
                                                                       coupling_transmat)
            P_vz_tm1_vw_t_O[:, :, t] = np.multiply(alpha_tm1_tm1, P_vz_tm1_vw_t_O_cond[:, :, t])

        beta[:, -1] = b_c_ot_nc[:, -1]
        alpha_T[:, -1] = np.multiply(alpha[:, -1], beta[:, -1])

        flag_nan = np.sum(np.sum(lsim_para.coupling_theta_IM, axis=1) < 0.0000000001) > 0;
        channel_numbers_index = np.concatenate([[-1], np.arange(C, C * C + 1, C) - 1])

        for t in range(T - 2, -1, -1):

            mat_mult[vec_mat_mult] = beta[:, t + 1]
            # beta_C2 = transitions_matrices * mat_mult;# compute from t=1 to t = T - 1

            beta_C = np.dot(P_vz_tm1_vw_t_O_cond[:, :, t + 1], mat_mult)

            if C > 1:
                temp = P_vz_tm1_vw_t_O[:, :, t + 1]
                temp_2d_f = np.divide(np.power(temp, 2), alpha[:, t + 1])
                temp_aplha = np.multiply(alpha[:, t], b_c_ot_nc[:, t])

                h_zee_temp = np.cumsum(np.power(temp_aplha, 2))
                # h_zee_all = h_zee_temp[state_numbers_index[1:]] - np.concatenate([[0], h_zee_temp[state_numbers_index[1:C]]])
                h_zee_all = h_zee_temp[state_numbers_index[1:]]  # instead of concatenation
                h_zee_all[1:] = h_zee_all[1:] - h_zee_temp[state_numbers_index[1:C]]  # instead of concatenation
                h_zee_all = np.repeat(h_zee_all, C)

                f_zee_temp1 = np.cumsum(temp_2d_f, axis=0)
                f_zee_temp = f_zee_temp1[state_numbers_index[1:], :]  # instead of concatenation
                f_zee_temp[1:, :] = f_zee_temp[1:, :] - f_zee_temp1[state_numbers_index[1:C],
                                                        :]  # instead of concatenation
                f_zee_temp = np.cumsum(f_zee_temp, axis=1)
                f_zee_all = f_zee_temp[:, state_numbers_index[1:]]
                f_zee_all[:, 1:] = f_zee_all[:, 1:] - f_zee_temp[:, state_numbers_index[1:C]]
                f_zee_all = f_zee_all.flatten('F') + 0.000001

                nume_temp1 = np.cumsum(np.divide(f_zee_all, (f_zee_all - h_zee_all)), axis=0)
                nume_temp = nume_temp1[channel_numbers_index[1:]]  # instead of concatenation
                nume_temp[1:] = nume_temp[1:] - nume_temp1[channel_numbers_index[1:C]]  # instead of concatenation
                nume_temp = np.multiply(np.repeat(nume_temp, C), h_zee_all)

                denume_temp2 = np.cumsum(np.divide(1, (f_zee_all - h_zee_all)), axis=0)
                denume_temp = denume_temp2[channel_numbers_index[1:]]  # instead of concatenation
                denume_temp[1:] = denume_temp[1:] - denume_temp2[channel_numbers_index[1:C]]  # instead of concatenation
                denume_temp = 1 + np.multiply(np.repeat(denume_temp, C), h_zee_all)

                g_zee_all = np.divide(nume_temp, denume_temp)
                d_hat = np.divide((f_zee_all - g_zee_all), (f_zee_all - h_zee_all))

                temp_delta = np.reshape(d_hat, (C, C))
                mat_mult[vec_mat_mult] = 1
                weights_subsystems = np.dot(mat_mult, temp_delta)

            else:
                mat_mult[vec_mat_mult] = 1
                weights_subsystems = mat_mult

            temp_sum = np.sum(np.multiply(weights_subsystems, beta_C), axis=1)
            temp_sum[temp_sum < 0.0000001] = 0.0000001
            beta[:, t] = np.multiply(b_c_ot_nc[:, t], temp_sum)

            if flag_nan:
                ind_nan = np.isnan(beta[:, t])
                beta[ind_nan, t] = b_c_ot_nc[ind_nan, t]

            alpha_T[:, t] = np.multiply(beta[:, t], alpha[:, t])

            temp_sum = np.cumsum(alpha_T[:, t])
            normalizer = temp_sum[state_numbers_index[1:]]  # instead of concatenation
            normalizer[1:] = normalizer[1:] - temp_sum[state_numbers_index[1:C]]  # instead of concatenation
            normalizer_vec = np.dot(mat_mult_one, normalizer)

            alpha_T[:, t] = np.divide(alpha_T[:, t], normalizer_vec)
            beta[:, t] = np.divide(beta[:, t], normalizer_vec)

        P_O_model = np.sum(np.log(P_ot_c_cond_past.flatten()))

        # return P_O_model ,alpha_out , beta_out , alpha_T_out , b_c_ot_nc_out , P_observ_cond_to_state_out , P_observ_cond_to_state_comp_out
        if flag_EM:
            return P_O_model, alpha, beta, alpha_T, b_c_ot_nc, P_observ_cond_to_state, P_observ_cond_to_state_comp
        else:
            alpha_out = pd.DataFrame(alpha, index=[lsim_para.channel_names, lsim_para.states_of_channels],
                                     columns=[np.arange(T)], dtype=float)
            beta_out = pd.DataFrame(beta, index=[lsim_para.channel_names, lsim_para.states_of_channels],
                                    columns=[np.arange(T)], dtype=float)
            alpha_T_out = pd.DataFrame(alpha_T, index=[lsim_para.channel_names, lsim_para.states_of_channels],
                                       columns=[np.arange(T)], dtype=float)
            b_c_ot_nc_out = pd.DataFrame(b_c_ot_nc, index=[lsim_para.channel_names, lsim_para.states_of_channels],
                                         columns=[np.arange(T)], dtype=float)
            P_observ_cond_to_state_out = pd.DataFrame(P_observ_cond_to_state,
                                                      index=[lsim_para.channel_names, lsim_para.states_of_channels],
                                                      columns=[np.arange(T)], dtype=float)

            P_observ_cond_to_state_comp_out = pd.DataFrame(
                P_observ_cond_to_state_comp.reshape((sum_states * max_gmm_num, T)),
                index=[lsim_para.channels_states_gmm_joint_1, lsim_para.channels_states_gmm_joint_2,
                       lsim_para.channels_states_gmm_joint_3],
                columns=[np.arange(T)], dtype=float)

            return P_O_model, alpha_out, beta_out, alpha_T_out, b_c_ot_nc_out, P_observ_cond_to_state_out, P_observ_cond_to_state_comp_out

    def em_lsim(self, obs, max_itration, extra_options):

        try:
            if len(obs.columns.levels) != 2:
                obs.columns = pd.MultiIndex.from_product([['obs:0001'], np.arange(obs.columns.shape[0])])
        except:
            obs.columns = pd.MultiIndex.from_product([['obs:0001'], np.arange(obs.columns.shape[0])])

        num_trials = len(obs.columns.levels[0])
        name_of_trails = obs.columns.levels[0]
        length_observation = np.ones(len(name_of_trails), dtype=int)
        for tr in range(num_trials):
            length_observation[tr] = np.int64(obs.loc[:, name_of_trails[tr]].shape[1])

        all_observation = np.array(obs, dtype=np.float64)
        T_all = np.int64(np.sum(length_observation))
        C = self.parameters.C
        lsim_para = self.parameters

        max_state_num = np.max(lsim_para.channel_state_num)
        max_gmm_num = np.max(lsim_para.num_gmm_component)
        dimension_numbers_reshape = np.concatenate([[0], np.cumsum(lsim_para.channel_obs_dim)])
        dimension_numbers_index = np.concatenate([[-1], np.cumsum(lsim_para.channel_obs_dim) - 1])
        state_numbers_index = np.concatenate([[-1], np.cumsum(lsim_para.channel_state_num) - 1])
        trial_time_index = np.concatenate([[-1], np.cumsum(length_observation) - 1])
        state_names = self.parameters.states_name_unique.copy()
        sum_states = state_numbers_index[-1] + 1
        ch_names = lsim_para.channels_name_unique
        state_names = lsim_para.states_name_unique
        gmm_names = lsim_para.gmm_names_unique

        # C = lsim_para.C
        flag_EM = True

        # GMM initialization
        for zee in range(C):

            Y = obs.loc[ch_names[zee], :]
            Y = np.array(Y)

            # fit a Gaussian Mixture Model with N components
            init_gmm_numbers = lsim_para.channel_state_num[zee] * lsim_para.num_gmm_component[zee]
            clf = mixture.GaussianMixture(n_components=init_gmm_numbers, covariance_type='diag', max_iter=10)
            clf.fit(Y.transpose())

            temp_sigma = clf.covariances_.copy()
            temp_mu = clf.means_.copy()
            P = clf.weights_.copy()

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

                lsim_para.gmm_para_P.loc[ch_names[zee], state_names[i]] = (lsim_para.gmm_para_P.loc[ch_names[zee], state_names[i]].values /
                                                                           np.nansum(lsim_para.gmm_para_P.loc[ch_names[zee], state_names[i]].values))
                lsim_para.pi_0.loc[ch_names[zee]] = 1 / lsim_para.channel_state_num[zee]
                lsim_para.transition_matrices.loc[:, ch_names[zee]] = 1 / lsim_para.channel_state_num[zee]

        lsim_para.coupling_theta_IM = np.ones((C, C)) / C

        alpha_T_trials = np.zeros((sum_states, T_all))
        alpha_trials = np.zeros((sum_states, T_all))
        beta_trials = np.zeros((sum_states, T_all))

        b_c_ot_nc_trials = np.zeros((sum_states, T_all))
        P_observ_cond_to_state_trials = np.zeros((sum_states, T_all))
        P_observ_cond_to_state_comp_trials = np.zeros((sum_states, max_gmm_num, T_all))

        gamma_trail_time_index = 1 + trial_time_index[0:-1]
        L_T_1 = T_all - num_trials
        P_O_model = np.zeros((num_trials))

        coupling_transition_all = np.zeros((sum_states, sum_states))

        mat_mult_one = np.zeros([sum_states, C])
        vec_index_x = []
        vec_index_y = []
        for zee in range(C):
            vec_index_x.extend(np.repeat(int(zee), lsim_para.channel_state_num[zee], axis=0))
            vec_index_y.extend(np.arange(state_numbers_index[zee] + 1, state_numbers_index[zee + 1] + 1))
        vec_mat_mult = (vec_index_y, vec_index_x)
        mat_mult_one[vec_mat_mult] = 1

        log_likelyhood = np.nan * np.ones((max_itration))

        for itr in range(max_itration):

            for zee in range(C):
                zee_index = np.arange(state_numbers_index[zee] + 1, state_numbers_index[zee + 1] + 1)
                coupling_transition_all[:, zee_index] = np.multiply(lsim_para.transition_matrices.loc[:, ch_names[zee]],
                                                                    np.dot(mat_mult_one,
                                                                           lsim_para.coupling_theta_IM[:, zee])[:,
                                                                    np.newaxis])

            sum3_zeta = 0

            for tr in np.arange(num_trials, dtype=int):
                P_O_model[tr], alpha_trials[:, trial_time_index[tr] + 1:trial_time_index[tr + 1] + 1], beta_trials[:,trial_time_index[tr] + 1: trial_time_index[ tr + 1] + 1], \
                alpha_T_trials[:,trial_time_index[tr] + 1: trial_time_index[tr + 1] + 1], b_c_ot_nc_trials[ :, trial_time_index[tr] + 1:trial_time_index[tr + 1] + 1], \
                P_observ_cond_to_state_trials[:, trial_time_index[tr] + 1:trial_time_index[tr + 1] + 1], P_observ_cond_to_state_comp_trials[:, :, trial_time_index[ tr] + 1:trial_time_index[ tr + 1] + 1] \
                    = self.forward_backward_lsim(obs.loc[:, name_of_trails[tr]], flag_EM)

                zeta_part1 = np.repeat(np.multiply(alpha_trials[:, trial_time_index[tr] + 1:trial_time_index[tr + 1]],
                                                   b_c_ot_nc_trials[:,
                                                   trial_time_index[tr] + 1:trial_time_index[tr + 1]])[:, np.newaxis,
                                       :], state_numbers_index[-1] + 1, axis=1)
                zeta_part2 = np.repeat(
                    beta_trials[:, trial_time_index[tr] + 2:trial_time_index[tr + 1] + 1][np.newaxis, :, :],
                    state_numbers_index[-1] + 1, axis=0)
                sum3_zeta += np.sum(np.multiply(zeta_part1, zeta_part2), axis=2)

            sum3_zeta = np.multiply(sum3_zeta, coupling_transition_all)

            gamma_P0 = alpha_T_trials[:, gamma_trail_time_index]
            lsim_para.pi_0.loc[:] = np.mean(gamma_P0, axis=1)[:, np.newaxis]
            temp_coupling = np.zeros(sum_states + 1)

            for zee in range(C):
                zee_index = np.arange(state_numbers_index[zee] + 1, state_numbers_index[zee + 1] + 1)

                sum3_Zeta_zee = sum3_zeta[:, zee_index]
                sum32_Zeta_zee = np.sum(sum3_Zeta_zee, axis=1)
                temp_Trans_zee = np.divide(sum3_Zeta_zee, sum32_Zeta_zee[:, np.newaxis])
                temp_Trans_zee[np.isnan(temp_Trans_zee)] = 1 / lsim_para.channel_state_num[zee]
                lsim_para.transition_matrices.loc[:, ch_names[zee]] = temp_Trans_zee

                temp_coupling[1:] = np.cumsum(sum32_Zeta_zee)
                lsim_para.coupling_theta_IM[:, zee] = (temp_coupling[state_numbers_index[1:] + 1] - temp_coupling[state_numbers_index[:-1] + 1]) / L_T_1

                for st in range(lsim_para.channel_state_num[zee]):
                    temp_gamma_obs = np.zeros((lsim_para.num_gmm_component[zee], T_all))

                    for k in range(lsim_para.num_gmm_component[zee]):
                        temp_gamma_obs[k, :] = np.divide(np.multiply(alpha_T_trials[zee_index[st], :],
                                                                     P_observ_cond_to_state_comp_trials[zee_index[st],
                                                                     k, :]),
                                                         P_observ_cond_to_state_trials[zee_index[st], :])
                        temp_gamma_obs[k, np.isnan(temp_gamma_obs[k, :])] = 0

                        Y_update = np.array(obs.loc[ch_names[zee]])
                        Y_update[np.isnan(Y_update)] = 0
                        weight_update = temp_gamma_obs[k, :]
                        if np.sum(weight_update > 0) < 2:
                            weight_update += 1

                        lsim_para.gmm_para_mu.loc[ch_names[zee], (state_names[st], gmm_names[k])] = np.divide(
                            np.sum(np.multiply(Y_update, weight_update), axis=1), np.sum(weight_update))

                        Recentering = np.subtract(Y_update, lsim_para.gmm_para_mu.loc[ch_names[zee], (state_names[st], gmm_names[k])].to_numpy()[:,np.newaxis])
                        lsim_para.gmm_para_sigma.loc[ch_names[zee], (state_names[st], gmm_names[k])] \
                            = 0.001 * np.nanvar(Recentering,axis=1) + np.diagonal(np.divide(
                                np.dot(np.multiply(Recentering, weight_update), Recentering.transpose()), np.sum(weight_update)))

                    lsim_para.gmm_para_P.loc[
                        ch_names[zee], (state_names[st], slice(gmm_names[0], gmm_names[k]))] = np.divide(
                        np.sum(temp_gamma_obs, axis=1), np.sum(temp_gamma_obs))

            lsim_para.pi_steady = np.sum(alpha_T_trials, axis=1) / T_all
            log_likelyhood[itr] = np.sum(P_O_model)

            if extra_options['plot'] and np.remainder(itr, 10) == 0:
                k = 0

        if extra_options['plot']:
            pd.options.plotting.backend = "plotly"

            df_loglik = pd.DataFrame(log_likelyhood, dtype=float)
            df_loglik.plot().show()

        # transition_matrices = np.array(lsim_para.transition_matrices)
        # pi_0 = np.array(lsim_para.pi_0)
        # coupling_tetha_IM = np.array(lsim_para.coupling_theta_IM)
        self.parameters = lsim_para
        return lsim_para

    def _select_P(self, pi_0):

        temp_rand = np.random.rand()
        return np.cumsum(pi_0).gt(temp_rand).loc[:].idxmax()

    def _gmm_gen(self, state, channel):

        P = self.parameters.gmm_para_P.loc[channel, state]
        P = P.dropna()

        if P.__len__() > 1:

            # P.index = P.index.droplevel()
            current_gmm = self._select_P(P)

        else:
            current_gmm = P.index[0]

        mu_temp = self.parameters.gmm_para_mu.loc[channel, (state, current_gmm)].to_numpy()
        sig_temp = self.parameters.gmm_para_sigma.loc[channel, (state, current_gmm)].to_numpy()
        # mu_temp = self.parameters.gmm_para_mu[ state, current_gmm].loc[channel]
        # sig_temp = self.parameters.gmm_para_sigma[ state, current_gmm].loc[channel]

        # sig_temp.columns = sig_temp.columns.droplevel()
        return np.sqrt(sig_temp) * np.random.randn(sig_temp.__len__()) + mu_temp

    def plot_chmm_timeseries(self, obs, channel_number):

        pd.options.plotting.backend = "plotly"

        df = pd.DataFrame(obs.loc[self.parameters.channels_name_unique[channel_number - 1]].values.transpose(),
                          columns=self.parameters.dim_names_unique[
                                  0:self.parameters.channel_obs_dim[channel_number - 1]], dtype=float)
        df.plot().show()

    def chmm_cart_prod(self):
        self.get_general_para()
        old_lsim_para = self.parameters
        C = 1
        channel_state_num = np.array([np.prod(old_lsim_para.channel_state_num)])
        channel_obs_dim = np.array([sum(old_lsim_para.channel_obs_dim)])
        num_gmm_component = np.array([1])
        eq_hmm = lsim(C, channel_state_num, channel_obs_dim, num_gmm_component)

        temp = np.ones(1)
        for c in range(old_lsim_para.C):
            temp = np.kron(temp, old_lsim_para.pi_0.loc[old_lsim_para.channels_name_unique[c]])

        eq_hmm.parameters.pi_0 = temp

        index_chmm_colomn = []
        for zee in range(old_lsim_para.C):
            temp_index = old_lsim_para.channel_state_num
            temp_index = temp_index[zee + 1:]
            temp_raw = np.kron(range(1, old_lsim_para.channel_state_num[zee] + 1),
                               np.ones((1, int(np.prod(temp_index)))))

            temp_index = old_lsim_para.channel_state_num
            temp_index = temp_index[:zee]
            index_chmm_colomn.append(np.kron(np.ones((1, int(np.prod(temp_index)))), temp_raw)[0].tolist())

        index_chmm_colomn_np = np.array(index_chmm_colomn, dtype=np.int64)

        for s in range(channel_state_num[0]):

            temp_index = index_chmm_colomn_np[:, s]

            temp = np.ones(1)
            for c in range(old_lsim_para.C):
                temp = np.kron(temp, self.general_chmm_para.transition_matrices.loc[
                    tuple(temp_index), old_lsim_para.channels_name_unique[c]])

            eq_hmm.parameters.transition_matrices.loc[
                (eq_hmm.parameters.channels_name_unique[0], eq_hmm.parameters.states_name_unique[s])] = temp.copy()

        dimension_numbers_index = np.concatenate([[-1], np.cumsum(old_lsim_para.channel_obs_dim) - 1])

        channels_name_unique = eq_hmm.parameters.channels_name_unique
        dim_names_unique = eq_hmm.parameters.dim_names_unique
        states_name_unique = eq_hmm.parameters.states_name_unique

        for c in range(old_lsim_para.C):
            for s in range(old_lsim_para.channel_state_num[c]):
                row_num = np.where(index_chmm_colomn_np[c, :] == s + 1)[0]

                mu_temp = old_lsim_para.gmm_para_mu.loc[
                    old_lsim_para.channels_name_unique[c], tuple([states_name_unique[s], 'gmm:001'])]
                bb = np.tile(mu_temp, [row_num.shape[0], 1]).transpose()
                eq_hmm.parameters.gmm_para_mu.loc[(channels_name_unique[0],
                                                   slice(dim_names_unique[dimension_numbers_index[c] + 1],
                                                         dim_names_unique[dimension_numbers_index[c + 1]])), [
                                                      states_name_unique[i] for i in row_num]] = bb.copy()

                sig_temp = old_lsim_para.gmm_para_sigma.loc[
                    old_lsim_para.channels_name_unique[c], tuple([states_name_unique[s], 'gmm:001'])]
                bb = np.tile(sig_temp, [row_num.shape[0], 1]).transpose()
                eq_hmm.parameters.gmm_para_sigma.loc[(channels_name_unique[0],
                                                      slice(dim_names_unique[dimension_numbers_index[c] + 1],
                                                            dim_names_unique[dimension_numbers_index[c + 1]])), [
                                                         states_name_unique[i] for i in row_num]] = bb.copy()

        return eq_hmm
