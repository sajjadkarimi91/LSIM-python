# https://realpython.com/python-first-steps/

import numpy as np
import pandas as pd


C = 3
channel_obs_dim = [4, 6, 3]
T = 100

temp_1 = np.random.randn(channel_obs_dim[0], T)
temp_2 = 1 + np.random.randn(channel_obs_dim[1], T)
temp_3 = 3 + np.random.randn(channel_obs_dim[2], T)

#%%

def make_channels_title(C_in, channel_obs_dim_in):
    channels_name_res = []
    dimension_of_channels_res = []
    channels_name_unique_res = []
    for c in range( C_in):
        c_ch = c+1
        channels_name_unique_res.append('channel:' + c_ch.__str__())
        for d in range(1, channel_obs_dim_in[c]+1):
            channels_name_res.append('channel:'+c_ch.__str__())
            dimension_of_channels_res.append(d)

    return channels_name_res, dimension_of_channels_res, channels_name_unique_res


channels_name, dimension_of_channels, channels_name_unique = make_channels_title(C, channel_obs_dim)

time_series = pd.DataFrame(np.concatenate([temp_1, np.concatenate([temp_2, temp_3])]),
                           index=[channels_name, dimension_of_channels])

print(time_series)

# time_series.loc[channels_name[0]]

for c in range( C):
    print(time_series.loc[channels_name_unique[c]])


aa = time_series.transpose().cov()  # np.dot(time_series, time_series.transpose())

aa_exp = pd.DataFrame(np.exp(aa), index=[channels_name, dimension_of_channels])

aa_hat = np.log(aa_exp)
# bb_hat = math.log1p(aa_exp) error
diff_0 = np.subtract(aa, aa_hat)

print(diff_0)

# https://realpython.com/python-first-steps/

number_list = [2,2,3,4]
mixed_list = ["Hi python",[1,2,3],"Ali"]
cat_list = number_list + mixed_list
cat_list.append("new")


print(cat_list.pop(2))
print(cat_list)

my_dict = {5: "ali", "name": "apple"}

print(my_dict["name"])

