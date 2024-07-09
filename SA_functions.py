import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pedophysics import predict, Soil
from PM import *

from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


def r2_inv(survey, dfsForward, coils, r2):

    for coil in coils:
        obsECa = survey.df[coil].values
        simECa = dfsForward[coil].values
        r2.loc[0, coil] = r2_score(simECa, obsECa)
    obsECa = survey.df[coils].values.flatten()
    simECa = dfsForward[coils].values.flatten()
    r2.loc[0, 'all'] = r2_score(simECa, obsECa)
    return r2


def check_uniformity_and_interpolate(df, profile_id_col, depth_col, *data_cols):
    # Initialize a DataFrame to store both interpolated and non-interpolated profiles
    all_profiles_df = pd.DataFrame()
    uniform_intervals = {}

    # Prepare the figure for plotting non-uniform depth intervals
    plt.figure(figsize=(10, 8))
    
    # Keep track of any profiles that have been plotted (i.e., non-uniform intervals)
    plotted_profiles = []

    for profile_id, group in df.groupby(profile_id_col):
        depth_diffs = group[depth_col].diff().dropna().abs()
        
        # Find the most common interval
        common_interval = depth_diffs.mode()[0]
        
        # Check if the depth differences are uniform
        if np.isclose(depth_diffs, common_interval).all():
            # Record the uniform interval
            uniform_intervals[profile_id] = common_interval
            # Add this profile's data as is to the all_profiles_df
            all_profiles_df = pd.concat([all_profiles_df, group])
        else:
            # Add profile to the list of plotted profiles for the combined histogram
            plotted_profiles.append(profile_id)
            #plt.hist(depth_diffs, bins=30, alpha=0.5, label=f'Profile ID {profile_id}')

            # Interpolate to make uniform intervals
            interp_funcs = {col: interp1d(group[depth_col], group[col], kind='linear', bounds_error=False, fill_value='extrapolate') for col in data_cols}
            new_depths = np.arange(group[depth_col].min(), group[depth_col].max() + common_interval, common_interval)
            interpolated_values = {col: interp_funcs[col](new_depths) for col in data_cols}
            interpolated_df = pd.DataFrame(interpolated_values, index=new_depths)
            interpolated_df[profile_id_col] = profile_id  # Add the profile ID

            # Copy other non-interpolated columns
            for col in df.columns.difference([profile_id_col, depth_col] + list(data_cols)):
                interpolated_df[col] = group.iloc[0][col]
            
            interpolated_df.reset_index(inplace=True)
            interpolated_df.rename(columns={'index': depth_col}, inplace=True)
            all_profiles_df = pd.concat([all_profiles_df, interpolated_df])
    
    # # Plot the combined histogram of depth intervals for all non-uniform profiles
    # if plotted_profiles:
    #     plt.title('Depth Interval Distributions for Non-uniform Profiles')
    #     plt.xlabel('Depth Intervals')
    #     plt.ylabel('Frequency')
    #     plt.legend()
    #     plt.show()
    # else:
    #     plt.close()

    # Reset index for the concatenated DataFrame
    all_profiles_df.reset_index(drop=True, inplace=True)

    return all_profiles_df, uniform_intervals


def deterministic(feature, target, df, Y0, f_ec, clay_mean, bd_mean, water_ec_hp_mean_t, t_mean_conv, clay_10cm, bd_10cm, water_ec_hp_10cm_t, t_10cm_conv, clay_50cm, bd_50cm, water_ec_hp_50cm_t, t_50cm_conv, t_conv, iters=100, round_n=3):
    df.reset_index(drop=True, inplace=True)

    ### Extract indices for 10 cm and 50 cm layers
    idx_layer_10 = df[df['depth'] == 10].index
    idx_layer_50 = df[df['depth'] == 50].index
    ### Select data at 10 cm depth
    X_layer_10 = df.loc[idx_layer_10, feature].values.reshape(-1, 1)
    Y_layer_10 = df.loc[idx_layer_10, target].values
    ### Select data at 50 cm depth
    X_layer_50 = df.loc[idx_layer_50, feature].values.reshape(-1, 1)
    Y_layer_50 = df.loc[idx_layer_50, target].values

    # Preallocate lists
    DR2_LS, MAE_LS, STD_LS, DRMSE_LS = [None] * iters, [None] * iters, [None] * iters, [None] * iters
    DR2_LT, MAE_LT, STD_LT, DRMSE_LT = [None] * iters, [None] * iters, [None] * iters, [None] * iters
    DR2_10, MAE_10, STD_10, DRMSE_10 = [None] * iters, [None] * iters, [None] * iters, [None] * iters
    DR2_50, MAE_50, STD_50, DRMSE_50 = [None] * iters, [None] * iters, [None] * iters, [None] * iters
    DR2_ID, MAE_ID, STD_ID, DRMSE_ID, ypred_ID_ = [None] * iters, [None] * iters, [None] * iters, [None] * iters, [None] * iters

    D0R2_LS, D0RMSE_LS = [None] * iters, [None] * iters
    D0R2_LT, D0RMSE_LT = [None] * iters, [None] * iters
    D0R2_ID, D0RMSE_ID = [None] * iters, [None] * iters

    y_ = [None] * iters
    
    for i in range(iters):
        ### Split data of 10cm layer and keep track of test indices
        X_train10, X_test10, y_train10, y_test10, idx_train10, idx_test10 = train_test_split(X_layer_10, Y_layer_10, idx_layer_10, test_size=0.3, random_state=i)
        ### Split data of 50cm layer and keep track of test indices
        X_train50, X_test50, y_train50, y_test50, idx_train50, idx_test50 = train_test_split(X_layer_50, Y_layer_50, idx_layer_50, test_size=0.3, random_state=i)

        ### Combine test indices from both layers
        idx_test = np.concatenate((idx_test10, idx_test50))
        Y0_test = Y0.ID0[idx_test]
        y_test = np.concatenate((y_test10, y_test50)).flatten()
        X_test = (np.concatenate((X_test10, X_test50))/1000).flatten()
        y_[i] = y_test

        ### Predict using layer together 
        LT = Soil( 
                    bulk_ec = X_test,
                    frequency_ec=f_ec,
                    clay = clay_mean,
                    bulk_density = bd_mean,
                    water_ec = water_ec_hp_mean_t,
                    temperature = t_mean_conv
                    )
        Dypred_LT = predict.Water(LT)
        DR2_LT[i] = round(r2_score(y_test, Dypred_LT), round_n)
        DRMSE_LT[i] = round(RMSE(y_test, Dypred_LT), round_n)
        D0R2_LT[i] = round(r2_score(Y0_test, Dypred_LT), round_n)
        D0RMSE_LT[i] = round(RMSE(Y0_test, Dypred_LT), round_n)
        MAE_LT[i] = round(mean_absolute_error(y_test, Dypred_LT), round_n)
        STD_LT[i] = round(np.std(Dypred_LT), round_n)

        ### Predict using 10 cm layer
        layer_10 = Soil( 
                    bulk_ec = X_test10.flatten()/1000,
                    frequency_ec=f_ec,
                    clay = clay_10cm,
                    bulk_density = bd_10cm,
                    water_ec = water_ec_hp_10cm_t,
                    temperature = t_10cm_conv
                    )
        Dypred_10 = predict.Water(layer_10)
        DR2_10[i] = round(r2_score(y_test10, Dypred_10), round_n)
        DRMSE_10[i] = round(RMSE(y_test10, Dypred_10), round_n)
        MAE_10[i] = round(mean_absolute_error(y_test, Dypred_10), round_n)
        STD_10[i] = round(np.std(Dypred_10), round_n)

        ### Predict using 50 cm layer
        layer_50 = Soil( 
                    bulk_ec = X_test50.flatten()/1000,
                    frequency_ec=f_ec,
                    clay = clay_50cm,
                    bulk_density = bd_50cm,
                    water_ec = water_ec_hp_50cm_t,
                    temperature = t_50cm_conv
                    )
        Dypred_50 = predict.Water(layer_50)
        DR2_50[i] = round(r2_score(y_test50, Dypred_50), round_n)
        DRMSE_50[i] = round(RMSE(y_test50, Dypred_50), round_n)
        MAE_50[i] = round(mean_absolute_error(y_test, Dypred_50), round_n)
        STD_50[i] = round(np.std(Dypred_50), round_n)


        ### Stochastic modelling for layers separate. 
        ### This is a combination of both layer's prediction
        Dypred_LS = np.concatenate((Dypred_10, Dypred_50))
        DR2_LS[i] = round(r2_score(y_test, Dypred_LS), round_n)
        DRMSE_LS[i] = round(RMSE(y_test, Dypred_LS), round_n)
        MAE_LS[i] = round(mean_absolute_error(y_test, Dypred_LS), round_n)
        STD_LS[i] = round(np.std(Dypred_LS), round_n)
        D0R2_LS[i] = round(r2_score(Y0_test, Dypred_LS), round_n)
        D0RMSE_LS[i] = round(RMSE(Y0_test, Dypred_LS), round_n)

        ### Predict using ideal samples
        filtered_df = df.loc[idx_test]
        
        ID =  Soil( 
                    bulk_ec = X_test,
                    frequency_ec=f_ec,
                    clay = filtered_df['clay'].values,
                    bulk_density = filtered_df['bd'].values,
                    water_ec = filtered_df['water_ec_hp_t'].values,
                    temperature = filtered_df['temp'].values+t_conv
                    )
        
        Dypred_ID = predict.Water(ID)
        ypred_ID_[i] = Dypred_ID
        DR2_ID[i] = round(r2_score(y_test, Dypred_ID), round_n)
        DRMSE_ID[i] = round(RMSE(y_test, Dypred_ID), round_n)
        MAE_ID[i] = round(mean_absolute_error(y_test, Dypred_ID), round_n)
        STD_ID[i] = round(np.std(Dypred_ID), round_n)
        D0R2_ID[i] = round(r2_score(Y0_test, Dypred_ID), round_n)
        D0RMSE_ID[i] = round(RMSE(Y0_test, Dypred_ID), round_n)


    return np.median(DR2_LT), np.median(DRMSE_LT), np.median(DR2_ID), np.median(DRMSE_ID), np.median(DR2_LS), np.median(DRMSE_LS), np.median(DR2_10), np.median(DRMSE_10), np.median(DR2_50), np.median(DRMSE_50), np.median(D0R2_LT), np.median(D0RMSE_LT), np.median(D0R2_ID), np.median(D0RMSE_ID), np.median(D0R2_LS), np.median(D0RMSE_LS), np.median(MAE_LS), np.median(MAE_LT), np.median(MAE_10), np.median(MAE_50), np.median(MAE_ID), np.median(STD_LS), np.median(STD_LT), np.median(STD_10), np.median(STD_50), np.median(STD_ID)
