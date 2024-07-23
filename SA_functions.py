import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pedophysics import predict, Soil
from PM import *
from FDEM import Initialize


from scipy import constants
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

#########################################################################################################################################

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
        MAE_10[i] = round(mean_absolute_error(y_test10, Dypred_10), round_n)
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
        MAE_50[i] = round(mean_absolute_error(y_test50, Dypred_50), round_n)
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


def r2_inv(s_rec, survey, dfsForward, r2):
    i =0

    for coil in s_rec.coils:
        obsECa = survey.df[coil].values
        simECa = dfsForward[coil].values

        r2.loc[i, coil] = r2_score(simECa, obsECa)
    obsECa = survey.df[s_rec.coils].values.flatten()
    simECa = dfsForward[s_rec.coils].values.flatten()
    r2.loc[i, 'all'] = r2_score(simECa, obsECa)
    return r2


def inv_samples(config, reg_meth, FM, MinM, alpha, s_rec, bounds):

    if MinM == 'ROPE':
        if config['constrain']:
            #print(f'Constrained inversion using {FM} with {MinM}, reg={reg_meth}, alpha={alpha}')
            s_rec.invert(forwardModel=FM, method=MinM, alpha=alpha, regularization=reg_meth, njobs = -1, bnds=bounds) # Not all options are for all solvers
        else:
            #print(f'Inversion using {FM} with {MinM}, reg={reg_meth}, alpha={alpha}')
            s_rec.invert(forwardModel=FM, method=MinM, alpha=alpha, regularization=reg_meth, njobs = -1)
            
    elif MinM == 'Gauss-Newton':
        if config['constrain']:
            #print(f'Constrained Inversion using {FM} with {MinM}, reg={reg_meth}, alpha={alpha}')
            s_rec.invert(forwardModel=FM, method=MinM, alpha=alpha, regularization=reg_meth, bnds=bounds)

        else: 
            #print(f'Inversion using {FM} with {MinM}, reg={reg_meth}, alpha={alpha}')
            s_rec.invert(forwardModel=FM, method=MinM, alpha=alpha, regularization=reg_meth)


def constrain_bounds(config, ec_cols_ref, ec_stats):

    bounds = []
    if config['constrain']:
        #if config['custom_bounds']:
        #    bounds = config['bounds']
        #else:
        for i, name in enumerate(ec_cols_ref):
            if ec_stats.loc['min_sd'][name] > 0:
                nmin = ec_stats.loc['min_sd'][name]
            elif ec_stats.loc['min'][name] > 0:
                nmin = ec_stats.loc['min'][name]
            else:
                nmin = 1
            nmax = ec_stats.loc['max_sd'][name]
            min_max = tuple([nmin,nmax])
            bounds.append(min_max)
        bounds = np.round(bounds, decimals=0)
        #    if not config['n_int'] and not config['custom_bounds']:
        #        bounds = bounds[1:]

    return bounds        


def generate_forward_model_inputs(df, profile_id_col, depth_col, res_col):
    models = {}  # Dictionary to store models by profile ID

    for profile_id, group in df.groupby(profile_id_col):
        # Assuming uniform interval after previous interpolation
        uniform_interval = abs(group[depth_col].diff().iloc[1])
        #print(uniform_interval)
        num_layers = len(group[res_col])
                # Thicknesses are the intervals between depths, except for the last value which does not define a new layer
        thick = np.full(num_layers - 1, uniform_interval)
        thick[0] = 2 * thick[0]
        # Conductivity is the inverse of resistivity
        con = group[res_col].values/1000
        # Permittivity is the epsilon_0 for all layers
        perm = np.full(num_layers, constants.epsilon_0)
        sus = np.zeros(num_layers)
        # Create model instance
        M = Initialize.Model(thick, sus[::-1], con[::-1], perm[::-1])
        
        # Store the model instance in the dictionary with the profile ID as the key
        models[profile_id] = M
    return models


def inv_config(s_site, config, interface):
    if s_site == 'P':
        config['coil_n'] = [0, 1]    # indexes of coils to remove (cf. emagpy indexing)
                                    # for Proefhoeve, coils 0 (HCP05) and 1 (PRP06) are best
                                    # removed, for Middelkerke coils 4 (HCP4.0) and 5 (PRP4.1)

        config['reference_profile'] = 15 # ID of ERT (conductivity) profile to be used 
                                    #  to generate starting model
                                    # For proefhoeve nr 15 is used, for middelkerke 65

    elif s_site == 'M':
        config['coil_n'] = [2, 3]    # indexes of coils to remove (cf. emagpy indexing)
                                    # for Proefhoeve, coils 0 (HCP05) and 1 (PRP06) are best
                                    # removed, for Middelkerke coils 4 (HCP4.0) and 5 (PRP4.1)

        config['reference_profile'] = 65 # ID of ERT (conductivity) profile to be used 
                                        #  to generate starting model
                                        # For proefhoeve nr 15 is used, for middelkerke 65

    config['n_int'] = True # if True custom interfaces are defined (via config['interface']), 
                            # otherwise reference profile interfaces are used

    #if interface == 'observed' or interface == 'observed-fixed':
    if interface == 'observed':
        config['interface'] = [0.3, 0.6, 1.0, 2.0 ] # depths to custom model interfaces

        #if site == 'M':
        #    config['bounds'] = [(5, 80), (50, 380), (76, 820), (100, 1000), (150, 1000)]
        #elif site == 'P':
        #    config['bounds'] = [(10, 55), (20, 120), (50, 335), (50, 250), (10, 50)] 

    #elif interface == 'log-fixed' or interface == 'log':
    elif interface == 'log':

        logint = np.geomspace(0.15, 2, num=7)
        logint[1:] += 0.15
        config['interface'] = logint.tolist()

        #if site == 'M':
        #    config['bounds'] = [(5, 80), (20, 300), (30, 380), (50, 350), (76, 600), (80, 700), (100, 1000), (130, 800)]
        #elif site == 'P':
        #    config['bounds'] = [(10, 55), (15, 100), (20, 160), (30, 200), (50, 335), (60, 300), (75, 500), (60, 500)] 


        # Define the interfaces depths between layers for starting model and inversion
        #           (number of layers = len(config['interface'])+1)
    config['n_int'] = True # if True custom interfaces are defined (via config['interface']), 
                            # otherwise reference profile interfaces are used

    # Inversion constraining
    # if constrained inversion is used, you can set custom EC bounds (and other params)

    config['custom_bounds'] = False
    '''
        config['bounds'] is the 'bnds' used in emagpy constraining
        if you fix 2 interface and fit 3 layer EC:
        [(layer1_ec_min, layer1_ec_max), 
        (layer2_ec_min, layer2_ec_max), 
        (layer3_ec_min, layer3_ec_max)]

        If you fit 2 interfaces and fit 3 layer EC:
        [(int1_min, int1_max), 
        (int2_min, int2_max), 
        (layer1_ec_min, layer1_ec_max), 
        (layer2_ec_min, layer2_ec_max), 
        (layer3_ec_min, layer3_ec_max)]

        autobounds MDK:[(22.877321099166068, 83.29767890083818), 
        (50.6150000000018, 381.79940055200086), 
        (75.5445000000012, 819.2347232074701), 
        (124.346000000001, 1108.1655185859772), 
        (188.4700000000009, 1025.8167426267287)]   
    '''
    # !!! ---

    # [  6.  43.]
    #  [ 27. 183.]
    #  [ 36. 307.]
    #  [ 63. 335.]
    #  [ 47. 337.]
    #  [ 82. 224.]
    #  [ 53. 103.]
    #  [ 39.  73.]

    if config['n_int'] == False and config['custom_bounds']:
        print('Check if bounds and number of interfaces match')

    # Geographic operations (if needed)
    c_transform = False
    c_utmzone = '31N'
    c_target_cs = 'EPSG:31370'

    # remove profiles at transect edges
    config['n_omit'] =  10 # number of profiles to exclude from the start
                        # and end of the ERT transect (none = 0) for the inversion
                        # a total of 60 profiles is available, for middelkerke
                        # 120 profiles are available  