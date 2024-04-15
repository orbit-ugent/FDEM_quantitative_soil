# Packages -----------------
import os
import sys
import shutil
import warnings
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy import constants
from matplotlib.path import Path

# Get notebook and parent dir
current_dir = os.path.dirname(os.path.abspath('__file__')) 
parent_dir = os.path.dirname(current_dir)

# Set path to pedophysics module 
pedophysics_code_path = os.path.join(parent_dir)
sys.path.insert(0, pedophysics_code_path)

import pedophysics
from pedophysics import predict, Soil
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import root
from scipy.stats import spearmanr
from scipy.optimize import minimize, differential_evolution
from IPython.display import clear_output
from utils.spatial_utils import utm_to_epsg, get_coincident
import pymel
from FDEM import Initialize
from FDEM import Modeling as Mod
from FDEM import Properties as Prop
from utils.profile_utils import merge_layers, plot_profile, check_uniformity_and_interpolate
from utils.profile_utils import smooth_profiles, clip_profiles_to_max_depth, plot_combined_profiles
from scipy.spatial import cKDTree

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from emagpy import Problem
from scipy.interpolate import interp1d

# Electromagnetic induction data inversion package
from plots import *
from PM import *
from resipy import Project

sys.path.insert(0,'../src/') # this add the emagpy/src directory to the PATH


def SA(site, cl, percent, FM, minm, alpha, remove_coil, start_avg, constrain):
    """
    site : 'M', 'P' Proefhoeve Middelkerke
    cl: 0.2, 0.3, 0.4
    percent: 10, 20, 30
    FM: 'FSeq', 'CS', 'FSlin' 
    minm: 'L-BFGS-B', 'CG', 'Nelder-Mead', 'ROPE'
    alpha: 0.02, 0.07, 0.2       
    remove_coil: True, False
    start_avg: True, False
    constrain: True, False
    """

    # Datetime for filename
    now = (datetime.datetime.now())
    now = now.strftime("%y%m%d_%H%M")

    # Define input datatype and source folder
    em_intype = 'rec'   # 'rec' = rECa transect; 'lin' = LIN ECa transect; 
                        # 'survey' = rEC full survey

    config = {}

    cal = 'calibrated' # 'non_calibrated', 'drift_calibrated'

    # User input

    datafolder = 'data' # data folder

    if site == 'P':
        config['instrument_code'] = 'Dualem-21HS' # instrument code
        instrument_code = '21HS' # 421S, '21HS'
        profile_prefix = 'proefhoeve'
        emfile_prefix = 'proefhoeve_21HS'
            
    else:
        profile_prefix = 'middelkerke'
        emfile_prefix = 'middelkerke_421S'
        # check if correct instrument (only 421S data available for Middelkerke)
        config['instrument_code'] = 'Dualem-421S'

########################################################################################################################
############################################# ERT INVERSION ################################################
    

    ERTdatadir = 'data/ERT/'
    ERT_path = ERTdatadir + profile_prefix+'-inv-ERT-'+str(cl)+'_'+str(percent)+'.csv'

    if os.path.exists(ERT_path):
        ERT_profiles = pd.read_csv(ERT_path)

    else: 
        if site == 'M':
            df = pd.read_csv(ERTdatadir + 'electrode_locations_Middelkerke.csv')
            elec = np.zeros((120, 3))
            elec[:, 0] = df['distance'].values

            # invert
            k = Project()
            k.createSurvey(ERTdatadir + '23082301.csv', ftype='Syscal')

        elif site == 'P':
            df = pd.read_csv(ERTdatadir + 'electrode_locations_Proefhoeve.csv')
            df = df[:-1]
            elec = np.zeros((60, 3))
            elec[:, 0] = df['distance'].values

            # invert
            k = Project()
            k.createSurvey(ERTdatadir + '23082201.csv', ftype='Syscal')

        k.setElec(elec)
        k.filterAppResist(vmin=0)
        k.filterRecip(percent=percent) 
        k.fitErrorPwl()
        k.createMesh('trian', cl = cl)

        # invert
        k.err = True  # use fitted error in the inversion
        k.invert()

        # extract profiles
        m = k.meshResults[0]
        dfs = []
        for i in range(df.shape[0]):
            row = df.loc[i, :]
            ie = m.df['X'].between(row['distance'] - 0.5, row['distance'] + 0.5) & m.df['Z'].gt(-5)
            sdf = m.df[ie][['Z', 'Resistivity(ohm.m)']]
            sdf['Z'] = sdf['Z'].round(1)
            sdf['Z'] = (sdf['Z'] * 2).round(1) / 2
            sdf = sdf.groupby('Z').mean().reset_index()
            sdf['easting'] = row['easting']
            sdf['northing'] = row['northing']
            sdf['ID'] = row['ID']
            dfs.append(sdf)

        ERT_profiles = pd.concat(dfs)
        ERT_profiles.to_csv(ERTdatadir + profile_prefix+'-inv-ERT-'+str(cl)+'_'+str(percent)+'.csv', index=False)


########################################################################################################################
############################################# 01 CALIBRATION ################################################

    #cal_folder = os.path.join(datafolder, 'calibrated')
    #os.makedirs(cal_folder, exist_ok=True) 
    config['raw'] = True

    Raw_emfile_prefix = 'Raw/'+emfile_prefix + '_raw'

    if config['raw']:
        emfile_prefix = emfile_prefix + '_raw'

    #raw_em_data = os.path.join(datafolder, f'{Raw_emfile_prefix}.csv')
    #samplocs = os.path.join(datafolder, f'{profile_prefix}_samps.csv')
    ###

    raw_em_data = os.path.join(datafolder, f'{Raw_emfile_prefix}.csv')
    #ert_file = os.path.join(datafolder, f'{profile_prefix}-inv-ERT-0.310.csv')
    samplocs = os.path.join(datafolder, f'{profile_prefix}_samps.csv')
    ###
    #print('ert_file', ert_file)
    #config['ert_file'] = ert_file
    config['em_file'] = raw_em_data

    config['transform'] = False
    config['utmzone'] = '31N'
    config['target_cs'] = 'EPSG:31370'

    # Profile selection criteria
    # filter out profiles that don't reach the min_depth)
    config['min_depth'] = -4.0 # minimum profile depth
    config['max_depth'] = -6.0
    config['surf_depth'] = -0.1
    # remove profiles at transect edges
    prof_excl = 10  # number of profiles to exclude from the start and end (none = 0)

    # Profile smoothing parameters 
    window_size = 1 # Define your window size for simple moving average filter (1 = no filtering)

    # Data import and structuring into dataframe
    #ert_p = pd.read_csv(config['ert_file'], sep=',', header=0)
    EM_data = pd.read_csv(config['em_file'], sep=',', header=0)
    ert_p = ERT_profiles
    save_fig = True

    if config['transform']:
        # Create a new filename with the target EPSG code
        EM_data = utm_to_epsg(EM_data, config['utmzone'], target_epsg=config['target_cs'])
        transformed_path = os.path.join(datafolder, f'{emfile_prefix}_Lam72.csv')
        #EM_data.to_csv(transformed_path, index=False)


##########################################################################################################

    # Group each ERT profile per electrode location
    # ---------------------------------------------#

    # Group the data by profile ID for efficient access to each profile
    profiles = ert_p.groupby('ID')

    # 2] Filter the profile Resistivity data with a simple moving average

    ert_p_smoothed = ert_p.groupby('ID').apply(lambda x: smooth_profiles(x, 'Resistivity(ohm.m)', 'Z', window_size))

    # 3] Filter out profiles that don't reach min_depth
    depth_filter = ert_p.groupby('ID').apply(lambda g: g['Z'].min() <= config['min_depth'])
    filtered_profile_ids = depth_filter.index[depth_filter]
    ert_p_depth_filtered = ert_p[ert_p['ID'].isin(filtered_profile_ids)]

    # 4] Exclude the first and last n profiles
    unique_ids = ert_p['ID'].unique()
    # This will select profiles excluding the first and last n
    selected_ids = unique_ids[prof_excl:-prof_excl]  # This is based on your earlier selection criteria
    ert_p_selected = ert_p_smoothed.loc[ert_p_smoothed['ID'].isin(selected_ids)]

    ert_final = clip_profiles_to_max_depth(ert_p_selected, config['max_depth'], config['surf_depth'])

    dataset_name = 'Resistivity(ohm.m)'  # The variable of interest

    # Reset the index without inserting it as a column
    ert_final = ert_final.reset_index(drop=True)

    # Now group by 'ID' and find the minimum 'Z' value for each group
    min_depths_per_profile = ert_final.groupby('ID')['Z'].max()

    # Convert the Series to a DataFrame for better display and manipulation
    min_depths_df = min_depths_per_profile.reset_index()
    min_depths_df.columns = ['Profile ID', 'Minimum Depth']
    #print(min_depths_df)

##########################################################################################################

    # Homogenise ERT profiles (assure uniform depth extent and intervals
    # ----------------------------------------------------------------- #

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

        # Reset index for the concatenated DataFrame
        all_profiles_df.reset_index(drop=True, inplace=True)

        return all_profiles_df, uniform_intervals

    # Columns containing the resistivity data
    data_columns = ['Resistivity(ohm.m)', 'Smoothed']
    # Assuming ert_final is your DataFrame with profile data
    all_profiles_df, uniform_intervals = check_uniformity_and_interpolate(
        ert_final, 'ID', 'Z', *data_columns
    )

    dataset_name = 'Resistivity(ohm.m)'  # The variable of interest

##########################################################################################################

    # Create forward model inputs per ERT profile (based on number of layers)
    # ---------------------------------------------------------------------- #
    def generate_forward_model_inputs(df, profile_id_col, depth_col, res_col):
        models = {}  # Dictionary to store models by profile ID

        for profile_id, group in df.groupby(profile_id_col):
            # Assuming uniform interval after previous interpolation
            uniform_interval = group[depth_col].diff().iloc[1]
            num_layers = len(group[res_col])

            # Thicknesses are the intervals between depths, except for the last value which does not define a new layer
            thick = np.full(num_layers - 1, uniform_interval)
            thick[0] = 2 * thick[0]
            # Conductivity is the inverse of resistivity
            con = 1 / group[res_col].values
            # Permittivity is the epsilon_0 for all layers
            perm = np.full(num_layers, constants.epsilon_0)

            # Susceptibility is 0.0 for all layers
            sus = np.zeros(num_layers)

            # Create model instance
            M = Initialize.Model(thick, sus[::-1], con[::-1], perm[::-1])
            
            # Store the model instance in the dictionary with the profile ID as the key
            models[profile_id] = M
        return models

    # Use the function with your DataFrame
    models = generate_forward_model_inputs(all_profiles_df, 'ID', 'Z', 'Smoothed')

##########################################################################################################

    # Set instrument properties to use in FWD modelling
    # ------------------------------------------------ #

    # Assigning to sensor object (S)
    #S = FDEM.CoilConfiguration(height, freq, x, ori, mom, 200)

    # Sensor settings
    config['instrument_height'] = 0.165     # instrument height
    config['instrument_orientation'] = 'HCP'    # instrument orientation

    instrument = Initialize.Instrument(config['instrument_code'],
                                        instrument_height=config['instrument_height'],
                                        instrument_orientation=config['instrument_orientation']
                                        )

    # Forward model the responses of each coil configuration at each of the profile locations
    # --------------------------------------------------------------------------------------

    forward_results = {}

    for profile_id, model in models.items():
        forward_results[profile_id] = {}
        for coil_id, cc in enumerate(instrument.cc_names):
            if 'QP' in cc:
                pair = Mod.Pair1D(getattr(instrument, cc), model)
                IP, QP = pair.forward()
                forward_results[profile_id][cc] = {'IP': IP, 'QP': QP}

##########################################################################################################

    # Check the forward model results for a given profile
    # IP and QP data are in ppm !!!
    profile_id = 13
    sensor_id = 'HCP1QP'
    ip_result = forward_results[profile_id][sensor_id]['IP']
    qp_result = forward_results[profile_id][sensor_id]['QP']

    #print(f"IP for profile {profile_id} with sensor {sensor_id}: {ip_result}")
    #print(f"QP for profile {profile_id} with sensor {sensor_id}: {qp_result}")

##########################################################################################################

    # Check location offset between ERT electrode locations and EMI measurements
    # --------------------------------------------------------------------------

    # Creating kd-tree for the EM data points
    tree = cKDTree(EM_data[['x', 'y']].values)

    # Dictionary to hold the nearest EM data points to each profile
    nearest_EM_data_points = {}

    for profile_id, profile_data in ert_final.groupby('ID'):
        if profile_id in all_profiles_df['ID'].unique():
            profile_location = np.array([profile_data['easting'].mean(), profile_data['northing'].mean()])

            # Query the tree for the nearest point
            distance, index = tree.query(profile_location, k=1)  # k=1 means we want the nearest point
            
            # Retrieve the EM data point that is closest to the profile location
            nearest_EM_data_points[profile_id] = EM_data.iloc[index]
            nearest_EM_data_points[profile_id]['p_ID'] = profile_id

############################################################################################################

    # Create dataframe from EM data over ERT transect for comparison and exporting
    EM_transect = pd.DataFrame.from_dict(nearest_EM_data_points, orient='index')

    # If you want to ensure that the p_ID is a column and not an index, you can reset the index:
    EM_transect.reset_index(drop=True, inplace=True)
    #EM_transect.columns

    # Forward model at transect locations
    forward_results = {}

    for profile_id, model in models.items():
        forward_results[profile_id] = {}
        for coil_id, cc in enumerate(instrument.cc_names):
            if 'QP' in cc:
                coilconfig = getattr(instrument, cc)
                # Forward model
                pair = Initialize.Properties.Modeling.Pair1D(coilconfig, model)
                IP, QP = pair.forward()
                # Convert fwd modelled result to LIN ECa to match sensor output
                eca = Initialize.Properties.qp_to_mcneill(coilconfig, QP[0], qp_input_units='ppm', eca_output_units='mS')
                forward_results[profile_id][cc] = eca

###############################################################################################################

    # Create a dataframe that combines the EM data along the transect and 
    #   the forward modelled EM responses based on ERT data
    # ---------------------------------------------------------------------------- #

    data_for_df = []

    # Iterate over the profile IDs in the nearest_EM_data_points dictionary
    for profile_id, em_data in nearest_EM_data_points.items():
        if profile_id in ert_final['ID'].unique():
            forward_data = forward_results[profile_id]
            
            # Construct the row for this profile
            row = {
                'ID': profile_id,
                'x_ert': em_data['x'],
                'y_ert': em_data['y'],
                'x_em': em_data['x'],
                'y_em': em_data['y']
            }
            
            # Add the EM sensor data from the 'nearest_EM_data_points'
            for cc in instrument.cc_names:
                if 'QP' in cc:
                    row[f'EM_{cc}'] = em_data[cc]

            # Add the forward modelled sensor data from the 'forward_results'
            for cc in instrument.cc_names:
                if 'QP' in cc:
                    row[f'forward_{cc}'] = forward_data[cc]

            # Add the row to our list
            data_for_df.append(row)

    # Create the DataFrame
    combined_df = pd.DataFrame(data_for_df)

############################################################################################################

    # Initialize a figure for subplots
    num_configs = len([name for name in instrument.cc_names if 'QP' in name])
    num_rows = num_configs // 2 + num_configs % 2
    num_cols = 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5), constrained_layout=True)
    fig.suptitle('Comparison of Measured vs. Modeled Data', fontsize=16)

    # Store regression parameters
    regression_params = {}

    # Loop through each coil configuration
    for i, coil_config in enumerate(name for name in instrument.cc_names if 'QP' in name):
        measured_col = f'EM_{coil_config}'
        modeled_col = f'forward_{coil_config}'

        # Perform linear regression
        X = combined_df[measured_col].values.reshape(-1, 1)
        y = combined_df[modeled_col].values
        regression = LinearRegression().fit(X, y)
        regression_params[coil_config] = (regression.coef_[0], regression.intercept_)

        # Predict values for regression line
        reg_line = regression.predict(X)

        # Plot
        row, col = divmod(i, 2)
        ax = axes[row, col]

        # Create the scatter plot
        ax.scatter(X, y, edgecolor='k', alpha=0.7)

        # # Annotate each point with the profile ID
        # for index, point in combined_df.iterrows():
        #     ax.text(point[measured_col], point[modeled_col], str(point['ID']))

        # Add a line of perfect agreement
        min_val = min(combined_df[measured_col].min(), combined_df[modeled_col].min())

        max_val = max(combined_df[measured_col].max(), combined_df[modeled_col].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')

        # Add the regression line
        ax.plot(X, reg_line, 'g-', label=f'Regression: {coil_config}')

        ax.set_title(coil_config)
        ax.axis('equal')

        min_limit = min(min_val, combined_df[modeled_col].min())
        max_limit = max(max_val, combined_df[modeled_col].max())
        ax.set_xlim(min_limit, max_limit)
        ax.set_ylim(min_limit, max_limit)

        ax.set_xlabel('Measured EM Data')
        ax.set_ylabel('Forward Modeled EM Data')
        ax.legend()

    # Hide any unused subplots
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axes.flatten()[j])

    # Show the plot
    #plt.savefig(cal_metaname)
    #plt.show()

    # Print regression equations
    # TODO store in metadata file
    df_calpar = pd.DataFrame(columns=['coilconfig', 'slope', 'intercept'])

    # Initialize an empty list to collect DataFrames
    dataframes = []

    # Loop over regression parameters
    for coil_config, params in regression_params.items():
        slope, intercept = params
        
        # dataframe creation per iteration
        df_temp = pd.DataFrame({
            'coilconfig': [coil_config],  
            'slope': [slope],
            'intercept': [intercept]
        })
        dataframes.append(df_temp)
        #print(f"{coil_config}: Mod = {slope:.3f} ECa + {intercept:.3f}")

    # Concatenate all the listed DataFrames into df.calpar
    df_calpar = pd.concat(dataframes, ignore_index=True)


    #cal_metaname = os.path.join(cal_folder, f'{emfile_prefix}_calibration_params.csv')
    #df_calpar.to_csv(cal_metaname)

#############################################################################################################

    # Implement EM calibration through regression parameters
    # ----------------------------------------------------- 

    cal_EM = EM_data.copy()
    cal_trans = EM_transect.copy()

    # Perform the calibration
    for coil_config, (slope, intercept) in regression_params.items():
        if coil_config in cal_EM.columns:  # Only apply if the coil_config is a column in the DataFrame
            #print(f"{coil_config}: Mod = {slope:.3f} ECa + {intercept:.3f}")
            cal_EM[coil_config] = slope * cal_EM[coil_config] + intercept
            cal_trans[coil_config] = slope * cal_trans[coil_config] + intercept
            #print('calibrated')

    cal_r_EM = cal_EM.copy()
    cal_r_trans = cal_trans.copy()
    EM_transect_r = EM_transect.copy()
    print('cal_r_EM', cal_r_EM)

    # Convert calibrated data to LIN ECa and rECa
    for cc in instrument.cc_names:  # Only apply if the coil_config is a column in the DataFrame
        if 'QP' in cc:
            coil_configuration = getattr(instrument, cc)
            ip_config = cc[:-2]+'IP'
            ## FULL SURVEY DATASET
            # convert to QP to prepare for rECa prediction
            qp = Prop.mcneill_to_qp(coil_configuration, 
                                    cal_EM[cc], 
                                    eca_input_units='mS', 
                                    qp_output_units='ppm')
            # perform rECa prediction
            print('cc', cc)
            print('ip_config', ip_config)
            print('cal_EM[ip_config]', cal_EM[ip_config])

            cal_r_EM[cc], _ = Prop.reca(coil_configuration, 
                                            qp, 
                                            cal_EM[ip_config]*1000, 
                                            precision=.001, 
                                            noise=0, 
                                            reference_eca=None, 
                                            original_msa=0, 
                                            alternative_msa=0, 
                                            maximum_eca=4)
            # convert from S/m to mS/m
            cal_r_EM[cc] = cal_r_EM[cc]*1000

            ## TRANSECT
            # convert to QP to prepare for rECa prediction
            qp = Prop.mcneill_to_qp(coil_configuration, 
                                    cal_r_trans[cc], 
                                    eca_input_units='mS', 
                                    qp_output_units='ppm')
            
            qp_uncal = Prop.mcneill_to_qp(coil_configuration, 
                                    EM_transect[cc], 
                                    eca_input_units='mS', 
                                    qp_output_units='ppm')
            
            # perform rECa prediction
            cal_r_trans[cc], _ = Prop.reca(coil_configuration, 
                                            qp, 
                                            cal_trans[ip_config]*1000, 
                                            precision=.001, 
                                            noise=0, 
                                            reference_eca=None, 
                                            original_msa=0, 
                                            alternative_msa=0, 
                                            maximum_eca=4)
            
            EM_transect_r[cc], _ = Prop.reca(coil_configuration, 
                                            qp_uncal, 
                                            EM_transect[ip_config]*1000, 
                                            precision=.001, 
                                            noise=0, 
                                            reference_eca=None, 
                                            original_msa=0, 
                                            alternative_msa=0, 
                                            maximum_eca=4)

            # convert from S/m to mS/m
            cal_r_trans[cc] = cal_r_trans[cc]*1000
            EM_transect_r[cc] = EM_transect_r[cc]*1000
            #print(f"Modeled robust {cc} ECa, median = {cal_r_trans[cc].median()} vs median = {cal_trans[cc].median()} ")

#############################################################################################################################

    # Data import dry down experiment
    #dry_down = os.path.join(datafolder, f'Dry_down.csv')
    #dry_d = pd.read_csv(dry_down, sep=',', header=0)

    #cal_folder = os.path.join(datafolder, 'calibrated')
    #em_survey = os.path.join(cal_folder, f'{emfile_prefix}_calibrated_rECa.csv')
    #em_survey = pd.read_csv(em_survey, sep=',', header=0)
    em_survey = cal_r_EM #### Link 01 to 02 files
    sampleprop = os.path.join(datafolder, f'{profile_prefix}_soil_analysis.csv')
    samples_analysis = pd.read_csv(sampleprop, sep=',', header=0)

    em_sample_prop = get_coincident(em_survey, samples_analysis)
    ds_c = em_sample_prop.copy()

########################################################################################################################

    #temp_dir = 'temp_emp_02' 
    #infile_name = 'infile_s02.csv'
    #os.makedirs(temp_dir, exist_ok=True)
    #temp_file = os.path.join(temp_dir,infile_name)

########################################################################################################################
############################################# INVERSION CONFIGURE INPUT ################################################

    # Remove coils for inversion?
    config['remove_coil'] = remove_coil    # set to True if you want to remove coils in the inversion process
    # Reference profile for starting model (conductivity values)
    config['start_avg'] = start_avg     # take average of input resistivity profiles per layer as starting model
                                    # if false, reference profile is taken as starting model
    config['constrain']=constrain

                                    
    if site == 'P':
        config['coil_n'] = [0, 1]    # indexes of coils to remove (cf. emagpy indexing)
                                    # for Proefhoeve, coils 0 (HCP05) and 1 (PRP06) are best
                                    # removed, for Middelkerke coils 4 (HCP4.0) and 5 (PRP4.1)

        config['bounds'] = [(10, 55), (20, 120), (50, 335), (50, 250), (10, 50)] 

        config['reference_profile'] = 15 # ID of ERT (conductivity) profile to be used 
                                    #  to generate starting model
                                    # For proefhoeve nr 15 is used, for middelkerke 65

    elif site == 'M':
        config['bounds'] = [(5, 80), (50, 380), (76, 820), (100, 1000), (150, 1000)]

        config['coil_n'] = [4, 5]    # indexes of coils to remove (cf. emagpy indexing)
                                    # for Proefhoeve, coils 0 (HCP05) and 1 (PRP06) are best
                                    # removed, for Middelkerke coils 4 (HCP4.0) and 5 (PRP4.1)

        config['reference_profile'] = 65 # ID of ERT (conductivity) profile to be used 
                                        #  to generate starting model
                                        # For proefhoeve nr 15 is used, for middelkerke 65

    # Inversion parameters
    config['FM'] = FM 
    config['opt_method'] = minm
    config['constrain']=True
    config['regularization'] = 'l2'
    config['alpha'] = alpha

    # Define the interfaces depths between layers for starting model and inversion
    #           (number of layers = len(config['interface'])+1)
    config['n_int'] = True # if True custom interfaces are defined (via config['interface']), 
                            # otherwise reference profile interfaces are used

    config['interface'] = [0.3, 
                        0.6, 
                        1.0,
                        2.0
                            ] # depths to custom model interfaces


    # remove profiles at transect edges
    config['n_omit'] =  10 # number of profiles to exclude from the start
                        # and end of the ERT transect (none = 0) for the inversion
                        # a total of 60 profiles is available, for middelkerke
                        # 120 profiles are available 

    config['custom_bounds'] = True

    if config['constrain']:
        if config['custom_bounds']:
            bounds = config['bounds']

    if config['n_int'] == False and config['custom_bounds']:
        print('Check if bounds and number of interfaces match')

    # Geographic operations (if needed)
    c_transform = False
    c_utmzone = '31N'
    c_target_cs = 'EPSG:31370'

    # ---------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------- #

    # Datetime for filename
    now = (datetime.datetime.now())
    now = now.strftime("%y%m%d_%H%M")

########################################################################################################################
############################################# LOAD DATA ################################################


    #inv_folder = os.path.join(datafolder, 'inverted')
    #os.makedirs(inv_folder, exist_ok=True) 
    #cal_folder = os.path.join(datafolder, 'calibrated')
    #ert_file = os.path.join(datafolder, f'{profile_prefix}-profiles.csv')
    #em_rec = os.path.join(cal_folder, f'{emfile_prefix}_transect_calibrated_rECa.csv')
    #em_lin = os.path.join(cal_folder,f'{emfile_prefix}_transect_calibrated_LIN.csv')
    #em_survey = os.path.join(cal_folder, f'{emfile_prefix}_calibrated_rECa.csv')
    #samplocs = os.path.join(datafolder, f'{profile_prefix}_samps.csv')
    em_rec = cal_r_trans ### link

    if em_intype == 'rec':
        infile = em_rec
    elif em_intype == 'survey':
        infile = em_survey
    else:
        infile = em_lin

    # Column names for emapgy input
    emp_21HS = [f"HCP0.5f9000{config['instrument_height']}", 'PRP0.6f9000h0.165', 'HCP1.0f9000h0.165', 'PRP1.1f9000h0.165',	'HCP2.0f9000h0.165', 'PRP2.1f9000h0.165',
                'HCP0.5f9000h0.165_inph', 'PRP0.6f9000h0.165_inph', 'HCP1.0f9000h0.165_inph',
                'PRP1.1f9000h0.165_inph', 'HCP2.0f9000h0.165_inph', 'PRP2.1f9000h0.165_inph'
                ]

    emp_421S = ['HCP1.0f9000h0.165', 'PRP1.1f9000h0.165',	'HCP2.0f9000h0.165', 'PRP2.1f9000h0.165', 'HCP4.0f9000h0.165', 'PRP4.1f9000h0.165', 
                'HCP1.0f9000h0.165_inph', 'PRP1.1f9000h0.165_inph', 'HCP2.0f9000h0.165_inph', 'PRP2.1f9000h0.165_inph',
                'HCP4.0f9000h0.165_inph', 'PRP4.1f9000h0.165_inph',
                ]

    # Datetime for filename
    now = (datetime.datetime.now())
    now = now.strftime("%y%m%d_%H%M")

    # 1.0 Data import and structuring into dataframe
    #ert_p = pd.read_csv(ert_file, sep=',', header=0)
    #em_rec = pd.read_csv(em_rec, sep=',', header=0)
    #em_lin = pd.read_csv(em_lin, sep=',', header=0)
    #em_survey = pd.read_csv(em_survey, sep=',', header=0)
    samples = pd.read_csv(samplocs, sep=',', header=0)


    #if c_transform:
        # Create a new filename with the target EPSG code
        #em_rec = utm_to_epsg(em_rec, c_utmzone, target_epsg=c_target_cs)
        #em_lin = utm_to_epsg(em_lin, c_utmzone, target_epsg=c_target_cs)
        #em_survey = utm_to_epsg(em_survey, c_utmzone, target_epsg=c_target_cs)

    instrument = Initialize.Instrument(config['instrument_code'],
                                        instrument_height=config['instrument_height'],
                                            instrument_orientation=config['instrument_orientation']
                                            )

    em_samples = get_coincident(em_survey, samples)

    # ---------------------------------------------------------------------------- #
    # Get ERT profiles
    # ---------------- #
    # Group the data by profile ID for efficient access to each profile
    profiles = ert_p.groupby('ID')

    # Exclude the first and last n_omit profiles
    unique_ids = ert_p['ID'].unique()

    if config['n_omit'] == 0:
        ert_final = ert_p.copy()
    else:
        if config['n_omit']*2 >= len(unique_ids):
            warnings.warn('!!! You removed all profiles !!! Change value for config[n_omit]')
            raise KeyboardInterrupt
        else:
            selected_ids = unique_ids[config['n_omit']:-config['n_omit']]
            ert_p = ert_p.loc[ert_p['ID'].isin(selected_ids)]
            ert_final = ert_p.copy()

    dataset_name = 'Resistivity(ohm.m)'  # The variable of interest

    # convert resistivity to conductivity and modify column names

    ert_final[dataset_name] = (1/ert_final[dataset_name])
    dc_corr = ert_final.copy()
    dc_corr[dataset_name] = predict.BulkEC(Soil(
                                                    frequency_ec = 9000,
                                                    bulk_ec_dc = dc_corr[dataset_name].values
                                                    ))

    ert_final.loc[:, dataset_name] = ert_final[dataset_name]*1000
    dc_corr.loc[:,dataset_name] = dc_corr[dataset_name]*1000
    ert_final = ert_final.rename(columns={"Resistivity(ohm.m)": "EC(mS/m)"})
    dc_corr = dc_corr.rename(columns={"Resistivity(ohm.m)": "EC(mS/m)"})

    # ------------------------------------------------------------------------------

    # Columns containing the resistivity data
    data_column = ['EC(mS/m)']
    # Assuming ert_final is your DataFrame with profile data
    all_profiles_df, uniform_intervals = check_uniformity_and_interpolate(
        dc_corr, 'ID', 'Z', *data_column
    )

    dataset_name = 'EC(mS/m)'  # The variable of interest

########################################################################################################################

    config['reference_profile'] = 11

    if config['reference_profile'] not in all_profiles_df['ID'].unique():
        warnings.warn("Warning: the reference profile ID does not exist. Provide correct profile ID.")
        raise KeyboardInterrupt
    else:
        profile_id = config['reference_profile']

    # Create new layer configuration for prior model based on ERT data
    if config['n_int']:
        new_int = config['interface']
        merged_df = merge_layers(all_profiles_df, new_int,'EC(mS/m)')
    else:
        merged_df = all_profiles_df
    comparedf = merged_df.copy()

    # Plot original and (merged and) DC corrected reference profile
    if config['n_int']:
        plot_title = 'Original vs merged & DC corrected data'
        first_in = .1
    else: 
        plot_title = 'Original vs DC corrected data'
        first_in = .0
    ert_eval = ert_final.copy()
    ert_eval['Z'] = ert_eval['Z'].values + first_in

    #plot_profile(ert_eval, profile_id, dataset_name, compare=True, compare_df = comparedf, compare_name = 'EC(mS/m)', block=True, plot_title=plot_title)

    # Get prior model info
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

    models = generate_forward_model_inputs(merged_df, 'ID', 'Z', 'EC(mS/m)')

########################################################################################################################

    # 
    # -------------------------------------------------------------------- #

    # 
    profile_data = merged_df[merged_df['ID'] == profile_id].copy()
    res_col = 'EC(mS/m)'
    depth = 'Z'
    max_ert_depth = ert_final['Z'].abs().max()

    # 
    # ------------------------------------------------------------------------------

    # A. Test run on the reference profile (config['reference_profile'])
    #       and plot the results

    if not config['n_int']:
        first_lay = profile_data[depth].iloc[-1].round(decimals=1)
        second_lay = profile_data[depth].iloc[-2].round(decimals=1)
        if first_lay == 0:
            profile_data[depth]=profile_data[depth] +second_lay
        else:
            profile_data[depth]=profile_data[depth] +first_lay
        thick = -profile_data[depth].iloc[1:].values
        #thick = -profile_data[depth].values
    else:
        thick = -profile_data[depth].values

    con = profile_data[res_col].values/1000
    ref_len = len(con)
    num_layers = len(con)
    perm = np.full(num_layers, constants.epsilon_0)
    sus = np.zeros(num_layers)

    # # Create model instance
    M = Initialize.Model(thick, sus[::-1], con[::-1], perm[::-1])

    # ----------------------------------------------------------------------

    dataset_name = 'EC(mS/m)'
    layers_interfaces = np.cumsum(models[profile_id].thick)
    layers_interfaces = np.insert(layers_interfaces, 0, 0)
    profile_data = ert_final[ert_final['ID'] == profile_id]

    conductivities = con*1000
    #print('conductivities', conductivities)

    ec_cols_ref = []
    if 'end' in config['interface']:
        config['interface'].remove('end')
    # Get conductivity stats for bounds
    if config['n_int']:
        if 'end' in ec_cols_ref:
            ec_cols_ref.remove('end')
        ec_cols_ref = config['interface']
        ec_cols_ref.append('end')
        mod_layers = thick[1:]
    else:
        if len(conductivities) == len(thick):
            mod_layers = thick[1:]
            #print(f"length modlayers = {len(mod_layers)} with {len(conductivities)} conductivities")
        elif len(conductivities) == (len(thick)+1):
            mod_layers = thick
            #print(f"length modlayers = {len(mod_layers)} with {len(conductivities)} conductivities")
        else:
            raise ValueError(f"Check length of conductivities ({len(conductivities)}) and layers ({len(thick)}) arrays!!")
        
        ec_cols_ref = np.round(layers_interfaces,decimals=1).tolist()
    ec_df = pd.DataFrame(columns=ec_cols_ref)

    # 
    for i in merged_df['ID'].unique(): 
        profile_data = merged_df[merged_df['ID'] == i].copy()
        if not config['n_int']:
            if abs(profile_data.iloc[0]['Z']) > max((list(map(abs, ec_cols_ref)))):
                #print(f'removed {profile_data.iloc[0]["z"]}')
                profile_data = profile_data.drop(profile_data.iloc[0].name)
            elif abs(profile_data.iloc[-1]['Z']) < 0.1:
                #print(f'removed {profile_data.iloc[-1]["z"]}')
                profile_data = profile_data.drop(profile_data.iloc[-1].name)
        res_col = 'EC(mS/m)'
        depth = 'Z' 
        con_m = profile_data[res_col].values
        layers_interfaces = np.cumsum(models[i].thick)
        layers_interfaces = np.insert(layers_interfaces, 0, 0)
        num_layers = len(con)
        perm = np.full(num_layers, constants.epsilon_0)
        sus = np.zeros(num_layers)

        first_lay = profile_data[depth].iloc[-1].round(decimals=1)
        second_lay = profile_data[depth].iloc[-2].round(decimals=1)

        if not config['n_int']:
            first_lay = profile_data[depth].iloc[-1].round(decimals=1)
            second_lay = profile_data[depth].iloc[-2].round(decimals=1)
            if first_lay == 0:
                profile_data[depth]=profile_data[depth] +second_lay
            else:
                profile_data[depth]=profile_data[depth] +first_lay
            thick = -profile_data[depth].iloc[1:].values
        else:
            thick = -profile_data[depth].values

        ec_df = pd.concat([ec_df, pd.DataFrame([np.flip(con_m)], columns=ec_cols_ref)])

    ec_df.reset_index(drop=True, inplace=True)

    ec_stats = ec_df.describe().loc[['min', 'max', 'std', '50%', 'mean']]
    ec_stats.rename(index={'50%': 'median'}, inplace=True)
    ec_stats.loc['min_sd'] = ec_stats.loc['min'] - 2 * ec_stats.loc['std']
    ec_stats.loc['max_sd'] = ec_stats.loc['max'] + 2 * ec_stats.loc['std']

    position = -thick


    # define parameters for inversion starting model
    # --------------------------------------------- #

    if not config['n_int']:
        minstat = np.flipud(ec_stats.loc['min'].values[1:])
        maxstat = np.flipud(ec_stats.loc['max'].values[1:])
        start_mod = ec_stats.loc['mean'].values[1:]
        boundcols = ec_cols_ref[:-1]
    else:
        minstat = np.flipud(ec_stats.loc['min'].values)
        maxstat = np.flipud(ec_stats.loc['max'].values)
        start_mod = ec_stats.loc['mean'].values

    #axr.legend()
    if config['constrain']:
        if config['custom_bounds']:
            bounds = config['bounds']
        else:
            bounds = []
            for i, name in enumerate(ec_cols_ref):
                if ec_stats.loc['min_sd'][name] > 0:
                    minp = ec_stats.loc['min_sd'][name]
                elif ec_stats.loc['min'][name] > 0:
                    minp = ec_stats.loc['min'][name]
                else:
                    minp = 10
                maxn = ec_stats.loc['max_sd'][name]
                min_max = tuple([minp,maxn])
                bounds.append(min_max)
            bounds = np.round(bounds, decimals=0)
            if not config['n_int'] and not config['custom_bounds']:
                bounds = bounds[1:]
            #print(f'autobounds = {bounds}')

########################################################################################################################

    # Perform inversion on sampling locations (to be used in pedophysical modelling)

    if 'code' in em_samples.columns:
        em_samples = em_samples.rename(columns={'code': 'ID'})

    temp_dir = 'temp_emp' 
    infile_name = 'survey_input.csv'
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, infile_name)

    i = instrument.niter
    n = 4
    em_samples.columns.values[n:n+i]

    if config['instrument_code'] == 'Dualem-21HS':
        new_columns = emp_21HS
    else:
        new_columns = emp_421S

    if len(new_columns) != i:
        raise ValueError("The length of new_columns must be equal to the number of columns to rename")
    else:
        em_samples.columns.values[n:n+i] = new_columns

    em_samples.to_csv(temp_file) # em_samples is just saved and called by Problem.createSurvey()

    # transect inversion settings

    s_rec = Problem()
    s_rec.createSurvey(temp_file)
    #t_rec.rollingMean(window=12)

    s_rec.setInit(
        depths0=np.flipud(mod_layers),
        conds0=conductivities
        )

    if config['remove_coil']:
        if type(config['coil_n']) == list:
            config['coil_n'] = sorted(config['coil_n'])
            for i in enumerate(config['coil_n']):
                r_coil = s_rec.coils[(config['coil_n'][i[0]]-i[0])]
                # print(f'removing {r_coil}')
                s_rec.removeCoil(config['coil_n'][i[0]]-i[0])
        else:
            s_rec.removeCoil(config['coil_n'])

    #print(f'Data used for inversion: {s_rec.coils}')

########################################################################################################################

    # invert using ROPE solver (RObust Parameter Estimation)
    warnings.filterwarnings('ignore')
    opt_meth = config['opt_method']
    inv_meth = config['FM']
    reg_meth = config['regularization']
    alph_param = config['alpha']
    if opt_meth in ['MCMC', 'ROPE']:
        if config['constrain']:
            
            print(f'Constrained inversion using {inv_meth} with {opt_meth}, reg={reg_meth}, alpha={alph_param}')
            s_rec.invert(forwardModel=config['FM'], method=opt_meth, 
                    regularization=reg_meth, alpha=alph_param, 
                    bnds=bounds
                    )

        else:
            print(f'Inversion using {inv_meth} with {opt_meth}, reg={reg_meth}, alpha={alph_param}')
            s_rec.invert(forwardModel=config['FM'], method=opt_meth, 
            regularization=reg_meth, alpha=alph_param, njobs=-1
            )

    else:
        print(f'Inversion using {inv_meth} with {opt_meth}, reg={reg_meth}, alpha={alph_param}')
        s_rec.invert(forwardModel='FSeq', method='Gauss-Newton', alpha=alph_param,regularization=reg_meth)
    #s_rec.showOne2one()

########################################################################################################################

    # 4.1: Plot the inversion results and put outcomes into a pandas dataframe
    # ------------------------------------------------------------------------
    csv_filename = f'{now}_{emfile_prefix}_inverted_samples_{opt_meth}_04.csv'

    # ******************************************************************** #

    # Plot inversion outcomes down to a max depth of 2 m, and plotting the data
    # based on their true coordinates along the transect (dist=True).
    #s_rec.showResults(dist=True, errorbar = True) 

    # Extracting the values from the first row of the transect.depths[0] array
    depth_values = s_rec.depths[0][0]

    # Creating the custom column names for layer_cols
    layer_cols = ['EC_{:.2f}'.format(d) for d in depth_values] + ['EC_end']

    # Combining the data from the 'x', 'y' columns and the transect.models[0] array
    data = np.c_[s_rec.surveys[0].df[['x', 'y']].values, s_rec.models[0]]

    # Creating the final dataframe with the desired column names
    ds_inv = pd.DataFrame(data, columns=['x', 'y'] + layer_cols)
    ds_inv['pos'] = em_samples['ID'].to_numpy()
    # ----------------------------------------------------------------------

    # Export the dataframe as a csv-file
    #outfile_transect = os.path.join(inv_folder, csv_filename)
    #ds_inv.to_csv(outfile_transect)

########################################################################################################################

    inv_columns = ds_inv.columns[3:-1]
    ds_c[inv_columns] = np.nan

    for idc, c in enumerate(inv_columns):

        for i in range(len(ds_inv.x)):
            ds_c.loc[ds_c.code == i+1, c] = ds_inv.loc[i, c]

    def closest_ec(row):
        depth = row['depth']
        # Filter columns that start with 'EC_' but not 'EC_end'
        ec_cols = [col for col in row.index if col.startswith('EC_') and col != 'EC_end']
        # Convert the part after 'EC_' to float and calculate the absolute difference with depth
        differences = {col: abs(depth/100 - float(col.split('_')[1])) for col in ec_cols}
        # Find the column name with the minimum difference
        closest_col = min(differences, key=differences.get)
        return row[closest_col]

    # Apply the function to each row
    ds_c['bulk_ec_inv'] = ds_c.apply(closest_ec, axis=1)

    #Obtain EC DC TC
    ds_c['bulk_ec_dc_tc_inv'] = predict.BulkECDCTC(Soil(temperature = ds_c.temp.values+273.15,
                                                        frequency_ec = 9e3,
                                                        bulk_ec = ds_c.bulk_ec_inv.values/1000))
    # Mean of input inverted EC DC TC values
    EC_mean = np.mean(ds_c['bulk_ec_dc_tc_inv'].values) 
    print('EC_mean', EC_mean)

########################################################################################################################
############################################ DETERMINISTIC MODELLING ###################################################

    # Caclculate Bulk EC from HydraProbe data at 50Mhz
    offset = 4
    water_perm = 80
    ds_c['bulk_ec_hp'] = logsdon(50e6, ds_c.rperm, ds_c.iperm)
    ds_c['bulk_ec_dc_hp'] = predict.BulkECDC(Soil(frequency_ec = 50e6,
                                                bulk_ec = ds_c.bulk_ec_hp.values))

    ds_c['bulk_ec_tc_hp'] = SheetsHendrickxEC( ds_c.bulk_ec_hp, ds_c.temp)
    ds_c['bulk_ec_dc_tc_hp'] = predict.BulkECDCTC(Soil(temperature = ds_c.temp.values,
                                                        bulk_ec_dc = ds_c.bulk_ec_dc_hp.values
                                                        ))

    # Caclculate Water EC from HydraProbe data at 50Mhz
    ds_c['water_ec_hp'] = Hilhorst(ds_c.bulk_ec_hp, ds_c.rperm, water_perm, offset)
    ds_c['water_ec_hp_t'] = WraithOr(ds_c.water_ec_hp, ds_c.temp)
    ds_c['iperm_water_t'] = ds_c.water_ec_hp_t/(50e6*2*pi*epsilon_0)


    ###################

    # -------------------------------------------------------------------------------------

    bulk_ec_inv_50cm = ds_c.bulk_ec_inv[ds_c['depth']==50]
    bulk_ec_inv_10cm = ds_c.bulk_ec_inv[ds_c['depth']==10]

    clay_50cm = np.mean(ds_c.clay[ds_c['depth']==50])
    clay_10cm = np.mean(ds_c.clay[ds_c['depth']==10])
    bd_50cm = np.mean(ds_c.bd[ds_c['depth']==50])
    bd_10cm = np.mean(ds_c.bd[ds_c['depth']==10])
    water_ec_hp_50cm = np.mean(ds_c.water_ec_hp[ds_c['depth']==50])
    water_ec_hp_10cm = np.mean(ds_c.water_ec_hp[ds_c['depth']==10])
    water_ec_hp_50cm_t = np.mean(ds_c.water_ec_hp_t[ds_c['depth']==50])
    water_ec_hp_10cm_t = np.mean(ds_c.water_ec_hp_t[ds_c['depth']==10])
    clay_mean = np.mean(ds_c.clay)
    bd_mean = np.mean(ds_c.bd)
    water_ec_hp_mean = np.mean(ds_c.water_ec_hp)
    water_ec_hp_mean_t = np.mean(ds_c.water_ec_hp_t)
    temp_50cm = np.mean(ds_c.temp[ds_c['depth']==50])
    temp_10cm = np.mean(ds_c.temp[ds_c['depth']==10])
    temp_mean = np.mean(ds_c.temp)
    vwc_50cm = np.mean(ds_c.vwc[ds_c['depth']==50])
    vwc_10cm = np.mean(ds_c.vwc[ds_c['depth']==10])
    vwc_mean = np.mean(ds_c.vwc)
    f_ec = 9000
    t_conv = 273.15
    t_mean_conv = temp_mean+t_conv
    t_50cm_conv = temp_50cm+t_conv
    t_10cm_conv = temp_10cm+t_conv

    Y0 = pd.read_csv(os.path.join('data/'+site+'Y0.csv'), sep=',', header=0) #link
########################################################################################################################

    def deterministic(feature, target, df, iters=100, round_n=3):
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

        f_ec = 9000
        t_conv = 273.15
        t_50cm_conv = temp_50cm+t_conv
        t_10cm_conv = temp_10cm+t_conv
        t_mean_conv = temp_mean+t_conv

        # Preallocate lists
        DR2_LS, DRMSE_LS = [None] * iters, [None] * iters
        DR2_LT, DRMSE_LT = [None] * iters, [None] * iters
        DR2_10, DRMSE_10 = [None] * iters, [None] * iters
        DR2_50, DRMSE_50 = [None] * iters, [None] * iters
        DR2_ID, DRMSE_ID, ypred_ID_ = [None] * iters, [None] * iters, [None] * iters

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

            ### Stochastic modelling for layers separate. 
            ### This is a combination of both layer's prediction
            Dypred_LS = np.concatenate((Dypred_10, Dypred_50))
            DR2_LS[i] = round(r2_score(y_test, Dypred_LS), round_n)
            DRMSE_LS[i] = round(RMSE(y_test, Dypred_LS), round_n)
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
            D0R2_ID[i] = round(r2_score(Y0_test, Dypred_ID), round_n)
            D0RMSE_ID[i] = round(RMSE(Y0_test, Dypred_ID), round_n)

        return np.median(DR2_LT), np.median(DRMSE_LT), np.median(DR2_ID), np.median(DRMSE_ID), np.median(DR2_LS), np.median(DRMSE_LS), np.median(DR2_10), np.median(DRMSE_10), np.median(DR2_50), np.median(DRMSE_50), np.median(D0R2_LT), np.median(D0RMSE_LT), np.median(D0R2_ID), np.median(D0RMSE_ID), np.median(D0R2_LS), np.median(D0RMSE_LS)

    feature = 'bulk_ec_inv'
    Dresults = {}
    target = 'vwc'

    DR2_LT, DRMSE_LT, DR2_ID, DRMSE_ID, DR2_LS, DRMSE_LS, DR2_10, DRMSE_10, DR2_50, DRMSE_50, D0R2_LT, D0RMSE_LT, D0R2_ID, D0RMSE_ID, D0R2_LS, D0RMSE_LS = deterministic(feature, target, ds_c)

###############################################################################################################################################

    filename = f"{now}_{site}_{FM}_{minm}_{alpha}_{cl}_{percent}_04_unc.json"
    filepath = os.path.join(temp_dir,filename)
    file = open(filepath, 'w')

    #file.write('\t"EC_00":"{}",'.format(EC_00) + '\n')
    #file.write('\t"EC_upper_p":"{}",'.format(EC_upper_p) + '\n')
    #file.write('\t"EC_lower_p":"{}",'.format(EC_lower_p) + '\n')

    #file.write('\t"sens_pedm_upper":"{}",'.format(sens_pedm_upper) + '\n')
    #file.write('\t"sens_pedm_lower":"{}",'.format(sens_pedm_lower) + '\n')

    file.write('\t"EC_mean":"{}",'.format(EC_mean) + '\n')
    #file.write('\t"ROPE_inv_upper_p":"{}",'.format(ROPE_inv_upper_p) + '\n')
    #file.write('\t"ROPE_inv_lower_p":"{}",'.format(ROPE_inv_lower_p) + '\n')

    file.write('\t"input file + path": "{}",'.format(infile) + '\n\n')
    file.write('\t"instrument": "{}",'.format(config['instrument_code'] ) + '\n')
    file.write('\t"instrument mode": "{}",'.format(config['instrument_orientation']) + '\n')
    file.write('\t"instrument height (m)": {:.3f},'.format(config['instrument_height']) + '\n')

    if config['remove_coil']:
        rem_coils = instrument.cc_names[config['coil_n']]
        file.write('\t"configurations not used in inversion": "{}",'.format(rem_coils) + '\n\n')

    file.write('\t"regularisation": "{}",'.format(config['regularization']) + '\n')
    file.write('\t"reference EC profile":"{}",'.format(config['reference_profile']) + '\n')

    file.write('\t"alpha parameter": "{}",'.format(alph_param) + '\n\n')
    file.write('\t"optimisation method":"{}",'.format(config['opt_method']) + '\n')
    file.write('\t"forward model": "{}",'.format(config['FM']) + '\n')
    file.write('\t"site": "{}",'.format(site) + '\n')
    file.write('\t"Dresults": "{}",'.format(Dresults) + '\n')

    if config['constrain']:
        file.write('\t "constrained inversion":' + '\n')
        if config['n_int']:
            file.write('\t"custom interface boundaries": "{}"\n'.format(config['interface']) + '\n')
        if config['custom_bounds']:
            file.write('\t"custom inversion constraints (bnds)": "{}" \n'.format(config['bounds']) + '\n')
        else:
            file.write('\t"automated inversion constraints (bnds)": "{}"\n'.format(bounds) + '\n')

    file.close()
    round_n = 3
    return round(DR2_LT, round_n), round(DRMSE_LT, round_n), round(DR2_ID, round_n), round(DRMSE_ID, round_n), round(DR2_LS, round_n), round(DRMSE_LS, round_n), round(DR2_10, round_n), round(DRMSE_10, round_n), round(DR2_50, round_n), round(DRMSE_50, round_n), round(D0R2_LT, round_n), round(D0RMSE_LT, round_n), round(D0R2_ID, round_n), round(D0RMSE_ID, round_n), round(D0R2_LS, round_n), round(D0RMSE_LS, round_n)