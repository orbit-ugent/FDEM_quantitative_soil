import sys
import datetime
import warnings
import os 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pedophysics import predict, Soil
#from PM import *
from FDEM import Initialize

from scipy import constants
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Get notebook and parent dir
current_dir = os.path.dirname(os.path.abspath('__file__')) 
parent_dir = os.path.dirname(current_dir)
# Set path to pedophysics module 
pedophysics_code_path = os.path.join(parent_dir)
sys.path.insert(0, pedophysics_code_path)
from utils.SA_utils import *
from pedophysics import predict, Soil
from utils.spatial_utils import get_coincident, get_stats_within_radius
from FDEM import Modeling as Mod
from FDEM import Properties as Prop
from utils.profile_utils import merge_layers
from utils.profile_utils import smooth_profiles, clip_profiles_to_max_depth
from scipy.spatial import cKDTree
from emagpy import Problem
#from utils.plots import *
from resipy import Project
from FDEM import Initialize
from scipy import constants
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.insert(0,'../src/') # this add the emagpy/src directory to the PATH


def SA(site, extract, sample_loc, interface, FM, MinM, alpha, remove_coil, start_avg, constrain):
    """
    site : 'M', 'P' Proefhoeve Middelkerke
    extract: 0.5, 2.5
    sample_loc: 'mean', 'closest'
    interface: 'Observed', 'Log-defined'
    FM: 'FSeq', 'CS', 'FSlin' 
    MinM: 'CG', 'ROPE'
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
    
    ERTdatadir = 'data/ERT/'
    ERT_path = ERTdatadir + profile_prefix+'-inv-ERT-'+str(extract)+'.csv'

    if os.path.exists(ERT_path):
        ERT_profiles = pd.read_csv(ERT_path)

    else: 
        print('########################################################################################################################'
        '############################################# 00 ERT INVERSION ################################################')

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
        k.filterRecip() 
        k.fitErrorPwl()
        k.createMesh('trian')

        # invert
        k.err = True  # use fitted error in the inversion
        k.invert()

        # extract profiles
        m = k.meshResults[0]
        dfs = []
        for i in range(df.shape[0]):
            row = df.loc[i, :]
            ie = m.dfm['X'].between(row['distance'] - extract, row['distance'] + extract) & m.dfm['Z'].gt(-5) # 0.5 and 2.5
            sdf = m.df[ie][['Z', 'Resistivity(ohm.m)']]
            sdf['Z'] = sdf['Z'].round(1)
            sdf['Z'] = (sdf['Z'] * 2).round(1) / 2
            sdf = sdf.groupby('Z').mean().reset_index()
            sdf['easting'] = row['easting']
            sdf['northing'] = row['northing']
            sdf['ID'] = row['ID']
            dfs.append(sdf)

        ERT_profiles = pd.concat(dfs)
        ERT_profiles.to_csv(ERTdatadir + profile_prefix+'-inv-ERT-'+str(extract)+'.csv', index=False)



    cal_data_dir = 'data/calibrated/'
    cal_path = cal_data_dir + emfile_prefix+'_raw_calibrated_rECa_'+str(extract)+'.csv'

    if os.path.exists(cal_path):
        em_survey = pd.read_csv(cal_path)

    else: 
        print('#####################################################################################################'
        '############################################# 01 CALIBRATION ################################################')

        #cal_folder = os.path.join(datafolder, 'calibrated')
        #os.makedirs(cal_folder, exist_ok=True) 

        if site == 'P':
            profile_prefix = 'proefhoeve'
            if config['instrument_code'] == 'Dualem-21HS':
                emfile_prefix = 'proefhoeve_21HS'
            else: 
                emfile_prefix = 'proefhoeve_421S'
        else:
            profile_prefix = 'middelkerke'
            emfile_prefix = 'middelkerke_421S'
            # check if correct instrument (only 421S data available for Middelkerke)
            if config['instrument_code'] == 'Dualem-21HS':
                config['instrument_code'] = 'Dualem-421S'
        #cal_folder = os.path.join(datafolder, 'calibrated')
        os.makedirs(cal_data_dir, exist_ok=True) 

        Raw_emfile_prefix = 'Raw/'+emfile_prefix + '_raw'

        raw_em_data = os.path.join(datafolder, f'{Raw_emfile_prefix}.csv')
        ert_file = os.path.join(datafolder, 'ERT/'+f'{profile_prefix}-inv-ERT-'+str(extract)+'.csv')
        #samplocs = os.path.join(datafolder, f'{profile_prefix}_samps.csv')
        ###
        #print('ert_file', ert_file)
        config['ert_file'] = ert_file
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
        ert_p = pd.read_csv(config['ert_file'], sep=',', header=0)
        EM_data = pd.read_csv(config['em_file'], sep=',', header=0)
        #print('ert_p', ert_p)
        save_fig = False


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

        # Plotting combined profiles
        #plot_combined_profiles(ert_final, ert_final, dataset_name)

    ##########################################################################################################

        # Homogenise ERT profiles (assure uniform depth extent and intervals
        # ----------------------------------------------------------------- #

        # Columns containing the resistivity data
        data_columns = ['Resistivity(ohm.m)', 'Smoothed']
        # Assuming ert_final is your DataFrame with profile data
        all_profiles_df, uniform_intervals = check_uniformity_and_interpolate(
            ert_final, 'ID', 'Z', *data_columns
        )

    ##########################################################################################################

        # Create forward model inputs per ERT profile (based on number of layers)
        # ---------------------------------------------------------------------- #

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
        #print('cal_r_EM', cal_r_EM)

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
                #print('cc', cc)
                #print('ip_config', ip_config)
                #print('cal_EM[ip_config]', cal_EM[ip_config])

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
        cal_r_EM.to_csv(cal_path, index=False)
        em_survey = cal_r_EM #### Link 01 to 02 files

    inv_folder = 'data/inverted/'
    inv_path = inv_folder + f'{emfile_prefix}_inverted_samples_{extract}_{sample_loc}_{interface}_{FM}_{MinM}_{alpha}_{remove_coil}_{start_avg}_{constrain}.csv'

    if os.path.exists(inv_path):
        ds_c = pd.read_csv(inv_path)

    else: # Failed and crashed inversions will be recalculated
        print('########################################################################################################################'
        '############################################# 02 INVERSION CONFIGURE INPUT ################################################')
        # User input

        #only_samples = True

        # Define input datatype and source folder
        #em_intype = 'rec'   # 'rec' = rECa transect; 'lin' = LIN ECa transect; 
                            # 'survey' = rEC full survey

        #config = {}
        #config['FM'] = FM #'CS', 'FSlin' or 'FSeq'
        #config['MinM'] = MinM
                                                # mMinimize = ['L-BFGS-B','TNC','CG','Nelder-Mead'] --> https://docs.scipy.org/doc/scipy/reference/optimize.html 
                                                # mMCMC = ['ROPE','DREAM', 'MCMC'] # ??? 'SCEUA' ??? --> https://spotpy.readthedocs.io/en/latest/ 
                                                # mOther = ['ANN','Gauss-Newton','GPS'] (ANN requires tensorflow)
        #config['alpha'] = alpha
        #config['remove_coil'] = remove_coil    # set to True if you want to remove coils in the inversion process
        # Reference profile for starting model (conductivity values)
        #config['start_avg'] = start_avg     # take average of input resistivity profiles per layer as starting model
                                        # if false, reference profile is taken as starting model
        #config['constrain'] = constrain

        # Sensor settings
        #config['instrument_code'] = 'Dualem-21HS' # instrument code
        #config['instrument_height'] = 0.165     # instrument height
        #config['instrument_orientation'] = 'HCP'    # instrument orientation
        config['regularization'] = 'l2'
        reg_meth = config['regularization']

        # Remove coils for inversion?

        n = 4                                    
        inv_config(site, config, interface)

        # ---------------------------------------------------------------------------- #
        # ---------------------------------------------------------------------------- #

        #if s_site == 'P':
        #    profile_prefix = 'proefhoeve'
        #    config['instrument_code'] == 'Dualem-21HS'
        #    emfile_prefix = 'proefhoeve_21HS'
            #else: 
            #    emfile_prefix = 'proefhoeve_421S'
        #else:
        #    profile_prefix = 'middelkerke'
        #    emfile_prefix = 'middelkerke_421S'
            # check if correct instrument (only 421S data available for Middelkerke)
        #    if config['instrument_code'] == 'Dualem-21HS':
        #        config['instrument_code'] = 'Dualem-421S'
        inv_folder = os.path.join(datafolder, 'inverted')
        os.makedirs(inv_folder, exist_ok=True) 
        #cal_folder = os.path.join(datafolder, 'calibrated')
        ert_file = os.path.join(ERTdatadir, f'{profile_prefix}-inv-ERT-{extract}.csv')
        em_rec = os.path.join(cal_data_dir, f'{emfile_prefix}_raw_transect_calibrated_rECa_{extract}.csv')
        em_lin = os.path.join(cal_data_dir, f'{emfile_prefix}_raw_transect_calibrated_{extract}.csv')
        em_survey = os.path.join(cal_data_dir, f'{emfile_prefix}_raw_calibrated_rECa_{extract}.csv')
        samplocs = os.path.join(datafolder, f'{profile_prefix}_samp_locations.csv')

        print('files read')
        #if em_intype == 'rec':
        #    infile = em_rec
        #elif em_intype == 'survey':
        #    infile = em_survey
        #else:
        #    infile = em_lin

        config['FM'] = FM #'CS', 'FSlin' or 'FSeq'
        config['MinM'] = MinM
                                                # mMinimize = ['L-BFGS-B','TNC','CG','Nelder-Mead'] --> https://docs.scipy.org/doc/scipy/reference/optimize.html 
                                                # mMCMC = ['ROPE','DREAM', 'MCMC'] # ??? 'SCEUA' ??? --> https://spotpy.readthedocs.io/en/latest/ 
                                                # mOther = ['ANN','Gauss-Newton','GPS'] (ANN requires tensorflow)
        config['alpha'] = alpha
        config['remove_coil'] = remove_coil    # set to True if you want to remove coils in the inversion process
        # Reference profile for starting model (conductivity values)
        config['start_avg'] = start_avg     # take average of input resistivity profiles per layer as starting model
                                        # if false, reference profile is taken as starting model
        config['constrain'] = constrain

        config['instrument_height'] = 0.165
        config['instrument_orientation'] = 'HCP'    # instrument orientation

        instrument = Initialize.Instrument(config['instrument_code'],
                                            instrument_height=config['instrument_height'],
                                            instrument_orientation=config['instrument_orientation']
                                            )

        # Column names for emapgy input
        emp_21HS = [f"HCP0.5f9000{config['instrument_height']}", 'PRP0.6f9000h0.165', 'HCP1.0f9000h0.165', 'PRP1.1f9000h0.165',	'HCP2.0f9000h0.165', 'PRP2.1f9000h0.165',
                    'HCP0.5f9000h0.165_inph', 'PRP0.6f9000h0.165_inph', 'HCP1.0f9000h0.165_inph',
                    'PRP1.1f9000h0.165_inph', 'HCP2.0f9000h0.165_inph', 'PRP2.1f9000h0.165_inph'
                    ]

        emp_421S = ['HCP1.0f9000h0.165', 'PRP1.1f9000h0.165',	'HCP2.0f9000h0.165', 'PRP2.1f9000h0.165', 'HCP4.0f9000h0.165', 'PRP4.1f9000h0.165', 
                    'HCP1.0f9000h0.165_inph', 'PRP1.1f9000h0.165_inph', 'HCP2.0f9000h0.165_inph', 'PRP2.1f9000h0.165_inph',
                    'HCP4.0f9000h0.165_inph', 'PRP4.1f9000h0.165_inph'
                    ]

        # Datetime for filename
        now = (datetime.datetime.now())
        now = now.strftime("%y%m%d_%H%M")

        # 1.0 Data import and structuring into dataframe
        ert_p = pd.read_csv(ert_file, sep=',', header=0)
        em_rec = pd.read_csv(em_rec, sep=',', header=0)
        em_lin = pd.read_csv(em_lin, sep=',', header=0)
        em_survey = pd.read_csv(em_survey, sep=',', header=0)
        samples = pd.read_csv(samplocs, sep=',', header=0)

        #if c_transform:
            # Create a new filename with the target EPSG code
        #    em_rec = utm_to_epsg(em_rec, c_utmzone, target_epsg=c_target_cs)
        #    em_lin = utm_to_epsg(em_lin, c_utmzone, target_epsg=c_target_cs)
        #    em_survey = utm_to_epsg(em_survey, c_utmzone, target_epsg=c_target_cs)

        if sample_loc == 'closest':
            em_samples = get_coincident(em_survey, samples)

        elif sample_loc == 'mean':
            if site == 'P':
                em_samples = get_stats_within_radius(em_survey, samples, 1)

            elif site == 'M':
                em_samples = get_stats_within_radius(em_survey, samples, 2)
        
        print('get ERT profiles')
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
        # #Plotting combined profiles
        # plot_combined_profiles(ert_final, all_profiles_df, 
        #                        data_column, compare_name = data_column)
        #print('em_rec', em_rec)


    ########################################################################################################################

        # config['reference_profile'] = 11
        pd.set_option('display.max_columns', None)

        if config['reference_profile'] not in all_profiles_df['ID'].unique():
            warnings.warn("Warning: the reference profile ID does not exist. Provide correct profile ID.")
            raise KeyboardInterrupt
        else:
            profile_id = config['reference_profile']

        print(' Create new layer configuration for prior model based on ERT data')
        # Create new layer configuration for prior model based on ERT data
        if config['n_int']:
            new_int = config['interface']
            #print('all_profiles_df', all_profiles_df)
            #print('new_int', new_int)
            merged_df = merge_layers(all_profiles_df, new_int,'EC(mS/m)')
        else:
            merged_df = all_profiles_df
        #print('merged_df', merged_df)

        # Plot original and (merged and) DC corrected reference profile
        #if config['n_int']:
        #    plot_title = 'Original vs merged & DC corrected data'
        #    first_in = .1
        #else: 
        #    plot_title = 'Original vs DC corrected data'
        #    first_in = .0
        #ert_eval = ert_final.copy()
        #ert_eval['Z'] = ert_eval['Z'].values + first_in

        #plot_profile(ert_eval, profile_id, dataset_name, compare=True, compare_df = comparedf, compare_name = 'EC(mS/m)', block=True, plot_title=plot_title)

        # Get prior model info
        models = generate_forward_model_inputs(merged_df, 'ID', 'Z', 'EC(mS/m)')

        print('forward model inputs generated')
    ########################################################################################################################

        # 
        # -------------------------------------------------------------------- #

        # 
        profile_data = merged_df[merged_df['ID'] == profile_id].copy()
        res_col = 'EC(mS/m)'
        depth = 'Z'
        #max_ert_depth = ert_final['Z'].abs().max()

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

        #ref_len = len(con)
        #num_layers = len(con)
        #perm = np.full(num_layers, constants.epsilon_0)
        #sus = np.zeros(num_layers)

        # # Create model instance
        #M = Initialize.Model(thick, sus[::-1], con[::-1], perm[::-1])

        # ----------------------------------------------------------------------

        dataset_name = 'EC(mS/m)'
        layers_interfaces = np.cumsum(models[profile_id].thick)
        layers_interfaces = np.insert(layers_interfaces, 0, 0)
        profile_data = ert_final[ert_final['ID'] == profile_id]

        #fig, axr = plt.subplots(figsize=(5, 10))
        #axr.set_xlabel('EC [mS/m]')
        #axr.set_ylabel('depth [m]')
        #axr.plot((profile_data[dataset_name]),profile_data['Z'], label='original (DC) ERT EC',)
        #if not config['n_int']: 
        #    axr.plot(con[:-1]*1000,-thick, '.', label='Model EC 9khz',color = 'red')
        #else:
        #    axr.plot(con*1000,-thick, '.', label='Model EC 9khz',color = 'red')
        #axr.set_title(f'Reference profile: ID {profile_id}')
        #print('ec_stats', ec_stats)
    
        conductivities = con*1000

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
            ec_df = pd.concat([ec_df, pd.DataFrame([np.flip(con_m)], columns=ec_cols_ref)])

        ec_df.reset_index(drop=True, inplace=True)

        ec_stats = ec_df.describe().loc[['min', 'max', 'std', '50%', 'mean']]
        ec_stats.rename(index={'50%': 'median'}, inplace=True)
        ec_stats.loc['min_sd'] = ec_stats.loc['min'] - 2 * ec_stats.loc['std']
        ec_stats.loc['max_sd'] = ec_stats.loc['max'] + 2 * ec_stats.loc['std']

        if not config['n_int']:

            start_mod = ec_stats.loc['mean'].values[1:]
        else:
            start_mod = ec_stats.loc['mean'].values

        #    layers_interfaces = np.cumsum(models[i].thick)
        #    layers_interfaces = np.insert(layers_interfaces, 0, 0)
        #    num_layers = len(con)
        #    perm = np.full(num_layers, constants.epsilon_0)
        #    sus = np.zeros(num_layers)

        #    first_lay = profile_data[depth].iloc[-1].round(decimals=1)
        #    second_lay = profile_data[depth].iloc[-2].round(decimals=1)

        #    if not config['n_int']:
        #        first_lay = profile_data[depth].iloc[-1].round(decimals=1)
        #        second_lay = profile_data[depth].iloc[-2].round(decimals=1)
        #        if first_lay == 0:
        #            profile_data[depth]=profile_data[depth] +second_lay
        #        else:
        #            profile_data[depth]=profile_data[depth] +first_lay
        #        thick = -profile_data[depth].iloc[1:].values
        #    else:
        #        thick = -profile_data[depth].values
            # if a == 1:
            #     fig, ax = plt.subplots(figsize=(5, 10))
            #     ax.set_xlabel('EC [mS/m]')
            #     ax.set_ylabel('depth [m]')
            #     profile_data = ert_final[ert_final['ID'] == i]
            #     ax.plot((profile_data[dataset_name]),profile_data['z'], label='original (DC) ERT EC',)
            #     if not config['n_int']:
            #         ax.plot(con[:-1],-thick, '.', label='Model EC 9khz',color = 'red')
            #     else:
            #         ax.plot(con,-thick, '.', label='Model EC 9khz',color = 'red')
            #     ax.set_title(f'profile {i}')
            #     ax.legend()
            #     a = a +1

        #position = -thick


        # define parameters for inversion starting model
        # --------------------------------------------- #

        #if not config['n_int']:
        #    minstat = np.flipud(ec_stats.loc['min'].values[1:])
        #    maxstat = np.flipud(ec_stats.loc['max'].values[1:])
        #    start_mod = ec_stats.loc['mean'].values[1:]
        #    boundcols = ec_cols_ref[:-1]
        #else:
        #    minstat = np.flipud(ec_stats.loc['min'].values)
#            maxstat = np.flipud(ec_stats.loc['max'].values)
 #           start_mod = ec_stats.loc['mean'].values

        bounds = constrain_bounds(config, ec_cols_ref, ec_stats)
        
    ########################################################################################################################

        if config['start_avg']:
            conductivities = start_mod
            if len(conductivities) == len(mod_layers):
                mod_layers = mod_layers[1:]
            elif len(conductivities) == (len(mod_layers)+1):
                mod_layers = mod_layers

    ########################################################################################################################

        # Perform inversion on sampling locations (to be used in pedophysical modelling)

        if 'code' in em_samples.columns:
            em_samples = em_samples.rename(columns={'code': 'ID'})

        # if config['n_omit'] != 0:
        #     unique_ids = em_input['ID'].unique()
        #     selected_ids = unique_ids[config['n_omit']:-config['n_omit']]
        #     em_input = em_input.loc[em_input['ID'].isin(selected_ids)]
        temp_dir = 'temp_emp' 

        infile_name = 'infile_s02.csv'
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir,infile_name)

        i = instrument.niter
        em_samples.columns.values[n:n+i]

        if config['instrument_code'] == 'Dualem-21HS':
            new_columns = emp_21HS
        else:
            new_columns = emp_421S

        if len(new_columns) != i:
            raise ValueError("The length of new_columns must be equal to the number of columns to rename")
        else:
            em_samples.columns.values[n:n+i] = new_columns

        em_samples.to_csv(temp_file)

        # transect inversion settings

        s_rec = Problem()
        s_rec.createSurvey(temp_file)
        #t_rec.rollingMean(window=12)

        s_rec.setInit(
            depths0=np.flipud(mod_layers),
            conds0=conductivities
            # fixedDepths=startmodel['fixedDepths'],
            # fixedConds=startmodel['fixedConds']
            )
        #print(np.flipud(mod_layers), conductivities)

        #shutil.rmtree(temp_dir)

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

        inv_samples(config, reg_meth, FM, MinM, alpha, s_rec, bounds)
        #s_rec.showOne2one()  

        survey = s_rec.surveys[0]
        dfsForward = s_rec.forward(forwardModel=FM)[0]
        r2 = pd.DataFrame(columns=np.r_[s_rec.coils, ['all']])
        coils = s_rec.coils

        r2inv = r2_inv(s_rec, survey, dfsForward, r2)['all']   
        print('r2inv', r2inv)

    ########################################################################################################################

        # 4.1: Plot the inversion results and put outcomes into a pandas dataframe
        # ------------------------------------------------------------------------
        #csv_filename = f'{emfile_prefix}_inverted_samples_{cl}_{percent}_{FM}_{MinM}_{alpha}_{remove_coil}_{start_avg}_{constrain}.csv'

        # ******************************************************************** #

        # Plot inversion outcomes down to a max depth of 2 m, and plotting the data
        # based on their true coordinates along the transect (dist=True).
        #s_rec.showResults(dist=True, errorbar = True) 
        #print('s_rec.depths', s_rec.depths)
        # Extracting the values from the first row of the transect.depths[0] array
        depth_values = s_rec.depths[0][0]

        # Creating the custom column names for layer_cols
        layer_cols = ['EC_{:.2f}'.format(d) for d in depth_values] + ['EC_end']

        # Combining the data from the 'x', 'y' columns and the transect.models[0] array
        data = np.c_[s_rec.surveys[0].df[['x', 'y']].values, s_rec.models[0]]

        # Creating the final dataframe with the desired column names
        ds_inv = pd.DataFrame(data, columns=['x', 'y'] + layer_cols)
        ds_inv['pos'] = em_samples['ID'].to_numpy()
        ds_inv['r2inv'] = r2inv[0]
        # ----------------------------------------------------------------------

        # Export the dataframe as a csv-file
        #outfile_transect = os.path.join(inv_folder, csv_filename)
        if r2inv[0] > 0:
            ds_inv.to_csv(inv_path)
            

        if r2inv[0] < 0:
            inv_path_failed = inv_folder + f'{emfile_prefix}_inverted_samples_{extract}_{sample_loc}_{interface}_{FM}_{MinM}_{alpha}_{remove_coil}_{start_avg}_{constrain}_failed.csv'
            ds_inv.to_csv(inv_path_failed)
        
    print('########################################################################################################################'
    '############################################ 03 DETERMINISTIC MODELLING ###################################################')

    #datafolder = 'data' # data folder
    #data_inv_folder = 'data/inverted'
    #data_sanalysis_folder = 'data/soil_analyses'

    #if s_site == 'P':
    #    profile_prefix = 'proefhoeve'
    #    emfile_prefix = 'proefhoeve_21HS'
        #else: 
        #    emfile_prefix = 'proefhoeve_421S'
    #else:
    #    profile_prefix = 'middelkerke'
    #    emfile_prefix = 'middelkerke_421S'

    inv_s = os.path.join(inv_path)

    # Define input datatype and source folder
    #em_intype = 'reca'   # 'reca', 'LIN' 
    #cal = 'calibrated' # 'calibrated', 'non_calibrated', 'drift_calibrated'

    #if cal == 'non_calibrated':
    #    em_intype = 'LIN' 
        
    #instrument_code = '21HS' # 421S, '21HS'

    if site == 'M':
        instrument_code = '421S'

    #cal_folder = os.path.join(datafolder, cal)
    em_survey = os.path.join(cal_data_dir, f'{emfile_prefix}_raw_calibrated_rECa_{extract}.csv')
    data_sanalysis_folder = 'data/soil_analyses'

    sampleprop = os.path.join(data_sanalysis_folder, f'{profile_prefix}_soil_analysis.csv')

    # Profile smoothing parameters 
    window_size = 1 # Define your window size for simple moving average filter (1 = no filtering)

    # 1.0 Data import and structuring into dataframe
    em_survey = pd.read_csv(em_survey, sep=',', header=0)
    #print('em_survey', em_survey.head())

    inverted = pd.read_csv(inv_s, sep=',', header=0)
    r2inv = inverted.r2inv[0]
    samples_analysis = pd.read_csv(sampleprop, sep=',', header=0)
    #print('samples', samples_analysis.head())

    em_sample_prop = get_coincident(em_survey, samples_analysis)

    #pd.options.future.infer_string = True
    ds = em_sample_prop.copy()

    # Caclculate Bulk EC from HydraProbe data at 50Mhz
    offset = 4
    water_perm = 80
    ds['bulk_ec_hp'] = logsdon(50e6, ds.rperm, ds.iperm)

    ds['bulk_ec_dc_hp'] = predict.BulkECDC(Soil(frequency_ec = 50e6,
                                                bulk_ec = ds.bulk_ec_hp.values))

    ds['bulk_ec_tc_hp'] = SheetsHendrickxEC( ds.bulk_ec_hp, ds.temp)
    ds['bulk_ec_dc_tc_hp'] = predict.BulkECDCTC(Soil(temperature = ds.temp.values,
                                                        bulk_ec_dc = ds.bulk_ec_dc_hp.values
                                                        ))

    # Caclculate Water EC from HydraProbe data at 50Mhz
    ds['water_ec_hp'] = Hilhorst(ds.bulk_ec_hp, ds.rperm, water_perm, offset)
    ds['water_ec_hp_t'] = WraithOr(ds.water_ec_hp, ds.temp)
    ds['iperm_water_t'] = ds.water_ec_hp_t/(50e6*2*pi*epsilon_0)

########################################################################################################################

    inv_columns = inverted.columns[3:-2]
    print('inv_columns', inv_columns)

    ds[inv_columns] = np.nan

    for idc, c in enumerate(inv_columns):

        for i in range(len(inverted.x)):
            ds.loc[ds.code == i+1, c] = inverted.loc[i, c]

    def closest_ec(row):
        depth = row['depth']
        # Filter columns that start with 'EC_' but not 'EC_end'
        ec_cols = [col for col in row.index if col.startswith('EC_') and col != 'EC_end']
        # Convert the part after 'EC_' to float and calculate the absolute difference with depth
        differences = {col: abs(depth/100 - float(col.split('_')[1])) for col in ec_cols}
        # Find the column name with the minimum difference
        closest_col = min(differences, key=differences.get)
        return row[closest_col]

    #print(ds.head())

    # Apply the function to each row
    ds['bulk_ec_inv'] = ds.apply(closest_ec, axis=1)

    ds['bulk_ec_dc_tc_inv'] = predict.BulkECDCTC(Soil(temperature = ds.temp.values+273.15,
                                                        frequency_ec = 9e3,
                                                        bulk_ec = ds.bulk_ec_inv.values/1000))*1000


    ds['ideal_bulk_ec'] = predict.BulkEC(
                                            Soil(temperature = ds.temp.values+273.15,
                                                water = ds.vwc.values,
                                                clay = ds.clay.values,
                                                bulk_density = ds.bd.values,
                                                water_ec = ds.water_ec_hp.values,
                                                frequency_ec = 9e3
                                                ))

    if instrument_code == '21HS':
        feature_set = [
            'HCPHQP',
            'HCP1QP',
            'HCP2QP',
            'PRPHQP',
            'PRP1QP',
            'PRP2QP',
            'bulk_ec_inv',
        #    'bulk_ec_dc_tc_inv',
        #    'bulk_ec_hp',
        #    'bulk_ec_dc_tc_hp'
        ]
    elif instrument_code == '421S':
        feature_set = [
            'HCP1QP',
            'HCP2QP',
            'HCP4QP',
            'PRP1QP',
            'PRP2QP',
            'PRP4QP',
            'bulk_ec_inv',
        #    'bulk_ec_dc_tc_inv',
        #    'bulk_ec_hp',
        #    'bulk_ec_dc_tc_hp'
        ]
        
    #folder_path = 'output_tables/'
    #file_name = 'ds_'+profile_prefix+'.csv'
    #ds.to_csv(folder_path + file_name, index=False)

###################################################################################################################

    #bulk_ec_inv_50cm = ds.bulk_ec_inv[ds['depth']==50]
    #bulk_ec_inv_10cm = ds.bulk_ec_inv[ds['depth']==10]

    clay_50cm = np.mean(ds.clay[ds['depth']==50])
    clay_10cm = np.mean(ds.clay[ds['depth']==10])
    bd_50cm = np.mean(ds.bd[ds['depth']==50])
    bd_10cm = np.mean(ds.bd[ds['depth']==10])
    #water_ec_hp_50cm = np.mean(ds.water_ec_hp[ds['depth']==50])
    #water_ec_hp_10cm = np.mean(ds.water_ec_hp[ds['depth']==10])
    water_ec_hp_50cm_t = np.mean(ds.water_ec_hp_t[ds['depth']==50])
    water_ec_hp_10cm_t = np.mean(ds.water_ec_hp_t[ds['depth']==10])
    clay_mean = np.mean(ds.clay)
    bd_mean = np.mean(ds.bd)
    #water_ec_hp_mean = np.mean(ds.water_ec_hp)
    water_ec_hp_mean_t = np.mean(ds.water_ec_hp_t)
    temp_50cm = np.mean(ds.temp[ds['depth']==50])
    temp_10cm = np.mean(ds.temp[ds['depth']==10])
    temp_mean = np.mean(ds.temp)
    #vwc_50cm = np.mean(ds.vwc[ds['depth']==50])
    #vwc_10cm = np.mean(ds.vwc[ds['depth']==10])
    #vwc_mean = np.mean(ds.vwc)
    f_ec = 9000
    t_conv = 273.15
    t_mean_conv = temp_mean+t_conv
    t_50cm_conv = temp_50cm+t_conv
    t_10cm_conv = temp_10cm+t_conv


###################################################################################################################

    Y0 = pd.read_csv('data/'+site+'Y0.csv', sep=',', header=0) # Link

    feature = 'bulk_ec_inv'
    #Dresults = {}
    target = 'vwc'

    DR2_LT, DRMSE_LT, DR2_ID, DRMSE_ID, DR2_LS, DRMSE_LS, DR2_10, DRMSE_10, DR2_50, DRMSE_50, D0R2_LT, D0RMSE_LT, D0R2_ID, D0RMSE_ID, D0R2_LS, D0RMSE_LS, MAE_LS, MAE_LT, MAE_10, MAE_50, MAE_ID, STD_LS, STD_LT, STD_10, STD_50, STD_ID = deterministic(feature, target, ds, Y0, f_ec, clay_mean, bd_mean, water_ec_hp_mean_t, t_mean_conv, clay_10cm, bd_10cm, water_ec_hp_10cm_t, t_10cm_conv, clay_50cm, bd_50cm, water_ec_hp_50cm_t, t_50cm_conv, t_conv)

    round_n = 3

    return round(DR2_LT, round_n), round(DRMSE_LT, round_n), round(DR2_ID, round_n), round(DRMSE_ID, round_n), round(DR2_LS, round_n), round(DRMSE_LS, round_n), round(DR2_10, round_n), round(DRMSE_10, round_n), round(DR2_50, round_n), round(DRMSE_50, round_n), round(D0R2_LT, round_n), round(D0RMSE_LT, round_n), round(D0R2_ID, round_n), round(D0RMSE_ID, round_n), round(D0R2_LS, round_n), round(D0RMSE_LS, round_n), round(MAE_LT, round_n), round(STD_LT, round_n),round(MAE_ID, round_n), round(STD_ID, round_n), round(MAE_LS, round_n), round(STD_LS, round_n), round(MAE_10, round_n), round(STD_10, round_n), round(MAE_50, round_n), round(STD_50, round_n),  r2inv


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
    print('generate_forward_model_inputs')
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