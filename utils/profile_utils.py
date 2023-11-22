# -*- coding: utf-8 -*-
# Packages ------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Apply a simple smooting function (sma) on a set of profiles
def smooth_profiles(df, value_col, depth_col, window_size, min_periods=1):
    """
    Apply a moving average smoothing on profile data while keeping the original
    first and last values of the profile.

    Parameters:
    - df: DataFrame containing profile data.
    - value_col: The name of the column with values to smooth.
    - depth_col: The name of the column with depth values.
    - window_size: The window size for the moving average in terms of depth units.
    - min_periods: Minimum number of observations in window required to have a value.

    Returns:
    - DataFrame with an additional column with the smoothed values, where the first
      and last values remain unchanged from the original data.
    """
    # Sort by depth_col to ensure correct rolling window application
    df_sorted = df.sort_values(by=depth_col).reset_index(drop=True)
    
    # Create a temporary series to apply the rolling window
    # leaving the first and last values out of the window calculation
    temp_series = pd.Series([np.nan] + df_sorted[value_col].tolist()[1:-1] + [np.nan])
    
    # Calculate the rolling window
    smoothed_values = temp_series.rolling(window=window_size, min_periods=min_periods, center=True).mean()
    
    # Fill in the NaNs with original values for the first and last data points
    smoothed_values.iloc[0] = df_sorted[value_col].iloc[0]
    smoothed_values.iloc[-1] = df_sorted[value_col].iloc[-1]
    
    # Assign the smoothed values back to the DataFrame
    df_sorted['Smoothed'] = smoothed_values.values
    
    return df_sorted


def clip_profiles_to_max_depth(df, max_depth, surf_depth):
    """
    Clip the profile data to a maximum depth.

    Parameters:
    - df: DataFrame containing the profile data.
    - max_depth: The maximum depth (negative value) to include in the profiles.

    Returns:
    - DataFrame with data beyond the max_depth excluded.
    """
    # Filter out rows where the 'z' column is less than the maximum depth
    clipped_df = df[df['z'] >= max_depth].copy()
    clipped_df = clipped_df[clipped_df['z'] <= surf_depth].copy()
    
    return clipped_df


# Plot a single profile with or without comparative profile data
def plot_profile(profile_df, profile_id, dataset_name, 
                 compare=False, compare_df=None, compare_name=None, subs=False,
                 xlims=None, plot_title =None, save_plot = False, plot_name = None):
    """
    Plot profile data, without or with comparative profile data

    Parameters:
    - original_df: DataFrame containing the original profile data.
    - profile_id: The ID of the profile to be plotted.
    - dataset_name: string with dataset column name .
    - compare_df: DataFrame containing the smoothed profile data. (optional)
    - compare_name: string with compare dataset column name. (optional)
    """
    profile_data = profile_df[profile_df['ID'] == profile_id]
    profile_label = str(profile_id)
    if '.0' in profile_label:
        profile_label = profile_label.split('.')[0]
        
    if compare:
            comparative_data = compare_df[compare_df['ID'] == profile_id]
            if subs:
                fig, axs = plt.subplots(2, 1, figsize=(16, 8))
                # Plot the original data
                axs[0].plot(profile_data[dataset_name], profile_data['z'], label=dataset_name)
                axs[0].set_title(f'Profile ID {profile_label} - Original Data')
                #axs[0].invert_yaxis()  # Invert y-axis to show depth correctly
                if xlims:
                     axs[0].set_xlim(xlims)
                axs[0].set_xlabel(dataset_name)
                axs[0].set_ylabel('Depth [m]')
                axs[0].grid(True)
                axs[0].legend()

                # Plot the comparative data
                axs[1].plot( comparative_data[compare_name], comparative_data['z'], label=compare_name, color='red')
                axs[1].plot( comparative_data[compare_name], '.', color='red', markersize=10)
                axs[1].set_title(f'Profile ID {profile_label} - {compare_name}')
                #axs[1].invert_yaxis()  # Invert y-axis to show depth correctly
                if xlims:
                    axs[1].set_xlim(xlims)
                axs[1].set_xlabel(f'{compare_name}')  # Note the corrected syntax here
                axs[1].set_ylabel('Depth (m)')
                axs[1].grid(True)
                axs[1].legend()
            else:
                fig, axs = plt.subplots(1, 1, figsize=(16, 8))
                # Plot the original data
                axs.plot(profile_data[dataset_name], profile_data['z'], label=dataset_name, color='blue')
                axs.plot( profile_data[dataset_name], profile_data['z'],'.', color='blue', markersize=10)
                axs.set_title(f'Profile ID {profile_label} - Original Data')
                #axs[0].invert_yaxis()  # Invert y-axis to show depth correctly
                if xlims:
                     axs.set_xlim(xlims)
                axs.set_xlabel(dataset_name)
                axs.set_ylabel('Depth [m]')
                axs.grid(True)
                axs.legend()

                # Plot the comparative data
                axs.plot( comparative_data[compare_name], comparative_data['z'], label=compare_name, color='red')
                axs.plot( comparative_data[compare_name], comparative_data['z'],'.', color='red', markersize=10)
                axs.set_title(f'Profile ID {profile_label} - {compare_name}')
                #axs[1].invert_yaxis()  # Invert y-axis to show depth correctly
                if xlims:
                     axs.set_xlim(xlims)
                axs.set_xlabel(f'{compare_name}')  # Note the corrected syntax here
                axs.set_ylabel('Depth (m)')
                axs.grid(True)
                axs.legend()
    else:
            # Plot the original data
            fig, axs = plt.subplots(1, 1, figsize=(8, 8))
            axs.plot(profile_data[dataset_name], profile_data['z'], label=dataset_name)
            axs.set_title(f'Profile ID {profile_label}')
            if xlims:
                axs.set_xlim(xlims)
            axs.set_xlabel(dataset_name)
            axs.set_ylabel('Depth [m]')
            axs.grid(True)
            axs.legend()

    if plot_title:
        fig.suptitle(plot_title, fontsize=14)

    plt.tight_layout()
    if save_plot:
        if plot_name:
            filename = plot_name + '.pdf'
        else: 
            filename = f'Profile_ID_{profile_label}.pdf'
        plt.savefig(filename)
        return filename
    plt.show()


# Plot two sets of profiles for visual comparison
def plot_combined_profiles(reference_df, comparison_df, 
                           reference_name, compare_name = None
                           ):
    """
    Plot comparative line plots between the original selected profile data and the smoothed profiles.
    All profiles will be plotted in one subplot for original data and one subplot for smoothed data,
    with synchronized x-axis limits.

    Parameters:
    - reference_df: DataFrame containing the selected profile data after filtering.
    - comparison_df: DataFrame containing the smoothed profile data.
    - reference_name: The column name of the reference dataset.
    - compare_name: The column name of the comparison dataset (if None, this is 
        equal to the reference name).
    """
    if compare_name:
         c_name = compare_name
    else:
         c_name = reference_name

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Plot the original data for all profiles
    for profile_id in reference_df['ID'].unique():
        profile_data = reference_df[reference_df['ID'] == profile_id]
        axs[0].plot(profile_data[reference_name], profile_data['z'], label=f'Profile {profile_id}')
        
    axs[0].set_title('Reference Profiles')
    axs[0].set_xlabel(reference_name)
    axs[0].set_ylabel('Depth (m)')
    axs[0].invert_yaxis()  # Invert y-axis to show depth correctly
    axs[0].xaxis.tick_top()
    axs[0].xaxis.set_label_position('top') 
    axs[0].grid(True)
    
    # Get axis limits from the first subplot
    xlims = axs[0].get_xlim()
    
    # Plot the smoothed data for all profiles
    for profile_id in comparison_df['ID'].unique():
        smoothed_data = comparison_df[comparison_df['ID'] == profile_id]
        axs[1].plot(smoothed_data[c_name], smoothed_data['z'], label=f'Profile {profile_id}')

    axs[1].set_title('Comparison Profiles')
    axs[1].set_xlabel(f'{c_name}')
    axs[1].set_xlim(xlims)  # Synchronize x-axis limits with the original profiles subplot
    axs[1].invert_yaxis()  # Invert y-axis to show depth correctly
    axs[1].xaxis.tick_top()
    axs[1].xaxis.set_label_position('top') 
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


# Assure that a set of profiles have the same depth interval and interpolate
#   where needed
def check_uniformity_and_interpolate(df, profile_id_col, depth_col, *data_cols, plot_hist=False):
    """
    Analyzes profile data for uniformity in depth intervals across different profiles.
    Profiles with non-uniform depth intervals are interpolated to create uniform intervals.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing geological profile data.
    profile_id_col (str): The column name in df that contains profile IDs.
    depth_col (str): The column name in df that contains depth information.
    *data_cols (str): Column names in df that contain data values to be analyzed.
    plot_hist (bool): Boolean to call for plotting of histograms of depth intervals 
                        per profile (optional)
    
    Returns:
    all_profiles_df (pandas.DataFrame): DataFrame with both original and interpolated profiles, 
                                        as appropriate, all with uniform interval.
    uniform_intervals (dict): Dictionary mapping profile IDs to their depth intervals.
    """

    # Initialize a DataFrame to store both interpolated and non-interpolated profiles
    all_profiles_df = pd.DataFrame()
    uniform_intervals = {}

    # Prepare the figure for plotting non-uniform depth intervals
    plt.figure(figsize=(10, 8))
    
    if plot_hist:
        # Keep track of any profiles that have been plotted (i.e., non-uniform intervals)
        plotted_profiles = []

    for profile_id, group in df.groupby(profile_id_col):
        # Calculate absolute differences between consecutive depths
        depth_diffs = group[depth_col].diff().dropna().abs()
        
        # Find the most common interval
        common_interval = depth_diffs.mode()[0]
        
        # Check if the depth differences are uniform
        if np.isclose(depth_diffs, common_interval).all():
            # Record the uniform interval
            uniform_intervals[profile_id] = common_interval
            # Add this profile's data 'as is' to the all_profiles_df
            all_profiles_df = pd.concat([all_profiles_df, group])
        else:
            if plot_hist:
                # Add profile to the list of plotted profiles for the combined histogram
                plotted_profiles.append(profile_id)
                plt.hist(depth_diffs, bins=30, alpha=0.5, label=f'Profile ID {profile_id}')

            # Interpolate data to make depth intervals uniform
            interp_funcs = {col: interp1d(group[depth_col], group[col], kind='linear', bounds_error=False, fill_value='extrapolate') for col in data_cols}
            new_depths = np.arange(group[depth_col].min(), group[depth_col].max() + common_interval, common_interval)
            interpolated_values = {col: interp_funcs[col](new_depths) for col in data_cols}
            interpolated_df = pd.DataFrame(interpolated_values, index=new_depths)
            interpolated_df[profile_id_col] = profile_id  # Add the profile ID

            # Copy other non-interpolated columns
            for col in df.columns.difference([profile_id_col, depth_col] + list(data_cols)):
                interpolated_df[col] = group.iloc[0][col]

            # Reset index and rename it to depth_col
            interpolated_df.reset_index(inplace=True)
            interpolated_df.rename(columns={'index': depth_col}, inplace=True)
            all_profiles_df = pd.concat([all_profiles_df, interpolated_df])

    # Reset index for the concatenated DataFrame
    all_profiles_df.reset_index(drop=True, inplace=True)

    return all_profiles_df, uniform_intervals


# Merging profile layers for forward and inverse models
def merge_layers(df, new_interfaces, dataset,
                 depth_col='z', x_col='easting', y_col='northing', id_col='ID'):
    """
    Merges profile layers in a DataFrame based on specified depth interfaces 
    and calculates mean values for a given dataset column. 
    Use-case: generate appropriate prior models for inversion based on soil 
                information or modelling data.

    Parameters:
    df (pd.DataFrame): The original DataFrame containing soil profile data.
    new_interfaces (list): List of new depth interfaces for merging layers.
    dataset (str): The name of the column in df for which the mean should be calculated (e.g., conductivity).
    depth_col (str, optional): The name of the column in df representing depth. Defaults to 'z'.
    x_col (str, optional): The name of the column in df representing easting coordinates. Defaults to 'easting'.
    y_col (str, optional): The name of the column in df representing northing coordinates. Defaults to 'northing'.
    id_col (str, optional): The name of the column in df representing profile IDs. Defaults to 'ID'.

    Returns:
    pd.DataFrame: A new DataFrame with merged layers and mean values for each depth range.
    """
        
    # Function to process each group
    def process_group(group_df):
        final_depth = new_interfaces[-1] + new_interfaces[0]
        new_depths = new_interfaces + [final_depth]

        group_df[depth_col] = abs(group_df['z'])

        merged_data = {
            depth_col: new_depths,
            x_col: [],
            y_col: [],
            dataset: [],
            id_col: []
        }

        start_depth = 0
        for end_depth in new_depths:
            layers_in_range = group_df[(group_df['z'] > start_depth) & (group_df['z'] <= end_depth)]
            
            mean_EC = layers_in_range[dataset].mean()

            x_value = layers_in_range[x_col].iloc[0] if not layers_in_range.empty else np.nan
            y_value = layers_in_range[y_col].iloc[0] if not layers_in_range.empty else np.nan
            pos_value = layers_in_range[id_col].iloc[0] if not layers_in_range.empty else np.nan

            merged_data[x_col].append(x_value)
            merged_data[y_col].append(y_value)
            merged_data[dataset].append(mean_EC)
            merged_data[id_col].append(pos_value)

            start_depth = end_depth

        # Create DataFrame and sort it by 'z' in descending order of absolute values
        merged_group_df = pd.DataFrame(merged_data)
        merged_group_df = merged_group_df.sort_values(by='z', key=lambda x: abs(x), ascending=False)
        return merged_group_df

    # Group by 'ID' and apply the processing function
    grouped = df.groupby(id_col)
    merged_groups = [process_group(group) for _, group in grouped]

    # Concatenate all processed groups
    merged_df = pd.concat(merged_groups, ignore_index=True)
    merged_df[depth_col] = -merged_df[depth_col]
    return merged_df