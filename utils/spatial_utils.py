#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Packages ------------------------
import warnings
import os
from io import StringIO
import subprocess
import tempfile
import shutil

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree

# ---------------------------------

def determine_epsg_from_utm_zone(zone_number, hemisphere='N'):
    """
    Determine the EPSG code for a given UTM zone and hemisphere.

    Parameters:
    - zone_number (int): The UTM zone number (1 to 60).
    - hemisphere (str): 'N' for Northern Hemisphere or 'S' for Southern Hemisphere.

    Returns:
    - str: The EPSG code corresponding to the UTM zone and hemisphere.

    Raises:
    - ValueError: If the zone_number is not within the valid range or hemisphere is not recognized.

    Example:
    >>> determine_epsg_from_utm_zone(31, 'N')
    'EPSG:32631'
    >>> determine_epsg_from_utm_zone(33, 'S')
    'EPSG:32733'
    """
    if not (1 <= zone_number <= 60):
        raise ValueError("UTM zone number must be between 1 and 60.")

    if hemisphere.upper() == 'N':
        return f'EPSG:326{zone_number:02d}'
    elif hemisphere.upper() == 'S':
        return f'EPSG:327{zone_number:02d}'
    else:
        raise ValueError("Hemisphere must be either 'N' (northern) or 'S' (southern).")
    

def split_utm_zone(utm_zone_str):
    """
    Splits a UTM zone string into the zone number and hemisphere.

    Parameters:
    - utm_zone_str (str): A UTM zone string (e.g., '31N').

    Returns:
    - tuple: A tuple containing the zone number (int) and the hemisphere (str).

    Raises:
    - ValueError: If the input does not match the UTM zone string pattern.

    Example:
    >>> split_utm_zone('31N')
    (31, 'N')
    """
    if not (utm_zone_str[:-1].isdigit() and utm_zone_str[-1] in ['N', 'S']):
        raise ValueError(f"Invalid UTM zone string format: {utm_zone_str}")

    zone_number = int(utm_zone_str[:-1])
    hemisphere = utm_zone_str[-1].upper()
    return zone_number, hemisphere

def determine_epsg_from_utm_zone(zone_number, hemisphere='N'):
    """
    Determine the EPSG code for a given UTM zone and hemisphere.

    Parameters:
    - zone_number (int): The UTM zone number (1 to 60).
    - hemisphere (str): 'N' for Northern Hemisphere or 'S' for Southern Hemisphere.

    Returns:
    - str: The EPSG code corresponding to the UTM zone and hemisphere.

    Raises:
    - ValueError: If the zone_number is not within the valid range or hemisphere is not recognized.

    Example:
    >>> determine_epsg_from_utm_zone(31, 'N')
    'EPSG:32631'
    >>> determine_epsg_from_utm_zone(33, 'S')
    'EPSG:32733'
    """
    type(zone_number)
    type(hemisphere)
    if not (1 <= zone_number <= 60):
        raise ValueError("UTM zone number must be between 1 and 60.")

    if hemisphere.upper() == 'N':
        return f'EPSG:326{zone_number:02d}'
    elif hemisphere.upper() == 'S':
        return f'EPSG:327{zone_number:02d}'
    else:
        raise ValueError("Hemisphere must be either 'N' (northern) or 'S' (southern).")

def get_utm_epsg_from_coords(x, y):
    """
    Determine the UTM EPSG code for given x and y coordinates.

    This function calculates the UTM zone from the given X (longitude)
    coordinate and assesses whether the location is in the northern or
    southern hemisphere using the Y (latitude) coordinate. It then
    constructs the appropriate EPSG code based on the UTM zone and
    hemisphere.

    Parameters:
    - x (float): The X coordinate (easting) in UTM meters.
    - y (float): The Y coordinate (northing) in UTM meters. Positive values
      indicate the northern hemisphere, and negative values indicate the
      southern hemisphere.

    Returns:
    - str: The EPSG code string corresponding to the calculated UTM zone
      and hemisphere, e.g., 'EPSG:32633' for UTM zone 33N.

    Note:
    - This function assumes the input coordinates are in meters and based
      on the WGS 84 datum.
    - The X coordinate must be within the range of -180 to +180 degrees
      longitude to correctly calculate the UTM zone.

    Example:
    >>> get_utm_epsg_from_coords(500000, 0)
    'EPSG:32631'
    """
    if not (-180 <= x <= 180):
        raise ValueError(f"The X coordinate {x} is out of bounds. Must be between -180 and 180.")

    zone_number = int((x + 180) // 6) + 1

    # Check if the zone number is within the valid UTM zone range
    if not (1 <= zone_number <= 60):
        warnings.warn(
            f"The computed UTM zone number {zone_number} is out of the valid range (1-60). "
            "This may indicate that the input coordinates are not in UTM meters.",
            UserWarning
        )

    # Northern hemisphere has positive Y coordinate (EPSG 32600 + zone number)
    # Southern hemisphere has negative Y coordinate (EPSG 32700 + zone number)
    if y >= 0:
        return f'EPSG:326{zone_number:02d}'
    else:
        return f'EPSG:327{zone_number:02d}'

def utm_to_epsg(df, UTM_zone, target_epsg='EPSG:31370', export=False, csv_file=None):
    """
    The function `utm_to_epsg` transforms UTM coordinates in a pandas DataFrame 
    to a specified coordinate system using geopandas, and optionally exports the 
    transformed data to a CSV file. It takes the following parameters:
    - df: A pandas DataFrame with columns 'x' and 'y' representing UTM coordinates.
    - UTM_zone: A string with the UTM zone structured as 'numberHemisphere'(e.g.,'31N')
    - target_epsg: A string representing the target EPSG code for transformation. 
        Default is 'EPSG:31370'.
    - export: A boolean indicating whether to export the transformed DataFrame 
        to a CSV file. Default is False.
    - csv_file: A string representing the path to the CSV file to be saved 
        (required if export is True).

    It returns a pandas DataFrame with transformed coordinates.

    Example:
    df_transformed = utm_to_epsg(input_df, target_epsg='EPSG:31370', export=True, csv_file='output.csv')
    """
    # Check if the 'x' and 'y' columns exist
    if 'x' not in df.columns or 'y' not in df.columns:
        raise ValueError("The DataFrame must contain 'x' and 'y' columns.")
    
    # TODO: add functionality for lat-lon UTM coordinates
    # # Automatically determine the UTM zone and EPSG code from the first row coordinates
    # first_x, first_y = df['x'].iloc[0], df['y'].iloc[0]  # Take the first x and y coordinates
    # original_epsg = get_utm_epsg_from_coords(first_x, first_y)
    
    original_epsg = determine_epsg_from_utm_zone(split_utm_zone(UTM_zone)[0],
                                                 split_utm_zone(UTM_zone)[1])
    
    # Convert the DataFrame to a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df['x'], df['y'])],
        crs=original_epsg
    )
    print(target_epsg)
    # Transform the coordinates
    gdf = gdf.to_crs(target_epsg)
    
    # Convert back to a regular DataFrame with just the target coordinates and other columns
    df_transformed = pd.DataFrame({
        'x': gdf.geometry.x,
        'y': gdf.geometry.y,
    }).join(df.drop(columns=['x', 'y']))
    
    # If export is True and a csv_file name is provided, export the transformed DataFrame to a CSV file
    if export and csv_file:
        # Create a new filename with the target EPSG code
        base, extension = os.path.splitext(csv_file)
        new_filename = f"{base}_EPSG{target_epsg.split(':')[1]}{extension}"
        
        # Export to CSV
        df_transformed.to_csv(new_filename, index=False)
        print(f"File saved as {new_filename}")
    elif export and not csv_file:
        raise ValueError("csv_file parameter must be provided if export is True.")
        
    return df_transformed

def get_coincident(df_in, df_query):
    '''
    Scipy method - knn search
    -------------------------
    Perform a knn-search on two dataframes to identify points that are closest
    between columns of an input dataframe (in the case below 'df'), and a query
    dataframe.
    '''

    # transform dataframes to numpy arrays and perform knn search
    data_in = np.array(list(zip(df_in['x'].values,df_in['y'].values)) )
    data_query = np.array(list(zip(df_query['x'].values,df_query['y'].values)) )
    #print('data_query', data_query)
    btree = cKDTree(data_in)
    dist, idx = btree.query(data_query, k=1) # k = number of neighbors; 
                                            # idx = index of the neighbors
                                            # dist = distance between the neighbors       

    # Concatenate the survey data values at the location closest to the 
    # queried dataset based on the indices 'idx' (df.iloc[idx]), and the queried 
    # data themselves.
    df_coincident = pd.concat(
        [
        df_in.iloc[idx],   # part of original dataframe that is closest to queried data                    
        df_query.set_index(df_in.iloc[idx].index) # reset index of the queried dataframe
        .rename(columns = {'x':'x_clhs','y':'y_clhs'}) # and rename the x-y columns
        ],
        axis=1 #concatenate along the correct axis (i.e., along the columns)
        )
    return df_coincident


def nn_interpolate(df_x, df_y, df_z, 
                   nx, ny, 
                   tempdir=True, 
                   tempdir_name=None,
                   df_out = False
                   ):
    '''
    NN Interpolation with nnbathy
    requires nnbathy.exe

    ref: https://github.com/sakov/nn-c
    Usage: nnbathy -i <XYZ file>
                -o <XY file> | -n <nx>x<ny> [-c|-s] [-z <zoom>]
                [-x <xmin xmax>] [-xmin <xmin>] [-xmax <xmax>]
                [-y <ymin ymax>] [-ymin <ymin>] [-ymax <ymax>]
                [-v|-T <vertex id>|-V]
                [-D [<nx>x<ny>]]
                [-L <dist>]
                [-N]
                [-P alg={l|nn|ns}]
                [-W <min weight>]
                [-% [npoints]]
    Options:
    -c               -- scale internally so that the enclosing minimal ellipse
                        turns into a circle (this produces results invariant to
                        affine transformations)
    -i <XYZ file>    -- three-column file with points to interpolate from
                        (use "-i stdin" or "-i -" for standard input)
    -n <nx>x<ny>     -- generate <nx>x<ny> output rectangular grid
    -o <XY file>     -- two-column file with points to interpolate in
                        (use "-o stdin" or "-o -" for standard input)
    -s               -- scale internally so that Xmax - Xmin = Ymax - Ymin
    -x <xmin> <xmax> -- set Xmin and Xmax for the output grid
    -xmin <xmin>     -- set Xmin for the output grid
    -xmax <xmax>     -- set Xmin for the output grid
    -y <ymin> <ymax> -- set Ymin and Ymax for the output grid
    -ymin <ymin>     -- set Ymin for the output grid
    -ymax <ymax>     -- set Ymin for the output grid
    -v               -- verbose / version
    -z <zoom>        -- zoom in (if <zoom> < 1) or out (<zoom> > 1) (activated
                        only when used in conjunction with -n)
    -D [<nx>x<ny>]   -- thin input data by averaging X, Y and 
    Z values within
                        every cell of the rectangular <nx>x<ny> grid (size
                        optional with -n)
    -L <dist>        -- thin input data by averaging X, Y and Z values within
                        clusters of consequitive input points such that the
                        sum of distances between points within each cluster
                        does not exceed the specified maximum value
    -N               -- do not interpolate, only pre-process
    -P alg=<l|nn|ns> -- use the following algorithm:
                            l -- linear interpolation
                            nn -- Sibson interpolation (default)
                            ns -- Non-Sibsonian interpolation
    -T <vertex id>   -- verbose; in weights output print weights associated
                        with this vertex only
    -V               -- very verbose / version
    -W <min weight>  -- restricts extrapolation by assigning minimal allowed
                        weight for a vertex (normally "-1" or so; lower
                        values correspond to lower reliability; "0" means
                        no extrapolation)
    -% [npoints]     -- print percent of the work done to standard error;
                        npoints -- total number of points to be done (optional
                        with -n)
    Description:
    `nnbathy' interpolates scalar 2D data in specified points using Natural
    Neighbours interpolation. The interpolated values are written to standard
    output.
    '''
    # ------------------------------------------------------------------------ #

    '''
    df_x: pandas DataFrame (column) with x coordinates (longitude, meters) 
    df_y: pandas DataFrame (column) with y coordinates (latitude, meters) 
    df_z: pandas DataFrame (column) with values to be interpolated 
    use_custom_tempdir: Boolean to decide whether to use a custom temporary directory
    custom_tempdir_name: Optional custom name for the temporary directory
    '''
    xyzfile = pd.concat([df_x, df_y, df_z], axis=1)
    
    if tempdir:
        temp_dir = tempdir_name or "temp_nn"
        os.makedirs(temp_dir, exist_ok=True)
        input_temp_file_path = os.path.join(temp_dir, "input_temp.txt")
        xyzfile.to_csv(input_temp_file_path, header=None, sep=' ', index=False)
    else:
        input_temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=True)
        #output_temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
        input_temp_file_path = input_temp_file.name
        xyzfile.to_csv(input_temp_file_path, header=None, sep=' ', index=False)
        input_temp_file.flush()
    
    output_temp_file_path = "output_temp.txt"

    grid_size = f"{ny}x{nx}" #latitude x longitude as string

    # Execute nnbathy with input as temporary file
    result = subprocess.Popen(['nnbathy.exe',
                               '-i', input_temp_file_path,
                               '-n', grid_size,
                               '-o', output_temp_file_path,
                               '-s'],
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()
    s = str(stdout, 'utf-8')
    data = StringIO(s)

    if result.returncode != 0:
        print(f"Error in nnbathy: {stderr}")
        return None

    # Read the output data
    interpolated_data = pd.read_csv(data, delimiter=' ', header=None)
    interpolated_data = interpolated_data.rename(columns={0: 'lat', 1: 'lon', 2: 'val'})

    #interpolated_data_sorted = interpolated_data.sort_values(by=['lon', 'lat'])
    grid = interpolated_data['val'].values.reshape(nx, ny)
    
    # Clean up temporary files and directories
    if tempdir:
        shutil.rmtree(temp_dir)
    else:
        input_temp_file.close()
        #output_temp_file.close()

    if df_out:
        return grid, interpolated_data
    else: 
        return grid
    

def get_stats_within_radius(df_input, df_query, radius, stat='mean'):
    """
    Function to find points within a specified radius from query points and calculate statistics on them.

    Parameters
    ----------
    df_input : pd.DataFrame
        DataFrame with the input data containing the 'x' and 'y' coordinates and other data columns.
    
    df_query : pd.DataFrame
        DataFrame with the query points containing the 'x' and 'y' coordinates.

    radius : float
        Radius within which to search for points from df_input around each point in df_query.

    stat : str
        The statistic to compute for the points within the radius; valid options are 'median' or 'mean'.

    Returns
    -------
    df_stats : pd.DataFrame
        DataFrame with the statistics of the data columns (excluding 'x' and 'y') for points in df_input
        within the radius of each point in df_query.
    """
    # Convert DataFrame columns to numpy arrays for spatial search
    data_in = np.array(list(zip(df_input['x'], df_input['y'])))
    data_query = np.array(list(zip(df_query['x'], df_query['y'])))
    #print('data_query', data_query)

    # Create a cKDTree object for efficient spatial search
    tree = cKDTree(data_in)

    # List to collect stats DataFrames
    stats_list = []

    # Loop over each query point to find input points within the radius
    for idx, point in enumerate(data_query):
        #print('idx, point', idx, point)
        indices = tree.query_ball_point(point, r=radius)
        #print('indices', indices)
        if indices:
            relevant_points = df_input.iloc[indices]
            if stat == 'median':
                stats = relevant_points.median().to_dict()
            elif stat == 'mean':
                stats = relevant_points.mean().to_dict()
            # Ensure the query point identifier is added to the stats
            stats.update({'query_index': idx})
            stats_list.append(pd.DataFrame(stats, index=[0]))

    # Concatenate all stats DataFrames
    if stats_list:
        df_stats = pd.concat(stats_list, ignore_index=True)
    else:
        df_stats = pd.DataFrame()
    
    # Add information from df_query to the closest points DataFrame
    df_query = df_query.reset_index(drop=True)
    df_query = df_query.rename(columns={'x': 'x_query', 'y': 'y_query'})
    df_radius = pd.concat([df_stats, df_query], axis=1)

    return df_radius