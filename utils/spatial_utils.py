#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Packages ------------------------
import warnings
import os

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
    dataframe (in the case below the calibration sample dataset 'd_cal').
    '''

    # transform dataframes to numpy arrays and perform knn search
    data_in = np.array(list(zip(df_in['x'].values,df_in['y'].values)) )
    data_query = np.array(list(zip(df_query['x'].values,df_query['y'].values)) )
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