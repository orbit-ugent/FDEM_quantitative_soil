import pandas as pd
import numpy as np

def hampel_1d(data, window_size=7, number_of_std=3):
    """
    Perform a Hampel filter on the data.

    Parameters
    ----------
    data : np.array
        data to be filtered.

    window_size : int, optional
        Number of data points in window.

    number_of_std : int, optional
        Number of standard deviations to use.

    Returns
    -------
    outlier_boolean : bool
        Boolean of outliers.

    filtered_data : np.array
        Filtered data.

    Reference
    ---------
    Hampel F. R., 1974,
        The influence curve and its role in robust estimation,
        Journal of the American Statistical Association 69, 382–393.
    """

    # Make copy so original is not edited
    filtered_data = pd.Series(data.copy())

    # Hampel filter
    rolling_median = filtered_data.rolling(window_size).median()
    deviation = np.abs(filtered_data - rolling_median)
    median_abs_deviation = deviation.rolling(window_size).median()
    threshold = number_of_std * 1.4826 * median_abs_deviation
    outlier_boolean = deviation > threshold
    filtered_data[outlier_boolean] = np.nan

    return outlier_boolean, filtered_data.values