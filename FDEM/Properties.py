#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Properties
==========
Frequency domain electromagnetic (FDEM) (apparent) physical properties.


References
----------
McNeill, J. D., 1980.
    Electromagnetic terrain conductivity measurement at low induction numbers,
    Technical Note TN-6, Geonics Ltd, Missisauga, Ontario, Canada.

Hanssens, D., Delefortrie, S., Bobe, C., Hermans, T., De Smedt, P., 2019.
    Improving the reliability of soil EC-mapping: Robust apparent electrical conductivity (reca)
    estimation in ground-based frequency domain electromagnetics. Geoderma 337, 1155-1163.

Guillemoteau, J., Sailhac, P., Boulanger, C., and J. Trules, 2015.
    Inversion of ground constant offset loop-loop electromagnetic data for a
    large range of induction numbers. Geophysics, 80, no. 1, E11-E21.

Huang, sphere3d., Won, I.J., 2000.
    Conductivity and Susceptibility Mapping Using Broadband Electromagnetic Sensors.
    Journal of Environmental & Engineering Geophysics 5(4), 31-41.


:AUTHOR: Daan Hanssens
:ORGANIZATION: Ghent University
:CONTACT: daan.hanssens@ugent.be
:REQUIRES: numpy, scipy, shapely
"""

# Import
import numpy as np
import scipy
import shapely

# Import package
from FDEM import Initialize, Modeling


def qp_to_mcneill(coil_configuration, qp, qp_input_units='ppm', eca_output_units='mS'):
    """
    Calculate the QP response (ppm or ppt) for McNeill's LIN ECa (mS/m or S/m).

    Parameters
    ----------
    coil_configuration : CoilConfiguration class
        A CoilConfiguration class.

    qp : np.array
        QP response data (ppm or ppt).

    qp_input_units : str, optional
        Unit of the QP input, either 'ppm' for ppm;
                                     'ppt' for ppt.

    eca_output_units : str
        Unit of the ECa output, either: 'mS' for mS/m;
                                        'S' for S/m.

    Returns
    -------
    mcneill_eca: np.array
        McNeill's LIN ECa (mS/m or S/m).
    """

    if (qp_input_units == 'ppm' and eca_output_units == 'mS') or (qp_input_units == 'ppt' and eca_output_units == 'S'):
        mcneill_eca = qp / scipy.constants.mu_0 / coil_configuration.angular_frequency / coil_configuration.spacing ** 2 * 4 / 1e3
    elif qp_input_units == 'ppt' and eca_output_units == 'mS':
        mcneill_eca = qp / scipy.constants.mu_0 / coil_configuration.angular_frequency / coil_configuration.spacing ** 2 * 4
    elif qp_input_units == 'ppm' and eca_output_units == 'S':
        mcneill_eca = qp / scipy.constants.mu_0 / coil_configuration.angular_frequency / coil_configuration.spacing ** 2 * 4 / 1e6
    else:
        # Error message
        raise ValueError('Input/output units should be defined correctly.')
    return mcneill_eca


def mcneill_to_qp(coil_configuration, mcneill_eca, eca_input_units='mS', qp_output_units='ppm'):
    """
    Calculate the QP response (ppm or ppt) for McNeill's LIN ECa (mS/m or S/m).

    Parameters
    ----------
    coil_configuration : CoilConfiguration class
        A CoilConfiguration class.

    mcneill_eca : np.array
        McNeill's LIN ECa (in mS/m or S/m).

    eca_input_units : str, optional
        Unit of the ECa input, either: 'mS' for mS/m;
                                       'S' for S/m.

    qp_output_units : str, optional
        Unit of the QP output, either 'ppm' for ppm;
                                      'ppt' for ppt.

    Returns
    -------
    qp : np.array
        QP response data (ppm or ppt).
    """

    if (eca_input_units == 'mS' and qp_output_units == 'ppm') or (eca_input_units == 'S' and qp_output_units == 'ppt'):
        qp = mcneill_eca * scipy.constants.mu_0 * coil_configuration.angular_frequency * coil_configuration.spacing ** 2 / 4 * 1e3
    elif eca_input_units == 'mS' and qp_output_units == 'ppt':
        qp = mcneill_eca * scipy.constants.mu_0 * coil_configuration.angular_frequency * coil_configuration.spacing ** 2 / 4
    elif eca_input_units == 'S' and qp_output_units == 'ppm':
        qp = mcneill_eca * scipy.constants.mu_0 * coil_configuration.angular_frequency * coil_configuration.spacing ** 2 / 4 * 1e6
    else:
        raise ValueError('Input/output units should be defined correctly.')
    return qp


def reca(coil_configuration, qp, ip, precision=.001, noise=0, reference_eca=None,
         original_msa=0, alternative_msa=0, maximum_eca=4):
    """
    Calculates the rECa (S/m) of an FDEM (QP and IP) dataset (ppm).

    Parameters
    ----------
    coil_configuration : object
        CoilConfiguration object.

    qp : np.array
        QP (quadrature-phase or out-of-phase) data (ppm).

    ip : np.array
        IP (in-phase) data (ppm).

    precision : float, optional
        Approximated required ECa precision (S/m), .001 by default.

    noise : float, optional
        Instrument noise level (ppm), 0 by default.

    reference_eca : float, optional
        Additional reference ECa estimation (S/m), None by default such that EG2015 ECa (Appendix B)
        algorithm is used to estimate the additional ECa value.

    original_msa : float, optional
        Homogeneous half-space MS (-) estimation used to generate the original ECa-QP curve (-), 0 by default.

    alternative_msa : float, optional
        Altered homogeneous half-space MS estimation used to generate the alternative ECa-QP_alt curve, 0 by default.

    maximum_eca : float, optional
        Maximum sampled homogeneous half-space EC value (S/m).

    Returns
    -------
    reca : np.array
        Robust apparent electrical conductivity (reca) (S/m).

    is_reca_robust : np.array, boolean
        Assessment of the robustness of the reca values.

    Cite
    ----
    Hanssens, D., Delefortrie, S., Bobe, C., Hermans, T., De Smedt, P., 2019.
        Improving the reliability of soil EC-mapping: Robust apparent electrical conductivity (reca)
        estimation in ground-based frequency domain electromagnetics. Geoderma 337, 1155-1163.

    :AUTHOR: Daan Hanssens
    :CONTACT: daan.hanssens@ugent.be
    """

    # Get extra information about input structure
    shape = np.shape(qp)
    size = np.size(qp)

    # Flatten initial matrices/arrays
    reshaped_qp = np.asarray(qp).flatten()
    reshaped_ip = np.asarray(ip).flatten()

    # Generate EC-QP curve
    [ec_range, forward_qp, forward_ip, non_robust_ECa] = _eca_qp_curve(coil_configuration,
                                                                       precision=precision,
                                                                       noise=noise,
                                                                       maximum_eca=maximum_eca,
                                                                       original_msa=original_msa,
                                                                       alternative_msa=alternative_msa
                                                                       )

    # Check monotony
    if np.all(np.diff(forward_qp) > 0):

        # Interpolation (pchip)
        intermediate_eca = scipy.interpolate.pchip_interpolate(forward_qp, ec_range, reshaped_qp)

    elif np.all(np.diff(forward_qp) < 0):

        # Interpolation (pchip) of flipped data, pchip requires sorted data
        intermediate_eca = scipy.interpolate.pchip_interpolate(np.flip(forward_qp, axis=0), np.flip(ec_range, axis=0), reshaped_qp)

    else:

        # Case no initial ECa estimation is included
        if reference_eca is None:

            # Initialize np.array
            reference_eca = np.zeros(size)

            # Loop data TODO vectorize
            for ii in range(size):

                if np.isnan(reshaped_qp[ii]):

                    # Assign NaN value
                    reference_eca[ii] = np.NaN

                else:

                    # EG2015 ECa method (Extended Guillemoteau et al., 2015; Appendix B)
                    E = 1 / 2 * np.sqrt((forward_qp - reshaped_qp[ii]) ** 2 + (forward_ip - reshaped_ip[ii]) ** 2)
                    reference_eca[ii] = ec_range[E.argmin()]

        else:

            # Create init ECa vector
            reference_eca = np.full(size, reference_eca)

        # Initialize curve shape
        ec_qp_curve = shapely.geometry.LineString(list(zip(ec_range.tolist(), forward_qp.tolist())))
        intermediate_eca = np.ones(size) * np.NaN

        # Determine reca (S/m) TODO vectorize
        for ii in range(size):

            # Check if value exists
            if np.isnan(reshaped_qp[ii]):

                # Set to NaN
                intermediate_eca[ii] = np.NaN

            else:

                # Calculate intersection
                intersect_line = shapely.geometry.LineString([(ec_range[0], reshaped_qp[ii]), (ec_range[-1], reshaped_qp[ii])])
                intersect = np.asarray(ec_qp_curve.intersection(intersect_line))

                # Case no intersection
                if not intersect.any():

                    # Set to NaN
                    intersect = np.NaN

                else:

                    # Get relevant data
                    if intersect.ndim > 1:

                        # Grab data
                        intersect = intersect[:, 0]

                        # Calculate nearest
                        near = np.abs(intersect - reference_eca[ii])

                        # Get intermediate_eca (S/m), i.e. get nearest
                        intermediate_eca[ii] = intersect[near.argmin()]

                    else:

                        # Get intermediate_eca (S/m)
                        intermediate_eca[ii] = intersect[0]

                # Clear variable
                del intersect

    # Check robustness
    #is_reca_robust = np.logical_not(np.any(np.abs(np.tile(non_robust_ECa, (size, 1)) -
    #                                              np.tile(intermediate_eca.reshape((size, 1)),
    #                                                      (1, non_robust_ECa.size))) < precision, axis=1))
    is_reca_robust = True

    # Reshape to original dimensions
    reca = np.reshape(intermediate_eca, shape)

    # Return output
    return reca, is_reca_robust


def _eca_qp_curve(coil_configuration, precision=.001, noise=0, maximum_eca=4,
                  minimum_eca=0.0001, original_msa=.0, alternative_msa=0):
    """
    Calculates the ECa-QP and -IP curve.

    Parameters
    ----------
    coil_configuration : object
        CoilConfiguration object.

    precision : float, optional
        Approximated required ECa precision (S/m).

    noise : float, optional
        FDEM instrument noise (ppm).

    maximum_eca : float, optional
        Maximum sampled homogeneous half-space EC value (S/m).

    minimum_eca : float, optional
        Minimum sampled homogeneous half-space EC value (S/m).

    original_msa : float, optional
        Homogeneous half-space MS (-) value used to generate the original ECa-QP curve, 0 by default.

    alternative_msa : float, optional
        Altered homogeneous MS value used to generate the alternative ECa-QP_alt curve, 0 by default.

    Returns
    -------
    eca : np.array
        eca (Apparent electrical conductivity) (S/m).

    qp : np.array
        QP (quadrature-phase or out-of-phase) data (ppm).

    ip : np.array
        IP (in-phase) data (ppm).

    non_robust_eca : np.array
        Non-robust ECa values (S/m).

    Cite
    ----
    Hanssens, D., Delefortrie, S., Bobe, C., Hermans, T., De Smedt, P., 2019.
        Improving the reliability of soil EC-mapping: Robust apparent electrical conductivity (reca)
        estimation in ground-based frequency domain electromagnetics. Geoderma 337, 1155-1163.

    :AUTHOR: Daan Hanssens
    :CONTACT: daan.hanssens@ugent.be
    """

    # Initialize model characteristics
    sus = np.array([original_msa])
    perm = np.array([scipy.constants.epsilon_0])
    thick = np.array([0])

    # Initial ECa range (S/m)
    number_of_ec_samples = int(np.round((maximum_eca - minimum_eca) * 1000))
    if number_of_ec_samples > 20000: raise ValueError('max eca is too high, make sure it is in S/m')  # Prevent overflow
    eca = np.linspace(minimum_eca, maximum_eca, number_of_ec_samples)

    # Initialize QP and IP response arrays
    ip = np.zeros(number_of_ec_samples)
    qp = np.zeros(number_of_ec_samples)

    # Loop samples
    for ii in range(number_of_ec_samples):

        # Update homogeneous half-space EC (S/m)
        con = np.array([eca[ii]])
        model = Initialize.Model(thick, sus, con, perm)

        # Calculate forward response (ppm)
        [ip[ii], qp[ii]] = Modeling.Pair1D(coil_configuration, model).forward()

    # Assess robustness of curve (if noise is present)
    if noise != 0:

        # Transform precision to indices
        precision_ind = int(precision * 1000)

        # Check if magnetic effects should be accounted for
        if alternative_msa != 0:

            # Initialize response arrays
            ip_alt = np.zeros(number_of_ec_samples)
            qp_alt = np.zeros(number_of_ec_samples)

            # Alter MS (-), calculate magnetic effect
            sus = sus + alternative_msa

            # Loop samples
            for ii in range(number_of_ec_samples):

                # Update homogeneous half-space EC (S/m)
                con = np.array([eca[ii]])
                model = Initialize.Model(thick, sus, con, perm)

                # Calculate forward response (ppm)
                [ip_alt[ii], qp_alt[ii]] = Modeling.Pair1D(coil_configuration, model).forward()

            # Calculate slope for required precision and magnetic effects
            qp_diff = np.abs(np.diff(qp[::precision_ind])) - (np.abs(qp[::precision_ind] - qp_alt[::precision_ind]))[:-1]

        else:

            # Calculate slope for required precision
            qp_diff = np.abs(np.diff(qp[::precision_ind]))

        # Account for FDEM instrument noise
        mask = qp_diff < noise

        # Grab non-robust ECa and QP values
        non_robust_eca = eca[:-precision_ind:precision_ind][mask]

    else:

        # Create vector, all values are de facto robust because 0 ppm noise
        non_robust_eca = np.ones(number_of_ec_samples)

    # Return ECa, QP, IP, non_robust_ECa
    return eca, qp, ip, non_robust_eca
