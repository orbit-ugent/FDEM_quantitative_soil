#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Initialize
==========
Initialize frequency domain electromagnetic (FDEM) Instrument, CoilConfiguration, Model and Survey class.

:AUTHOR: Daan Hanssens
:ORGANIZATION: Ghent University
:CONTACT: daan.hanssens@ugent.be

:REQUIRES: numpy
"""

# Import
import numpy as np

# Import package
from FDEM import Properties


class Instrument: 
    """object class for FDEM instruments defining instrument type (brand, coil
    configurations (geometry+spacing), operating frequencies) and field setup 
    (elevation above surface, instrument orientation (HCP/VCP))
    """
    def __init__(self, instrument_code, instrument_height, instrument_orientation='HCP'):
        """
        Frequency domain electromagnetic (FDEM) instrument class.

        Parameters
        ----------
        instrument_code : {'Dualem-21HS', 'Dualem-21S', 'Dualem-421S', 'Dualem-642S'}
            Set instrument code as for corresponding DUALEM sensor.

        instrument_height : float
            Height (expressed in meters) of horizontal instrument above ground.

        instrument_orientation : {'HCP', 'VCP'}, optional
            Set instrument orientation as: 'HCP' for HCP-PRP orientations;
                                           'VCP' for VCP-NULL orientations.
        """

        self.instrument_orientation = instrument_orientation
        self.height = instrument_height
        self.set_instrument_code(instrument_code)
        self.set_instrument_orientation(instrument_orientation)

    def set_instrument_height(self, instrument_height):
        """
        Set height (expressed in meters) of horizontal instrument above ground.
        """
        
        self.height = instrument_height
        self._update_instrument_cc_details()

    def set_instrument_code(self, instrument_code):
        """
        Set specific instrument variables based on associated instrument code.


        Instrument details are specified here. Only a minor amount of instruments were added.
        """

        # Update class
        self._clear_instrument_cc_details()
        self.code = instrument_code

        # DUALEM-21HS
        if instrument_code == 'Dualem-21HS':
            self.frequencies = np.ones(12) * 9000.  # Hz
            self.spacings = np.array([.5, .6, 1., 1.1, 2., 2.1,
                                      .5, .6, 1., 1.1, 2., 2.1])
            self.cc_names = np.array(['HCPHQP', 'PRPHQP', 'HCP1QP', 'PRP1QP', 'HCP2QP', 'PRP2QP',
                                      'HCPHIP', 'PRPHIP', 'HCP1IP', 'PRP1IP', 'HCP2IP', 'PRP2IP'],
                                     dtype=str)
            self.noise = np.array([30, 30, 50, 50, 100, 100,
                                   30, 30, 50, 50, 100, 100])
            self.niter = 12

        # DUALEM-21S
        elif instrument_code == 'Dualem-21S':
            self.frequencies = np.ones(8) * 9000.  # Hz
            self.spacings = np.array([1., 1.1, 2., 2.1,
                                      1., 1.1, 2., 2.1])
            self.cc_names = np.array(['HCP1QP', 'PRP1QP', 'HCP2QP', 'PRP2QP',
                                      'HCP1IP', 'PRP1IP', 'HCP2IP', 'PRP2IP'],
                                     dtype=str)
            self.noise = np.array([50, 50, 100, 100,
                                   50, 50, 100, 100])
            self.niter = 8

        # DUALEM-421S
        elif instrument_code == 'Dualem-421S':
            self.frequencies = np.ones(12) * 9000.  # Hz
            self.spacings = np.array([1., 1.1, 2., 2.1, 4., 4.1,
                                      1., 1.1, 2., 2.1, 4., 4.1])
            self.cc_names = np.array(['HCP1QP', 'PRP1QP', 'HCP2QP', 'PRP2QP', 'HCP4QP', 'PRP4QP',
                                      'HCP1IP', 'PRP1IP', 'HCP2IP', 'PRP2IP', 'HCP4IP', 'PRP4IP'],
                                     dtype=str)
            self.noise = np.array([50, 50, 100, 100, 200, 200,
                                   50, 50, 100, 100, 200, 200])
            self.niter = 12

        # DUALEM-642S
        elif instrument_code == 'Dualem-642S':
            self.frequencies = np.ones(12) * 9000.  # Hz
            self.spacings = np.array([2., 2.1, 4., 4.1, 6., 6.1,
                                      2., 2.1, 4., 4.1, 6., 6.1])
            self.cc_names = np.array(['HCP2QP', 'PRP2QP', 'HCP4QP', 'PRP4QP', 'HCP6QP', 'PRP6QP',
                                      'HCP2IP', 'PRP2IP', 'HCP4IP', 'PRP4IP', 'HCP6IP', 'PRP6IP'],
                                     dtype=str)
            self.noise = np.array([100, 100, 200, 200, 400, 400,
                                   100, 100, 200, 200, 400, 400])
            self.niter = 12
        
        # DUALEM-1S
        elif instrument_code == 'Dualem-1S':
            self.frequencies = np.ones(12) * 9000.  # Hz
            self.spacings = np.array([1., 1.1,
                                      1., 1.1])
            self.cc_names = np.array(['HCP1QP', 'PRP1QP',
                                      'HCP1IP', 'PRP1IP'],
                                     dtype=str)
            self.noise = np.array([100, 100,
                                   100, 100])
            self.niter = 12

        else:
            raise ValueError("Instrument code '{}' is not defined.".format(instrument_code))

        # Set additional class variables
        self.moments = np.ones(len(self.frequencies))
        self.angular_frequencies = np.pi * 2 * self.frequencies
        self.set_instrument_orientation(self.instrument_orientation)
        self._update_instrument_cc_details()

    def set_instrument_orientation(self, instrument_orientation):
        """
        Set specific instrument variables based on instrument orientation and changes in orientation.
        """

        # Get coil configuration names and delete previous
        self._clear_instrument_cc_details()
        alt_cc_names = self.cc_names

        # Check new orientation
        if instrument_orientation == 'HCP':

            # Change configuration names
            alt_cc_names = np.core.defchararray.replace(alt_cc_names, 'VCP', 'HCP')
            alt_cc_names = np.core.defchararray.replace(alt_cc_names, 'NULL', 'PRP')

            # Change coil orientations
            coil_orientations = ['ZZ', 'ZX', 'ZZ', 'ZX', 'ZZ', 'ZX', 'ZZ', 'ZX', 'ZZ', 'ZX', 'ZZ', 'ZX']

        elif instrument_orientation == 'VCP':

            # Change configuration names
            alt_cc_names = np.core.defchararray.replace(alt_cc_names, 'HCP', 'VCP')
            alt_cc_names = np.core.defchararray.replace(alt_cc_names, 'PRP', 'NULL')

            # Change coil orientations
            coil_orientations = ['YY', 'YX', 'YY', 'YX', 'YY', 'YX', 'YY', 'YX', 'YY', 'YX', 'YY', 'YX']

        else:

            raise ValueError('No valid instrument orientation category.')

        # Update class
        self.cc_names = alt_cc_names
        self.instrument_orientation = instrument_orientation
        self.orientations = np.array(coil_orientations, dtype=str)
        self._update_instrument_cc_details()

    def _clear_instrument_cc_details(self):
        """
        Clear specific coil configuration class within instrument class. This 
        internal function assures that when a wrong instrument code has been set 
        this is corrected (clear, update and set)
        """
        if hasattr(self, 'cc_names'):
            for cc_name in self.cc_names:
                if hasattr(self, cc_name):
                    delattr(self, cc_name)

    def _update_instrument_cc_details(self):
        """
        Update specific coil configuration class within instrument class, based
        on the instrument orientation.
        """
        for ii, cc_name in enumerate(self.cc_names):
            setattr(self, cc_name, CoilConfiguration(self.height,
                                                     self.frequencies[ii],
                                                     self.spacings[ii],
                                                     self.orientations[ii],
                                                     self.moments[ii],
                                                     self.noise[ii])
                    )


class CoilConfiguration:
    def __init__(self, height, frequency, spacing, orientation, moment, noise):
        """
        Frequency domain electromagnetic (FDEM) coil class.

        Parameters
        ----------
        height : float
            Instrument height (m).

        frequency: float
            Instrument operating frequency (Hz).

        spacing: float
            Coil spacing (m).

        orientation : str
            Coil orientations represented in a Cartesian (X, Y, Z) coordinate system.

        moment : float
            Magnetic moment of transmitter coil.

        noise : float
            Noise level of coil configuration (ppm).
        """

        self.height = height
        self.frequency = frequency
        self.spacing = spacing
        self.orientation = orientation
        self.noise = noise
        self.moment = moment
        self.angular_frequency = np.pi * 2 * frequency
        self.xyz = np.array([spacing, 0, -height])


class Survey:
    def __init__(self, dataframe, instrument, qp_input_units='mS', ip_input_units='ppt'):
        """
        Frequency domain electromagnetic (FDEM) survey class.

        Parameters
        ----------
        dataframe : pd.DataFrame
            data as DataFrame type with following starting header ['x', 'y', 'z', 't',],
            where x, y and z are the respective coordinates and t is the (relative or absolute) timestamp.

        instrument : Instrument class
            An instrument class.

        qp_input_units : {'mS', 'S', 'ppt', 'ppm'}, optional
            Input units of QP response as: 'mS' in mS/m in McNeill ECa;
                                           'S' in S/m in McNeill ECa;
                                           'ppt' in ppt;
                                           'ppm' in ppm.

        ip_input_units = {'ppt', 'ppm'}, optional
            Input units of IP response as: 'ppt' in ppt;
                                           'ppm' in ppm.
        """

        self.instrument = instrument
        self.set_dataframe(dataframe)
        self.qp_units = qp_input_units
        self.ip_units = ip_input_units

    def set_dataframe(self, dataframe):
        """
        Set dataframe of survey class and check header conditions.
        """

        start_columns_list = ['x', 'y', 'z', 't']
        start_columns_array = np.array(start_columns_list)

        if (dataframe.columns.values[:4] == start_columns_array).all().any():
            self.dataframe = dataframe
            self.dataframe.columns = np.append(start_columns_array, self.instrument.cc_names)
            self.gps_dataframe = dataframe[start_columns_list]
            for col in self.dataframe.columns:
                setattr(self, col, dataframe[col].values)
        else:
            raise ValueError("DataFrame.columns does not start with ['x', 'y', 'z', 't',].")

    def set_dataframe_to_qp(self, qp_units='ppm'):
        """
        Calculate QP response from McNeill's LIN ECa in DataFrame.

        Parameters
        ----------
        qp_units = {'ppm', 'ppt'}, optional
            Units of calculated QP response as: 'ppt' in ppt;
                                                'ppm' in ppm.
        """

        # Check
        if (qp_units != 'ppt') and (qp_units != 'ppm'):
            raise ValueError("QP units are set to {} and should be in 'ppm' or 'ppt'.".format(qp_units))

        if (self.qp_units == 'mS') or (self.qp_units == 'S'):
            for cc_name in self.instrument.cc_names:
                if 'QP' in cc_name:
                    cc = getattr(self.instrument, cc_name)
                    self.dataframe.loc[:, cc_name] = Properties.mcneill_to_qp(cc, self.dataframe[cc_name].values,
                                                                              eca_input_units=self.qp_units,
                                                                              qp_output_units=qp_units
                                                                              )
        elif self.qp_units != qp_units:
            for cc_name in self.instrument.cc_names:
                if 'QP' in cc_name:
                    if qp_units == 'ppm':
                        self.dataframe.loc[:, cc_name] *= 1e3
                    elif qp_units == 'ppt':
                        self.dataframe.loc[:, cc_name] /= 1e3
        else:
            print("The data is already in QP ({}).".format(self.qp_units))
        self.qp_units = qp_units
        self.set_dataframe(self.dataframe)

    def set_dataframe_to_reca(self, eca_units='mS'):
        """
        Calculate rECa from QP response in DataFrame.

        Parameters
        ----------
        eca_units : {'mS', 'S'}, optional
            Units of calculated rECa as: 'mS' in mS/m;
                                         'S' in S/m.
        """

        # Check
        if (eca_units != 'mS') and (eca_units != 'S'):
            raise ValueError("ECa units are set to {} and should be in 'mS' or 'S'.".format(eca_units))

        # Check QP data
        if self.qp_units != 'ppm':
            self.set_dataframe_to_qp('ppm')

        # Set to rECa
        for cc_name in self.instrument.cc_names:
            if 'QP' in cc_name:
                cc = getattr(self.instrument, cc_name)
                self.dataframe.loc[:, cc_name], _ = Properties.reca(cc, self.dataframe[cc_name].values,
                                                                    ip=self.dataframe[cc_name].values,
                                                                    precision=.001,
                                                                    noise=0,
                                                                    reference_eca=.005,
                                                                    original_msa=0,
                                                                    alternative_msa=0,
                                                                    maximum_eca=4
                                                                    )

        # Check output ECa
        self.qp_units = 'S'
        if eca_units == 'mS':
            self.set_dataframe_to_lin_eca(eca_units=eca_units)
        self.set_dataframe(self.dataframe)

    def set_dataframe_to_lin_eca(self, eca_units='mS'):
        """
        Calculate McNeill's LIN ECa from QP response in DataFrame.

        Parameters
        ----------
        eca_units : {'mS', 'S'}, optional
            Units of calculated McNeill's LIN ECa as: 'mS' in mS/m;
                                                      'S' in S/m.
        """

        # Check
        if (eca_units != 'mS') and (eca_units != 'S'):
            raise ValueError("ECa units are set to {} and should be in 'mS' or 'S'.".format(eca_units))

        if (self.qp_units == 'ppt') or (self.qp_units == 'ppm'):
            for cc_name in self.instrument.cc_names:
                if 'QP' in cc_name:
                    cc = getattr(self.instrument, cc_name)
                    self.dataframe.loc[:, cc_name] = Properties.qp_to_mcneill(cc, self.dataframe[cc_name].values,
                                                                              qp_input_units=self.qp_units,
                                                                              eca_output_units=eca_units
                                                                              )
        elif self.qp_units != eca_units:
            for cc_name in self.instrument.cc_names:
                if 'QP' in cc_name:
                    if eca_units == 'mS':
                        self.dataframe.loc[:, cc_name] *= 1e3
                    elif eca_units == 'S':
                        self.dataframe.loc[:, cc_name] /= 1e3
        else:
            print("The data is already in ECa ({}/m).".format(self.qp_units))
        self.qp_units = eca_units
        self.set_dataframe(self.dataframe)

    def set_dataframe_to_ip(self, ip_units='ppm'):
        """
        Calculate IP response IP data in DataFrame.

        Parameters
        ----------
        ip_units = {'ppm', 'ppt'}, optional
            Units of calculated IP response as: 'ppt' in ppt;
                                                'ppm' in ppm.
        """

        # Check
        if (ip_units != 'ppm') and (ip_units != 'ppt'):
            raise ValueError("IP units are set to {} and should be in 'ppm' or 'ppt'.".format(ip_units))

        if self.ip_units != ip_units:
            for cc_name in self.instrument.cc_names:
                if 'IP' in cc_name:
                    if ip_units == 'ppm':
                        self.dataframe.loc[:, cc_name] *= 1e3
                    elif ip_units == 'ppt':
                        self.dataframe.loc[:, cc_name] /= 1e3
        else:
            print("The data is already in IP ({}).".format(self.ip_units))
        self.ip_units = ip_units
        self.set_dataframe(self.dataframe)


class Model:
    def __init__(self, layer_thickness, magnetic_susceptibility, electrical_conductivity, dielectric_permittivity):
        """
        Model characteristics (1D) class.

        Parameters
        ----------
        layer_thickness : np.array
            Layer(s) thickness (m).

        magnetic_susceptibility : np.array
            Magnetic susceptibility of layer(s) (-).

        electrical_conductivity : np.array
            Electrical conductivity of layer(s) (F/m).

        dielectric_permittivity : np.array
            Dielectric permittivity of layer(s) (S/m).
        """

        self.thick = np.array(layer_thickness)  # Layer(s) thickness (m)
        self.sus = np.array(magnetic_susceptibility)  # Susceptibility of layer(s) (-)
        self.perm = np.array(dielectric_permittivity)  # Permittivity of layer(s) (F/m)
        self.con = np.array(electrical_conductivity)  # Conductivity of layer(s) (S/m)

    @property
    def depth(self):
        """
        Calculate depth (m) of model.
        """

        return np.cumsum(self.thick)

    @property
    def nr_layers(self):
        """
        Calculate number of layers.
        """

        return self.depth.size

class Common:
    def __init__(self):
        """
        Create a Common object.

        Parameters
        ----------
        filename : filename of stored Common (.csv).
        """

