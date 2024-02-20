#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modeling
========
Frequency domain electromagnetic modeling.


Cite
----
Hanssens, D., Delefortrie, S., De Pue, J., Van Meirvenne, M., and P. De Smedt, 2019.
    Frequency-Domain Electromagnetic Forward and Sensitivity Modeling: Practical Aspects of modeling
    a Magnetic Dipole in a Multilayered Half-Space. IEEE Geoscience and Remote Sensing Magazine, 7(1), 74-85


:AUTHOR: Daan Hanssens
:ORGANIZATION: Ghent University
:CONTACT: daan.hanssens@ugent.be
:REQUIRES: numpy, scipy, copy
"""

# Import
import numpy as np
import copy
import scipy


class Pair1D:
    def __init__(self, coil_configuration, model, method='RC'):
        """
            Pairs Coil Configuration to Model for 1D forward and sensitivity calculation.

            Parameters
            ----------
            coil_configuration : CoilConfiguration object
                CoilConfiguration class.

            model : Model object
                Model class.

            method : str, optional
                Reflection coefficient ('RC'; default) or Propagation Matrix ('PM').
        """

        self.cc = coil_configuration
        self.model = model
        self.method = method
        if method not in ['RC', 'PM']:
            raise ValueError(
                "Choose an appropriate method: 'PM' for Propagation Matrix or 'RC' for Reflection Coefficient")
        self.original_model = copy.deepcopy(self.model)

    def _original_model(self, parameter):
        """
            Store original profile with sensitivity parameter

            Parameters
            ----------
            parameter
                Sensitivity of physical property ('con','sus','perm')

            Returns
            -------
            _original_model
                Model with set sensitivity parameter
        """

        return getattr(self.original_model, parameter)

    def _reset_model(self):
        """
        Reset model (for sensitivity calculation)
        """

        self.model = copy.deepcopy(self.original_model)

    def forward(self):
        """
        Calculate the forward response (ppm) of a given layered half-space and loop-loop configuration.

        Returns
        -------
        forward_ip : np.array
            IP response (ppm).

        forward_qp : np.array
            QP response (ppm).
        """

        # Get magnetic fields
        [h, hn] = self.magnetic_fields()

        # Normalization of magnetic field (ppm)
        h_normalized = 1e6 * h / hn

        # Get forward response (ppm)
        forward_ip = np.real(h_normalized)
        forward_qp = np.imag(h_normalized)

        return forward_ip, forward_qp

    def sensitivity(self, parameter, perturbation_factor=1e-2):
        """
        Calculate the sensitivity distribution of a given layered half-space and loop-loop configuration.

        Parameters
        ----------
        parameter : str
            Sensitivity of physical property ('con', 'sus', 'perm')

        perturbation_factor : float, optional
            Perturbation factor (default: 1e-2)

        Returns
        -------
        sensitivity_ip_1 : np.array
            IP sensitivity distribution

        sensitivity_qp_1 : np.array
            QP sensitivity distribution

        error : float
            Estimated error on sensitivity
        """

        # Initialize
        nlay = self._original_model(parameter).size
        forward_ip_pos = np.zeros(nlay)
        forward_qp_pos = np.zeros(nlay)
        forward_ip_neg = np.zeros(nlay)
        forward_qp_neg = np.zeros(nlay)

        # Get original response
        [forward_ip_original, forward_qp_original] = self.forward()

        # Loop over layer(s)
        for ii in range(nlay):

            # Copy original model
            model_original = copy.deepcopy(self._original_model(parameter))

            # Get altered response (forward)
            perturbation = model_original[ii] * perturbation_factor
            model_original[ii] = model_original[ii] + perturbation
            setattr(self.model, parameter, model_original)  # Set Perturbed profile
            [forward_ip_pos[ii], forward_qp_pos[ii]] = self.forward()

            # Get altered response (backward)
            model_original = copy.deepcopy(self._original_model(parameter))
            model_original[ii] = model_original[ii] - perturbation
            setattr(self.model, parameter, model_original)  # Set Perturbed profile
            [forward_ip_neg[ii], forward_qp_neg[ii]] = self.forward()

            # Reset Model
            self._reset_model()

        # First derivative
        sensitivity_ip_1 = (forward_ip_pos - forward_ip_original) / perturbation
        sensitivity_qp_1 = (forward_qp_pos - forward_qp_original) / perturbation

        # Second derivative
        sensitivity_ip_2 = (forward_ip_pos - 2 * forward_ip_original + forward_ip_neg) / perturbation ** 2
        sensitivity_qp_2 = (forward_qp_pos - 2 * forward_qp_original + forward_qp_neg) / perturbation ** 2

        # Estimate maximum error
        error = [np.max(sensitivity_ip_2) * perturbation / 2, np.max(sensitivity_qp_2) * perturbation / 2]

        return sensitivity_ip_1, sensitivity_qp_1, error

    def magnetic_fields(self):
        """
        Calculate magnetic fields for an x-directed transmitter and different X,Y,Z-receiver orientations
        ('XX','XY','XZ'), Y-directed transmitter and different X,Y,Z-receiver orientations ('YX','YY','YZ') and
        Z-directed transmitter and different x,y,z-receiver orientations ('ZX','ZY','ZZ').

        Returns
        -------
        h : np.array
            Magnetic field (A/m).

        hn : np.array
            Magnetic field used for normalization (A/m).
        """

        if self.cc.orientation == 'ZZ':

            # Calculate Hzz (HCP)
            h = self.cc.moment / (4 * np.pi) * self._digital_filter(self._rte_p2, 0)
            hn = self.cc.moment / (4 * np.pi) * self._digital_filter(self._rte_02, 0)

        elif self.cc.orientation == 'ZY':

            # Calculate Hzy (NULL)
            h = -self.cc.moment / (4 * np.pi) * self.cc.xyz[1] / \
                self.cc.spacing * self._digital_filter(self._rte_n2, 1)
            hn = self.cc.moment / (4 * np.pi) * self._digital_filter(self._rte_02, 0)

        elif self.cc.orientation == 'ZX':

            # Calculate Hzx (PRP)
            h = -self.cc.moment / (4 * np.pi) * self.cc.xyz[0] / \
                self.cc.spacing * self._digital_filter(self._rte_n2, 1)
            hn = self.cc.moment / (4 * np.pi) * self._digital_filter(self._rte_02, 0)

        elif self.cc.orientation == 'XX':

            # Calculate Hxx (VCA)
            h = -self.cc.moment / (4 * np.pi) * (
                    1 / self.cc.spacing - 2 * self.cc.xyz[0] ** 2 / self.cc.spacing ** 3) * self._digital_filter(self._rte_n1, 1) - \
                self.cc.moment / (4 * np.pi) * self.cc.xyz[0] ** 2 / self.cc.spacing ** 2 * self._digital_filter(
                    self._rte_n2, 0)
            hn = -self.cc.moment / (4 * np.pi) * (
                    1 / self.cc.spacing - 2 * self.cc.xyz[0] ** 2 / self.cc.spacing ** 3) * self._digital_filter(self._rte_01, 1) - \
                 self.cc.moment / (4 * np.pi) * self.cc.xyz[0] ** 2 / self.cc.spacing ** 2 * self._digital_filter(
                     self._rte_02, 0)

        elif self.cc.orientation == 'XY':

            # Calculate Hxy (NULL)
            h = self.cc.moment / (2 * np.pi) * self.cc.xyz[0] * self.cc.xyz[
                1] / self.cc.spacing ** 3 * self._digital_filter(self._rte_n1, 1) - \
                self.cc.moment / (4 * np.pi) * self.cc.xyz[0] * self.cc.xyz[
                    1] / self.cc.spacing ** 2 * self._digital_filter(self._rte_n2, 0)
            hn = -self.cc.moment / (4 * np.pi) * (
                    1 / self.cc.spacing - 2 * self.cc.xyz[0] ** 2 / self.cc.spacing ** 3) * self._digital_filter(self._rte_01, 1) - \
                 self.cc.moment / (4 * np.pi) * self.cc.xyz[0] ** 2 / self.cc.spacing ** 2 * self._digital_filter(
                     self._rte_02, 0)

        elif self.cc.orientation == 'XZ':

            # Calculate Hxz (PRP)
            h = self.cc.moment / (4 * np.pi) * self.cc.xyz[0] / self.cc.spacing * self._digital_filter(self._rte_p2,
                                                                                              1)
            hn = -self.cc.moment / (4 * np.pi) * (
                    1 / self.cc.spacing - 2 * self.cc.xyz[0] ** 2 / self.cc.spacing ** 3) * self._digital_filter(self._rte_01, 1) - \
                 self.cc.moment / (4 * np.pi) * self.cc.xyz[0] ** 2 / self.cc.spacing ** 2 * self._digital_filter(
                     self._rte_02, 0)

        elif self.cc.orientation == 'YX':

            # Calculate Hyx (NULL)
            h = self.cc.moment / (2 * np.pi) * self.cc.xyz[0] * self.cc.xyz[
                1] / self.cc.spacing ** 3 * self._digital_filter(self._rte_n1, 1) - \
                self.cc.moment / (4 * np.pi) * self.cc.xyz[0] * self.cc.xyz[
                    1] / self.cc.spacing ** 2 * self._digital_filter(self._rte_n2, 0)
            hn = -self.cc.moment / (4 * np.pi) * (
                    1 / self.cc.spacing - 2 * self.cc.xyz[1] ** 2 / self.cc.spacing ** 3) * self._digital_filter(self._rte_01, 1) - \
                 self.cc.moment / (4 * np.pi) * self.cc.xyz[1] ** 2 / self.cc.spacing ** 2 * self._digital_filter(
                     self._rte_02, 0)

        elif self.cc.orientation == 'YY':

            # Calculate Hyy (VCP)
            h = -self.cc.moment / (4 * np.pi) * (
                    1 / self.cc.spacing - 2 * self.cc.xyz[1] ** 2 / self.cc.spacing ** 3) * self._digital_filter(self._rte_n1, 1) - \
                self.cc.moment / (4 * np.pi) * self.cc.xyz[1] ** 2 / self.cc.spacing ** 2 * self._digital_filter(
                    self._rte_n2, 0)
            hn = -self.cc.moment / (4 * np.pi) * (
                    1 / self.cc.spacing - 2 * self.cc.xyz[1] ** 2 / self.cc.spacing ** 3) * self._digital_filter(self._rte_01, 1) - \
                 self.cc.moment / (4 * np.pi) * self.cc.xyz[1] ** 2 / self.cc.spacing ** 2 * self._digital_filter(
                     self._rte_02, 0)

        elif self.cc.orientation == 'YZ':

            # Calculate Hyz (NULL)
            h = self.cc.moment / (4 * np.pi) * self.cc.xyz[1] / self.cc.spacing * self._digital_filter(self._rte_p2,
                                                                                              1)
            hn = -self.cc.moment / (4 * np.pi) * (
                    1 / self.cc.spacing - 2 * self.cc.xyz[1] ** 2 / self.cc.spacing ** 3) * self._digital_filter(self._rte_01, 1) - \
                 self.cc.moment / (4 * np.pi) * self.cc.xyz[1] ** 2 / self.cc.spacing ** 2 * self._digital_filter(
                     self._rte_02, 0)

        else:

            raise ValueError('Transmitter-receiver orientation unknown: should be a case sensitive X, Y, Z combination')

        return h, hn

    def _digital_filter(self, function_name, order):
        """
            Solves the Hankel Transform of the zeroth or first order by using a Guptasarma and Singh filtering routine.

            Parameters
            ----------
            function_name
                Name of function

            order
                Order of filter

            Returns
            -------
            y
                Solved Hankel Transform
        """

        # Load Guptasarma and Singh filter
        if order == 0:

            # Load 120-point filter
            filter_a = -8.3885
            filter_s = 0.090422646867
            filter_w = np.array(
                [9.62801364263000e-07, -5.02069203805000e-06, 1.25268783953000e-05, -1.99324417376000e-05,
                 2.29149033546000e-05, -2.04737583809000e-05, 1.49952002937000e-05, -9.37502840980000e-06,
                 5.20156955323000e-06, -2.62939890538000e-06, 1.26550848081000e-06, -5.73156151923000e-07,
                 2.76281274155000e-07, -1.09963734387000e-07, 7.38038330280000e-08, -9.31614600001000e-09,
                 3.87247135578000e-08, 2.10303178461000e-08, 4.10556513877000e-08, 4.13077946246000e-08,
                 5.68828741789000e-08, 6.59543638130000e-08, 8.40811858728000e-08, 1.01532550003000e-07,
                 1.26437360082000e-07, 1.54733678097000e-07, 1.91218582499000e-07, 2.35008851918000e-07,
                 2.89750329490000e-07, 3.56550504341000e-07, 4.39299297826000e-07, 5.40794544880000e-07,
                 6.66136379541000e-07, 8.20175040653000e-07, 1.01015545059000e-06, 1.24384500153000e-06,
                 1.53187399787000e-06, 1.88633707689000e-06, 2.32307100992000e-06, 2.86067883258000e-06,
                 3.52293208580000e-06, 4.33827546442000e-06, 5.34253613351000e-06, 6.57906223200000e-06,
                 8.10198829111000e-06, 9.97723263578000e-06, 1.22867312381000e-05, 1.51305855976000e-05,
                 1.86329431672000e-05, 2.29456891669000e-05, 2.82570465155000e-05, 3.47973610445000e-05,
                 4.28521099371000e-05, 5.27705217882000e-05, 6.49856943660000e-05, 8.00269662180000e-05,
                 9.85515408752000e-05, 0.000121361571831000, 0.000149454562334000, 0.000184045784500000,
                 0.000226649641428000, 0.000279106748890000, 0.000343716968725000, 0.000423267056591000,
                 0.000521251001943000, 0.000641886194381000, 0.000790483105615000, 0.000973420647376000,
                 0.00119877439042000, 0.00147618560844000, 0.00181794224454000, 0.00223860214971000,
                 0.00275687537633000, 0.00339471308297000, 0.00418062141752000, 0.00514762977308000,
                 0.00633918155348000, 0.00780480111772000, 0.00961064602702000, 0.0118304971234000, 0.0145647517743000,
                 0.0179219149417000, 0.0220527911163000, 0.0271124775541000, 0.0333214363101000, 0.0408864842127000,
                 0.0501074356716000, 0.0612084049407000, 0.0745146949048000, 0.0900780900611000, 0.107940155413000,
                 0.127267746478000, 0.146676027814000, 0.162254276550000, 0.168045766353000, 0.152383204788000,
                 0.101214136498000, -0.00244389126667000, -0.154078468398000, -0.303214415655000, -0.297674373379000,
                 0.00793541259524000, 0.426273267393000, 0.100032384844000, -0.494117404043000, 0.392604878741000,
                 -0.190111691178000, 0.0743654896362000, -0.0278508428343000, 0.0109992061155000, -0.00469798719697000,
                 0.00212587632706000, -0.000981986734159000, 0.000444992546836000, -0.000189983519162000,
                 7.31024164292000e-05, -2.40057837293000e-05, 6.23096824846000e-06, -1.12363896552000e-06,
                 1.04470606055000e-07])

        elif order == 1:

            # Load 140-point filter
            filter_a = -7.91001919
            filter_s = 0.087967143957
            filter_w = np.array(
                [-6.76671159511000e-14, 3.39808396836000e-13, -7.43411889153000e-13, 8.93613024469000e-13,
                 -5.47341591896000e-13, -5.84920181906000e-14, 5.20780672883000e-13, -6.92656254606000e-13,
                 6.88908045074000e-13, -6.39910528298000e-13, 5.82098912530000e-13, -4.84912700478000e-13,
                 3.54684337858000e-13, -2.10855291368000e-13, 1.00452749275000e-13, 5.58449957721000e-15,
                 -5.67206735175000e-14, 1.09107856853000e-13, -6.04067500756000e-14, 8.84512134731000e-14,
                 2.22321981827000e-14, 8.38072239207000e-14, 1.23647835900000e-13, 1.44351787234000e-13,
                 2.94276480713000e-13, 3.39965995918000e-13, 6.17024672340000e-13, 8.25310217692000e-13,
                 1.32560792613000e-12, 1.90949961267000e-12, 2.93458179767000e-12, 4.33454210095000e-12,
                 6.55863288798000e-12, 9.78324910827000e-12, 1.47126365223000e-11, 2.20240108708000e-11,
                 3.30577485691000e-11, 4.95377381480000e-11, 7.43047574433000e-11, 1.11400535181000e-10,
                 1.67052734516000e-10, 2.50470107577000e-10, 3.75597211630000e-10, 5.63165204681000e-10,
                 8.44458166896000e-10, 1.26621795331000e-09, 1.89866561359000e-09, 2.84693620927000e-09,
                 4.26886170263000e-09, 6.40104325574000e-09, 9.59798498616000e-09, 1.43918931885000e-08,
                 2.15798696769000e-08, 3.23584600810000e-08, 4.85195105813000e-08, 7.27538583183000e-08,
                 1.09090191748000e-07, 1.63577866557000e-07, 2.45275193920000e-07, 3.67784458730000e-07,
                 5.51470341585000e-07, 8.26916206192000e-07, 1.23991037294000e-06, 1.85921554669000e-06,
                 2.78777669034000e-06, 4.18019870272000e-06, 6.26794044911000e-06, 9.39858833064000e-06,
                 1.40925408889000e-05, 2.11312291505000e-05, 3.16846342900000e-05, 4.75093313246000e-05,
                 7.12354794719000e-05, 0.000106810848460000, 0.000160146590551000, 0.000240110903628000,
                 0.000359981158972000, 0.000539658308918000, 0.000808925141201000, 0.00121234066243000,
                 0.00181650387595000, 0.00272068483151000, 0.00407274689463000, 0.00609135552241000,
                 0.00909940027636000, 0.0135660714813000, 0.0201692550906000, 0.0298534800308000, 0.0439060697220000,
                 0.0639211368217000, 0.0916763946228000, 0.128368795114000, 0.173241920046000, 0.219830379079000,
                 0.251193131178000, 0.232380049895000, 0.117121080205000, -0.117252913088000, -0.352148528535000,
                 -0.271162871370000, 0.291134747110000, 0.317192840623000, -0.493075681595000, 0.311223091821000,
                 -0.136044122543000, 0.0512141261934000, -0.0190806300761000, 0.00757044398633000, -0.00325432753751000,
                 0.00149774676371000, -0.000724569558272000, 0.000362792644965000, -0.000185907973641000,
                 9.67201396593000e-05, -5.07744171678000e-05, 2.67510121456000e-05, -1.40667136728000e-05,
                 7.33363699547000e-06, -3.75638767050000e-06, 1.86344211280000e-06, -8.71623576811000e-07,
                 3.61028200288000e-07, -1.05847108097000e-07, -1.51569361490000e-08, 6.67633241420000e-08,
                 -8.33741579804000e-08, 8.31065906136000e-08, -7.53457009758000e-08, 6.48057680299000e-08,
                 -5.37558016587000e-08, 4.32436265303000e-08, -3.37262648712000e-08, 2.53558687098000e-08,
                 -1.81287021528000e-08, 1.20228328586000e-08, -7.10898040664000e-09, 3.53667004588000e-09,
                 -1.36030600198000e-09, 3.52544249042000e-10, -4.53719284366000e-11])

        else:

            raise ValueError('Digital filter order should be 0 or 1.')

        # Get (complex) lambda values
        n_f = filter_w.size
        ind = np.arange(n_f)
        l1 = 1.0 / self.cc.spacing
        l2 = 10.0 ** (filter_a + ind * filter_s)
        l = l1 * l2
        l = l.astype('complex128')

        # Evaluate function at lambda
        YF = function_name(l)

        # Calculate output, considering weights and r
        y = (np.dot(YF[None, :], filter_w[:, None])) / self.cc.spacing
        y = y[0]  # reduce dimensionality

        return y

    def _reflection_coefficient(self, lambdaa):
        """
        Calculate the reflection coefficient for a given layered half-space and lambda value.
        """

        # Calculate mu
        mu = scipy.constants.mu_0 * (1.0 + self.model.sus[None, :])

        # Calculate u0 and u
        u0 = np.sqrt(lambdaa ** 2 - self.cc.angular_frequency ** 2 * scipy.constants.mu_0 * scipy.constants.epsilon_0)
        u = np.sqrt(lambdaa[:, None] ** 2 -
                    self.cc.angular_frequency ** 2 * mu * self.model.perm[None, :] +
                    1j * self.cc.angular_frequency * mu * self.model.con[None, :])

        # Calculate y and create yhat
        y0 = u0 / (1j * self.cc.angular_frequency * scipy.constants.mu_0)
        y = u / (1j * self.cc.angular_frequency * mu)
        yhat = copy.deepcopy(y)

        # In case of half-space
        if self.model.con.size == 1:

            # Calculate rte
            rte = (y0 - yhat[:, 0]) / (y0 + yhat[:, 0])

        # In case of more complex space w0000t
        else:

            # Recursive formula
            for i in range(self.model.nr_layers - 1)[::-1]:
                # Calculate tanh
                tanh_uh = np.tanh(u[:, i] * self.model.thick[i])

                # Calculate yhat
                num = yhat[:, i + 1] + y[:, i] * tanh_uh
                den = y[:, i] + yhat[:, i + 1] * tanh_uh
                yhat[:, i] = y[:, i] * num / den

            # Calculate rte
            rte = (y0 - yhat[:, 0]) / (y0 + yhat[:, 0])

        return rte

    def _propagation_matrix(self, lambdaa):
        """
        Calculates the P(2,1)/P(1,1) ratio of the propagation matrix for a given layered half-space and lambda
        value.
        """

        # Get information
        n_f = lambdaa.size
        n_l = self.model.sus.size

        # Calculate mu
        mu = scipy.constants.mu_0 * (1.0 + self.model.sus)

        # Calculate u0 and u
        u0 = np.sqrt(lambdaa ** 2 - self.cc.angular_frequency ** 2 * scipy.constants.mu_0 * scipy.constants.epsilon_0)
        u = np.sqrt(lambdaa[:, None] ** 2 -
                    self.cc.angular_frequency ** 2 * mu[None, :] * self.model.perm[None, :] +
                    1j * self.cc.angular_frequency * mu[None, :] * self.model.con[None, :])

        # Calculate M1 (for first layer)
        M = np.zeros((2, 2, n_f, n_l), dtype='complex128')
        M[:, :, :, 0] = np.array([[0.5 * (1 + (scipy.constants.mu_0 * u[:, 0]) / (mu[0] * u0)),
                                   0.5 * (1 - (scipy.constants.mu_0 * u[:, 0]) / (mu[0] * u0))],
                                  [0.5 * (1 - (scipy.constants.mu_0 * u[:, 0]) / (mu[0] * u0)),
                                      0.5 * (1 + (scipy.constants.mu_0 * u[:, 0]) / (mu[0] * u0))]])

        # Calculate Mn
        M[:, :, :, 1:] = np.array([[0.5 * (1 + (mu[:-1] * u[:, 1:]) / (mu[1:] * u[:, :-1])),
                                    0.5 * (1 - (mu[:-1] * u[:, 1:]) / (mu[1:] * u[:, :-1]))],
                                   [0.5 * (1 - (mu[:-1] * u[:, 1:]) / (mu[1:] * u[:, :-1]))
                                    * np.exp(-2 * u[:, :-1] * self.model.thick[:-1]),
                                    0.5 * (1 + (mu[:-1] * u[:, 1:]) / (mu[1:] * u[:, :-1]))
                                    * np.exp(-2 * u[:, :-1] * self.model.thick[:-1])]])

        # Dot product vector
        PM = copy.deepcopy(M[:, :, :, 0])
        for iL in range(1, n_l):
            for iF in range(n_f):
                PM[:, :, iF] = np.dot(PM[:, :, iF], M[:, :, iF, iL])

        # Get ratio
        PP = PM[1, 0, :] / PM[0, 0, :]

        return PP

    def _rte_01(self, lambdaa):
        """
        Additional lambda functions for Hankel calculation of Primary and Secondary magnetic fields.
        """

        # Define variable u0
        u0 = np.sqrt(lambdaa ** 2 - self.cc.angular_frequency ** 2 * scipy.constants.mu_0 * scipy.constants.epsilon_0)

        # Output
        y = (np.exp(-u0 * (self.cc.xyz[2] + self.cc.height))) * lambdaa

        return y

    def _rte_02(self, lambdaa):
        """
        Additional lambda functions for Hankel calculation of Primary and Secondary magnetic fields.
        """

        # Define variable u0
        u0 = np.sqrt(lambdaa ** 2 - self.cc.angular_frequency ** 2 * scipy.constants.mu_0 * scipy.constants.epsilon_0)

        # Output
        y = (np.exp(-u0 * (self.cc.xyz[2] + self.cc.height))) * lambdaa ** 3 / u0

        return y

    def _rte_n1(self, lambdaa):
        """
        Additional lambda functions for Hankel calculation of Primary and Secondary magnetic fields.
        """

        # Calculate rTE
        if self.method == 'RC':
            rTE = self._reflection_coefficient(lambdaa)
        elif self.method == 'PM':
            rTE = self._propagation_matrix(lambdaa)

        # Define variable u0
        u0 = np.sqrt(lambdaa ** 2 - self.cc.angular_frequency ** 2 * scipy.constants.mu_0 * scipy.constants.epsilon_0)

        # Output
        y = (-rTE * np.exp(u0 * (self.cc.xyz[2] - self.cc.height))) * lambdaa

        return y

    def _rte_n2(self, lambdaa):
        """
        Additional lambda functions for Hankel calculation of Primary and Secondary magnetic fields.
        """

        # Calculate rTE
        if self.method == 'RC':
            rTE = self._reflection_coefficient(lambdaa)
        elif self.method == 'PM':
            rTE = self._propagation_matrix(lambdaa)

        # Define variable u0
        u0 = np.sqrt(lambdaa ** 2 - self.cc.angular_frequency ** 2 * scipy.constants.mu_0 * scipy.constants.epsilon_0)

        # Output
        y = (-rTE * np.exp(u0 * (self.cc.xyz[2] - self.cc.height))) * lambdaa ** 2

        return y

    def _rte_p2(self, lambdaa):
        """
        Additional lambda functions for Hankel calculation of Primary and Secondary magnetic fields.
        """

        # Calculate rTE
        if self.method == 'RC':
            rTE = self._reflection_coefficient(lambdaa)
        elif self.method == 'PM':
            rTE = self._propagation_matrix(lambdaa)

        # Define variable u0
        u0 = np.sqrt(lambdaa ** 2 - self.cc.angular_frequency ** 2 * scipy.constants.mu_0 * scipy.constants.epsilon_0)

        # Output
        y = (rTE * np.exp(u0 * (self.cc.xyz[2] - self.cc.height))) * lambdaa ** 3 / u0

        return y


def sphere3d(skin, radius, depth, offset, mu_r, coil_spacing, x0, configuration='zz'):
    """
    Calculate normalized magnetic field H in ppm (QP and IP) for 'zz' or 'zx' configuration.

    Parameters
    ----------
    skin : float
        Skin depth (m).

    radius : float
        Radius of sphere (m).

    depth : float
        Depth of sphere center (m).

    offset : float
        Lateral offset of sphere center (m).

    mu_r : float
        Relative magnetic permeability (-).

    coil_spacing : float
        Coil spacing (m).

    x0 : float
        Distance along axis (m).

    configuration : {'zz', 'zx'}, optional
        Determine coil configuration PRP ('zx') or HCP ('zz').

    Returns
    -------
    h_qp : float
        Magnetic field QP (ppm).

    h_ip : float
        Magnetic field IP (ppm).

    References
    ----------
    Frischknecht, F.C., Labson, V.F., Spies, B.R., Anderson, W.L., 1991.
        Profiling methods using small sources. In: M.N. Nabighian (Ed.),
        Electromagnetic methods in applied geophysics. Society of Exploration Geophysicists, USA, pp. 105-269

    Grant, F.S., West, G.F., 1965.
        Interpretation theory in applied geophysics. McGraw-Hill Book Co., New York
    """

    # Theta
    theta_0 = np.arctan(x0 / np.sqrt(depth ** 2 + offset ** 2))
    theta_r = np.arctan((x0 - coil_spacing) / np.sqrt(depth ** 2 + offset ** 2))
    theta = theta_0 - theta_r

    # Radials
    ksi = np.arctan(offset / depth)
    r = np.sqrt(depth ** 2 + offset ** 2 + (x0 - coil_spacing) ** 2)
    r0 = np.sqrt(depth ** 2 + offset ** 2 + x0 ** 2)

    # m
    mr0 = np.cos(ksi) * np.cos(theta_0)
    mt0 = np.cos(ksi) * np.sin(theta_0)
    mf0 = - np.sin(ksi)

    # Other
    ka = radius / skin * (1 + 1j)
    za = (1 / 2 - 2 * mu_r + ka) / (1 / 2 + mu_r + ka)

    # sphere3d
    Hrr = - mr0 * za * radius ** 3. / (r * r0) ** 3. * 2. * np.cos(theta)
    Hrt = - mr0 * za * radius ** 3. / (r * r0) ** 3. * np.sin(theta)
    Htr = mt0 * za * radius ** 3. / (r * r0) ** 3. * np.sin(theta)
    Htt = - mt0 * za * radius ** 3. / (r * r0) ** 3. * 1 / 2. * np.cos(theta)
    Hff = mf0 * za * radius ** 3. / (r * r0) ** 3. * 1 / 2.

    # Grab configuration
    if configuration == 'zx':

        Hzx = np.sin(theta_r) * (Hrr + Htr) - np.cos(theta_r) * (Hrt + Htt)
        hzx_qp = 1e6 * coil_spacing ** 3. * np.imag(Hzx)
        hzx_ip = 1e6 * coil_spacing ** 3. * np.real(Hzx)

        return hzx_qp, hzx_ip

    elif configuration == 'zz':

        Hzz = np.cos(ksi) * np.cos(theta_r) * (Hrr + Htr) + np.cos(ksi) * np.sin(theta_r) * (Hrt + Htt) + np.sin(ksi) * Hff
        hzz_qp = -1e6 * coil_spacing ** 3. * np.imag(Hzz)
        hzz_ip = -1e6 * coil_spacing ** 3. * np.real(Hzz)

        return hzz_qp, hzz_ip

    else:

        raise ValueError('Configuration is unknown')
