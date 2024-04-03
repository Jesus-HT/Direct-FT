# -*- coding: cp1252 -*-

# -----------------------------------------------------------------------------------------------------------------------
"""
                SUBJECT: Fast Fourier Transform and damped oscillator for biological periodic patterns.
            Description: Finding the periodic pattern (angular frequency and period) of somite formation,
                    for human and mice, as well as, for the circadian and menstrual cycles.

 Article title: Application of physical concepts in molecular biology: Determination of oscillation periods of
                biological systems

 Authors: Jesús Hernández-Trujillo, Marco Franco-Pérez, Alejandro Pisanty-Baruch
"""
# -----------------------------------------------------------------------------------------------------------------------

# Libraries to be used
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.fftpack import fft
from lmfit import Model
import pandas as pd
import os
from scipy.interpolate import CubicSpline


"""Classes"""

class raw_data_tratment:  # (Regular) Data preparation for further analysis
    """
        Each object data is defined in terms of its weighted data.
        Arguments:
            file: file which contain experimental (regular) data
        """

    def __init__(self, file, detrend_lvl, save_plts, in_time_scale, do_splines, label):
        """
            Constructor reads the data from a csv file
            Arguments:
                file:  filename
                detrend_lvl: level of decorrelation to be applied
                save_plts: indicates if plots must be saved or not
                in_time_scale: input timescale, the one comming from the raw data
        """
        # Reading and filtering data with pandas
        df = pd.read_csv(file)                  # Creating dataframe
        self.time_data = df['Time']             # Extracting data time from the column headed 'Time'
        self.exp_data = df['EXPERIMENT']        # Extracting data measurments from the column headed 'EXPERIMENT'
        self.save_plts = save_plts              # Indicates if initial data plots will be constructed
        self.detrend_level = detrend_lvl                    # indicates the level of decorrelation to be applied
        self.do_splines = do_splines
                             # name of the experiment from the csv file
        self.init_t_sc = in_time_scale                          # indicates the timescale of the raw data
        self.label = label

        if self.label == None:
            self.name = str(file.strip(".csv"))
        else:
            self.name = str(file.strip(".csv")) + '_' + str(self.label)

        self.folder_name = 'Results_for_' + str(self.name)

        try:
            os.makedirs(self.folder_name)
        except Exception:
            pass

        # Indicating to user which data_file is under analysis
        print('****** Computing frequencies and periods for', self.name, 'experiment ******')

        if self.save_plts == True:
            self.plots()  # Building and saving data plots

    def data_preparation(self):

        """
            Perform cubic splines if desired and also normalizes the data be the sample mean .
            Takes raw experimental values
            return new data
        """

        data = self.exp_data
        time = self.time_data
        time = time.values.tolist()


        if self.do_splines == True:
            splin_data = CubicSpline(self.time_data, self.exp_data)  #, bc_type='natural')
            time = np.linspace(min(self.time_data), max(self.time_data) + 0.8, 100)
            data = splin_data(time)

        mean = np.mean(data)           # mean of the data
        weighted_data = [100 * (x - mean) / mean for x in data]   # data weighted by the mean
        return time, weighted_data

    def data_for_analysis(self):
        """
            Apply de-trending at ith order to the raw regular data.
            It will take the object property weighted data as argument, as well as, time dataframe
            Returns:
                any ith-order (even zero) de-correlated data and its corresponding time
        """
        t, data = self.data_preparation()
        i = 1
        ith_order_detrend = data
        tr = t
        while i <= self.detrend_level:  # this loop apply the i-esim de-correlation procedure indicated by the user
            ith_order_detrend = [float(ith_order_detrend[x] - ith_order_detrend[x - 1])
                                for x in range(1, len(ith_order_detrend))]
            tr = t[i:]

            i = 1 + i
        return ith_order_detrend, tr

    def plots(self):
        """ If indicated by user, it will plot graphs corresponding to the data preparation, previous to the numerical
        fitting"""
        dfa = self.data_for_analysis()
        data_output, t = dfa[0], dfa[1]

        # Plot 1) Time vs Experimental Data
        plotter(self.time_data, self.exp_data, 'Time in ' + self.init_t_sc, 'Measurement',
                '1_Experimental_Data_Plot_', self.name, [], [], 'b',
                'Raw Data', self.folder_name)

        # A plot is also built for the ith > 0 de-correlated data
        if self.detrend_level > 0:
            # Plot 2) Time vs output data
            plotter(t, data_output, 'Time in ' + self.init_t_sc, 'Measurement',
                    '2_WorkingData_Plot_', self.name, [],[], 'b',
                    'Decorrelated data', self.folder_name)


class data_analyzer:
    """
        Each object is a set of data with methods for the required analysis.
        Arguments:
            time: time dataframe.
            income_data: weighted and de-correlated data to be analyzed.
            in_time_scale: timescale coming from the experimental measurements.
            final_time_scale: desired timescale for the final results
            save_plts: set True to save all the available plots.
            verbosity: set True to print final results in screen
        """

    def __init__(self, time, income_data, name, in_time_scale, final_time_scale, save_plts, verbosity,
                 avoiding_spurious_T):
        self.time_list = time
        self.ready_data_list = income_data
        self.name = name
        self.save_plts = save_plts
        self.init_t_sc = in_time_scale
        self.end_t_sc = final_time_scale
        self.folder_name = 'Results_for_' + str(self.name)
        self.verbosity = verbosity
        self.avoiding_spurious_T = avoiding_spurious_T
        try:
            os.makedirs(self.folder_name)
        except Exception:
            pass

    def conver_factor(self):
        """
        time conversion factor. Apply the indicated change in timescale (among seconds/minutes/hours)
        :return: conversion factor
        """
        s = 'seconds'
        m = 'minutes'
        h = 'hours'

        from_sec_to_min, from_min_to_sec, from_sec_to_hrs, from_hrs_to_sec = 1 / 60, 60, 1 / 3600, 3600
        from_min_to_hrs, from_hrs_to_min = 1 / 60, 60
        factor = 1

        # each condition apply depending on the desired time conversion
        if self.init_t_sc == s and self.end_t_sc == m:
            factor = from_sec_to_min
        if self.init_t_sc == m and self.end_t_sc == s:
            factor = from_min_to_sec
        if self.init_t_sc == s and self.end_t_sc == h:
            factor = from_sec_to_hrs
        if self.init_t_sc == h and self.end_t_sc == s:
            factor = from_hrs_to_sec
        if self.init_t_sc == m and self.end_t_sc == h:
            factor = from_min_to_hrs
        if self.init_t_sc == h and self.end_t_sc == m:
            factor = from_hrs_to_min
        return factor

    def fft(self, save_frequencies):
        """
        Performs the Fourier transform of the data.
        If indicated, create fft plots and save the frequency spectra in a .csv file
        Takes the ith decorrelated data. Timescale converter will be used to perform the analysis at
        the final timescale
        :argument
            save_frequencies: Set true to save all the frequencies found in the FFT analysis
        :return: Fourier frequencies spectra and Amplitudes
        """
        # Prepare data and define the spacing for the frequencies
        spacing = abs(float(self.time_list[2] - self.time_list[1]))  # Spacing between data reading.
        size = len(self.ready_data_list)  # Sample's size.
        n = np.arange(size)  # Arr. of the sample's size, to store the data.
        sr_output_time = 1 / spacing  # Sample rate .
        T = size / sr_output_time  # Sampling period.
        freq_sample = n / T  # Sampling frequency.

        trans = fft(self.ready_data_list)  # Transform data (amplitudes).
        n_oneside = size // 2  # Split the sampling space (to avoid aliasing).
        freq_oneside = freq_sample[:n_oneside]  # Store frequencies avoiding aliasing.
        list_amplituds = np.abs(trans[:n_oneside])  # Retrieve real positive part of the fft.

        # Getting the angular frequency and the period of the frequency with the highest FFT Amplitude
        omega = 2 * np.pi * (np.argmax(list_amplituds)) / (size * self.conver_factor() * spacing)
        period = 2 * np.pi / omega

        # If the most intense period is very close to the overall experiment time (avoiding spurious Periods):
        time_list_a = [item * self.conver_factor() for item in self.time_list]
        s = time_list_a[len(time_list_a) - 1] * 0.99

        if self.avoiding_spurious_T == True and period > s :
            freq_oneside = np.array(freq_oneside)
            freq_oneside = np.delete(freq_oneside, np.argmax(list_amplituds))
            list_amplituds = np.delete(list_amplituds, np.argmax(list_amplituds))
            frequency = freq_oneside[np.argmax(list_amplituds)] / self.conver_factor()
            omega = 2 * np.pi * frequency
            period = 1 / frequency


        if self.verbosity == True:
            print('Fast Fourier Transform fitting results for the Max Amplitude')
            print("Max Amplitud Omega = ", round(omega, 3), " in rad /", str(self.end_t_sc))
            print("Corresponding Period = ", round(period, 4), " in ", str(self.end_t_sc), '\n')

        if self.save_plts == True:
            # Plot 4) Amplitudes vs Fourier frequency
            plotter(freq_oneside, list_amplituds, 'Freq (' + str(self.end_t_sc) + '[-1])',
                    'FFT Amplitude |X(freq)|', '4_FFT_Plot_', self.name, [],
                    [], 'b', 'FFT plot', self.folder_name)

        if save_frequencies == True:
            # Saving results in a .csv file
            self.save_results(list_amplituds, freq_oneside)

        return freq_oneside, np.abs(trans[:n_oneside]), omega

    def save_results(self, list_amps, f_oneside):
        """
       Save amplitudes, frequencies and periods coming from the FFT analysis
       Arguments:
           list_amps: ist of amplitudes from the fft
           f_oneside: frequencies without aliasing
       :return:
           save maximum frequencies in a csv file
       """

        # opening file for storing frequencies
        f = open(self.folder_name + '/' + 'FFT_Results_' + str(self.name) + '.csv', 'w')
        # setting columns heads
        f.write('Max_Frequency (' + str(self.end_t_sc) + '[-1]), '
                                                         'Amplitud, Amplitud ratio, Period (' + str(
            self.end_t_sc) + ')\n')

        max_amplitude = max(list_amps)  # Max amplitude value.
        for index in range(len(list_amps) - 1):
            freq = f_oneside[index]  # Particular frequency values.
            if freq > 0:  # avoiding a zero frequency value.
                ff_amplitude = list_amps[index]  # Local frequency.
                freq_h = freq / (self.conver_factor())  # Local frequency at the final timescale required by user.
                amp_ratio = ff_amplitude / max_amplitude  # Amplitude ratio respect the maximum amplitude of the set.
                period_h = 1 / freq_h  # Period at the final timescale.

                # Storing frequencies in the indicated .csv file.
                f_h = round(freq_h, 4)
                f_a = round(ff_amplitude, 4)
                f_ar = round(amp_ratio, 4)
                t_h = round(period_h, 4)
                f.write('{0}, {1}, {2}, {3}\n'.format(f_h, f_a, f_ar, t_h))
        f.close()

    def damped_fit(self, trial_omega):
        """
        Perform de damped oscillator fit.
        Takes the ith order de-correlated data. The fit is performed at the final time scale.
        :return:
            omega and period
            Plots the damped movement function
        """
        final_time = [x * self.conver_factor() for x in self.time_list]  # applying timescale factor to all elements

        def decay_sine(t, amp, beta, omega, phi):
            """
            Decay function of a sine wave
            Arguments:
                t: a float, the time
                amp: a float, the amplitude
                beta: a float, the decay coefficient
                omega: a float, the angular frequency
                phi: a float, the propagation phase factor
            Returns:
                Angular frequency and period after the fit
            """
            return amp * np.exp(- beta * t) * np.sin(omega * t + phi)

        # Define the function as a model
        mod = Model(decay_sine)

        # Initialize parameters fot the fitting (those can be tuned to improve results)
        params = mod.make_params(amp=5.0, beta=0.005, omega= 0.95 * trial_omega, phi=0)
        params['phi'].max =  2 * np.pi
        ##params['omega'].min = 1.2
        params['amp'].min = 1

        # Perform numerical fit
        result = mod.fit(self.ready_data_list, params, t=final_time)
        res = result.params.valuesdict()  # a generated dictionary which contains the values of the adjusted parameters
        omega = round(float(res['omega']), 2)
        period = round(2 * np.pi / omega, 2)

        # Saving Results In File
        f = open(self.folder_name + '/' + 'DAMPED_Fitting_Results__' + str(self.name) + '.csv', 'w')
        # setting columns heads
        f.write('Angular_Frequency (' + str(self.end_t_sc) + '[-1]), '
                                                         'Period (' + str(
            self.end_t_sc) + ')\n')

        f_w = round(omega, 4)
        f_p = round(period, 4)
        f.write('{0}, {1}\n'.format(f_w, f_p))
        f.close()

        if self.verbosity == True:
            print('Damped oscillator fitting results')
            print('omega = ', omega, ' in ', str(self.end_t_sc), '[-1]')
            print('period = ', period, ' in ', str(self.end_t_sc), '\n')

        if self.save_plts == True:  # Building plot (if indicated by user)
            plotter(final_time, self.ready_data_list, 'Time (' + str(self.end_t_sc) + ')', 'Measurement',
                    '5_Damping_Fit_plot_', str(self.name), final_time, result.best_fit, 'bo',
                    'omega =' + str(omega) + ',  period = ' + str(period) + ' (time units in ' + str(self.end_t_sc) +
                    ')', self.folder_name)
        return omega, period


"""Functions"""

def plotter(x_list, y_list, label_x, label_y, plot_name_file, name, second_x, second_y, type1, title, folder_name):
    """
    Plots (and saves as pdf) the corresponding graphs. It is called four times:
    1) Time vs experimental data
    2) Time vs weighted data
    3) time vs decorrelated data
    4) Amplitudes vs fourier frequency
    Arguments:
        x_list: a list of floats, corresponding to the x-axis
        y_list: a list of floats, corresponding to the y-axis
        label_x: a string, corresponding to the x-axis label
        label_y: a string, corresponding to the y-axis label
        plot_name_file: name of the corresponding pdf file
        name: a string, corresponding to the pdf's filename
        second_x: a list of floats, a second set of x values to plot
        second_y: a list of floats, a second set of y values to plot
        type1: a string, corresponding to the type of plot (dashes, point, solid, etc)
        title: a string, corresponding to the title of the graph
        folder_name: folder where the pdf file will be stores
    """
    file_name = folder_name + '/' + plot_name_file + str(name) + '.pdf'
    with PdfPages(file_name) as pdf:
        plt.figure(figsize=(12, 6))
        plt.plot(x_list, y_list, str(type1))
        plt.plot(second_x, second_y, 'r')
        plt.xlabel(str(label_x))
        plt.ylabel(str(label_y))
        plt.title(title)
        pdf.savefig()
        plt.close()


# Applying the whole model to a particular file
def get_results(file_name, detrend_lvl, initial_t_scale, final_t_scale, damping_fit, do_cubic_splines, verbosity, label,
                avoid_spurious_T):
    """
        Apply the whole model to data contained in a csv file
        Arguments:
            file_name: file containing the data
            detrend_lvl: de-trending level required by user
            initial_t_scale: experimental timescale
            final_t_scale: desired timescale for the final results (periods and frequencies)
        :return:
            none, apply the model
        """
    file_name = str(file_name)

    # building object constituted by the data set
    data_preparation = raw_data_tratment(file=file_name, detrend_lvl=detrend_lvl, save_plts=True,
                                         in_time_scale=initial_t_scale, do_splines=do_cubic_splines, label=label)

    # Extracting, weightening and detrending data, using corresponding class methods
    ready_data, name = data_preparation.data_for_analysis(), data_preparation.name
    working_data, working_time = ready_data[0], ready_data[1]  # data & time to be analyzed numerically.

    # building object for numerical procedures
    frequencies_calculation = data_analyzer(working_time, working_data, name, in_time_scale=initial_t_scale,
                                            final_time_scale=final_t_scale, save_plts=True, verbosity=verbosity,
                                            avoiding_spurious_T=avoid_spurious_T)
    x = frequencies_calculation.fft(save_frequencies=True)  # Performing the Fast Fourier Transform

    if damping_fit == True:
        frequencies_calculation.damped_fit(x[2])  # Performing damping fit if indicated by user


# main execution
def main():
    """ Perform numerical fits to the indicated files.
        NOTE: In the.csv file, the head of the column data must be "EXPERIMENT", while the head of the time data
        must be "Time"
        + detrend_lvl is the desired ith de-trending to be applied (only restricted by the data size)
        + initial_t_scale is the time units at which experimental data was measured (coming from the initial csv file)
        + final_t_scale is time units at which frequencies and periods will be calculated
        + damping fit indicates if a damping oscillator fitting will be applied to the data
        + verbosity = true: print all results in screen
        + label = any particular word to be added in files name that will be generated by this algorithm
        + avoiding_spurious_T = True: it avoids the computation of a period very close to the total time length with up
                                to a 1% difference"""

    # Calculation of angular frequencies and periods of circadian oscillations.
    get_results('osc_circadian_raw.csv', detrend_lvl=0, initial_t_scale='hours',
                final_t_scale='hours', damping_fit=True, do_cubic_splines=True, verbosity=True, label=None,
                avoid_spurious_T=False)

    # Calculation of angular frequencies and periods of the oscillatory expression of the Estradiol Rhythm  in human
    # menstrual cycle
    get_results('estradiol_raw.csv', detrend_lvl=0, initial_t_scale='hours',
                final_t_scale='hours', damping_fit=True, do_cubic_splines=True, verbosity=True, label=None,
                avoid_spurious_T=False)

    # Calculation of angular frequencies and periods of the oscillatory expression of the HES7 gene in human
    # **Raw data
    get_results('hes7_raw.csv', detrend_lvl=0, initial_t_scale='minutes',
                final_t_scale='hours', damping_fit=True, do_cubic_splines=False, verbosity=True, label=None,
                avoid_spurious_T=True)
    # **First order detrended data
    get_results('hes7_raw.csv', detrend_lvl=1, initial_t_scale='minutes',
                final_t_scale='hours', damping_fit=True, do_cubic_splines=False, verbosity=True, label='Detrend',
                avoid_spurious_T=False)
main()
