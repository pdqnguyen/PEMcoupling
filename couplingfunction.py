""""
Classes for holding coupling functions, ASDs, and estimated ambients.


CLASSES:

    CouplingData --- Contains coupling function data, corresponding sensor/DARM ASDs, and sensor metadata.
    CompositeCouplingData --- Contains composite coupling function, corresponding sensor/DARM backgrounds, and sensor metadata.


Usage Notes:
    gwpy.time required for doing gps time calculation.

"""

from gwpy.time import *
from math import *
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
plt.switch_backend('Agg')
from matplotlib import rc
rc('text',usetex=True)

from textwrap import wrap
import os
import time
import datetime
import sys





############################################################################################################



class CoupFunc(object):
    """
    Coupling function data for a single sensor/DARM combination, for a single injection time and background time.
    
    Attributes
    ----------
    name : str
        Name of PEM sensor.
    freqs : array
        Frequency array.
    factors : array
        Coupling factor arrays in physical units and in counts.
    factors_in_counts : array
        Coupling factor arrays in counts.
    flags : array
        Coupling factor flags ('Real', 'Upper limit', 'Thresholds not met', 'No data').
    sens_bg : array
        ASD of sensor during background.
    sens_inj : array
        ASD of sensor during injection.
    darm_bg : array
        ASD of DARM during background.
    darm_inj : array
        ASD of DARM during injection.
    ambients : array
        Estimated ambient ASD for this sensor.
    t_bgb : int
        Background start time.
    t_inj : int
        Injection start time.
    df : float
        Frequency bandwidth in Hz.
    unit : str
        Sensor unit of measurement
    qty : str
        Quantity measured by sensor.
    type : str
        Coupling type associated with sensor.
    
    Methods
    -------
    get_channel_info
        Use channel name to get unit of measurement, name of quantity measured by sensor, and coupling type.
    get_factors_in_counts
        Convert coupling factors from physical units to raw counts
    plot
        Plot coupling function from the data.
    specplot
        Plot sensor and DARM spectra with estimated ambient.
    to_csv
        Save data to csv file.
    """
    
    def __init__(self, name, freqs, factors, flags, sens_bg, sens_inj, darm_bg, darm_inj, t_bg, t_inj, calibration_factors=None):
        """
        Set CouplingData attributes.
        
        Parameters
        ----------
        name : str
            Name of PEM sensor.
        freqs : array
            Frequency array.
        factors : array
            Coupling factor arrays in physical units and in counts.
        flags : array
            Coupling factor flags ('Real', 'Upper limit', 'Thresholds not met', 'No data').
        sens_bg : array
            ASD of sensor during background.
        sens_inj : array
            ASD of sensor during injection.
        darm_bg : array
            ASD of DARM during background.
        darm_inj : array
            ASD of DARM during injection.
        ambients : array
            Estimated ambient ASD for this sensor.
        t_bgb : int
            Background start time.
        t_inj : int
            Injection start time.
        calibration_factors : dict
            Calibration factors for sensors that were calibrated. Used for converting back to counts.
        """
        
        # Coupling function data
        self.name = name
        self.freqs = np.asarray(freqs)
        self.factors = np.asarray(factors)
        self.factors_in_counts = self.get_factors_in_counts(calibration_factors)
        self.flags = np.asarray(flags)
        
        # ASDs
        self.sens_bg = sens_bg
        self.sens_inj = sens_inj
        self.darm_bg = darm_bg[:len(self.freqs)]
        self.darm_inj = darm_inj[:len(self.freqs)]
        self.ambients = self.factors * self.sens_bg
        
        # Metadata
        self.t_bg = t_bg
        self.t_inj = t_inj
        self.df = self.freqs[1] - self.freqs[0]
        self.calibration_factors = calibration_factors
        self.unit, self.qty, self.type = self.get_channel_info()
        
    def get_channel_info(self):
        """
        Get basic information about the channel's measurements
        
        Returns
        -------
        info : list of strings
            Units, quantity name, and coupling type of this sensor.
        """
        
        units_dict = {'MIC': 'Pa', 'MAG': 'T', 'RADIO': 'ADC', 'SEIS': 'm', \
                  'ISI': 'm', 'ACC': 'm', 'HPI': 'm', 'ADC': 'm'}
        quantity_dict = {'Pa': 'Pressure', 'T': 'Magnetic Field', 'm': 'Displacement', \
                         '(\\mu m/s^2)': 'Acceleration', 'ADC': 'Displacement'}
        type_dict = {'MIC': 'Acoustic', 'MAG': 'Magnetic', 'RADIO': 'RF', 'SEIS': 'Seismic', \
                 'ISI': 'Seismic', 'ACC': 'Vibrational', 'HPI': 'Seismic', 'ADC': 'Vibrational'}

        for x in units_dict.keys():
            if x in self.name:
                info = [units_dict[x], quantity_dict[units_dict[x]], type_dict[x]]
                return info
        info = ['Counts', '', '']
        return info
    
    def get_factors_in_counts(self, calibration_factors=None):
        """
        Reverse-calibrates a coupling function to convert from physical units to raw counts.
        Recall that coupling function units are inversely related to sensor units,
        and calibration factors have units [sensor units / counts], so one *multiplies*
        the coupling function by the calibration factor to reverse the calibration.
        
        Parameters
        ----------
        calibration_factors : dict
            Channel names and their corresponding calibration factors.
            
        Returns
        -------
        factors_in_counts : array
            Coupling function in raw counts.
        """
        
        if calibration_factors is None:
            return None
        if self.name not in calibration_factors.keys():
            # No calibration was done, so coupling function is already in counts
            factors_in_counts = self.factors
        else:
            calibration = calibration_factors[self.name]
            # Reverse any velocity-to-displacement or acceleration-to-displacement conversion
            if ('SEIS' in self.name) or ('ISI' in self.name) or ('HPI' in self.name):
                calibration /= (2 * pi * self.freqs)
            elif ('ACC' in self.name) or ('ADC' in self.name):
                calibration /= (2 * pi * self.freqs)**2
            factors_in_counts = self.factors * calibration
        return factors_in_counts
    
    def plot(
        self, path, in_counts=False, ts=None, upper_lim=True,
        freq_min=None, freq_max=None,
        factor_min=None, factor_max=None,
        fig_w=15, fig_h=6
    ):
        """
        Export a coupling function plot from the data
        
        Parameters
        ----------
        path : str
            Target directory.
        in_counts : bool
            If True, convert coupling function to counts, and treat data as such.
        ts : time.time object
            Timestamp object.
        upper_lim : bool
            If True, include upper limits in the plot. Otherwise just plot measured (real) values.
        freq_min : float
            Minimum frequency (x-axis).
        freq_max : float
            Maximum frequency (x_axis).
        factor_min : float
            Minimum value of coupling factor axis (y-axis).
        factor_max : float
            Maximum value of coupling factor axis (y-axis).
        fig_w : float or int
            Figure width.
        fig_h : float or int
            Figure height.
        """
        
        if in_counts:
            factors = self.factors_in_counts
            unit = 'Counts'
            type_c = ''
        else:
            factors = self.factors
            unit = self.unit
            type_c = self.type
        
        # Create time stamp if not provided
        if ts is None:
            ts = time.time()


        # X-AXIS LIMIT
        if freq_min is None:
            freq_min = self.freqs[0]
        if freq_max is None:
            freq_max = self.freqs[-1]


        # Y-AXIS LIMITS
        factor_values = factors[np.isfinite(factors) & (factors > 1e-30)]
        if len(factor_values) == 0:
            print('Warning: No coupling factors found for channel ' + self.name)
            return
        if factor_min is None:
            factor_min = min(factor_values)/3
        if factor_max is None:
            factor_max = max(factor_values)*3


        # ORGANIZE DATA FOR COUPLING FUNCTION PLOT
        real = [[],[]]
        upper = [[],[]]
        for x in range(len(self.freqs)):
            if (self.flags[x] == 'Upper Limit'):
                upper[1].append(factors[x])
                upper[0].append(self.freqs[x])
            elif (self.flags[x] == 'Real'):
                real[1].append(factors[x])
                real[0].append(self.freqs[x])


        # PLOT SIZE OPTIONS BASED ON CHANNEL TYPE
        ms = (8. if 'MAG' in self.name else 4.)
        edgew_circle = (.7 if 'MAG' in self.name else .5)
        edgew_triangle = (1 if 'MAG' in self.name else .7)
        ms_triangle = ms * (.8 if 'MAG' in self.name else .6)


        # CREATE FIGURE FOR COUPLING FUNCTION PLOT
        fig = plt.figure()
        ax = fig.add_subplot(111)

        fig.set_figheight(fig_h)
        fig.set_figwidth(fig_w)

        p1 = ax.get_position()
        p2 = [p1.x0, p1.y0+0.02, p1.width, p1.height-0.02]
        ax.set_position(p2)


        # PLOT COUPLING FUNCTION DATA
        plt.plot(
            real[0],
            real[1], 
            'o', 
            color='lime', 
            markersize=ms, 
            markeredgewidth=edgew_circle, 
            label='Measured', 
            zorder=6
        )
        if upper_lim:
            plt.plot(
                upper[0], 
                upper[1],
                '^',
                markersize=ms_triangle, 
                markerfacecolor='none',
                markeredgecolor='0.0',
                markeredgewidth=edgew_triangle, 
                label='Upper Limits', 
                zorder=2
            )


        # CREATE LEGEND, LABELS, AND TITLES
        legend = plt.legend(prop={'size':18}, loc=1)
        legend.get_frame().set_alpha(0.5)

        plt.ylabel('{0} Coupling [m/{1}]'.format(type_c,unit), size=18)
    #    plt.ylabel(str(type_c)+r' Coupling $\left[\mathrm{{{0}}}/\mathrm{{{1}}}\right]$'.format('m',unit), size=18)
        plt.xlabel(r'Frequency [Hz]', size=18)

        plt.title(self.name.replace('_DQ','').replace('_',' ') + ' - Coupling Function', size=20)
        plt.suptitle(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), fontsize = 14, y=1.01)

        str_quiet_time = 'Background Time: {}\n({})'.format(from_gps(self.t_bg), self.t_bg) + ' '*11
        str_inj_time = 'Injection Time: {}\n' + ' '*25 + '({})'.format(from_gps(self.t_inj), self.t_inj)

        plt.figtext(.9,.01, str_quiet_time, ha='right', fontsize = 12, color = 'b')
        plt.figtext(.1,.01, str_inj_time, fontsize = 12, color = 'b')
        plt.figtext(.5,0.0, 'Band Width: {:1.3f} Hz'.format(self.df), ha='center', va='top', fontsize = 12, color = 'b')
        plt.figtext(.05,.96, 'Injection name:\n{} '.format(path.split('/')[-1].replace('_','\_')),
                    fontsize = 12, color = 'b')
        plt.figtext(.95,.99, 'Measured coupling factors: {}'.format(len(real[1])), ha='right', fontsize = 12, color = 'b')
        if upper_lim:
            plt.figtext(.95,.96, 'Upper limit coupling factors: {}'.format(len(upper[1])), ha='right', fontsize = 12, color = 'b')


        # CUSTOMIZE AXES
        plt.xlim([freq_min, freq_max])
        plt.ylim([factor_min, factor_max])
        ax.set_xscale('log', nonposx = 'clip')
        ax.set_yscale('log', nonposy = 'clip')
        ax.autoscale(False)
        plt.grid(b=True, which='major', color='0.0', linestyle=':', linewidth=1, zorder=0)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='0.6', linestyle=':', zorder=0)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)


        # EXPORT PLOT
        filename = self.name[self.name.index('-')+1:].replace('_DQ','') +'_coupling_plot.png'
        if unit == 'Counts':
            filename = filename.replace('_plot','_counts_plot')

        if path is None: 
            path = datetime.datetime.fromtimestamp(ts).strftime('DATA_%Y-%m-%d_%H:%M:%S')
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = path + '/' + filename
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()

        return fig

    def specplot(
        self,
        path=None, ts=None, est_amb=True, show_darm_threshold=True, upper_lim=True,
        freq_min=None, freq_max=None,
        spec_min=None, spec_max=None,
        fig_w=12, fig_h=6
    ):
        """
        Export an estimated ambient plot from the data.
        
        Parameters
        ----------
        path : str
            Target directory.
        ts : time.time object
            Timestamp object.
        est_amb : bool
            If True, show estimated ambient super-imposed on DARM spectrum.
        show_darm_threshold: bool
            If True, draw a dashed spectrum representing one order of magnitude below DARM.
        upper_lim : bool
            If True, include upper limits in the plot. Otherwise just plot measured (real) values.
        freq_min : float
            Minimum frequency (x-axis).
        freq_max : float
            Maximum frequency (x_axis).
        spec_min : float
            Minimum value of coupling factor axis (y-axis).
        spec_max : float
            Maximum value of coupling factor axis (y-axis).
        fig_w : float or int
            Figure width.
        fig_h : float or int
            Figure height.
        """

        if ts is None:
            ts = time.time()

        loc_freq_min, loc_freq_max = 0, -1
        while self.freqs[loc_freq_min] < freq_min: loc_freq_min += 1
        while self.freqs[loc_freq_max] > freq_max: loc_freq_max -= 1
        darm_y = list(self.darm_bg[loc_freq_min:loc_freq_max]) + list(self.darm_inj[loc_freq_min:loc_freq_max])
        amp_max_spec = max(darm_y)
        amp_min_spec = min(darm_y)

        # Y-AXIS LIMITS FOR SENSOR SPECTROGRAM
        spec_sens_min = np.min(self.sens_bg[loc_freq_min:loc_freq_max]) / 2
        spec_sens_max = np.max(self.sens_inj[loc_freq_min:loc_freq_max]) * 4

        # Y-AXIS LIMITS FOR DARM/EST AMB SPECTROGRAM
        amb_values = self.ambients[np.isfinite(self.ambients) & (self.ambients>0)]
        amb_min = np.min(amb_values) if np.any(amb_values) else self.darm_bg.min()/10.
        if spec_min is None:
            if show_darm_threshold:
                spec_min = min([amb_min, min(self.darm_bg)])/4
            else:
                spec_min = amp_min_spec/4
        if spec_max is None:
            spec_max = amp_max_spec*2


        # CREATE FIGURE FOR SPECTRUM PLOTS
        if fig_w is None:
            fig_w = 14
        if fig_h is None:
            fig_h = 6
        fig = plt.figure(figsize=(fig_w, fig_h))


        # PLOT SENSOR SPECTRA
        ax1 = fig.add_subplot(211)

        plt.plot(
            self.freqs, 
            self.sens_inj, 
            color='r', 
            label='Injection', 
            zorder=5
        )
        plt.plot(
            self.freqs, 
            self.sens_bg, 
            color='k', 
            label='Background', 
            zorder=5
        )

        # CREATE LEGEND AND LABELS
        plt.legend()
        ylabel = self.qty + '[' + self.unit.replace('(', '').replace(')', '') + '/Hz$^{1/2}$]'
        plt.ylabel(ylabel, size=18)
        plt.title(self.name.replace('_',' ') + ' - Spectrum', size=20)

        # CUSTOMIZE AXES
        plt.xlim([freq_min, freq_max])
        plt.ylim([spec_sens_min, spec_sens_max])
        if ( (freq_max / freq_min) > 5 ):
            ax1.set_xscale('log', nonposx = 'clip')
        ax1.set_yscale('log', nonposy = 'clip')
        ax1.autoscale(False)
        plt.grid(b=True, which='major', color='0.0', linestyle=':', linewidth=1, zorder=0)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='0.6', linestyle=':', zorder=0)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax1.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        p1 = ax1.get_position()
        p2 = [p1.x0, p1.y0 + 0.02, p1.width, p1.height]
        ax1.set_position(p2)


        # PLOT DARM LINES
        ax2 = fig.add_subplot(212)

        plt.plot(self.freqs, self.darm_inj, '-', color = 'r', label = 'DARM during injection', zorder=3)
        plt.plot(self.freqs, self.darm_bg, '-', color = '0.1', label = 'DARM background', zorder=4)
        if show_darm_threshold == True:
            plt.plot(self.freqs, self.darm_bg / 10., '--', color = 'k', label='DARM background / 10', zorder=2)


        # PLOT ESTIMATED AMBIENT
        if est_amb:
            real_amb = [[],[]]
            upper_amb = [[],[]]

            for y in range(len(self.freqs)):
                if self.flags[y] == 'Upper Limit':
                    upper_amb[1].append(self.ambients[y])
                    upper_amb[0].append(self.freqs[y])
                elif self.flags[y] == 'Real':
                    real_amb[1].append(self.ambients[y])
                    real_amb[0].append(self.freqs[y])

            plt.plot(
                real_amb[0], 
                real_amb[1], 
                'o', 
                color='lime', 
                markersize=6, 
                markeredgewidth=.5, 
                label='Est. Amb.',
                zorder=6
            )
            if upper_lim:
                plt.plot(
                    upper_amb[0], 
                    upper_amb[1], 
                    '^', 
                    markersize = 5, 
                    markerfacecolor='none', 
                    markeredgecolor='0.3',
                    markeredgewidth = .8, 
                    label='Upper Limit Est. Amb.', 
                    zorder=5
                )


        # CREATE LEGEND AND LABELS
        legend = plt.legend(prop={'size':12}, loc=1)
        legend.get_frame().set_alpha(0.5)
        plt.ylabel('DARM ASD [m/Hz$^{1/2}$]', size=18)
        plt.xlabel('Frequency [Hz]', size=18)
        if est_amb:
            plt.title(self.name.replace('_DQ','').replace('_',' ') + ' - Estimated Ambient', size=20)
        else:
            plt.title('DARM - Spectrum', size=20)
        
        
        # FIGURE TEXT
        # Supertitle (timestamp)
        plt.suptitle(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), fontsize = 16, y=1.01)
        # Miscellaneous captions
        str_quiet_time = 'Background Time: '+str(from_gps(self.t_bg)) + '\n({})'.format(self.t_bg) + ' '*11
        str_inj_time = 'Injection Time: '+str(from_gps(self.t_inj)) + '\n' + ' '*25 + '({})'.format(self.t_inj)
        plt.figtext(.9,.01, str_quiet_time, ha='right', fontsize = 12, color = 'b')
        plt.figtext(.1,.01, str_inj_time, fontsize = 12, color = 'b')
        plt.figtext(.5,0.0, 'Band Width: {:1.3f} Hz'.format(self.df), ha='center', va='top', fontsize = 14, color = 'b')
        plt.figtext(.1,.97, 'Injection name:\n{} '.format(path.split('/')[-1].replace('_','\_')),
                    fontsize = 12, color = 'b')
        if est_amb:
            plt.figtext(.9,.99, 'Measured coupling factors: {}'.format(len(real_amb[1])), ha='right', \
                        fontsize = 14, color = 'b')
            if upper_lim:
                plt.figtext(.9,.965, 'Upper limit coupling factors: {}'.format(len(upper_amb[1])), ha='right', \
                            fontsize = 14, color = 'b')


        # CUSTOMIZE AXES
        plt.xlim([freq_min, freq_max])
        plt.ylim([spec_min, spec_max])
        if ( (freq_max / freq_min) > 5 ):
            ax2.set_xscale('log', nonposx = 'clip')
        ax2.set_yscale('log', nonposy = 'clip')
        ax2.autoscale(False)
        plt.grid(b=True, which='major', color='0.0', linestyle=':', linewidth=1, zorder=0)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='0.6', linestyle=':', zorder=0)
        for tick in ax2.xaxis.get_major_ticks():
            tick.label.set_fontsize(18)
        for tick in ax2.yaxis.get_major_ticks():
            tick.label.set_fontsize(18)


        # EXPORT PLOT
        if path is None:
            path = datetime.datetime.fromtimestamp(ts).strftime('DATA_%Y-%m-%d_%H:%M:%S')
        if not os.path.exists(path):
            os.makedirs(path)
        filename = path + '/' + self.name[self.name.index('-')+1:].replace('_DQ','')+'_spectrum.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        return fig

    def to_csv(self, filename, coherence_data=None):
        """
        Save data to CSV file
        
        Parameters
        ----------
        filename : str
            Name of save file.
        coherence_data : dict
            Dictionary of sensor names and their corresponding coherence data
        """
        
        with open(filename, 'w') as file:

            # HEADER
            header = 'frequency,factor,factor_counts,flag,sensBG,darmBG'
            if coherence_data is not None:
                header += 'coherence'
            file.write(header + '\n')

            # DATA
            for i in range(len(self.freqs)):
                # Line of formatted data to be written to csv
                line = '{0:.2f},{1:.2e},{2:.2e},{3},{4:.2e},{5:.2e}'.format(
                    self.freqs[i],
                    self.factors[i],
                    self.factors_in_counts[i],
                    self.flags[i],
                    self.sens_bg[i],
                    self.darm_bg[i]
                )
                if coherence_data is not None:
                    line += ',{:.2e}'.format(coherence_data[i])
                file.write(line + '\n')
                
        return



    


#################################################################################################
#################################################################################################







class CompositeCoupFunc(object):
    """
    Composite coupling function data for a single sensor
    
    Attributes
    ----------
    name : str
        Name of PEM sensor.
    freqs : array
        Frequency array.
    factors : array
        Coupling factors in physical units.
    factors_in_counts : array
        Coupling factors in raw counts.
    flags : array
        Coupling factor flags ('Real', 'Upper limit', 'Thresholds not met', 'No data').
    injections : array
        Names of injections corresponding to each coupling factor.
    sens_bg : array
        ASD of sensor background.
    darm_bg : array
        ASD of DARM background.
    ambients : array
        Estimated ambient ASD for this sensor.
    df : float
        Frequency bandwidth in Hz
    unit : str
        Sensor unit of measurement
    qty : str
        Quantity measured by sensor
    type : str
        Coupling type associated with sensor.
    
    Methods
    -------
    get_channel_info
        Use channel name to get unit of measurement, name of quantity measured by sensor, and coupling type.
    plot
        Plot coupling function from the data.
    ambientplot
        Plot DARM spectrum with estimated ambient of the sensor.
    to_csv
        Save data to csv file.
    """
    
    def __init__(self, name, freqs, factors, factors_in_counts, flags, injections, sens_bg, darm_bg):
        """
        Parameters
        ----------
        name : str
            Name of PEM sensor.
        freqs : array
            Frequency array.
        factors : array
            Coupling factors in physical units.
        factors_in_counts : array
            Coupling factors in raw counts.
        flags : array
            Coupling factor flags ('Real', 'Upper limit', 'Thresholds not met', 'No data').
        injections : array
            Names of injections corresponding to each coupling factor.
        sens_bg : array
            ASD of sensor background.
        darm_bg : array
            ASD of DARM background.
        """
        # Coupling function data
        self.name = name
        self.freqs = np.asarray(freqs)
        self.factors = np.asarray(factors)
        self.factors_in_counts = np.asarray(factors_in_counts)
        self.flags = np.asarray(flags)
        self.injections = np.asarray(injections)
        
        # ASDs
        self.sens_bg = sens_bg
        self.darm_bg = darm_bg[:len(self.freqs)]
        self.ambients = self.factors * self.sens_bg
        
        # Metadata
        self.df = self.freqs[1] - self.freqs[0]
        self.unit, self.qty, self.type = self.get_channel_info()
        
        
    def get_channel_info(self):
        """
        Get basic information about the channel's measurements
        
        Returns
        -------
        unit : str
            Sensor unit of measurement
        qty : str
            Quantity measured by sensor
        type : str
            Coupling type associated with sensor.
        """
        
        units_dict = {'MIC': 'Pa', 'MAG': 'T', 'RADIO': 'ADC', 'SEIS': 'm', \
                  'ISI': 'm', 'ACC': 'm', 'HPI': 'm', 'ADC': 'm'}
        quantity_dict = {'Pa': 'Pressure', 'T': 'Magnetic Field', 'm': 'Displacement', \
                         '(\\mu m/s^2)': 'Acceleration', 'ADC': 'Displacement'}
        type_dict = {'MIC': 'Acoustic', 'MAG': 'Magnetic', 'RADIO': 'RF', 'SEIS': 'Seismic', \
                 'ISI': 'Seismic', 'ACC': 'Vibrational', 'HPI': 'Seismic', 'ADC': 'Vibrational'}

        for x in units_dict.keys():
            if x in self.name:
                return units_dict[x], quantity_dict[units_dict[x]], type_dict[x]

        return 'Counts', '', ''
    
    
    def plot(
        self, filename,
        freq_min, freq_max, factor_min, factor_max,
        in_counts=False, split_injections=True, upper_lim=True,
        fig_w=None, fig_h=None
    ):
        """
        Plot composite coupling function from the data.
        
        Parameters
        ----------
        filename : str
            Target file name.
        freq_min : float
            Minimum frequency (x-axis).
        freq_max : float
            Maximum frequency (x_axis).
        factor_min : float
            Minimum value of coupling factor axis (y-axis).
        factor_max : float
            Maximum value of coupling factor axis (y-axis).
        in_counts : bool
            If True, convert coupling function to counts, and treat data as such
        split_injections : bool
            If True, split data by injection, using different colors for each.
        upper_lim : bool
            If True, include upper limits in the plot. Otherwise just plot measured (real) values.
        fig_w : float or int
            Figure width.
        fig_h : float or int
            Figure height.
        """
        
        # Use factors and sensor units in counts
        if in_counts:
            factors = self.factors_in_counts
            coupling_type = ''
            unit = 'Counts'
        else:
            factors = self.factors
            coupling_type = self.type
            unit = self.unit
        
        # Different plot marker sizes for magnetometers
        if 'MAG' in self.name:
            ms = 10.
            edgew_circle = 0.7
            edgew_triangle = 1.
            ms_triangle = 8.
            lgd_size = 16
        else:
            ms = 5.
            edgew_circle = 0.4
            edgew_triangle = 0.7
            ms_triangle = 3.
            lgd_size = 12
        

        #### CREATE FIGURE ####
        
        if fig_w is None:
            fig_w = 9
        if fig_h is None:
            fig_h = 6
        fig = plt.figure(figsize=(fig_w,fig_h))
        ax = fig.add_subplot(111)


        if split_injections:

            #### COLOR MAP ####

            # Generate & discretize color map for distinguishing injections
            injection_names = sorted(set(self.injections))
            if None in injection_names:
                injection_names.remove(None)
            colors = cm.jet(np.linspace(0.05, 0.95, len(injection_names)))
            colorsDict = {injection: colors[i] for i, injection in enumerate(injection_names)}


            #### PLOTTING LOOP ####

            lgd_patches = []
            for injection in injection_names:

                # Color by injection
                c = colorsDict[injection]
                lgd_patches.append(mpatches.Patch(color=c, label=injection.replace('_', '\_')))

                mask_injection = (self.injections == injection) & (self.factors > 0)
                mask_real = mask_injection & (self.flags == 'Real')
                if in_counts:
                    mask_upper = mask_injection & ((self.flags == 'Upper Limit') | (self.flags == 'Thresholds not met'))
                else:
                    mask_upper = mask_injection & (self.flags == 'Upper Limit')

                plt.plot(
                    self.freqs[mask_real],
                    factors[mask_real],
                    'o',
                    markersize=ms,
                    color=tuple(c[:-1]),
                    markeredgewidth=edgew_circle,
                    zorder=2
                )

                if upper_lim:
                    plt.plot(
                        self.freqs[mask_upper],
                        factors[mask_upper],
                        '^',
                        markersize=ms_triangle,
                        markeredgewidth=edgew_triangle,
                        color='none',
                        markeredgecolor=tuple(c[:-1]),
                        zorder=1
                    )

        else:
            lgd_patches = [mpatches.Patch(color='r',label='Measured Value\n(Signal seen in both\nsensor and DARM)')]

            mask_real = self.flags == 'Real'
            mask_upper = self.flags == 'Upper Limit'
            mask_null = self.flags == 'Thresholds not met'

            ms = 7 if 'MAG' in self.name else 3
            plt.plot(self.freqs[mask_real], factors[mask_real], 'r.', ms=ms, zorder=3)
            if upper_lim:
                lgd_patches.append(mpatches.Patch(color='b',label='Upper Limit\n(Signal seen in\nsensor but not DARM)'))
                plt.plot(self.freqs[mask_upper], factors[mask_upper], '.', color='b', ms=ms, zorder=2)
                if in_counts:
                    plt.plot(self.freqs[mask_null], factors[mask_null], '.', color='c', ms=ms, zorder=1)
                    lgd_patches.append(mpatches.Patch(color='c',label='Maximal Upper Limit\n(Signal not seen\nin sensor nor DARM)'))


        #### SET AXIS STYLE ####

        plt.ylim([factor_min, factor_max])
        plt.xlim([freq_min, freq_max])
        ax.set_yscale('log', nonposy = 'clip')
        ax.set_xscale('log', nonposx = 'clip')
        ax.autoscale(False)
        plt.grid(b=True, which='major', color='0.0', linestyle=':', linewidth=1, zorder=0)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='0.6', linestyle=':', zorder=0)


        #### SET AXIS LABELS ####

        # AXIS NAME LABELS
        plt.ylabel('{0} Coupling [m/{1}]'.format(coupling_type, unit), size=22)
        plt.xlabel('Frequency [Hz]', size=22)

        # AXIS TICK LABELS
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(25)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(25)

        # TITLE
        if 'XYZ' in self.name:
            title = self.name[:-4].replace('_',' ') + ' (Quadrature sum of X, Y, and Z components)\nComposite Coupling Function ' +\
            '(Lowest at each frequency over multiple injection locations)'
        else:
            title = self.name.replace('_DQ','').replace('_',' ') + ' - Composite Coupling Function' + \
            '\n(Lowest at each frequency over multiple injection locations)'
        ttl = plt.title(title, size=22)
        ttl.set_position([.7,1.05])


        #### CREATE LEGEND ####

        lgd = plt.legend(handles=lgd_patches, prop={'size':lgd_size}, bbox_to_anchor=(1.025,1), loc=2, borderaxespad=0.)
        fig.canvas.draw()

        if split_injections:
            # TEXT BELOW LEGEND
            leg_pxls = lgd.get_window_extent()
            ax_pxls = ax.get_window_extent()
            fig_pxls = fig.get_window_extent()
            # Convert back to figure normalized coordinates to create new axis
            pad = 0.01
            ax2 = fig.add_axes([leg_pxls.x0/fig_pxls.width, ax_pxls.y0/fig_pxls.height,\
                                leg_pxls.width/fig_pxls.width, (leg_pxls.y0-ax_pxls.y0)/fig_pxls.height-pad])
            # Eliminate all the tick marks
            ax2.tick_params(axis='both', left='off', top='off', right='off',
                            bottom='off', labelleft='off', labeltop='off',
                            labelright='off', labelbottom='off')
            ax2.axis('off')
            # Add text (finally)
            ax2_pxls = ax2.get_window_extent()
            ax2_pxls = ax2.get_window_extent()
            caption1 = r'$\textbf{Circles}$ represent measured coupling factors, i.e. where a signal was seen in both sensor and DARM.'
            if 'MAG' in self.name:
                caption2 = r'$\textbf{Triangles}$ represent upper limit coupling factors, i.e. where a signal was seen in the sensor only.'
            else:
                caption2 = r'$\textbf{Triangles}$ represent upper limit coupling factors, i.e. where a signal was not seen in DARM.'
            text_width = 10 + max([len(name) for name in injection_names])
            caption1 = '\n'.join(wrap(caption1, text_width))
            caption2 = '\n'.join(wrap(caption2, text_width))
            ax2.text(0.02, .98, caption1 + '\n' + caption2, size=lgd_size, va='top')


        #### EXPORT PLOT ####

        #creating file name and the directory in which it will reside:
        mydpi = fig.get_dpi()
        plt.savefig(filename, bbox_inches='tight', dpi=mydpi*2)
        plt.close()

        return

    
    def ambientplot(
        self, filename,
        freq_min, freq_max, amb_min, amb_max,
        gw_signal='darm', split_injections=True, gwinc=None,
        fig_w=14, fig_h=6
    ):
        """
        Plot DARM spectrum and composite estimated ambient from the data.
        
        Parameters
        ----------
        filename : str
            Target file name.
        freq_min : float
            Minimum frequency (x-axis).
        freq_max : float
            Maximum frequency (x_axis).
        factor_min : float
            Minimum value of coupling factor axis (y-axis).
        factor_max : float
            Maximum value of coupling factor axis (y-axis).
        gw_signal : 'darm' or 'strain'
            Treat GW channel as DARM (meters/Hz^(1/2)) or strain (1/Hz^(1/2)).
        split_injections : bool
            If True, split data by injection, using different colors for each.
        snr : float
            If not None, show a DARM spectrum divided by this SNR for comparison.
        gwinc : tuple of arrays
            Frequency and ASD arrays of GWINC estimate.
        fig_w : float or int
            Figure width.
        fig_h : float or int
            Figure height.
        """
        
        # Different plot marker sizes for magnetometers
        if 'MAG' in self.name:
            ms = 10.
            edgew_circle = 0.7
            edgew_triangle = 1.
            ms_triangle = 8.
            lgd_size = 16
        else:
            ms = 5.
            edgew_circle = 0.4
            edgew_triangle = 0.7
            ms_triangle = 3.
            lgd_size = 12
        
        
        #### CREATE PLOT ####

        # CREATE FIGURE
        if fig_w is None:
            fig_w = 14
        if fig_h is None:
            fig_h = 6
        fig = plt.figure(figsize=(fig_w,fig_h))

        ax = fig.add_subplot(111)

        # COLORMAP FOR DIFFERENT INJECTIONS
        injection_names = sorted(set(self.injections))
        if None in injection_names:
            injection_names.remove(None)
        colors = cm.jet(np.linspace(0.05, .95, len(injection_names)))
        colorsDict = {injection: colors[i] for i, injection in enumerate(injection_names)}

        # DARM VS STRAIN
        if gw_signal.lower() == 'strain':
            darm_bg = self.darm_bg / 4000.
            gwinc[1] = gwinc[1] / 4000.
            ambients = self.ambients / 4000.
        else:
            darm_bg = self.darm_bg
            ambients = self.ambients

        # PLOT DARM
        darmline1, = plt.plot(
            self.freqs,
            darm_bg,
            'k-',
            lw=.7,
            label='DARM background',
            zorder=3
        )
        lgd_patches = [darmline1]

        if gwinc is not None:
            gwincline, = plt.plot(
                gwinc[0],
                gwinc[1],
                color='0.5',
                lw=3,
                label='GWINC, 125 W, No Squeezing',
                zorder=1)
            lgd_patches.append(gwincline)


        # MAIN LOOP FOR PLOTTING AMBIENTS
        if split_injections:
            for injection in injection_names:
                c = colorsDict[injection]
                lgd_patches.append(mpatches.Patch(color=tuple(c[:-1]), label=injection.replace('_','\_')))

                mask_injection = (self.injections == injection) & (ambients > 0)
                mask_real = mask_injection & (self.flags == 'Real')
                mask_upper = mask_injection & (self.flags == 'Upper Limit')

                if np.any(mask_real):
                    plt.plot(
                        self.freqs[mask_real],
                        ambients[mask_real],
                        'o',
                        markersize=ms,
                        color=tuple(c[:-1]),
                        markeredgewidth=edgew_circle,
                        label=injection + ' (Above Threshold)',
                        zorder = 4
                    )

                if np.any(mask_upper):
                    plt.plot(
                        self.freqs[mask_upper],
                        ambients[mask_upper],
                        '^',
                        markersize=ms_triangle,
                        color='none',
                        markeredgewidth=edgew_triangle,
                        markeredgecolor=tuple(c[:-1]),
                        label=injection + ' (Below Threshold)',
                        zorder=3
                    )

        else:
            lgd_patches.append(mpatches.Patch(color='r',label='Measured Value\n(Signal seen in both sensor and DARM)'))
            lgd_patches.append(mpatches.Patch(color='b',label='Upper Limit\n(Signal seen in sensor but not DARM)'))
            lgd_patches.append(mpatches.Patch(color='c',label='Upper Limit\n(Signal not seen in sensor nor DARM)'))

            mask_real = self.flags == 'Real'
            mask_upper = self.flags == 'Upper Limit'

            ms = 7 if 'MAG' in self.name else 3
            plt.plot(self.freqs[mask_real], ambients[mask_real], 'r.', ms=ms, zorder=3)
            plt.plot(self.freqs[mask_upper], ambients[mask_upper], '.', color='b', ms=ms, zorder=2)



        #### SET AXES STYLE ####

        plt.ylim([amb_min, amb_max])
        plt.xlim([freq_min, freq_max])
        ax.set_yscale('log', nonposy = 'clip')
        ax.set_xscale('log', nonposx = 'clip')
        ax.autoscale(False)
        plt.grid(b=True, which='major', color='0.0', linestyle=':', linewidth=1, zorder=0)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='0.6', linestyle=':', zorder=0)


        #### SET LABELS ####

        # AXIS NAME LABELS
        if gw_signal.lower() == 'darm':
            plt.ylabel('DARM ASD [m/Hz$^{1/2}$]', size=22)
        else:
            plt.ylabel('Strain [Hz$^{-1/2}$]', size=22)
        plt.xlabel('Frequency [Hz]', size=22)

        # AXIS TICK LABELS
        for tick, label in zip(ax.xaxis.get_major_ticks(), ax.get_xticklabels()):
            tick.label.set_fontsize(25)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(25)

        # TITLE
        if 'XYZ' in self.name:
            title = self.name[:-4].replace('_',' ') + ' (Quadrature sum of X, Y, and Z components)' +\
            '\nComposite Estimated Ambient'

        else:
            title = self.name.replace('DQ','').replace('_',' ') + ' - Composite Estimated Ambient'
        title += '\n(Obtained from lowest coupling function over multiple injection locations)'
        ttl = plt.title(title, size=22)
        ttl.set_position([.7,1.05])


        #### CREATE LEGEND ####

        lgd = plt.legend(handles=lgd_patches, prop={'size':lgd_size}, bbox_to_anchor=(1.025,1), loc=2, borderaxespad=0.)
        fig.canvas.draw()

        # TEXT BELOW LEGEND
        leg_pxls = lgd.get_window_extent()
        ax_pxls = ax.get_window_extent()
        fig_pxls = fig.get_window_extent()
        # Convert back to figure normalized coordinates to create new axis
        pad = 0.01
        ax2 = fig.add_axes([leg_pxls.x0/fig_pxls.width, ax_pxls.y0/fig_pxls.height,\
                            leg_pxls.width/fig_pxls.width, (leg_pxls.y0-ax_pxls.y0)/fig_pxls.height-pad])
        # Eliminate all the tick marks
        ax2.tick_params(axis='both', left='off', top='off', right='off',
                        bottom='off', labelleft='off', labeltop='off',
                        labelright='off', labelbottom='off')
        ax2.axis('off')
        # Add text (finally)
        ax2_pxls = ax2.get_window_extent()
        text_width = (15 + max([len(name) for name in injection_names])) / .9
        if split_injections:
            caption = 'Ambient estimates are made by multiplying coupling factors by injection-free sensor levels. ' +\
            r'$\textbf{Circles}$ indicate estimates from measured coupling factors, i.e. where the injection signal was seen in ' +\
            'the sensor and in DARM. '
            if 'MAG' in self.name:
                caption += r'$\textbf{Triangles}$ represent upper limit coupling factors, i.e. where a signal was seen in the sensor only. '
            else:
                caption += r'$\textbf{Triangles}$ represent upper limit coupling factors, i.e. where a signal was not seen in DARM. '
            caption += 'For some channels, at certain frequencies the ambient estimates are upper limits ' +\
            'because the ambient level is below the sensor noise floor.'
        else:
            caption = 'Ambient estimates are made by multiplying coupling factors by injection-free sensor levels. ' +\
            'For some channels, at certain frequencies the ambient estimates are upper limits because the ambient ' +\
            'level is below the sensor noise floor.'
        caption = '\n'.join(wrap(caption, text_width))
        ax2.text(0.02, .98, caption, size=lgd_size*.9, va='top')


        #### EXPORT PLOT ####

        my_dpi = fig.get_dpi()
        plt.savefig(filename, bbox_inches='tight', dpi=2*my_dpi)
        plt.close()





    def to_csv(self, filename):
        """
        Save data to a csv file.
        
        Parameters
        ----------
        filename : str
            Target file name.
        """

        with open(filename, 'w') as file:

            header = 'frequency,factor,factor_counts,flag,darm\n'
            file.write(header)
            for i in range(len(self.freqs)):
                row = [
                    '{:.2f}'.format(self.freqs[i]),
                    '{:.2e}'.format(self.factors[i]),
                    '{:.2e}'.format(self.factors_in_counts[i]),
                    self.flags[i],
                    '{:.2e}'.format(self.darm_bg[i])
                    ]
                line = ','.join(row) + '\n'
                file.write(line)

        return