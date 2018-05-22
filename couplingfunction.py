""""
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
plt.switch_backend('Agg')
from textwrap import wrap
import os
import time
import datetime
import sys
import logging
try:
    from channel import ChannelInfoBase
except ImportError:
    print('')
    logging.error('Failed to load PEM coupling modules. Make sure you have all of these in the right place!')
    print('')
    raise

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
    t_bg : int
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
    
    def __init__(self, name, freqs, factors, flags, sens_bg, darm_bg, sens_inj=None, darm_inj=None,\
                 t_bg=0, t_inj=0, unit='', calibration=None, factors_in_counts=None):
        """        
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
        darm_bg : array
            ASD of DARM during background.
        sens_inj : array
            ASD of sensor during injection.
        darm_inj : array
            ASD of DARM during injection.
        ambients : array
            Estimated ambient ASD for this sensor.
        t_bg : int
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
        self.flags = np.asarray(flags, dtype=object)
        self.sens_bg = np.asarray(sens_bg)
        self.sens_inj = np.asarray(sens_inj)
        self.darm_bg = np.asarray(darm_bg)
        self.darm_inj = np.asarray(darm_inj)
        self.ambients = self.factors * self.sens_bg
        self.t_bg = int(t_bg)
        self.t_inj = int(t_inj)
        self.df = self.freqs[1] - self.freqs[0]
        self.channel = ChannelInfoBase(self.name, unit)
        self.calibration = calibration
        if factors_in_counts is not None:
            self.factors_in_counts = np.asarray(factors_in_counts)
        elif self.calibration is not None:
            self.factors_in_counts = self.factors * self.calibration
        else:
            self.factors_in_counts = self.factors
    
    def _get_factors_in_counts(self, calibration_factor=None):
        """
        Reverse-calibrate a coupling function to convert from physical units to raw counts.
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
        if calibration_factor is None:
            return None
        else:
            sensor_units = {'SEI': 'm/s', 'ACC': 'm/s2', 'MIC': 'Pa', 'MAG': 'T'}
            system_units = {'HPI': 'm/s', 'ISI': 'm/s'}
            if self.channel.sensor in sensor_units.keys():
                calibration_unit = sensor_units[self.channel.sensor]
            elif self.channel.system in system_units.keys():
                calibration_unit = system_units[self.channel.system]
            else:
                calibration_unit = ''
            calibration_array = np.ones_like(self.factors) * calibration_factor
            # Convert m/s and m/s2 into displacement
            if calibration_unit == 'm/s':
                calibration_array /= self.freqs * (2*np.pi) # Divide by omega
            elif calibration_unit == 'm/s2':
                calibration_array /= (self.freqs * (2 * np.pi))**2 # Divide by omega^2
            # Apply calibration and assign new (physical units)
            factors_in_counts = self.factors * calibration_array
        return factors_in_counts
    
    def bin_data(self, width):
        """
        Return a logarithmic binned coupling function.

        Parameters
        ----------
        width : float
            Bin width as fraction of frequency.

        Returns
        -------
        data_binned : CoupFunc object
            New CoupFunc instance containing the binned data.
        """
        scale = 1. + width
        bin_edges = [self.freqs[self.freqs > 0.].min()]
        while bin_edges[-1] <= self.freqs.max():
            bin_edges.append(bin_edges[-1] * scale)
        freqs_binned = np.zeros(len(bin_edges)-1)
        factors_binned = np.zeros_like(freqs_binned)
        factors_counts_binned = np.zeros_like(freqs_binned)
        flags_binned = np.array(['No data'] * len(freqs_binned), dtype=object)
        sens_bg_binned = np.zeros_like(freqs_binned)
        darm_bg_binned = np.zeros_like(freqs_binned)
        for i in range(len(bin_edges)-1):
            freq_min, freq_max = (bin_edges[i], bin_edges[i+1])
            freqs_binned[i] = (freq_max + freq_min) / 2.
            window = (self.freqs >= freq_min) & (self.freqs < freq_max)
            if np.any(window):
                factor_max_idx = self.factors[window].argmax()
                factors_binned[i] = self.factors[window][factor_max_idx]
                factors_counts_binned[i] = self.factors_in_counts[window][factor_max_idx]
                flags_binned[i] = self.flags[window][factor_max_idx]
                sens_bg_binned[i] = self.sens_bg[window][factor_max_idx]
                darm_bg_binned[i] = self.darm_bg[window][factor_max_idx]
            else:
                sens_bg_binned[i] = self.sens_bg[i]
                darm_bg_binned[i] = self.darm_bg[i]
        data_binned = self.__class__(self.name, freqs_binned, factors_binned, flags_binned, sens_bg_binned, darm_bg_binned,\
                                     sens_inj=self.sens_inj, darm_inj=self.darm_inj, t_bg=self.t_bg, t_inj=self.t_inj,\
                                     calibration=self.calibration, factors_in_counts=factors_counts_binned)
        return data_binned
    
    def plot(self, path, in_counts=False, ts=None, upper_lim=True, freq_min=None, freq_max=None,\
             factor_min=None, factor_max=None, fig_w=15, fig_h=6):
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
            unit = self.channel.unit
            type_c = self.channel.coupling
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
        for i in range(len(self.freqs)):
            if (self.flags[i] == 'Upper Limit'):
                upper[1].append(factors[i])
                upper[0].append(self.freqs[i])
            elif (self.flags[i] == 'Real'):
                real[1].append(factors[i])
                real[0].append(self.freqs[i])
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
        plt.xlabel(r'Frequency [Hz]', size=18)
        plt.title(self.name.replace('_DQ','').replace('_',' ') + ' - Coupling Function', size=20)
        plt.suptitle(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), fontsize = 14, y=1.01)
        str_quiet_time = 'Background Time: {}\n({})'.format(from_gps(self.t_bg), self.t_bg)
        str_inj_time = 'Injection Time: {}\n({})'.format(from_gps(self.t_inj), self.t_inj)
        plt.figtext(.95,0, str_quiet_time, ha='right', fontsize = 12, color = 'b')
        plt.figtext(.05,0, str_inj_time, fontsize = 12, color = 'b')
        plt.figtext(.5,0, 'Band Width: {:1.3f} Hz'.format(self.df), ha='center', va='top', fontsize = 12, color = 'b')
        plt.figtext(.05,.96, 'Injection name:\n{} '.format(path.split('/')[-1]), fontsize = 12, color = 'b')
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
        if in_counts:
            filename = filename.replace('_plot','_counts_plot')
        if path is None: 
            path = datetime.datetime.fromtimestamp(ts).strftime('DATA_%Y-%m-%d_%H:%M:%S')
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = path + '/' + filename
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()
        return fig

    def specplot(self, path=None, ts=None, est_amb=True, show_darm_threshold=True, upper_lim=True,\
                 freq_min=None, freq_max=None, spec_min=None, spec_max=None, fig_w=12, fig_h=6):
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
        ylabel = self.channel.qty + '[' + self.channel.unit.replace('(', '').replace(')', '') + '/Hz$^{1/2}$]'
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
    
    def __init__(self, name, freqs, factors, factors_in_counts, flags, injections, sens_bg=None, darm_bg=None, ambients=None, unit=''):
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
        self.darm_bg = darm_bg
        if ambients is None and self.sens_bg is not None:
            self.ambients = self.factors * self.sens_bg
        else:
            self.ambients = ambients
        # Metadata
        self.df = self.freqs[1] - self.freqs[0]
        self.channel = ChannelInfoBase(self.name, unit)
    
    def plot(self, filename, in_counts=False, split_injections=True, upper_lim=True,\
             freq_min=None, freq_max=None, factor_min=None, factor_max=None, fig_w=9, fig_h=6):
        """
        Plot composite coupling function from the data.
        
        Parameters
        ----------
        filename : str
            Target file name.
        in_counts : bool, optional
            If True, convert coupling function to counts, and treat data as such
        split_injections : bool, optional
            If True, split data by injection, using different colors for each.
        upper_lim : bool, optional
            If True, include upper limits in the plot. Otherwise just plot measured (real) values.
        freq_min : float, optional
            Minimum frequency (x-axis).
        freq_max : float, optional
            Maximum frequency (x_axis).
        factor_min : float, optional
            Minimum value of coupling factor axis (y-axis).
        factor_max : float, optional
            Maximum value of coupling factor axis (y-axis).
        fig_w : float or int, optional
            Figure width.
        fig_h : float or int, optional
            Figure height.
        """
        # Use factors and sensor units in counts
        if in_counts:
            factors = self.factors_in_counts
            coupling_type = ''
            unit = 'Counts'
        else:
            factors = self.factors
            coupling_type = self.channel.coupling
            unit = self.channel.unit
            if unit == 'm/s2':
                unit = '(m/s^2)'
            elif unit == 'm/s':
                unit = '(m/s)'
        # Plot limits if not provided
        if freq_min is None:
            freq_min = self.freqs[0]
        if freq_max is None:
            freq_max = self.freqs[-1]
        if factor_min is None:
            factor_min = factors.min() / 1.2
        if factor_max is None:
            factor_max = factors.max() * 1.2
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
                lgd_patches.append(mpatches.Patch(color=c, label=injection))
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
                    lgd_patches.append(mpatches.Patch(color='c',label='Thresholds Not Met\n(Signal not seen\nin sensor nor DARM)'))
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
            caption1 = 'CIRCLES represent measured coupling factors, i.e. where a signal was seen in both sensor and DARM.'
            if 'MAG' in self.name:
                caption2 = 'TRIANGLES represent upper limit coupling factors, i.e. where a signal was seen in the sensor only.'
            else:
                caption2 = 'TRIANGLES represent upper limit coupling factors, i.e. where a signal was not seen in DARM.'
            text_width = 10 + max([len(name) for name in injection_names])
            caption1 = '\n'.join(wrap(caption1, text_width))
            caption2 = '\n'.join(wrap(caption2, text_width))
            ax2.text(0.02, .98, caption1 + '\n' + caption2, size=lgd_size, va='top')
        #### EXPORT PLOT ####
        mydpi = fig.get_dpi()
        plt.savefig(filename, bbox_inches='tight', dpi=mydpi*2)
        plt.close()
        return

    def ambientplot(self, filename, gw_signal='darm', split_injections=True, gwinc=None, darm_data=None,\
                    freq_min=None, freq_max=None, amb_min=None, amb_max=None, fig_w=9, fig_h=6):
        """
        Plot DARM spectrum and composite estimated ambient from the data.
        
        Parameters
        ----------
        filename : str
            Target file name.
        gw_signal : 'darm' or 'strain', optional
            Treat GW channel as DARM (meters/Hz^(1/2)) or strain (1/Hz^(1/2)).
        split_injections : bool, optional
            If True, split data by injection, using different colors for each.
        gwinc : tuple of arrays, optional
            Frequency and ASD arrays of GWINC estimate, for plotting.
        darm_data: tuple of arrays, optional
            Frequency and ASD arrays of DARM spectrum, for plotting.
        freq_min : float, optional
            Minimum frequency (x-axis).
        freq_max : float, optional
            Maximum frequency (x_axis).
        factor_min : float, optional
            Minimum value of coupling factor axis (y-axis).
        factor_max : float, optional
            Maximum value of coupling factor axis (y-axis).
        fig_w : float or int, optional
            Figure width.
        fig_h : float or int, optional
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
            fig_w = 9
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
        # DARM DATA
        if darm_data is None:
            darm_freqs, darm_bg = [self.freqs, self.darm_bg]
        else:
            darm_freqs, darm_bg = darm_data
        # DARM VS STRAIN
        if gw_signal.lower() == 'strain':
            darm_bg = darm_bg / 4000.
            ambients = self.ambients / 4000.
        else:
            ambients = self.ambients
        # PLOT LIMITS
        if freq_min is None:
            freq_min = self.freqs[(self.flags == 'Real') | (self.flags == 'Upper Limit')][0]
        if freq_max is None:
            freq_max = self.freqs[(self.flags == 'Real') | (self.flags == 'Upper Limit')][-1]
        if amb_min is None:
            amb_min = ambients.min() / 1.2
        if amb_max is None:
            amb_max = ambients.max() * 1.2
        # PLOT DARM
        darmline1, = plt.plot(
            darm_freqs,
            darm_bg,
            'k-',
            lw=.7,
            label='DARM background',
            zorder=3
        )
        lgd_patches = [darmline1]
        if gwinc is not None:
            gwinc_freqs = gwinc[0]
            if gw_signal.lower() == 'strain':
                gwinc_amp = gwinc[1] / 4000
            else:
                gwinc_amp = gwinc[1]
            gwincline, = plt.plot(
                gwinc_freqs,
                gwinc_amp,
                color='0.5',
                lw=3,
                label='GWINC, 125 W, No Squeezing',
                zorder=1)
            lgd_patches.append(gwincline)
        # MAIN LOOP FOR PLOTTING AMBIENTS
        if split_injections:
            for injection in injection_names:
                c = colorsDict[injection]
                lgd_patches.append(mpatches.Patch(color=tuple(c[:-1]), label=injection))
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
            lgd_patches.append(mpatches.Patch(color='c',label='Thresholds Not Met\n(Signal not seen in sensor nor DARM)'))
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
            'CIRCLES indicate estimates from measured coupling factors, i.e. where the injection signal was seen in ' +\
            'the sensor and in DARM. '
            if 'MAG' in self.name:
                caption += 'TRIANGLES represent upper limit coupling factors, i.e. where a signal was seen in the sensor only. '
            else:
                caption += 'TRIANGLES represent upper limit coupling factors, i.e. where a signal was not seen in DARM. '
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
        return
        
    def to_csv(self, filename):
        """
        Save data to a csv file.
        
        Parameters
        ----------
        filename : str
            Target file name.
        """
        with open(filename, 'w') as file:
            header = 'frequency,factor,factor_counts,flag,ambient,darm\n'
            file.write(header)
            for i in range(len(self.freqs)):
                row = [
                    '{:.2f}'.format(self.freqs[i]),
                    '{:.2e}'.format(self.factors[i]),
                    '{:.2e}'.format(self.factors_in_counts[i]),
                    self.flags[i],
                    '{:.2e}'.format(self.ambients[i]),
                    '{:.2e}'.format(self.darm_bg[i])
                    ]
                line = ','.join(row) + '\n'
                file.write(line)
        return