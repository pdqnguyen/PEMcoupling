from gwpy.time import from_gps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
plt.switch_backend('Agg')
from textwrap import wrap
import re
import os
import time
import datetime
import sys
import logging
try:
    from pemchannel import PEMChannelASD
    from utils import smooth_ASD
except ImportError:
    print('')
    logging.error('Failed to load PEM coupling modules. Make sure you have all of these in the right place!')
    print('')
    raise

class CoupFunc(PEMChannelASD):
    """
    Composite coupling function data for a single sensor
    
    Attributes
    ----------
    name : str
        Name of PEM sensor.
    freqs : array
        Frequency array.
    values : array
        Coupling factors in physical units.
    values_in_counts : array
        Coupling factors in raw counts.
    flags : array
        Coupling factor flags ('Real', 'Upper limit', 'Thresholds not met', 'No data').
    sens_bg : array
        ASD of sensor background.
    sens_inj : array
        ASD of sensor injection.
    darm_bg : array
        ASD of DARM background.
    darm_inj : array
        ASD of DARM injection.
    ambients : array
        Estimated ambient ASD for this sensor.
    df : float
        Frequency bandwidth in Hz
    calibration : float
        Calibration factor, used to convert back to raw counts if counts coupling function not provided.
    unit : str
        Sensor unit of measurement
    qty : str
        Quantity measured by sensor
    coupling : str
        Coupling type associated with sensor.
    
    Methods
    -------
    load
        Load coupling function from csv/txt file
    compute
        Compute coupling function from PEM and DARM ASDs
    plot
        Plot coupling function from the data.
    ambientplot
        Plot DARM spectrum with estimated ambient of the sensor.
    to_csv
        Save data to csv file.
    """
    
    def __init__(self, name, freqs, values, flags, sens_bg, darm_bg, sens_inj=None, darm_inj=None,\
                 t_bg=0, t_inj=0, injection_name=None, unit='', calibration=None, values_in_counts=None):
        super(CoupFunc, self).__init__(name, freqs, values)
        self.flags = np.asarray(flags, dtype=object)
        self.sens_bg = np.asarray(sens_bg)
        self.sens_inj = np.asarray(sens_inj)
        self.darm_bg = np.asarray(darm_bg)
        self.darm_inj = np.asarray(darm_inj)
        self.ambients = self.values * self.sens_bg
        self.t_bg = int(t_bg)
        self.t_inj = int(t_inj)
        self.injection_name = injection_name
        self.df = self.freqs[1] - self.freqs[0]
        self.calibration = calibration
        if values_in_counts is not None:
            self.values_in_counts = np.asarray(values_in_counts)
        elif self.calibration is not None:
            self.values_in_counts = self.values * self.calibration
        else:
            self.values_in_counts = self.values
    
    @classmethod
    def load(cls, filename, channelname=None):
        """
        Loads csv coupling function data into a CoupFunc object.

        Parameters
        ----------
        filename : str
            Name of coupling function data file.

        Returns
        -------
        cf : couplingfunction.CoupFunc object
        """
        try:
            data = pd.read_csv(filename, delimiter=',')
        except IOError:
            print('')
            logging.warning('Invalid file or file format: ' + filename)
            print('')
        cf = cls(channelname, data.frequency, data.factor, data.flag,\
                 sens_bg=data.sensBG, darm_bg=data.darmBG, sens_inj=data.sensINJ,\
                 values_in_counts=data.factor_counts)
        return cf
    
    @classmethod
    def compute(
        cls, ASD_bg, ASD_inj, ASD_darm_bg, ASD_darm_inj,
        darm_factor=2, sens_factor=2, local_max_width=0,
        smooth_params=None, notch_windows = [], fsearch=None, injection_name=None, verbose=False
    ):
        """
        Calculates coupling factors from sensor spectra and DARM spectra.

        Parameters
        ----------
        ASD_bg : ChannelASD object
            ASD of PEM sensor during background.
        ASD_inj : ChannelASD object
            ASD of PEM sensor during injection.
        ASD_darm_bg : ChannelASD object
            ASD of DARM during background.
        ASD_darm_inj : ChannelASD object
            ASD of DARM during injection.
        darm_factor : float, int, optional
            Coupling factor threshold for determining measured coupling factors vs upper limits. Defaults to 2.
        sens_factor : float, int, optional
            Coupling factor threshold for determining upper limit vs no injection. Defaults to 2.
        local_max_width : float, int, optional
            Width of local max restriction. E.g. if 2, keep only coupling factors that are maxima within +/- 2 Hz.
        smooth_params : tuple, optional
            Injection smooth parameter, background smoothing parameter, and logarithmic smoothing.
        notch_windows : list, optional
            List of notch window frequency pairs (freq min, freq max).
        fsearch : None, float, optional
            Only compute coupling factors near multiples of this frequency. If none, treat as broadband injection.
        verbose : {False, True}, optional
            Print progress.

        Returns
        -------
        cf : CoupFunc object
            Contains sensor name, frequencies, coupling factors, flags ('real', 'upper limit', or 'Thresholds not met'), and other information.
        """

        # Gather relevant data
        name = ASD_bg.name
        unit = str(ASD_bg.unit)
        calibration = ASD_bg.calibration
        t_bg = ASD_bg.t0
        t_inj = ASD_inj.t0
        freqs = ASD_inj.freqs
        bandwidth = ASD_bg.df
        sens_bg = ASD_bg.values
        darm_bg = ASD_darm_bg.values
        sens_inj = ASD_inj.values
        darm_inj = ASD_darm_inj.values
        # "Reference background": this is smoothed w/ same smoothing as injection spectrum.
        # Thresholds use both sens_bg and sens_bg_ref for classifying coupling factors.
        sens_bg_ref = np.copy(sens_bg)
        # OUTPUT ARRAYS
        factors = np.zeros(len(freqs))
        flags = np.array(['No data']*len(freqs), dtype=object)
        loop_range = np.array(range(len(freqs)))
        # SMOOTH SPECTRA
        if smooth_params is not None:
            inj_smooth, base_smooth, log_smooth = smooth_params
            sens_bg_ref = smooth_ASD(freqs, sens_bg, inj_smooth, log_smooth)
            sens_bg = smooth_ASD(freqs, sens_bg, base_smooth, log_smooth)
            darm_bg = smooth_ASD(freqs, darm_bg, base_smooth, log_smooth)
            sens_inj = smooth_ASD(freqs, sens_inj, inj_smooth, log_smooth)
            darm_inj = smooth_ASD(freqs, darm_inj, base_smooth, log_smooth)
        ###########################################################################################
        # DETERMINE LOOP RANGE
        # Zero out low-frequency saturated signals
        # This is done only if a partially saturated signal has produced artificial excess power at low freq
        # The cut-off is applied to the coupling function by replacing sens_inj with sens_bg below the cut-off freq
        # This doesn't affect the raw data, so plots will still show excess power at low freq
        cutoff_idx = 0
        if ('ACC' in name) or ('MIC' in name):
            freq_sat = 10 # Freq below which a large excess in ASD will trigger the coupling function cut-off
            coup_freq_min = 30 # Cut-off frequency (Hz)
            ratio = sens_inj[freqs < freq_sat] / sens_bg[freqs < freq_sat]
            if ratio.mean() > sens_factor:
                # Apply cut-off to data by treating values below the cut-off as upper limits
                while freqs[cutoff_idx] < coup_freq_min:
                    factors[cutoff_idx] = darm_bg[cutoff_idx] / sens_bg[cutoff_idx]
                    flags[cutoff_idx] = 'Thresholds not met'
                    cutoff_idx += 1
            loop_range = loop_range[loop_range >= cutoff_idx]
        # Keep only freqs that are within (0.5*local_max_width) of the specified freqs, given by fsearch
        if (fsearch is not None) and (local_max_width > 0):
            loop_freqs = freqs[loop_range]
            f_mod = loop_freqs % fsearch
            # Distance of each freq from nearest multiple of fsearch
            prox = np.column_stack((f_mod, fsearch - f_mod)).min(axis=1)
            fsearch_mask = prox < (0.5*local_max_width)
            loop_range = loop_range[fsearch_mask]
        # Notch DARM lines and MIC 60 Hz resonances
        if name[10:13] == 'MIC':
            # Add notch windows for microphone 60 Hz resonances
            notches = notch_windows + [[f-2.,f+2.] for f in range(120, int(max(freqs)), 60)]
        else:
            notches = notch_windows
        if len(notches) > 0:
            loop_freqs = freqs[loop_range]
            # Skip freqs that lie b/w any pair of freqs in notch_windows
            notch_mask = sum( (loop_freqs>wn[0]) & (loop_freqs<wn[1]) for wn in notches) < 1
            loop_range = loop_range[notch_mask]
        ##########################################################################################
        # COMPUTE COUPLING FACTORS WHERE APPLICABLE
        sens_ratio = sens_inj / np.maximum(sens_bg, sens_bg_ref)
        darm_ratio = darm_inj / darm_bg
        for i in loop_range:
            # Determine coupling factor status
            sens_above_threshold = sens_ratio[i] > sens_factor
            darm_above_threshold = darm_ratio[i] > darm_factor
            if darm_above_threshold and sens_above_threshold:
                # Sensor and DARM thresholds met --> measureable coupling factor
                factors[i] = np.sqrt(darm_inj[i]**2 - darm_bg[i]**2) / np.sqrt(sens_inj[i]**2 - sens_bg[i]**2)
                flags[i] = 'Real'
            elif sens_above_threshold:
                # Only sensor threshold met --> upper limit coupling factor
                # Can't compute excess power for DARM, but can still do so for sensor
                factors[i] = darm_inj[i] / np.sqrt(sens_inj[i]**2 - sens_bg[i]**2)
                flags[i] = 'Upper Limit'
            elif fsearch is None:
                # Below-threshold upper limits; for broad-band injections (not searching specific frequencies)
                # No excess power in either DARM nor sensor, just assume maximum sensor contribution
                factors[i] = darm_inj[i] / sens_inj[i] # Reproduces DARM in estimated ambient plot
                flags[i] = 'Thresholds not met'
            else:
                # Leave this factor as "No Data"
                pass
        ###########################################################################################
        # LOCALIZED COUPLING FACTORS
        if local_max_width > 0:
            w_locmax = int( local_max_width / bandwidth ) # convert Hz to bins
            for i in range(len(factors)):
                lo = max([0, i - w_locmax])
                hi = min([i + w_locmax + 1, len(factors)])
                local_factors = factors[lo:hi]
                local_flags = flags[lo:hi]
                local_factors_real = [factor for j,factor in enumerate(local_factors) if local_flags[j] == 'Real']
                if 'Real' not in local_flags:
                    # No real values nearby -> keep only if this is a local-max-upper-limit
                    if not (flags[i] == 'Upper Limit' and factors[i] == max(local_factors)):
                        factors[i] = 0
                        flags[i] = 'No data'
                elif not (flags[i] == 'Real' and factors[i] == max(local_factors_real)):
                    # Keep only if local max and real
                    factors[i] = 0
                    flags[i] = 'No data'
        ###########################################################################################
        # OUTLIER REJECTION
        # Clean up coupling functions by demoting marginal values; only for broad-band coupling functions
        elif smooth_params is not None:
            base_smooth = smooth_params[1]
            new_factors = np.copy(factors)
            new_flags = np.copy(flags)
            loop_range_2 = loop_range[(flags[loop_range] == 'Real') | (flags[loop_range] == 'Upper Limit')]
            for i in loop_range_2:
                N = np.round(freqs[i] * base_smooth / 100).astype(int) # Number of nearby values to compare to
                lower_ = max([0, i-int(N/2)])
                upper_ = min([len(freqs), i+N-int(N/2)+1])
                nearby_flags = flags[lower_ : upper_] # Flags of nearby coupling factors
                if sum(flags[i] == nearby_flags) < (N/2.):
                    # This is an outlier, demote this point to a lower flag
                    if (flags[i] == 'Real'):
                        new_factors[i] = darm_inj[i] / np.sqrt(sens_inj[i]**2 - sens_bg[i]**2)
                        new_flags[i] = 'Upper Limit'
                    elif (flags[i] == 'Upper Limit') and (fsearch is None):
                        new_factors[i] = darm_inj[i] / sens_inj[i]
                        new_flags[i] = 'Thresholds not met'
            factors = new_factors
            flags = new_flags
        cf = cls(name, freqs, factors, flags, sens_bg, darm_bg, sens_inj=sens_inj, darm_inj=darm_inj,\
                 t_bg=t_bg, t_inj=t_inj, injection_name=injection_name, unit=unit, calibration=calibration)
        return cf
    
    def plot(self, filename, in_counts=False, ts=None, upper_lim=True, freq_min=None, freq_max=None,\
             factor_min=None, factor_max=None, fig_w=15, fig_h=6):
        """
        Export a coupling function plot from the data
        
        Parameters
        ----------
        filename : str
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
            factors = self.values_in_counts
            unit = 'Counts'
            type_c = ''
        else:
            factors = self.values
            unit = str(self.unit)
            type_c = self.coupling
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
        plt.figtext(.05,.96, 'Injection name:\n{} '.format(self.injection_name), fontsize = 12, color = 'b')
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
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        return fig

    def specplot(self, filename, ts=None, est_amb=True, show_darm_threshold=True, upper_lim=True,\
                 freq_min=None, freq_max=None, spec_min=None, spec_max=None, fig_w=12, fig_h=6):
        """
        Export an estimated ambient plot from the data.
        
        Parameters
        ----------
        filename : str
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
        try:
            float(freq_min)
        except:
            freq_min = self.freqs[0]
        try:
            float(freq_max)
        except:
            freq_max = self.freqs[-1]
        loc_freq_min, loc_freq_max = 0, -1
        while self.freqs[loc_freq_min] < freq_min:
            loc_freq_min += 1
        while self.freqs[loc_freq_max] > freq_max:
            loc_freq_max -= 1
        darm_y = list(self.darm_bg[loc_freq_min:loc_freq_max]) + list(self.darm_inj[loc_freq_min:loc_freq_max])
        amp_max_spec = max(darm_y)
        amp_min_spec = min(darm_y)
        # Y-AXIS LIMITS FOR SENSOR SPECTROGRAM
        spec_sens_min = np.min(self.sens_bg[loc_freq_min:loc_freq_max]) / 2
        spec_sens_max = np.max(self.sens_inj[loc_freq_min:loc_freq_max]) * 4
        # Y-AXIS LIMITS FOR DARM/EST AMB SPECTROGRAM
        amb_values = self.ambients[np.isfinite(self.ambients) & (self.ambients>0)]
        amb_min = np.min(amb_values) if np.any(amb_values) else self.darm_bg.min()/10.
        try:
            float(spec_min)
        except TypeError:
            if show_darm_threshold:
                spec_min = min([amb_min, min(self.darm_bg)])/4
            else:
                spec_min = amp_min_spec/4
        try:
            float(spec_max)
        except TypeError:
            spec_max = amp_max_spec*2
        # CREATE FIGURE FOR SPECTRUM PLOTS
        try:
            float(fig_w)
        except TypeError:
            fig_w = 14
        try:
            float(fig_h)
        except TypeError:
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
        ylabel = self.qty + '[' + str(self.unit).replace('(', '').replace(')', '') + '/Hz$^{1/2}$]'
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
        plt.figtext(.1,.97, 'Injection name:\n{} '.format(self.injection_name), fontsize = 12, color = 'b')
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
            header = 'frequency,factor,factor_counts,flag,sensINJ,sensBG,darmBG'
            if coherence_data is not None:
                header += 'coherence'
            file.write(header + '\n')
            # DATA
            for i in range(len(self.freqs)):
                # Line of formatted data to be written to csv
                line = '{0:.2f},{1:.2e},{2:.2e},{3},{4:.2e},{5:.2e},{6:.2e}'.format(
                    self.freqs[i],
                    self.values[i],
                    self.values_in_counts[i],
                    self.flags[i],
                    self.sens_inj[i],
                    self.sens_bg[i],
                    self.darm_bg[i]
                )
                if coherence_data is not None:
                    line += ',{:.2e}'.format(coherence_data[i])
                file.write(line + '\n')
        return