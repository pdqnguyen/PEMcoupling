""""
FUNCTIONS:
    coupling_function --- Computes coupling functions for multiple sensors, returning a list of CouplingData objects.
    coherence --- Computes GWPy coherence b/w sensor and DARM.
    composite_coupling_function --- Extracts lowest coupling function from lists of coupling functions.        

Usage Notes:
    GWPy is used in coherence function.
"""

from gwpy.timeseries import TimeSeries
from gwpy.time import *
from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
plt.switch_backend('Agg')
from textwrap import wrap
import os
import time
import datetime
import sys
import logging
# pemcoupling modules
try:
    from couplingfunction import CoupFunc, CompositeCoupFunc
    from preprocess import smooth_ASD
except ImportError:
    print('')
    logging.error('Failed to load PEM coupling modules. Make sure you have all of these in the right place!')
    print('')
    raise

def coupling_function(
    ASD_bg, ASD_inj, ASD_darm_bg, ASD_darm_inj,
    darm_factor=2, sens_factor=2, local_max_width=0,
    smooth_params=None, notch_windows = [], fsearch=None, verbose=False
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
    unit = ASD_bg.channel.unit
    calibration = ASD_bg.calibration
    time_bg = ASD_bg.t0
    time_inj = ASD_inj.t0
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
    ###########################################################################################
    # Create a CouplingData object from this data
    cf = CoupFunc(
        name, freqs, factors, flags, sens_bg, darm_bg,
        sens_inj=sens_inj, darm_inj=darm_inj, t_bg=time_bg, t_inj=time_inj,
        unit=unit, calibration=calibration
    )
    return cf

def coherence(
    gw_channel, ifo, sensor_time_series,
    t_start, t_end, FFT_time, overlap_time,
    cohere_plot, thresh1, thresh2, path, ts
):
    """
    Coherence between DARM signal and sensor signal. Notifies user if a certain amount of the data exhibits poor coherence.
    This coherence data will be reported in the csv output file.

    Parameters
    ----------
    gw_channel : str
        Either 'strain_channel' or 'deltal_channel'.
    ifo : str
        Interferometer name, 'H1' or 'L1'.
    sensor_time_series : list
        Sensor TimeSeries objects.
    t_start : float, int
        Start time of time series segment.
    t_end : float, int
        End time of time series segment.
    FFT_time :
        Length (seconds) of FFT averaging windows.
    overlap_time : float, int
        Overlap time (seconds) of FFT averaging windows.
    cohere_plot : bool
        If True, save coherence plot to path.
    thresh1 : float, int
        Coherence threshold. Used only for notifications.
    thresh2 : float, int
        Threshold for how much of coherence data falls below thresh1. Used only for notifications.
    path : str
        Output directory
    ts : time.time object
        Time stamp of main routine; important for output plots and directory name.
    
    Returns
    -------
    coherence_dict : dict
        Dictionary of channel names and coherences (gwpy FrequencySeries object).
    """
    
    if type(sensor_TS) != list:
        sensor_TS = [sensor_TS]
    ts1 = time.time()
    thresh_single = thresh1*(1e-2)
    coherence_dict = {}
    for i, sensor_timeseries in enumerate(sensor_TS):
        # COMPUTE COHERENCE
        if cal_from == 'strain_channel':
            darm = TimeSeries.fetch(interfer+':GDS-CALIB_STRAIN', iT, fT)
        elif cal_from == 'deltal_channel':
            darm = TimeSeries.fetch(interfer+':CAL-DELTAL_EXTERNAL_DQ', iT, fT)
        coherence = sensor_timeseries.coherence(darm, FFT_t, overlap)
        coherences = np.asarray(coherence.value)
        freqs = np.asarray(coherence.frequencies.value)
        coherence_dict[x.name] = coherences    # Save to output dictionary
        # PLOT COHERENCE DATA
        if cohere_plot:            
            # Create plot
            ax = plt.subplot(111)
            plt.plot(freqs, coherences, '-', color = 'b')
            plt.ylabel('Coherence') 
            plt.xlabel('Frequency [Hz]')
            plt.ylim([0, 1])
            plt.xlim([10, max(freqs)])
            ax.set_xscale('log', nonposx = 'clip')
            ax.autoscale(False)
            plt.grid(b=True, which='major',color='0.75',linestyle=':',zorder = 1)
            plt.minorticks_on()
            plt.grid(b=True, which='minor',color='0.65',linestyle=':', zorder = 1)
            plt.title(sensor_timeseries.name[7:].replace('_DQ','') + ' and ' + darm.name[7:].replace('_DQ',''))
            plt.suptitle(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), fontsize = 10)
            # Export plot
            if path is None:
                path = datetime.datetime.fromtimestamp(ts).strftime('DATA_%Y-%m-%d_%H:%M:%S')
            if not os.path.exists(path):
                os.makedirs(path)
            path = path + '/' + sensor_timeseries.name[7:].replace('_DQ', '') + '_coherence_plot'
            plt.savefig(path)
            plt.close()
        # COHERENCE REPORT
        # Reports coherence results if overall coherence falls below threshold
        pts_above_threshold = np.count_nonzero(coherences > thresh_single) # How many points are above thresh1%
        pts_total = float(len(coherences))
        percent_above_threshold = (pts_above_threshold / pts_total) * (10**2)
        if percent_above_threshold < thresh2:
            print('\n')
            print('Less than {:.1f}% of {} data has a coherence of {:.1f}% or greater.'.format(thresh2, sensor_timeseries.name, thresh1))
            print('Only {:.3f}% of the data fit this criteria.'.format(percent_cohere))
        else:
            print('\n')
            print('Coherence thresholds passed for channel {}.'.format(sensor_timeseries.name))
    ts2 = time.time() - ts1
    print('Coherence(s) calculated. (Runtime: {:.3f} s)'.format(ts2))    
    return coherence_dict

def composite_coupling_function(cf_data_list, injection_names, local_max_window=0, freq_lines=None):
    """
    Selects for each frequency bin the "nearest" coupling factor, i.e. the lowest across multiple injection locations.
    
    Parameters
    ----------
    cf_data_list : list
        CoupFunc objects to be processed.
    injection_names : list
        Names of injections.
    local_max_window : int, optional
        Local max window in number of frequency bins.
    freq_lines : dict, optional
        Frequency lines for each injection name.

    Returns
    -------
    comp_cf : CompositeCoupFunc object
        Contains composite coupling function tagged with flags and injection names
    """
    
    if type(cf_data_list) != list or type(injection_names) != list:
        print('\nError: Parameters "cf_data_list" and "injection_names" must be lists.\n')
        sys.exit()
    if len(cf_data_list) != len(injection_names):
        print('\nError: Parameters "cf_data_list" and "injection_names" must be equal length.\n')
        sys.exit()    
    freqs = cf_data_list[0].freqs
    N_rows = len(freqs)
    channel_name = cf_data_list[0].name
    factor_cols = [cf_data.factors for cf_data in cf_data_list]
    factor_counts_cols = [cf_data.factors_in_counts for cf_data in cf_data_list]
    flag_cols = [cf_data.flags for cf_data in cf_data_list]
    injection_cols = [[n]*N_rows for n in injection_names]    
    sens_bg = np.mean([cf_data.sens_bg for cf_data in cf_data_list], axis=0)
    darm_bg = np.mean([cf_data.darm_bg for cf_data in cf_data_list], axis=0)    
    if len(cf_data_list) == 1:
        # Only one injection; trivial
        cf = cf_data_list[0]
        factors = cf.factors
        factors_counts = cf.factors_in_counts
        flags = cf.flags
        injs = np.asarray(injection_cols[0])
        comp_data = CompositeCoupFunc(channel_name, freqs, factors, factors_counts, flags, injs, sens_bg, darm_bg)
        return comp_data    
    # Stack columns in matrices
    matrix_fact = np.column_stack(tuple(factor_cols))
    matrix_fact_counts = np.column_stack(tuple(factor_counts_cols))
    matrix_flag = np.column_stack(tuple(flag_cols))
    matrix_inj = np.column_stack(tuple(injection_cols))    
    # Output lists
    factors = np.zeros(N_rows)
    factors_counts = np.zeros(N_rows)
    flags = np.array(['No data'] * N_rows, dtype=object)
    injs = np.array([None] * N_rows)
    for i in range(N_rows):
        # Assign rows to arrays
        factor_row = np.asarray(matrix_fact[i,:])
        factor_counts_row = np.asarray(matrix_fact_counts[i,:])
        flag_row = np.asarray(matrix_flag[i,:])
        inj_row = np.asarray(matrix_inj[i,:])        
        # Local data, relevant if local minimum search is applied
        i1 = max([0, i - local_max_window])
        i2 = min([i + local_max_window + 1, N_rows])
        local_factors = np.ravel(matrix_fact[ i1 : i2 , : ])
        local_flags = np.ravel(matrix_flag[ i1 : i2 , : ])
        mask_zero = (local_flags != 'No data')
        local_factors_nonzero = local_factors[mask_zero]
        local_flags_nonzero = local_flags[mask_zero]
        # Separate each column into 'Real', 'Upper Limit', and 'Thresholds not met' lists
        flag_types = ['Real', 'Upper Limit', 'Thresholds not met']
        factors_dict, factors_counts_dict, injs_dict = [{}, {}, {}]
        for f in flag_types:
            factors_dict[f] = factor_row[flag_row == f]
            factors_counts_dict[f] = factor_counts_row[flag_row == f]
            injs_dict[f] = inj_row[flag_row == f]
        # Conditions for each type of coupling factor
        flag = 'No data'
        for f in flag_types:
            if len(factors_dict[f]) > 0:
                if min(factors_dict[f]) == min(local_factors_nonzero):
                    flag = f
        # Assign lowest coupling factor and injection; assign flag based on above conditions
        if flag != 'No data':
            idx = np.argmin(factors_dict[flag])
            factors[i] = factors_dict[flag][idx]
            factors_counts[i] = factors_counts_dict[flag][idx]
            flags[i] = flag
            injs[i] = injs_dict[flag][idx]
    # Truncate low-freq injections when higher-freq injections are present
    # This fixes the issue of overlapping upper limits at high-freq in magnetic coupling functions
    if type(freq_lines) is dict:
        if all(name in list(freq_lines.keys()) for name in injection_names):
            freq_lines[None] = 0.
            flines = sorted(set(freq_lines.values()))    # List of injection fundamental frequencies
            freq_ranges = [(flines[i], flines[i+1]) for i in range(1, len(flines) - 1)]
            freq_ranges.append((flines[-1], 1e10))   # Between highest injection freq and "infinity"
            # Loop over frequency range pairs
            for fmin, fmax in freq_ranges:
                # Find upper limits within this freq range
                idx = np.where((flags == 'Upper Limit') & (freqs > fmin) & (freqs <= fmax))[0]
                # Find first instance of an injection whose fundamental freq is fmin
                if len(idx) > 0:
                    f0_arr = np.array([freq_lines[inj] for inj in injs])
                    try:
                        start_idx = np.where(f0_arr >= fmin)[0][0]
                    except IndexError:
                        continue
                    idx = idx[idx >= start_idx]
                    if len(idx) > 0:
                        for i in idx:
                            if f0_arr[i] < fmin:
                                # Zero out this data pt if it's from an injection whose f0 is below this freq range
                                factors[i] = 0.
                                factors_counts[i] = 0.
                                flags[i] = 'No data'
                                injs[i] = None
    comp_cf = CompositeCoupFunc(channel_name, freqs, factors, factors_counts, flags, injs, sens_bg=sens_bg, darm_bg=darm_bg)
    return comp_cf

def max_coupling_function(freqs, factor_df, flag_df):
    """
    Determine maximum coupling function.
    
    Parameters
    ----------
    freqs : array-like
        Frequency array.
    factor_df : pandas.DataFrame
        Dataframe where each column is a coupling function for a channel.
    flag_df : pandas.DataFrame
        Corresponding flags for the coupling factors in factor_df.
        
    Returns
    -------
    max_factor_df : pandas.DataFrame
        Maximum coupling function with corresponding frequencies, flags, and channels.
    """
    
    above_thresh = (flag_df == 'Real') | (flag_df == 'Upper Limit')
    below_thresh = ~above_thresh
    row_above_thresh = np.any(above_thresh, axis=1)
    row_below_thresh = np.all(below_thresh, axis=1)
    factor_df_above_thresh = factor_df.copy()
    factor_df_below_thresh = factor_df.copy()
    factor_df_above_thresh[~above_thresh] = 0.
    factor_df_below_thresh[~below_thresh] = 0.
    max_channels = pd.Series(index=factor_df.index)
    max_channels[row_above_thresh] = factor_df_above_thresh.idxmax(axis=1)
    max_channels[row_below_thresh] = factor_df_below_thresh.idxmax(axis=1)
    has_data = ~pd.isnull(max_channels)
    max_channels = max_channels[has_data]
    max_factor_df = pd.DataFrame(index=factor_df.index[has_data])
    max_factor_df['frequency'] = freqs[has_data]
    max_factor_df['factor'] = factor_df[has_data].lookup(max_channels.index, max_channels.values)
    max_factor_df['flag'] = flag_df[has_data].lookup(max_channels.index, max_channels.values)
    max_factor_df['channel'] = max_channels.values
    return max_factor_df

def max_estimated_ambient(freqs, amb_df, flag_df):
    """
    Determine maximum estimated ambient.
    
    Parameters
    ----------
    freqs : array-like
        Frequency array.
    amb_df : pandas.DataFrame
        Dataframe where each column is an estimated ambient (ASD) for a channel.
    flag_df : pandas.DataFrame
        Corresponding flags for the estimated ambients in amb_df.
        
    Returns
    -------
    max_amb_df : pandas.DataFrame
        Maximum ambient with corresponding frequencies, flags, and channels.
    """
    
    above_thresh = (flag_df == 'Real') | (flag_df == 'Upper Limit')
    below_thresh = ~above_thresh
    row_above_thresh = np.any(above_thresh, axis=1)
    row_below_thresh = np.all(below_thresh, axis=1)
    amb_df_above_thresh = amb_df.copy()
    amb_df_below_thresh = amb_df.copy()
    amb_df_above_thresh[~above_thresh] = 0.
    amb_df_below_thresh[~below_thresh] = 0.
    max_channels = pd.Series(index=amb_df.index)
    max_channels[row_above_thresh] = amb_df_above_thresh.idxmax(axis=1)
    max_channels[row_below_thresh] = amb_df_below_thresh.idxmax(axis=1)
    has_data = ~pd.isnull(max_channels)
    max_channels = max_channels[has_data]
    max_amb_df = pd.DataFrame(index=amb_df.index[has_data])
    max_amb_df['frequency'] = freqs[has_data]
    max_amb_df['amb'] = amb_df[has_data].lookup(max_channels.index, max_channels.values)
    max_amb_df['flag'] = flag_df[has_data].lookup(max_channels.index, max_channels.values)
    max_amb_df['channel'] = max_channels.values
    return max_amb_df

def smooth_comp_data(data, width):
    """
    Gaussian smooth a composite coupling function.
    
    Parameters
    ----------
    data: CompositeCoupFunc object
        Data to be smoothed.
    width: float
        Width (standard deviation) for Gaussian smoothing kernel.
    
    Returns
    -------
    smooth_data: CompositeCouplingData object
        Result of smoothing.
    """
    # Apply smooothing
    factors = gaussian_smooth(data.freqs, data.factors, width, data.flags)
    factors_counts = gaussian_smooth(data.freqs, data.factors_in_counts, width, data.flags)
    sens_bg = gaussian_smooth(data.freqs, data.sens_bg, width, data.flags)
    # Make new data object
    smooth_data = CompositeCoupFunc(
        data.name, data.freqs, factors, factors_counts,
        data.flags, data.injections, sens_bg, data.darm_bg
    )
    return smooth_data

def gaussian_smooth(x, y, width, flags=None):
    """
    Gaussian smooth an ASD.
    
    Parameters
    ----------
    x: array
        Frequencies in Hz.
    y: array
        ASD values to be smoothed.
    width: float
        Standard deviation of Gaussian kernel, in Hz.
    flags: {None, array}, optional
        If provided, use flags to filter data for smoothing (only smoothe measured values and upper limits).
    
    Returns
    -------
    y_smooth: array
        Smoothed ASD values.
    """
    
    if width == 0. or width is None:
        return y
    gaussian = lambda x, mu, sig: np.exp( - (x-mu)**2. / (2.*sig**2.) )
    normGaussian = lambda x, mu, sig: gaussian(x,mu,sig)/(np.nansum(gaussian(x,mu,sig)))    
    y_smooth = np.array(y, copy=True)
    for i in range(len(x)):
        f = flags[i] if flags is not None else 'Real'
        if f == 'Real' or f == 'Upper Limit':
            w = width*x[i]
            dx = x[i+1]-x[i] if (i+1<len(x)) else x[i]-x[i-1]
            w_idx = 1*int(np.ceil(w/dx))
            x_wndw = np.copy(x)[ max([0,i-w_idx]) : min([len(x),i+w_idx]) ]
            y_wndw = np.copy(y)[ max([0,i-w_idx]) : min([len(x),i+w_idx]) ]
            x_wndw[y_wndw == 0.] = np.nan
            y_wndw[y_wndw == 0.] = np.nan
            y_smooth[i] = np.nansum( y_wndw * normGaussian(x_wndw, x[i], w) )
    return y_smooth