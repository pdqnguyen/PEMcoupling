from gwpy.timeseries import TimeSeries
from scipy import signal
import numpy as np
import ConfigParser
import os
import time
import datetime
import sys

def get_time_series(channel_names, t_start, t_end, return_failed=False):
    """
    Extracts time series data for a list of channel(s)
    
    Parameters
    ----------
    channel_names: list of strings
        Names of channels to load.
    t_start: float, int
        Start time of time series segment.
    t_end: float, int
        End time of time series segment.
    return_failed: {False, True}
        If True, return list of channels that failed to load.
        
    Returns
    -------
    time_series_list: list
        Gwpy TimeSeries objects.
    channels_failed: list
        Names of channels that failed to load.
    """
    
    time_series_list = []
    channels_failed = []
    for name in channel_names:
        try:
            time_series_list.append(TimeSeries.fetch(name, t_start, t_end))
        except RuntimeError:
            channels_failed.append(name)
        
    if len(time_series_list) == 0:
        print('\nWarning: No channels were successfully extracted.')
        print('Check that channel names and times are valid.')
    elif len(channels_failed) > 0:
        print('\nWarning: Failed to extract {} of {} channels:'.format(len(chans_failed), len(l)))
        for c in channels_failed:
            print(c)
        print('Check that channel names and times are valid.')

    if return_failed:
        return time_series_list, channels_failed
    else:
        return time_series_list

def get_calibrated_DARM(ifo, gw_channel, t_start, t_end, FFT_time, overlap_time, calibration_file):
    """
    Imports and calibrates DARM from whichever state it is in, to displacement.

    Parameters
    ----------
    cal_from: str
        GW channel to use, must be 'strain_channel' or 'deltal_channel'.
    t_start: float, int
        Start time of time series segment.
    t_end: float, int
        End time of time series segment.
    FFT_t: float
        FFT time (seconds).
    overlap_t: float
        FFT overlap time (seconds).
    cal_file: str
        Name of DARM calibration file to be used.
    
    Returns
    -------
    calibrated_darmASD: FrequencySeries
        Calibrated ASD of imported gravitational wave channel.
    """

    if gw_channel == 'strain_channel':
        
        strain = TimeSeries.fetch(ifo + ':GDS-CALIB_STRAIN', t_start, t_end)
        strainASD = strain.asd(FFT_time, overlap_time)
        darmASD = strainASD * 4000.    # Strain to DARM, multiply by 4 km

    elif gw_channel == 'deltal_channel':
        
        darm = TimeSeries.fetch(ifo + ':CAL-DELTAL_EXTERNAL_DQ', t_start, t_end)
        darmASD = darm.asd(FFT_time, overlap_time)
        
        # Load DARM calibration file
        try:
            get_cal = open(calibration_file,'r')
        except:
            print('\nError: Calibration file ' + calibration_file + ' not found.\n')
            sys.exit()
        lines = get_cal.readlines()
        get_cal.close()
        
        # Get frequencies and calibration factors
        cal_freqs = np.zeros(len(lines))
        dB_ratios = np.zeros(len(lines))
        for i, line in enumerate(lines):
            values = line.split()
            cal_freqs[i] = float(values[0])
            dB_ratios[i] = float(values[1])
        cal_factors = 10.0 ** (dB_ratios / 20.) # Convert to calibration factors
        
        # Interpolate calibration factors then apply to DARM ASD
        adjusted_ratios = np.interp(darmASD.frequencies.value, cal_freqs, cal_factors)
        darmASD = darmASD * adjusted_ratios

    else:
        print('Please correctly specify GW channel calibration method in the configuration file.')
        print('To calibrate from the strain-calibrated channel, H1:GDS-CALIB_STRAIN, write "strain_channel".')
        print('To calibrate from H1:CAL-DELTAL_EXTERNAL_DQ, write "deltal_channel".')
        sys.exit()
    
    return darmASD