from gwpy.timeseries import TimeSeries
import numpy as np
import pandas as pd
from scipy.io import loadmat
import logging

def get_time_series(channel_names, t_start, t_end, return_failed=False):
    """
    Extract time series data for a list of channel(s).
    
    Parameters
    ----------
    channel_names : list of strings
        Names of channels to load.
    t_start : float, int
        Start time of time series segment.
    t_end : float, int
        End time of time series segment.
    return_failed : {False, True}
        If True, return list of channels that failed to load.
        
    Returns
    -------
    time_series_list : list
        Gwpy TimeSeries objects.
    channels_failed : list
        Names of channels that failed to load, only return if return_failed is True.
    """
    
    time_series_list = []
    channels_failed = []
    for name in channel_names:
        try:
            time_series_list.append(TimeSeries.fetch(name, t_start, t_end))
        except RuntimeError:
            channels_failed.append(name)
            print('')
            logging.warning('Channel failed to load: ' + name + ' at time ' + str(t_start) + ' to ' + str(t_end) + '.')
            print('')
        else:
            logging.info('Channel successfully extracted: ' + name + ' at time ' + str(t_start) + ' to ' + str(t_end) + '.')
    if len(time_series_list) == 0:
        print('')
        logging.warning('No channels were successfully extracted. Check that channel names and times are valid.', warn=True)
        print('')
    elif len(channels_failed) > 0:
        N_failed, N_chans = [len(channels_failed), len(time_series_list)]
        print('Failed to extract {} of {} channels:'.format(N_failed, N_chans))
        for c in channels_failed:
            print(c)
        print('Check that these channel names and times are valid.')
    if return_failed:
        return time_series_list, channels_failed
    else:
        return time_series_list

def get_calibrated_DARM(ifo, gw_channel, t_start, t_end, FFT_time, overlap_time, calibration_file):
    """
    Import DARM channel, convert it to ASD, and calibrate it.

    Parameters
    ----------
    ifo: str
        Interferometer, 'H1' or 'L1'.
    gw_channel : str
        GW channel to use, must be 'strain_channel' or 'deltal_channel'.
    t_start : float, int
        Start time of time series segment.
    t_end : float, int
        End time of time series segment.
    FFT_time : float, int
        FFT time (seconds).
    overlap_time : float, int
        FFT overlap time (seconds).
    calibration_file : str
        Name of DARM calibration file to be used.
    
    Returns
    -------
    darm_asd : FrequencySeries
        Calibrated ASD of imported gravitational wave channel.
    """

    if gw_channel == 'strain_channel':
        strain = TimeSeries.fetch(ifo + ':GDS-CALIB_STRAIN', t_start, t_end)
        strain_asd = strain.asd(FFT_time, overlap_time)
        darm_asd = strain_asd * 4000.    # Strain to DARM, multiply by 4 km
    elif gw_channel == 'deltal_channel':
        darm = TimeSeries.fetch(ifo + ':CAL-DELTAL_EXTERNAL_DQ', t_start, t_end)
        darm_asd = darm.asd(FFT_time, overlap_time)
        # Load DARM calibration file
        try:
            with open(calibration_file,'r') as get_cal:
                lines = get_cal.readlines()
        except IOError:
            print('')
            logging.error('Calibration file ' + calibration_file + ' not found.')
            print('')
            raise
        # Get frequencies and calibration factors
        cal_freqs = np.zeros(len(lines))
        dB_ratios = np.zeros(len(lines))
        for i, line in enumerate(lines):
            values = line.split()
            cal_freqs[i] = float(values[0])
            dB_ratios[i] = float(values[1])
        cal_factors = 10.0 ** (dB_ratios / 20.) # Convert to calibration factors
        # Interpolate calibration factors then apply to DARM ASD
        adjusted_ratios = np.interp(darm_asd.frequencies.value, cal_freqs, cal_factors)
        darm_asd = darm_asd * adjusted_ratios
    else:
        print('')
        logging.error('Invalid input ' + str(gw_channel) + ' for arg gw_channel.')
        print('')
        print('Please correctly specify GW channel calibration method in the configuration file.')
        print('To calibrate from the strain-calibrated channel, H1:GDS-CALIB_STRAIN, write "strain_channel".')
        print('To calibrate from H1:CAL-DELTAL_EXTERNAL_DQ, write "deltal_channel".')
        raise NameError(gw_channel)
    return darm_asd

def get_gwinc(filename):
    """
    Uses scipy.io.loadmat to load gwinc data.
    
    Parameters
    ----------
    filename : str
        Name of gwinc data file.
        
    Returns
    -------
    gwinc : list
        Frequency array and GWINC ASD array.
    """
    logging.info('Importing GWINC data.')
    try:
        gwinc_mat = loadmat(filename)
    except IOError:
        print('')
        logging.warning('GWINC data file ' + filename + ' not found.')
        print('')
        raise
    try:
        gwinc = [
            np.asarray(gwinc_mat['nnn']['Freq'][0][0][0]),
            np.sqrt(np.asarray(gwinc_mat['nnn']['Total'][0][0][0])) * 4000. # Convert strain PSD to DARM ASD
        ]
        logging.info('Successfully imported GWINC data.')
        return gwinc
    except:
        print('')
        logging.warning('GWINC data file ' + filename + ' not formatted correctly.')
        print('')
        return None