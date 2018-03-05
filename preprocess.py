""""
CLASSES:
    ChannelASD -- Contains name, frequencies, amplitude values, and start time for a sensor ASD.
FUNCTIONS:
    get_time_series --- Extracts time series data for a list of channel(s).
    get_calibrated_darm --- Imports and calibrates DARM from whichever state it is in, to displacement.
    reject_saturated --- Removes saturated channels from list of channels.
    notch60Hz --- Applies a filter to time series to notch 60Hz resonances.
    convert_to_ASD --- Converts a list of TimeSeries data into a list of amplitude density spectra.
    quad_sum_ASD --- Generates a quad-sum channel for every set of tri-axial sensors.
    get_calibration_factor --- Determines calibration factor associated with a specific channel.
    calibrate_sensor --- Calibrates the ASDs according to channel sensor type.
    smooth_ASD --- Sliding-average smoothing function for cleaning up noisiness in a spectrum.
Usage Notes:
    GWPy is used for Channel, TimeSeries, and time explicitly. FrequencySeries is also relevant when ASDs are generated.
    scipy.signal is used in notch60Hz for filtering.
"""

from gwpy.timeseries import TimeSeries
import numpy as np
import ConfigParser
from scipy import signal
import os
import time
import datetime
import sys

#===============================
#### OBJECT CLASSES
#===============================

class ChannelASD(object):
    """
    Calibrated amplitude spectral density of a single channel.
    Note: this uses the Channel function from gwpy.detector to retrieve channel info.
    
    Attributes:
    name: str
        Channel name.
    freqs: array
        Frequencies.
    value: array
        Calibrated ASD values.
    t0: float, int
        Start time of channel.
    """
    
    def __init__(self, name, freqs, values, t0=None):
        self.name = name
        self.freqs = np.asarray(freqs)
        self.values = np.asarray(values)
        self.t0 = t0
        self.df = self.freqs[1] - self.freqs[0]

    def unit(self):
        # Sensor unit of measurement
        chan = Channel(self.name)
        sig = chan.signal
        units_dict = {'MIC': 'Pa', 'MAG': 'T', 'RADIO': 'ADC', 'SEIS': 'm', \
                      'ISI': 'm', 'ACC': 'm', 'HPI': 'm'}
        for x in units_dict:
            if x in sig:
                return units_dict[x]
            elif x == chan.system:
                return units_dict[x]
        #If no units are found, return unknown
        return 'Unknown Units'

    def sensor(self):
        # Sensor type
        chan = Channel(self.name)
        sig = chan.signal
        types = {'MIC': 'MIC', 'MAG': 'MAG', 'RADIO': 'RADIO', 'SEIS': 'SEIS', \
                 'ISI':'SEIS', 'HPI':'SEIS', 'ACC': 'ACC'}
        for x in types:
            if x in sig:
                return units_dict[x]
            elif x == channel.system:
                        return units_dict[x]
        # If no units are found, return unknown
        return 'Unknown Type'

    def system(self):
        # Name of sensor system
        return Channel(self.name).system
    
    def crop(self, fmin, fmax):
        """
        Crop ASD between fmin and fmax.
        
        Parameters
        ----------
        fmin: float, int
            Minimum frequency (Hz).
        fmax: float, int
            Maximum frequency (Hz).
        """
        
        try:
            fmin = float(fmin)
            fmax = float(fmax)
        except:
            print('\nError: .crop method for ChannelASD object requires float inputs for fmin, fmax.\n')
        # Determine start and end indices from fmin and fmax
        if fmin > self.freqs[0]:
            start = int(float(fmin - self.freqs[0]) / float(self.df))
        else:
            start = None
        if fmax <= self.freqs[-1]:
            end = int(float(fmax - self.freqs[0]) / float(self.df))
        else:
            end = None
        # Crop data to start and end
        self.freqs = self.freqs[start:end]
        self.values = self.values[start:end]

#===============================
#### DATA IMPORT FUNCTIONS
#===============================

def get_time_series(channel_names, t_start, t_end, return_failed=False):
    """
    Extracts time series data for a list of channel(s).
    
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

#=======================================
#### PREPROCESSING FUNCTIONS
#=======================================

def reject_saturated(time_series_list, verbose=False):
    """
    Rejects saturated sensors (currently just accelerometers) based on if the time series exceeding a threshold value.
    
    Parameters
    ----------
    time_series_list: list
        TimeSeries objects.
    verbose: {False, True}, optional
        If True, report saturated channels.
        
    Returns
    -------
    time_series_unsaturated: list
        TimeSeries objects that did not exceed the saturation threhold.
    """
    
    bad_channels = []
    time_series_unsaturated = []
    for time_series in time_series_list:
        name = time_series.channel.name
        if ('ACC' in name or 'ADC' in name) and (time_series.value.max() >= 32000):
            bad_channels.append(name)
        else:
            time_series_unsaturated.append(time_series)
    
    # Report results
    num_saturated = len(bad_channels)
    num_channels = len(time_series_list)
    if (num_saturated > 0):
        print('{}/{} channels rejected due to signal saturation'.format(num_saturated, num_channels))
        if verbose:
            print('The following ACC channels were rejected: ')
            for bad_chan in bad_channels:
                print(bad_chan)
            
    return time_series_unsaturated

def notch_60Hz(time_series_list):
    """
    Notch-filters 60-Hz line and resonances in MIC, MAG, and DARM signals.
    
    Parameters
    ----------
    time_series_list: list
        TimeSeries object(s).

    Returns
    -------
    time_series_list: list
        TimeSeries object(s) with 60-Hz lines notched.
    """
    
    bw_dict = {'MIC': [.2, .18], 'MAG': [.2, .18], 'DELTAL': [.2,.18], 'CALIB_STRAIN': [.2,.18]} # pass and stop bands
    g_dict = {'MIC': 20, 'MAG': 20, 'DELTAL': 6, 'CALIB_STRAIN': 6} # stop gains
    
    for i, time_series in enumerate(time_series_list):
        nyq = time_series.sample_rate.value / 2 # Nyquist frequency
        
        # Choose pass- and stop-band frequencies based on channel type
        try:
            chan_type = next(key for key in bw_dict.keys() if key in time_series.channel.name)
        except:
            continue
        df1, df2 = bw_dict[chan_type]
        
        # Only notch 60Hz for DARM; notch 60 Hz and resonances for sensors
        if chan_type in ['DELTAL', 'CALIB_STRAIN']:
            frange = [60]
        else:
            frange = range(60, min(1000,int(nyq)), 60)
        for f in frange:
            wp = [(f-df1)/nyq, (f+df1)/nyq]
            ws = [(f-df2)/nyq, (f+df2)/nyq]
            notch_filt = signal.iirdesign(wp=wp, ws=ws, gpass=1, gstop=g_dict[chan_type], ftype='ellip', output='zpk')
            time_series_list[i] = time_series.filter(notch_filt, filtfilt=True)
        
    return time_series_list

def convert_to_ASD(time_series_list, FFT_time, overlap_time):
    """
    Converts a list of TimeSeries data into a list of amplitude density spectra.
    
    Parameters
    ----------
    ts_list: list
        TimeSeries objects.
    FFT_time: float, int
        Length of FFT averaging windows (seconds).
    overlap_time:
        Amount of overlap between FFT averaging windows (seconds).
        
    Returns
    -------
    asd_list: list
        Amplitude spectral densities as gwpy FrequencySeries objects.
    """
    
    asd_list = [ts.asd(FFT_time, overlap_time) for ts in time_series_list]
    return asd_list

def quad_sum_ASD(asd_list, replace_original=False):
    """
    Finds quadrature sum of multi-axial sensors (i.e. magnetometers).
        
    Parameters
    ----------
    asd_list: list
        FrequencySeries objects with '_X', '_Y', or '_Z' in their names
    replace_original: bool
        If True, remove original single-component channels from ASD list.
        
    Returns
    -------
    asd_list_qsum: list
        Quadrature-summed FrequencySeries object(s).
    """
    
    qsum_names = ['MAG_LVEA_INPUTOPTICS', 'MAG_LVEA_OUTPUTOPTICS', 'MAG_LVEA_VERTEX', \
                  'MAG_EBAY_LSCRACK', 'MAG_EBAY_SUSRACK', 'MAG_EBAY_SEIRACK', \
                  'MAG_VEA_FLOOR']
    
    asd_list_qsum = []
    for qs_n in qsum_names:
        asd_axes = [asd.copy() for asd in asd_list if (qs_n in asd.channel.name and 'QUAD' not in asd.channel.name)]
        if len(asd_axes) == 3:
            qsum = np.sqrt( sum([asd**2 for asd in asd_axes]) )
            qsum.name = qsum.name[:-5] + '_XYZ'
            qsum.channel.name = qsum.channel.name[:-5] + '_XYZ'
            asd_list_qsum.append(qsum)
            
    asd_list_out = []
    names = [asd.name for asd in asd_list_qsum]
    names_sorted = sorted(names)
    while len(names_sorted) > 0:
        n = names_sorted.pop(0)
        asd_list_out.append(asd_list_qsum[names.index(n)])
    
    return asd_list_out

def get_calibration_factor(channel_names, ifo, calibration_file):
    """
    Determines calibration factor associated with a specific channel.
    NOTE: this function assumes that the general calibration for unit conversion is done elsewhere in the code.
    
    Parameters
    ----------
    channel_names: list of strings
        Names of channels to be calibrated.
    ifo: str
        Interferometer name, 'H1' or 'L1'.
    calibration_file: str
        Name of calibration file for PEM sensors.
        
    Returns
    -------
    chan_factor: float
        Calibration factor.
    """
    
    # Load calibration file
    try:
        calib_file = open(calibration_file,'r')
    except:
        print('\nError: Sensor calibration file ' + calibration_file + ' not found.\n')
        sys.exit()
    lines = calib_file.readlines()[1:] # Skip header
    calib_file.close()
    
    calib_factor_dict = {}
    for line in lines:
        info = line.split(',')
        if ifo in info[0]:
            chan = info[0]
            calib_raw = info[2]
            if calib_raw != "" and calib_raw != " ":
                num = 0
                factor = ''
                if not calib_raw[0].isdigit():
                    calib_raw = calib_raw[1:]
                while calib_raw[num].isdigit() or (calib_raw[num] == '.'):
                    factor = factor + calib_raw[num]
                    num += 1
                if len(factor) > 0:
                    calib_factor_dict[chan] = factor
                else:
                    calib_factor_dict[chan] = 1
            else:
                calib_factor_dict[chan] = 1
    
    
    # Make sure channel_names is treated as a list for the looping
    if type(channel_names) == str:
        channel_names = [channel_names]
    
    # Match calibration factors with channels
    calibration_factors = {}
    for name in channel_names:
        if name.replace('_DQ', '') in calib_factor_dict.keys():
            calibration_factors[name] = float(calib_factor_dict[name.replace('_DQ', '')])
        else:
            calibration_factors[name] = 1.
    
    return calibration_factors

def calibrate_sensor(asd_list, ifo, calibration_file, verbose=False):
    """
    Calibrates the ASDs according to channel sensor type.
    
    Parameters
    ----------
    asd_list: list
        Uncalibrated PEM sensor ASD(s) as FrequencySeries object(s).
    ifo: str
        Interferometer name, 'H1' or 'L1'.
    verbose: {False, True}, optional
        If True, print progress.
        
    Returns
    -------
    calibrated_asd_list: list
        Calibrated ASD(s) as FrequencySeries object(s).
    calibration_factors: dict
        Channel names and corresponding calibration factors.
    uncalibrated_channels: list
        Channels with no calibration factors found.
    """
    
    ts1 = time.time()
    
    if type(asd_list) == list:
        not_list = False
    else:
        not_list = True
        asd_list= [asd_list]
    
    # Get dictionary of calibration factors from calibration file
    channel_names = [asd.channel.name for asd in asd_list]
    calibration_factors = get_calibration_factor(channel_names, ifo, calibration_file)
        
    calibrated_asd_list = []
    uncalibrated_channels = []
    for raw_asd in asd_list:
        ts11 = time.time()
        channel = raw_asd.channel
        sig = channel.signal
        freqs = np.asarray(raw_asd.frequencies.value)
        amps = raw_asd.value # Hz^(-1/2)
        
        factor = calibration_factors[channel.name]
        
        # Apply calibration according to channel type
            
        if 'MIC' in sig:
            # Records air pressure (Pa)
            factor *= 1e-5    # calib factor is given in atm
            cal_asd = raw_asd * factor
            calibrated_asd_list.append(cal_asd)
            calibration_factors[channel.name] = factor
            
        elif 'MAG' in sig:
            # Records Tesla (T)
            loc = np.argmax(freqs >= 1) # Index of first freq value >= 1. Gives 0 if all freqs < 1.
            factor *= 1e-12    # calib factor is given in picoTesla
            factor_arr = np.zeros_like(freqs)
            for j in range(loc, len(freqs)):
                factor_arr[j] = factor
            cal_asd = raw_asd * factor_arr
            calibrated_asd_list.append(cal_asd)
            calibration_factors[channel.name] = factor
            
        elif 'RADIO' in sig:
            # Records voltage (V)
            cal_asd = raw_asd * factor
            calibrated_asd_list.append(cal_asd)
            calibration_factors[channel.name] = factor

        elif ('SEIS' in sig) or (channel.system == 'ISI') or (channel.system == 'HPI'):
            # Records velocity (m/s) --> Convert to displacement by dividing by omega = (2*pi*freq)
            freqs1 = list(freqs)
            freqs_new = [10**-10]
            freqs_new += freqs1[1:]
            freqs_new = np.asarray(freqs_new)
            denom = freqs_new*(2*np.pi)
            cal_asd = raw_asd * factor / denom
            calibrated_asd_list.append(cal_asd)
            calibration_factors[channel.name] = factor

        elif ('ACC' in sig) or ('ADC' in sig):
            # Records acceleration (m/s^2) --> Convert to displacement by dividing by omega^2 = (2*pi*freq)^2
            freqs1 = list(freqs)
            freqs_new = [10**-10]
            freqs_new += freqs1[1:]
            freqs_new = np.asarray(freqs_new)
            factor *= 1e-6    # calib factor is given in micrometers
            denom = (freqs_new*(2*np.pi))**2    # convert acceleration to displacement (m/s^2 --> m)
            cal_asd = raw_asd * factor / denom
            calibrated_asd_list.append(cal_asd)
            calibration_factors[channel.name] = factor
        
        else:
            uncalibrated_channels.append(channel.name)
            calibrated_asd_list.append(raw_asd)
        
        
        ts12 = time.time() - ts11
        if verbose:
            print('{} calibrated. (Runtime: {:.2f} s)'.format(channel.name, ts12))
    
    ts2 = time.time() - ts1
    if not_list:
        if verbose:
            print('Channel spectrum calibrated. (Runtime: {:.3f} s)'.format(ts2))
        return calibrated_asd_list[0], calibration_factors
    else:
        if verbose:
            print('Channel spectra calibrated. (Runtime: {:.3f} s)'.format(ts2))
        return calibrated_asd_list, calibration_factors, uncalibrated_channels