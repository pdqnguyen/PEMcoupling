""""
FUNCTIONS:
    reject_saturated --- Removes saturated channels from list of channels.
    convert_to_ASD --- Converts a list of TimeSeries data into a list of amplitude density spectra.
    quad_sum_ASD --- Generates a quad-sum channel for every set of tri-axial sensors.
    get_calibration_factors --- Finds calibration factors for list of channels.
    calibrate_sensors --- Calibrates ASDs according to channel sensor type.
    smooth_ASD --- Sliding-average smoothing function for cleaning up noisiness in a spectrum.
"""

from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
import numpy as np
import time
import logging
from pemchannel import PEMChannelASD
from utils import quad_sum_names

def reject_saturated(time_series_list, verbose=False):
    """
    Reject saturated sensors based on whether the time series exceeds 32000 ADC counts.
    
    Parameters
    ----------
    time_series_list : list
        TimeSeries objects.
    verbose : {False, True}, optional
        If True, report saturated channels.
        
    Returns
    -------
    time_series_unsaturated : list
        TimeSeries objects that did not exceed the saturation threhold.
    """
    
    bad_channels = []
    time_series_unsaturated = []
    for time_series in time_series_list:
        name = time_series.channel.name
        if ('ACC' in name or 'ADC' in name) and (time_series.value.max() >= 32000):
            bad_channels.append(name)
            logging.info('Channel rejected due to saturation: ' + name)
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

def convert_to_ASD(time_series_list, FFT_time, overlap_time):
    """
    Convert a list of TimeSeries data into a list of amplitude density spectra in the form of ChannelASD objects.
    
    Parameters
    ----------
    ts_list : list
        Gwpy TimeSeries objects.
    FFT_time : float, int
        Length of FFT averaging windows (seconds).
    overlap_time : float, int
        Amount of overlap between FFT averaging windows (seconds).
        
    Returns
    -------
    asd_list : list
        Amplitude spectral densities as ChannelASD objects.
    """
    
    asd_list = []
    for ts in time_series_list:
        asd1 = ts.asd(FFT_time, overlap_time)
        name = asd1.channel.name.replace('_DQ', '')
        asd2 = PEMChannelASD(name, asd1.frequencies.value, asd1.value, t0=asd1.epoch.value)
#         asd2 = ChannelASD(name, asd1.frequencies.value, asd1.value, t0=asd1.epoch.value)
        logging.info('Channel converted to ASD: ' + name + ' with FFT time ' + str(FFT_time) + ' and overlap time ' + str(overlap_time))
        asd_list.append(asd2)
    return asd_list

def get_calibration_factors(channel_names, calibration_file):
    """
    Read a calibration file to determine the calibration factor of each channel.
    
    Parameters
    ----------
    channel_names : list of strings
        Names of channels to be calibrated.
    calibration_file : str
        Name of calibration file for PEM sensors.
        
    Returns
    -------
    calibration_factors : dict
        Channel names and calibration factors.
    uncalibrated_channesls : list
        Channels with no calibration factors found.
    """
    
    # Clean up channel names
    channel_names = [c.replace('_DQ', '').replace(' ','') for c in channel_names]
    # Reference data for substituting and removing parts of strings
    remove_strings = ['<sup>', '</sup>', '"', '&', '^', ';']
    replace_strings = {
        ' mm': ' x 10-3 m',
        ' mum': ' x 10-6 m',
        ' um': ' x 10-6 m',
        ' mu': ' x 10-6 m',
        ' nm': ' x 10-9 m',
        ' muT': ' x 10-6 T',
        ' uT': ' x 10-6 T',
        ' nT': ' x 10-9 T',
        ' pT': ' x 10-12 T',
        ' mPa': ' x 10-3 Pa',
        ' muPa': ' x 10-6 Pa',
        ' uPa': ' x 10-6 Pa',
    }
    units = ['m', 'm/s', 'm/s2', 'Pa', 'T']
    # Load calibration file
    try:
        with open(calibration_file,'r') as calib_file:
            lines = calib_file.readlines()
    except IOError:
        print('')
        logging.warning('Sensor calibration file ' + calibration_file + ' not found.')
        print('')
        return {}, []
    calibration_factors = {}
    calibration_channels = []
    for line in lines:
        info = line.split(',')
        cal_channel = info[0].replace('_DQ', '').replace(' ','')   # Clean up calibration channel name
        for channel in channel_names:
            # Only grab data for channels in channel_names
            if channel == cal_channel:
                logging.info('Parsing calibration data for channel ' + channel + '.')
                calib_raw_string = info[2]
                for x in remove_strings:
                    calib_raw_string = calib_raw_string.replace(x, '')
                for key, value in replace_strings.items():
                    calib_raw_string = calib_raw_string.replace(key, value)
                if len(calib_raw_string.replace(' ','')) > 0:
                    if any(unit in calib_raw_string for unit in units):
                        for unit in units:
                            if unit in calib_raw_string:
                                calib_raw_string = calib_raw_string[:calib_raw_string.find(unit)].replace(' ', '')
                                break
                    else:
                        i = 0
                        while calib_raw_string[i].isdigit() or (calib_raw_string[i] in ['.', ' ', 'x', '-']):
                            i += 1
                        calib_raw_string = calib_raw_string[:i].replace(' ','')
                    calib_split = calib_raw_string.split('x')
                    calib_factor = float(calib_split[0])
                    if len(calib_split) > 1:
                        if calib_split[1][:2] == '10':
                            exponent = calib_split[1][2:]
                            calib_factor *= 10 ** int(exponent)
                    calibration_factors[channel] = calib_factor
                    logging.info('Calibration factor acquired for channel ' + channel + '.')
                break
    uncalibrated_channels = []
    for c in channel_names:
        if c not in calibration_factors.keys():
            uncalibrated_channels.append(c)
            logging.info('No calibration factor acquired for channel ' + channel + '.')
    return calibration_factors, uncalibrated_channels

def calibrate_sensors(asd_list, calibration_file, verbose=False):
    """
    Calibrate ASDs according to channel sensor type.
    
    Parameters
    ----------
    asd_list : list
        Uncalibrated PEM sensor ASD(s) as FrequencySeries object(s) or ChannelASD objects.
    calibration_file : str
        Name of calibration file for PEM sensors.
    verbose : {False, True}, optional
        If True, print progress.
        
    Returns
    -------
    calibrated_asd_list : list
        Calibrated ASD(s) as FrequencySeries object(s).
    calibration_factors : dict
        Channel names and corresponding calibration factors.
    uncalibrated_channels : list
        Channels with no calibration factors found.
    """
    
    ts1 = time.time()
    
    if type(asd_list) == list:
        not_list = False
    else:
        not_list = True
        asd_list= [asd_list]
    channel_names = [asd.name for asd in asd_list]
    calibration_factors, uncalibrated_channels = get_calibration_factors(channel_names, calibration_file)
    calibrated_asd_list = []
    for i, asd in enumerate(asd_list):
        name = channel_names[i]
        if isinstance(asd, FrequencySeries):
            asd = PEMChannelASD(name, asd.frequencies.value, asd.value, t0=asd.epoch.value)
            logging.info('Channel converted from gwpy FrequencySeries to ChannelASD object: ' + name + '.')
        if name in calibration_factors.keys():
            asd.calibrate(calibration_factors[name])
            logging.info('Channel calibrated: ' + name + '.')
        calibrated_asd_list.append(asd)
    ts2 = time.time() - ts1
    if verbose:
        print('Channel spectra calibrated. (Runtime: {:.3f} s)'.format(ts2))
    return calibrated_asd_list, calibration_factors, uncalibrated_channels

def quad_sum_ASD(asd_list, replace_original=False):
    """
    Create a quadrature sum ASD of multi-axial sensors (i.e. magnetometers).
        
    Parameters
    ----------
    asd_list : list
        FrequencySeries objects with '_X', '_Y', or '_Z' in their names
    replace_original : bool
        If True, remove original single-component channels from ASD list.
        
    Returns
    -------
    asd_list_qsum : list
        Quadrature-summed FrequencySeries object(s).
    """
    
    channel_names = [asd.name for asd in asd_list]
    qsum_dict = quad_sum_names(channel_names)
    asd_list_qsum = []
    for name, axes in qsum_dict.items():
        asd_axes = [asd for asd in asd_list if asd.name in axes]
        logging.info('Generating quad sum ASD for ' + name + '.')
        qsum_values = np.sqrt( sum([asd.values**2 for asd in asd_axes]) )
        qsum = PEMChannelASD(name, asd_axes[0].freqs, qsum_values, t0=asd_axes[0].t0,\
                          unit=asd_axes[0].unit, calibration=asd_axes[0].calibration)
        asd_list_qsum.append(qsum)
    return asd_list_qsum