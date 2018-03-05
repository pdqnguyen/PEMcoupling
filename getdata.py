from gwpy.timeseries import TimeSeries
from scipy import signal
import numpy as np
import ConfigParser
import os
import time
import datetime
import sys

def get_channel_list(file_name, ifo, station='ALL', search=None, verbose=False):
    """
    Get list of channels from a .txt file.
    
    Parameters
    ----------
    file_name: str
        Name of file containing channel list.
    ifo: str
        Interferometer name, 'H1' or 'L1'.
    station: {'ALL', 'CS', 'EX', 'EY'} optional
        CS, EX, EY, or ALL (default ALL).
    
    Returns
    -------
    channels: list of strings
        Full channel names, e.g. 'H1:PEM-CS_ACC_BEAMTUBE_MCTUBE_Y_DQ'.
    """
    try:
        file_chans = open(file_name.replace(' ',''), 'r')
    except:
        print('\nError: Channel list ' + file_name + ' not found.\n')
        sys.exit()
    # Read and sort channel names
    channels = []
    for c in file_chans.readlines():
        if (station in ['CS', 'EX', 'EY']) and (c[:9] == ifo + ':PEM-' + station):
            channels.append(c.replace('\n', ''))
        elif (station == 'ALL') and (c[:2] == ifo):
            channels.append(c.replace('\n', ''))
    file_chans.close()
    # Report if no channels found
    if len(channels)==0:
        print('\nError: No channels found from ' + file_chans + '.\n')
        sys.exit()
    channels.sort()
    if (search is not None):
        channels = search_channels(channels, search, verbose=verbose)
    if len(channels) == 0:
        print('\nError: No channels found matching search entry.\n')
        sys.exit()
    return channels

def get_channel_names(file_name, ifo, station='ALL'):
    """
    Get list of channels from a .txt file.
    
    Parameters
    ----------
    file_name: str
        Name of file containing channel list.
    ifo: str
        Interferometer name, 'H1' or 'L1'.
    station: {'ALL', 'CS', 'EX', 'EY'} optional
        CS, EX, EY, or ALL (default ALL).
    
    Returns
    -------
    channels: list of strings
        Full channel names, e.g. 'H1:PEM-CS_ACC_BEAMTUBE_MCTUBE_Y_DQ'.
    """
    
    try:
        file_chans = open(file_name.replace(' ',''), 'r')
    except:
        print('\nError: Channel list ' + file_name + ' not found.\n')
        sys.exit()
        
    # Read and sort channel names
    channels = []
    for c in file_chans.readlines():
        if (station in ['CS', 'EX', 'EY']) and (c[:9] == ifo + ':PEM-' + station):
            channels.append(c.replace('\n', ''))
        elif (station == 'ALL') and (c[:2] == ifo):
            channels.append(c.replace('\n', ''))
    file_chans.close()
    
    return channels

def search_channels(channels, search, verbose=False):
    """
    Search from a list of channel names for matching channels.
    
    Parameters
    ----------
    channels: list of strings
        Full channel names, e.g. 'H1:PEM-CS_ACC_BEAMTUBE_MCTUBE_Y_DQ'.
    search: str
        Search entry.
        
    Returns
    -------
    channel_search_results: list of strings
        Channel names matching search entry.
    """
    
    channel_search_results = []
    search = search.upper()
    if (',/' in search) or ('/,' in search) or ('//' in search) or (',,' in search)\
    or search[-1] == '/' or search[-1] == ',':
        print('\nError: Invalid search entry.')
        print('Entry cannot have adjacent commas and slashes or commas and slashes at the end.')
        sys.exit()
        
    search_or = search.split('/') # Read forward slashes as OR
    if verbose:
        print('\nSearching for channels containing:')
        search_print = " OR ".join(["( '"+s_o.replace(",","' AND '")+"' )" for s_o in search_or]).replace("'-","NOT '")
        print(search_print)
    
    for c in channels:
        for s_o in search_or:
            search_and = s_o.split(',') # Read commas as AND
            match = 0
            for s_a in search_and:
                if (s_a[0] != '-') and (s_a in c):
                    match += 1
                elif (s_a[0] == '-') and (s_a[1:] not in c):
                    match+= 1
            if match == len(search_and):
                channel_search_results.append(c)
    return channel_search_results

def get_times(ifo, station='ALL', times=None, injection_list=None, dtt=None, dtt_list=None, injection_search=None):
    """
    Create a table of injection names/times from input times, a list of injection names/times, a DTT file, or a list of DTT files.
    We have multiple ways of getting injection/background times. In order of priority:
        1) Command line option, must be a single bkgd time and a single inj time
        2) Times list, provided as a 3-column (name, bkgd time, inj time) csv file
        4) DTT file(s) in .xml format, parsed by get_times_DTT to create names and times
    Times will be placed into lists, even if only a single pair given
    
    Parameters
    ----------
    ifo: str
        Inteferometer name, 'H1' or 'L1'.
    station = {'ALL', 'CS', 'EX', 'EY'}
        Station name. If 'ALL', get times for all stations at given ifo.
    times: tuple, optional
        Background time and injection time, in GPS. None by default.
    injection_list: str, optional
        Name of config file (.txt) containing injection names and times (separated by station). None by default.
    dtt: str, optional
        Name of a single DTT file to reference for background/injection times. None by default.
    dtt_list: str, optional
        Name of a .txt file containing DTT filenames.
    
    Returns
    -------
    injection_table: list of lists
        Each sub-list contains injection name (str), background time (int), and injection time (int).
    """

    # (1) COMMAND LINE OPTION - SINGLE PAIR OF <BACKGROUND TIME> <INJECTION TIME>
    if times is not None:
        injection_table = [['', int(times[0]), int(times[1])]]
    # (2) COMMAND LINE OR CONFIG OPTION - TXT TABLE OF INJECTION NAMES, BKGD TIMES, INJECTION TIMES
    elif injection_list is not None:
        injection_table = get_times_config(injection_list, ifo, station)
    # (3) COMMAND LINE OPTION - DTT FILE (.XML)
    elif (dtt is not None) or (dtt_list is not None):
        # Get DTT filename(s)
        if dtt is not None:
            # One or more DTT files provided directly in command line
            dtt_names = dtt.split(',')
        elif dtt_list is not None:
            # txt file containing a list of DTT files
            try:
                dtt_list_file = open(dtt_list)
                dtt_names = dtt_list_file.readlines().replace('/n', '')
            except:
                print('\nError: dtt list file ' + dtt_list + ' not found.')
                sys.exit()
        # Search for DTT file(s)
        injection_table = get_times_DTT(dtt_names)
        print('\nInjection data found for {} DTT files.'.format(len(injection_table)))
    else:
        print('\nError: No times or lists given to get_times.\n')
        sys.exit()
    # SEARCH FOR SPECIFIC INJECTIONS BY NAME
    if (injection_search is not None):
        # Get injections (rows of injection_table) that match the given injection_search option
            injection_table = search_injections(injection_table, injection_search, verbose)
    return injection_table

def get_times_config(file_name, ifo, station='ALL'):
    """
    Get injection names and times from a .txt table
        
    Parameters
    ----------
    file_name: str
        Name of times config file (.txt).
    ifo: str
        Inteferometer name, 'H1' or 'L1'.
    station = {'ALL', 'CS', 'EX', 'EY'}
        Station name. If 'ALL', get times for all stations at given ifo.
    
    Returns
    -------
    injection_table: list of lists
        Each sub-list contains injection name (str), background time (int), and injection time (int).
    """
    
    ifo_name = ('LHO' if ifo == 'H1' else 'LLO')
    
    config_file = ConfigParser.ConfigParser()
    try:
        config_file.read(file_name)
    except:
        print('\nError: Could not find the time list file ' + file_name + '.\n')
        sys.exit()
    
    if station == 'ALL':
        injections_raw = ''
        for s in ['CS', 'EX', 'EY']:
            injections_raw += config_file.get(ifo_name, s)
    else:
        injections_raw = config_file.get(ifo_name, station)
    
    injections_split = injections_raw.strip().split('\n')
    injection_table = []
    for row in injections_split:
        name, bg_time, inj_time = row.split(',')
        injection_table.append([name, int(bg_time), int(inj_time)])
    
    return injection_table

def get_times_DTT(file_names, channels=[], return_channels=False):
    """
    Get injection names and times from a single .xml file (saved from DTT).
        
    Parameters
    ----------
    file_name: str
        DTT file name (.xml).
    channels: list of strings
        Channel list to add channels found in DTT file.
    return_channels: {False, True}, optional
        If True, return new channel list.
        
    Returns
    -------
    injection_table: list of lists
        Each sub-list contains injection name (str), background time (int), and injection time (int).
    chans: list of strings
        Input channel list, with new channels appended.
    """
    
    if type(file_names) != list:
        if '*' in file_names:
            # Find all file names matching wildcard file name provided
            try:
                file_names = subprocess.check_output(['ls ' + file_names], shell=True).splitlines()
            except:
                print('\nError: DTT file(s) ' + file_names + ' not found.\n')
                sys.exit()
        else:
            # Just one file name, but treat as a list anyway
            file_names = [file_names]
    
    injection_table = []
    
    for file_name in file_names:
        
        # Get injection name from .xml or vice versa
        injection_name = file_name.split('/')[-1]
        if file_name[-4:] != '.xml':
            file_name += '.xml'
        else:
            injection_name = injection_name.replace('.xml', '')
            
        # Open DTT file, report a warning if not found
        try:
            DTT_file = open(file_name)
        except:
            print('\nWarning: File not found: ' + file)
            return
        
        # Main loop for getting times and channels
        lines = DTT_file.readlines()
        times = {}
        for i, line in enumerate(lines):
            line = line.replace(' ','').replace('\n','')
            
            # Add time to times dictionary if a time is found in this line
            if '"GPS"' in line:
                tname = line[line.index('TimeName') + 10: line.index('Type')-1]
                t = line[line.index('"GPS"') + 6: line.index('</Time>')]
                if (t not in times.values()) & (tname not in times.keys()):
                    times[tname] = str(int(float(t)))
                elif (t not in times.values()):
                    times[tname + '_1'] = str(int(float(t)))
                    
            # Append channel name if found in this line
            if return_channels and ('"MeasurementChannel[' in line):
                channel_name = line[line.index('"channel"') + 10: line.index('</Param>')]
                if len(channel_name) > 0:
                    if ('STRAIN' not in channel_name)\
                    and ('DELTAL' not in channel_name)\
                    and (channel_name not in channels):
                        # Add to list if channel is not GW channel and not in list yet
                        channels.append(c)
                        
        DTT_file.close()
        try:
            injection_table.append([injection_name, int(times['t0']), int(times['t0_1'])])
        except:
            print('\nWarning: No background/injection times found in file ' + xn)
    
    if return_channels:
        return injection_table, channels
    else:
        return injection_table

def search_injections(injection_table, search, verbose=False):
    """
    Search from a table of injection names and times for matching injection names.
    
    Parameters
    ----------
    injection_table: list of lists
        Each row consists of injection name, background time, and injection time for an injection.
    search: str
        Search entry.
        
    Return
    ------
    injection_table_new:
        Rows of input injection_table with name matching search entry.
    """
    if (',/' in search) or ('/,' in search) or ('//' in search) or (',,' in search)\
    or search[-1] == '/' or search[-1] == ',':
        print('\nError: Invalid search entry.')
        print('Entry cannot have adjacent commas and slashes or commas and slashes at the end.')
        sys.exit()
    
    injection_table_new = []

    search_or = search.split('/') # Read forward slashes as OR
    if verbose:
        print('\nSearching for injections containing:')
        search_print = " OR ".join(["( '"+s_o.replace(",","' AND '")+"' )" for s_o in search_or]).replace("'-","NOT '")
        print(search_print)
    for row in injection_table:
        injection_name = row[0].lower()
        for s_o in search_or:
            search_and = s_o.split(',') # Read commas as AND
            match = 0
            for s_a in search_and:
                if (s_a[0] != '-') and (s_a.lower() in injection_name):
                    match += 1
                elif (s_a[0] == '-') and (s_a[1:].lower() not in injection_name):
                    match+= 1
            if match == len(search_and):
                injection_table_new.append(row)
    return injection_table_new

def freq_search(name_inj):
    """
    Get fundamental frequency of an injection if is a magnetic injection.
    """
    freq_search = None # Fundamental frequency of injection lines
    if 'mag' in name_inj.lower():
        # Search for either one of two patterns:
        # (1) 'p' between two numbers (eg 7p1)
        # (2) number followed by 'Hz' (eg 7Hz)
        freq_str = re.search('([0-9]+p[0-9]+)|([0-9]+Hz)', name_inj, re.IGNORECASE).group(0)
        if freq_str is not None:
            # If found, get fundamental frequency from search result
            freq_str = re.sub('Hz', '', freq_str, flags=re.IGNORECASE)
            freq_str = re.sub('p', '.', freq_str, flags=re.IGNORECASE)
            freq_search = float(freq_str)
            print('Fundamental frequency found: ' + freq_str +' Hz.')
            print('Computing coupling factors near integer multiples of this frequency only.')
    return freq_search

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