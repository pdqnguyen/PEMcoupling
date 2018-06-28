"""
Functions for determining parameters for coupling function calculations in PEMcoupling.py, e.g. FFT parameters, coupling factor thresholds, channel lists, and injection lists.
"""

import ConfigParser
import sys
import subprocess
import re
import logging

def get_arg_params(args):
    """
    Parse arguments, terminating the script if user input is invalid.
    
    Parameters
    ----------
    args : tuple
        Arguments from argparser.
    
    Returns
    -------
    ifo : str
        Interferometer name.
    station : str
        Station name, CS, EX, EY, or ALL.
    config_name : str
        Name of configuration file to use, based on injection_type.
    """
    
    logging.info('Parsing arguments.')
    if len(args) == 0:
        print('')
        logging.warning('No arguments given. Running interactive mode.')
        print('')
        ifo_input = raw_input('Interferometer (H1/L1): ').upper()
        station_input = raw_input('Station (CS/EX/EY/ALL): ').upper()
        injection_type = raw_input('Interferometer (MAGNETIC/VIBRATIONAL/ACOUSTIC): ').upper()
    elif len(args) != 3:
        print('')
        logging.error('Exactly 3 arguments required (interferometer, station, injection type).')
        print('')
        sys.exit()
    else:
        ifo_input, station_input, injection_type = [s.upper() for s in args]
    # Make sure interferometer input is valid
    ifo_dict = {'H1': 'H1', 'LHO': 'H1', 'L1': 'L1', 'LLO': 'L1'}
    try:
        ifo = ifo_dict[ifo_input]
    except KeyError:
        print('')
        logging.error('Argument "ifo" must be one of "H1", "LHO", "L1", or "LHO" (not case-sensitive).')
        print('')
        sys.exit()
    # Make sure station input is valid
    if station_input in ['CS', 'EX', 'EY', 'ALL']:
        station = station_input
    else:
        print('')
        logging.error('Argument "station" must be one of "CS", "EX", "EY", or "ALL" (not case-sensitive).')
        print('')
        sys.exit()
    # Make sure injection type is valid
    if injection_type in ['MAG', 'MAGNETIC']:
        config_name = 'config_files/config_magnetic.txt'
    elif injection_type in ['VIB', 'VIBRATIONAL', 'ACOUSTIC']:
        config_name = 'config_files/config_vibrational.txt'
    else:
        print('')
        logging.error('Argument "injection_type" must be one of "mag", "magnetic", "vib", "vibrational", or '+\
                      '"acoustic" (not case-sensitive).')
        print('')
        sys.exit()
    return ifo, station, config_name

def get_config_params(config_name):
    """
    Parse config file and return a dictionary of the config options and values.
    
    Parameters
    ----------
    config_name : str
        Name of config file.
    
    Returns
    -------
    config_dict : dict
        Dictionary containing a sub-dictionary of config options/values for each section in the config file.
    """
    
    # Read config file
    config = ConfigParser.ConfigParser()
    logging.info('Opening config file.')
    try:
        config.read(config_name)
    except IOError:
        print('')
        logging.warning('Configuration file ' + config_name + ' not found.')
        print('')
        return {}
    # Check for missing/empty config file
    logging.info('Checking if missing/empty config file.')
    if len(config.sections()) == 0:
        print('')
        logging.warning('Configuration file ' + config_name + ' not found.')
        print('')
        return {}
    if all( [len(config.options(x)) == 0 for x in config.sections()] ):
        print('')
        logging.warning('Configuration file ' + config_name + ' is empty.')
        print('')
        return {}
    # Read config inputs into sub-dictionaries of config_dict (separated by config section)
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {option: config.get(section, option) for option in config.options(section)}
    # Categories for converting numerical config options to appropriate types
    float_options = [
        'darm_factor_threshold', 'sens_factor_threshold', 'local_max_width',\
        'duration', 'fft_overlap_pct', 'band_width',\
        'coup_fig_height', 'coup_fig_width', 'spec_fig_height', 'spec_fig_width',\
        'data_freq_min', 'data_freq_max', 'plot_freq_min', 'plot_freq_max',\
        'coup_y_min', 'coup_y_max', 'spec_y_min', 'spec_y_max',\
        'coherence_threshold', 'percent_data_threshold',\
        'ratio_min_frequency', 'ratio_max_frequency',\
        'coupling_function_binning',\
        'comp_fig_height', 'comp_fig_width',\
        'comp_freq_min', 'comp_freq_max',\
        'comp_y_min', 'comp_y_max'
    ]
    int_options = ['fft_avg']
    bool_options = [
        'fft_rounding', 'smoothing_log', 'quad_sum',\
        'darm/10', 'spec_plot', 'spec_plot_name', 'est_amb_plot',\
        'coherence_calculator', 'coherence_spectrum_plot',\
        'ratio_plot', 'ratio_avg', 'ratio_max',\
        'composite_coupling', 'upper_lim', 'comp_est_amb_plot'\
    ]
    float_required = ['darm_factor_threshold', 'sens_factor_threshold' ]
    int_required = ['fft_avg']
    # Convert numerical and boolean config options to appropriate types (float, int, bool)
    logging.info('Parsing numerical and boolean config options.')
    for sub_dict in config_dict.values():
        for option, value in sub_dict.items():
            if option in float_options:
                try:
                    sub_dict[option] = float(value)
                except ValueError:
                    sub_dict[option] = None
                    if option in float_required:
                        print('')
                        logging.error('Float input required for option ' + option + '.')
                        raise
            elif option in int_options:
                try:
                    sub_dict[option] = int(value)
                except ValueError:
                    sub_dict[option] = None
                    if option in int_required:
                        print('')
                        logging.error('Integer input required for option ' + option + '.')
                        print('')
                        raise
            elif option in bool_options:
                sub_dict[option] = (value.lower() in ['on', 'true', 'yes'])
            elif '_smoothing' in option:
                try:
                    sub_dict[option] = [float(v) for v in value.split(',')]
                except ValueError:
                    print('')
                    logging.error('Input for ' + option + ' should be list of three floats (e.g. ACC_smothing: 5, 0.5, 0.5).')
                    print('')
                    raise
            elif 'notch' in option:
                if value is not None:
                    try:
                        with open(value) as notch_file:
                            notch_data = notch_file.read().split('\n')
                            notch_list = []
                            for row in notch_data:
                                try:
                                    notch_list.append(list(map(float, row.split(','))))
                                except ValueError:
                                    print('')
                                    logging.error('Data in DARM notch files must be floats.')
                                    print('')
                                    raise
                            sub_dict[option] = notch_list
                    except IOError:
                        logging.error('Cannot open file ' + value + ' provided by ' + option + ' in config file.')
                        sub_dict[option] = [[]]
                else:
                    sub_dict[option] = []
    return config_dict

def get_channel_list(file_name, ifo, station='ALL', search=None, verbose=False):
    """
    Get list of channels from a .txt file.
    
    Parameters
    ----------
    file_name : str
        Name of file containing channel list.
    ifo : str
        Interferometer name, 'H1' or 'L1'.
    station : {'ALL', 'CS', 'EX', 'EY'} optional
        CS, EX, EY, or ALL (default ALL).
    
    Returns
    -------
    channels: list of strings
        Full channel names, e.g. 'H1:PEM-CS_ACC_BEAMTUBE_MCTUBE_Y_DQ'.
    """
    
    channels = get_channel_names(file_name, ifo, station)
    channels.sort()
    if (search is not None):
        logging.info('Performing channel search.')
        channels = search_channels(channels, search, verbose=verbose)
    if len(channels) == 0:
        print('')
        logging.warning('No channels found matching search entry.')
        print('')
    return channels

def get_channel_names(file_name, ifo, station='ALL'):
    """
    Get list of channels from a .txt file.
    
    Parameters
    ----------
    file_name : str
        Name of file containing channel list.
    ifo : str
        Interferometer name, 'H1' or 'L1'.
    station : {'ALL', 'CS', 'EX', 'EY'} optional
        CS, EX, EY, or ALL (default ALL).
    
    Returns
    -------
    channels : list of strings
        Full channel names, e.g. 'H1:PEM-CS_ACC_BEAMTUBE_MCTUBE_Y_DQ'.
    """
    
    logging.info('Opening channel list file.')
    try:
        with open(file_name.replace(' ',''), 'r') as file:
            lines = file.readlines()
    except IOError:
        print('')
        logging.error('Channel list ' + file_name + ' not found.')
        print('')
        return []
    # Read and sort channel names
    channels = []
    for c in lines:
        if (station in ['CS', 'EX', 'EY']) and (c[:9] == ifo + ':PEM-' + station):
            channels.append(c.replace('\n', ''))
        elif (station == 'CS') and (c[:6] == ifo + ':IMC'):
            channels.append(c.replace('\n', ''))
        elif (station == 'ALL') and (c[:2] == ifo):
            channels.append(c.replace('\n', ''))
    if len(channels)==0:
        print('')
        logging.warning('No channels found from ' + file_chans + '.')
        print('')
        return channels
    channels.sort()
    return channels

def search_channels(channels, search, verbose=False):
    """
    Search from a list of channel names for matching channels.
    
    Parameters
    ----------
    channels : list of strings
        Full channel names, e.g. 'H1:PEM-CS_ACC_BEAMTUBE_MCTUBE_Y_DQ'.
    search : str
        Search entry.
        
    Returns
    -------
    channel_search_results : list of strings
        Channel names matching search entry.
    """
    
    channel_search_results = []
    search = search.upper()
    if (',/' in search) or ('/,' in search) or ('//' in search) or (',,' in search)\
    or search[-1] == '/' or search[-1] == ',':
        print('')
        logging.error('Invalid search entry. Entry cannot have adjacent commas and slashes or commas and slashes at the end.')
        print('')
        return channels
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

def get_times(ifo, station='ALL', times=None, dtt=None, dtt_list=None, injection_list=None, injection_search=None):
    """
    Create a table of injection names/times from input times, a list of injection names/times, a DTT file, or a list of DTT files.
    We have multiple ways of getting injection/background times. In order of priority:
        1) Command line option, must be a single bkgd time and a single inj time
        2) Times list, provided as a 3-column (name, bkgd time, inj time) csv file
        4) DTT file(s) in .xml format, parsed by get_times_DTT to create names and times
    Times will be placed into lists, even if only a single pair given
    
    Parameters
    ----------
    ifo : str
        Inteferometer name, 'H1' or 'L1'.
    station = {'ALL', 'CS', 'EX', 'EY'}
        Station name. If 'ALL', get times for all stations at given ifo.
    times : tuple, optional
        Background time and injection time, in GPS. None by default.
    dtt : str, optional
        Name of a single DTT file to reference for background/injection times. None by default.
    dtt_list : str, optional
        Name of a .txt file containing DTT filenames.
    injection_list : str, optional
        Name of config file (.txt) containing injection names and times (separated by station). None by default.
    injection_search : str, optional
        Search entry to feed into search_injections.
    
    Returns
    -------
    injection_table : list of lists
        Each sub-list contains injection name (str), background time (int), and injection time (int).
    """

    # (1) COMMAND LINE OPTION - SINGLE PAIR OF <BACKGROUND TIME> <INJECTION TIME>
    if times is not None:
        logging.info('Getting times from command line input.')
        injection_table = [['', int(times[0]), int(times[1])]]
    # (2) COMMAND LINE OPTION - DTT FILE (.XML)
    elif (dtt is not None) or (dtt_list is not None):
        logging.info('Getting DTT files from command line input.')
        # Get DTT filename(s)
        if dtt is not None:
            if ('.xml' in dtt):
                # One or more DTT files provided directly in command line, split by commas
                dtt_names = dtt.split(',')
            elif '*' in dtt:
                # Wildcard in command line input; use ls to find DTT files
                try:
                    dtt_names = subprocess.check_output(['ls ' + dtt], shell=True).splitlines()
                except:
                    dtt_names = []
                    print('')
                    logging.error('DTT file(s) ' + str(dtt) + ' not found.')
                    print('')
                    sys.exit()
            else:
                # Name of directory containing DTT files
                try:
                    if dtt[-1] != '/':
                        dtt += '/'
                    dtt_names = subprocess.check_output(['ls ' + dtt + '*.xml'], shell=True).splitlines()
                except:
                    dtt_names = []
                    print('')
                    logging.error('DTT directory' + str(dtt) + ' not found, or it does not contain .xml files.')
                    print('')
                    sys.exit()
        elif dtt_list is not None:
            # txt file containing a list of DTT files
            try:
                with open(dtt_list) as dtt_list_file:
                    dtt_names = dtt_list_file.readlines()
                    if len(dtt_names) == 0:
                        print('')
                        logging.error('DTT files found.')
                        print('')
            except IOError:
                dtt_names = []
                print('')
                logging.error('DTT list file ' + dtt_list + ' not found.')
                print('')
                raise
        # Search for DTT file(s)
        if len(dtt_names) == 0:
            injection_table = []
        else:
            logging.info('List of DTT files created.')
            injection_table = get_times_DTT(dtt_names)
            print('\nInjection data found for {} DTT file(s).'.format(len(injection_table)))
    # (3) COMMAND LINE OR CONFIG OPTION - TXT TABLE OF INJECTION NAMES, BKGD TIMES, INJECTION TIMES
    elif injection_list is not None:
        try:
            with open(injection_list) as injection_file:
                lines = injection_file.readlines()
                if any(['[' in line for line in lines]):
                    injection_table = get_times_config(injection_list, ifo, station)
                else:
                    injection_table = get_times_txt(injection_list)
        except IOError:
            print('')
            logging.error('Injection list file ' + injection_list + ' not found.')
            print('')
            injection_table = []
    else:
        print('')
        logging.error('No times or lists given to get_times.')
        print('')
        injection_table = []
    # SEARCH FOR SPECIFIC INJECTIONS BY NAME
    if (injection_search is not None):
        # Get injections (rows of injection_table) that match the given injection_search option
            injection_table = search_injections(injection_table, injection_search, verbose=True)
    return injection_table

def get_times_DTT(file_names, return_channels=False):
    """
    Get injection names and times from a single .xml file (saved from DTT).
        
    Parameters
    ----------
    file_name : str
        DTT file name (.xml).
    channels : list of strings
        Channel list to add channels found in DTT file.
    return_channels : {False, True}, optional
        If True, return new channel list.
        
    Returns
    -------
    injection_table : list of lists
        Each sub-list contains injection name (str), background time (int), and injection time (int).
    chans : list of strings
        Input channel list, with new channels appended.
    """
    
    if type(file_names) != list:
        file_names = [file_names]
    injection_table = []
    injection_times = [] # This is to make sure injection times aren't mistaken as background times
    channels = []
    logging.info('Reading DTT files for times.')
    for file_name in file_names:
        # Get injection name from .xml or vice versa
        injection_name = file_name.split('/')[-1]
        if file_name[-4:] != '.xml':
            file_name += '.xml'
        else:
            injection_name = injection_name.replace('.xml', '')
        # Open DTT file, report a warning if not found
        try:
            with open(file_name) as DTT_file:
                lines = DTT_file.readlines()
        except IOError:
            print('')
            logging.warning('File not found: ' + str(file))
            print('')
            continue
        # Main loop for getting times and channels
        times = {'Start': [], 'Time':[], 't0': [], 'TestTime': []}
        for i, line in enumerate(lines):
            line = line.replace(' ','').replace('\n','')            
            # Add time to times dictionary if a time is found in this line
            if '"GPS"' in line:
                tname = line[line.index('TimeName') + 10: line.index('Type')-1]
                t = line[line.index('"GPS"') + 6: line.index('</Time>')]
                t = int(float(t))
                if (t not in times[tname]):
                    times[tname].append(t)
            # Append channel name if found in this line
            if return_channels and ('"MeasurementChannel[' in line):
                channel_name = line[line.index('"channel"') + 10: line.index('</Param>')]
                if len(channel_name) > 0:
                    if ('STRAIN' not in channel_name)\
                    and ('DELTAL' not in channel_name)\
                    and (channel_name not in channels):
                        # Add to list if channel is not GW channel and not in list yet
                        channels.append(channel_name)
        DTT_file.close()
        if 'TestTime' not in times.keys() or 't0' not in times.keys():
            print('')
            logging.warning('Missing background or injection time found in file ' + file_name)
            print('')
        else:
            t0_list = [t for t in times['t0'] if (t not in injection_times)]
            if len(times['TestTime']) == 0 or len(t0_list) == 0:
                print('')
                logging.warning('Missing background or injection time found in file ' + file_name)
                print('')
            else:
                time_inj = max(times['TestTime'])
                injection_times.append(time_inj)
                time_bg = max([t for t in times['t0'] if (t not in injection_times)])
                injection_table.append([injection_name, time_bg, time_inj])
    logging.info('Finished reading DTTs.')
    if len(injection_table) == 0:
        print('')
        logging.error('No injection times found in ' + file_name + '.')
        print('')
    if return_channels:
        return injection_table, channels
    else:
        return injection_table

def get_times_config(file_name, ifo, station='ALL'):
    """
    Get injection names and times from a config file.
        
    Parameters
    ----------
    file_name : str
        Name of times config file (.txt).
    ifo : str
        Inteferometer name, 'H1' or 'L1'.
    station = {'ALL', 'CS', 'EX', 'EY'}
        Station name. If 'ALL', get times for all stations at given ifo.
    
    Returns
    -------
    injection_table : list of lists
        Each sub-list contains injection name (str), background time (int), and injection time (int).
    """
    
    ifo_name = ('LHO' if ifo == 'H1' else 'LLO')
    config_file = ConfigParser.ConfigParser()
    logging.info('Reading injection times config file.')
    try:
        config_file.read(file_name)
    except IOError:
        print('')
        logging.error('Could not find the time list file ' + file_name + '.')
        print('')
        return []
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
    if len(injection_table) == 0:
        print('')
        logging.warning('No injection times found in ' + file_name + '. Check that file format is valid.')
        print('')
    return injection_table

def get_times_txt(file_name):
    """
    Get injection names and times from a .txt table.
    """
    
    injection_table = []
    logging.info('Reading injection names/times file.')
    try:
        with open(file_name) as file:
            lines = file.readlines()
    except IOError:
        print('')
        logging.error('Injection names/times file ' + file_name + ' not found.')
        print('')
        return []
    for line in lines:
        try:
            row = line.replace('\n', '').split(',')
            injection_table.append([row[0], int(row[1]), int(row[2])])
        except ValueError:
            pass
    logging.info('Finished reading injection names/times file.')
    if len(injection_table) == 0:
        print('')
        logging.warning('No injection times found in ' + file_name + '. Check that file format is valid.')
        print('')
    return injection_table

def search_injections(injection_table, search, verbose=False):
    """
    Search from a table of injection names and times for matching injection names.
    
    Parameters
    ----------
    injection_table : list of lists
        Each row consists of injection name, background time, and injection time for an injection.
    search : str
        Search entry.
        
    Return
    ------
    injection_table_new:
        Rows of input injection_table with name matching search entry.
    """
    
    logging.info('Searching injection table for search entry ' + search + '.')
    if (',/' in search) or ('/,' in search) or ('//' in search) or (',,' in search)\
    or search[-1] == '/' or search[-1] == ',':
        print('')
        logging.warning('Invalid search entry. Entry cannot have adjacent commas and slashes or commas and slashes at the end.')
        print('')
        return injection_table
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
    logging.info('Finished searching injection table.')
    return injection_table_new

def get_FFT_params(duration, band_width, fft_overlap_pct, fft_avg, fft_rounding=True, verbose=False):
    """
    Get parameters for calculating ASDs.
    
    Parameters
    ----------
    duration : float, int
        Duration of TimeSeries segment in seconds. If not None, this is used to get FFT time instead of band_width.
    band_width : float, int
        Bandwidth in Hz, used if duration is None.
    fft_overlap_pct : float
        FFT overlap percentage, e.g. 0.5 for 50% overlap.
    fft_avg : int
        Number of FFT averages to take over time segment.
    fft_rounding : {True, False}, optional
        If True, round FFT time to nearest second.
    
    Returns
    -------
    fft_time : float, int
        FFT time in seconds, calculated from duration or bandwidth.
    overlap_time : float
        FFT overlap time in seconds, calculated from FFT time and overlap percentage.
    duration : float, int
        Duration of TimeSeries segment in seconds. If input duration was None, this is calculated from FFT_time.
    band_width : float
        Bandwidth in Hz. If input duration not None, this is calculated from FFT time.
    """
    
    if duration is not None:
        logging.info('Duration given; determining overlap time and bandwidth.')
        # Duration takes precedence; use it to get overlap time and bandwidth
        over_prcnt = fft_overlap_pct*(10**-2)
        fft_time = duration/(1+((1-over_prcnt)*(fft_avg-1)))
        # The algorithm used for calculating ffts works well  with integers, better when even, best when powers of 2.
        # Having fft_times and overlap_times be weird floats will take a lot of time. Thats why theres a rounding option.
        if fft_rounding:
            fft_time1 = fft_time
            fft_time = round(fft_time1)
            if verbose:
                print('fft time rounded from '+str(fft_time1)+' s to '+str(fft_time)+' s.')
        overlap_time = fft_time*over_prcnt
        dur = duration
        band_width = 1.0/fft_time
        if verbose:
            print('Band width is: '+str(band_width))
        print('')
    else:
        logging.info('Bandwidth given; determining overlap time and duration.')
        # No duration; use band width to get duration and overlap time
        fft_time = 1/band_width
        if fft_rounding == True:
            fft_time1 = fft_time
            fft_time = round(fft_time1)
            print('fft time rounded from '+str(fft_time1)+' to '+str(fft_time)+'.')
        over_prcnt = fft_overlap_pct*(10**-2)
        dur = fft_time*(1+((1-over_prcnt)*(fft_avg-1)))
        overlap_time = fft_time*over_prcnt
        if verbose:
            print('Band width is: '+str(band_width+' Hz.'))
        print('')
    logging.info('FFT parameters acquired.')
    return fft_time, overlap_time, duration, band_width

def freq_search(name_inj, verbose=False):
    """
    Get fundamental frequency of an injection if is a magnetic injection.
    
    Parameters
    ----------
    name_inj : str
    
    Returns
    -------
    out : float
        Fundamental frequency (Hz).
    """
    
    out = None # Fundamental frequency of injection lines
    if 'mag' in name_inj.lower():
        # Search for either one of two patterns:
        # (1) 'p' between two numbers (eg 7p1)
        # (2) number followed by 'Hz' (eg 7Hz)
        freq_str = re.search('([0-9]+p[0-9]+)|([0-9]+Hz)', name_inj, re.IGNORECASE).group(0)
        if freq_str is not None:
            # If found, get fundamental frequency from search result
            freq_str = re.sub('Hz', '', freq_str, flags=re.IGNORECASE)
            freq_str = re.sub('p', '.', freq_str, flags=re.IGNORECASE)
            out = float(freq_str)
            logging.info('Fundamental frequency found: ' + freq_str +' Hz.')
            if verbose:
                print('Fundamental frequency found: ' + freq_str +' Hz.')
                print('Coupling will only be computed near harmonics of this frequency.')
    return out