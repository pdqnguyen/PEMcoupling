"""
Functions for determining parameters for coupling function calculations in PEMcoupling.py, e.g. FFT parameters, coupling factor thresholds, channel lists, and injection lists.
"""

import ConfigParser
import sys

def get_arg_params(args):
    """
    Parse arguments, terminating the script if user input is invalid.
    
    Parameters
    ----------
    args: tuple
        Arguments from argparser.
    
    Returns
    -------
    ifo: str
        Interferometer name.
    station: str
        Station name, CS, EX, EY, or ALL.
    config_name: str
        Name of configuration file to use, based on injection_type.
    """
    
    if len(args) != 3:
        print('\nError: Exactly 3 arguments required (configuration file, interferometer, and station).\n')
        sys.exit()
    ifo_input, station_input, injection_type = args
    # Make sure interferometer input is valid
    if ifo_input.lower() in ['h1', 'lho']:
        ifo = 'H1'
    elif ifo_input.lower() in ['l1', 'llo']:
        ifo = 'L1'
    else:
        print('\nError: 1st argument "ifo" must be one of "H1", "LHO", "L1", or "LHO" (not case-sensitive).\n')
        sys.exit()
    # Make sure station input is valid
    if station_input.upper() in ['CS', 'EX', 'EY', 'ALL']:
        station = station_input.upper()
    else:
        print('\nError: 2nd argument "station" must be one of "CS", "EX", "EY", or "ALL" (not case-sensitive).\n')
        sys.exit()
    # Make sure injection type is valid
    if injection_type.lower() in ['mag', 'magnetic']:
        config_name = 'config_files/config_magnetic.txt'
    elif injection_type.lower() in ['vib', 'vibrational', 'acoustic']:
        config_name = 'config_files/config_vibrational.txt'
    else:
        print('\nError: 3rd argument "injection_type" must be one of "mag", "magnetic", "vib", "vibrational", or '+\
              '"acoustic" (not case-sensitive).\n')
        sys.exit()
    return ifo, station, config_name

def get_config_params(config_name):
    """
    Parse config file and return a dictionary of the config options and values.
    
    Parameters
    ----------
    config_name: str
        Name of config file.
    
    Returns
    -------
    config_dict: dict
        Dictionary containing a sub-dictionary of config options/values for each section in the config file.
    """
    
    # Read config file
    config = ConfigParser.ConfigParser()
    try:
        config.read(config_name)
    except:
        print('\nError: Configuration file ' + config_name + ' not found.\n')
        sys.exit()
    # Check for missing/empty config file
    if len(config.sections()) == 0:
        print('\nError: Configuration file ' + config_name + ' not found.\n')
        sys.exit()
    if all( [len(config.options(x)) == 0 for x in config.sections()] ):
        print('\nError: Configuration file ' + config_name + ' is empty.\n')
        sys.exit()
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
    # Convert numerical config options to appropriate types (floats and ints)
    for sub_dict in config_dict.values():
        for option, value in sub_dict.items():
            if option in float_options:
                try:
                    sub_dict[option] = float(value)
                except:
                    sub_dict[option] = None
                    if option in float_required:
                        print('\nError: float input required for option ' + option + '.\n')
                        sys.exit()
            elif option in int_options:
                try:
                    sub_dict[option] = int(value)
                except:
                    sub_dict[option] = None
                    if option in int_required:
                        print('\nError: int input required for option ' + option + '.\n')
                        sys.exit()
            elif option in bool_options:
                sub_dict[option] = (value.lower() in ['on', 'true', 'yes'])
            elif '_smoothing' in option:
                try:
                    sub_dict[option] = [float(v) for v in value.split(',')]
                except:
                    print('\nError: Input for ' + option + ' should be list of three floats (e.g. ACC_smothing: 5, 0.5, 0.5).\n')
                    sys.exit()
            elif 'notch' in option:
                if value is not None:
                    try:
                        notch_file = open(value)
                    except:
                        print('\nError: Cannot open file ' + value + ' provided in config file.\n')
                        sys.exit()
                    notch_data = notch_file.read().split('\n')
                    notch_list = []
                    for row in notch_data:
                        try:
                            notch_list.append(map(float, row.split(',')))
                        except:
                            print('\nError: Data in DARM notch files must be floats.\n')
                            sys.exit()
                    sub_dict[option] = notch_list
                    notch_file.close()
                else:
                    sub_dict[option] = []
    return config_dict

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
            injection_table = search_injections(injection_table, injection_search, verbose=True)
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

def get_FFT_params(duration, band_width, fft_overlap_pct, fft_avg, fft_rounding=True, verbose=False):
    """
    Get parameters for calculating ASDs.
    
    Parameters
    ----------
    duration: float, int
        Duration of TimeSeries segment in seconds. If not None, this is used to get FFT time instead of band_width.
    band_width: float, int
        Bandwidth in Hz, used if duration is None.
    fft_overlap_pct: float
        FFT overlap percentage, e.g. 0.5 for 50% overlap.
    fft_avg: int
        Number of FFT averages to take over time segment.
    fft_rounding: {True, False}, optional
        If True, round FFT time to nearest second.
    
    Returns
    -------
    fft_time: float, int
        FFT time in seconds, calculated from duration or bandwidth.
    overlap_time: float
        FFT overlap time in seconds, calculated from FFT time and overlap percentage.
    duration: float, int
        Duration of TimeSeries segment in seconds. If input duration was None, this is calculated from FFT_time.
    band_width: float
        Bandwidth in Hz. If input duration not None, this is calculated from FFT time.
    """
    
    # Duration takes precedence; use it to get overlap time and bandwidth
    if duration is not None:
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
    
    # No duration; use band width to get duration and overlap time
    else:
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
    
    return fft_time, overlap_time, duration, band_width