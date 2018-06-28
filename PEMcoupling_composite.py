from optparse import OptionParser
import ConfigParser
import numpy as np
import pandas as pd
from scipy.io import loadmat
import time
import datetime
import sys
import subprocess
import logging
# pemcoupling modules
try:
    from coupling.coupfunc import CoupFunc
    from coupling.coupfunccomposite import CoupFuncComposite
    from coupling.getparams import get_channel_list, freq_search
    from coupling.loaddata import get_gwinc
    from coupling.savedata import export_composite_coupling_data
    from coupling.utils import quad_sum_names
except ImportError:
    print('')
    logging.error('Failed to load PEM coupling modules. Make sure you have all of these in the right place!')
    print('')
    raise

def get_composite_coup_func(cf_list, injection_names, comp_dict, directory,\
                            injection_freqs=None, gwinc=None, smoothing_width=0.005, verbose=False):
    """
    Calculates a composite coupling function from multiple coupling functions, and saves the results.
    
    Parameters
    ----------
    cf_list : list
        CoupFunc objects containing coupling functions, each from a different injection time/location.
    injection_names : list
        Names of injections corresponding to each coupling function.
    comp_dict : dict
        Options for processing composite coupling functions.
    directory : str
        Output directory.
    inejction_freqs : dict, optional
        Fundamental frequency of each injection, for magnetic line injections.
    gwinc : list, optional
        Frequencies and ASD values of GWINC spectrum for plotting.
    smoothing_width : float, optional
        Gaussian smoothing width.
    verbose : {False, True}, optional
    """
    
    channel_name = cf_list[0].name
    if injection_freqs is None:
        injection_freqs = {n: None for n in injection_names}
    freqs = np.mean([coup_func.freqs for coup_func in cf_list], axis=0)
    darm = np.mean([coup_func.darm_bg for coup_func in cf_list], axis=0)
    band_widths = [coup_func.df for coup_func in cf_list]
    column_len = [coup_func.freqs.shape for coup_func in cf_list]
    if (any(bw != band_widths[0] for bw in band_widths) or any([k != column_len[0] for k in column_len])):
        print('\nError: Coupling data objects have unequal data lengths.')
        print('If all the band_widths are the same, this should not be an issue.\n')
    local_max_window = int(comp_dict['local_max_width'] / band_widths[0]) # Convert from Hz to bins
    # Create composite coupling function
    comp_cf = CoupFuncComposite.compute(cf_list, injection_names, local_max_window=local_max_window, freq_lines=injection_freqs)
#     comp_cf = analysis.composite_coupling_function(cf_list, injection_names, local_max_window=local_max_window, freq_lines=injection_freqs)
    # Gaussian smoothing of final result
    smooth_chans = ['ACC', 'MIC', 'WFS']
    # Export results
    path = directory + '/CompositeCouplingFunctions'
    export_composite_coupling_data(
        comp_cf, freqs, darm, gwinc, injection_names, path,
        upper_lim=comp_dict['upper_lim'], est_amb_plot=comp_dict['comp_est_amb_plot'],
        freq_min=comp_dict['comp_freq_min'], freq_max=comp_dict['comp_freq_max'],
        factor_min=comp_dict['comp_y_min'], factor_max=comp_dict['comp_y_max'],
        fig_w=comp_dict['comp_fig_width'], fig_h=comp_dict['comp_fig_height'],
        verbose=verbose
    )
    #### APPLY ALL THE ABOVE STEPS TO BINNED DATA ####
    cf_binning = None #comp_dict['coupling_function_binning']
    if cf_binning is not None:
        # Bin original coupling function data
        cf_binned_list = [coup_func.bin_data(cf_binning) for coup_func in cf_list]
        # Keep track of binned frequencies and DARM separately for plotting an unbinned DARM
        freqs_binned = np.mean([coup_func.freqs for coup_func in cf_binned_list], axis=0)
        darm_binned = np.mean([coup_func.darm_bg for coup_func in cf_binned_list], axis=0)
        # Composite coupling function
        comp_cf_binned = CoupFuncComposite.compute(cf_binned_list, injection_names, local_max_window=local_max_window)
        # Data export
        path_binned = path + 'Binned'
        export_composite_coupling_data(
            comp_cf_binned, freqs, darm, gwinc, injection_names, path_binned,
            upper_lim=comp_dict['upper_lim'], est_amb_plot=comp_dict['comp_est_amb_plot'],
            freq_min=comp_dict['comp_freq_min'], freq_max=comp_dict['comp_freq_max'],
            factor_min=comp_dict['comp_y_min'], factor_max=comp_dict['comp_y_max'],
            verbose=verbose
        )
    if verbose:
        print('\nLowest (composite) coupling function complete for ' + channel_name)
    return

##############################################################################################################

if __name__ == "__main__":
    t1 = time.time()
    parser = OptionParser()
    parser.add_option("-C", "--channel_list", dest = "channel_list",
                      help = "Txt file containing full channel names.")
    parser.add_option("-c", "--channel_search", dest= "channel_search",
                      help = "Channel search keys separated by commas ',' (for AND) and forward slashes '/' (for OR) "+\
                      "(AND takes precedence over OR). Use minus signs '-' to exclude a string (i.e. NOT).")
    parser.add_option("-o", "--output", dest = "directory", 
                      help = "Directory containing coupling function data. Composite coupling functions will be saved "+\
                      "in a sub-directory there.")
    parser.add_option("-v", "--verbose", action = "store_true", dest = "verbose", default = False,
                      help = "The porgram will give additional information about its procedures and show runtime "+\
                      "for specifc executions.")
    (options, args) = parser.parse_args()
    verbose = options.verbose
    #### PARSE ARGUMENTS ####
    ifo_input, station_input, injection_type = args
    if ifo_input.lower() in ['h1', 'lho']:
        ifo = 'H1'
    elif ifo_input.lower() in ['l1', 'llo']:
        ifo = 'L1'
    else:
        print('\nError: Argument "ifo" must be one of "H1", "LHO", "L1", or "LHO" (not case-sensitive).')
        sys.exit()
    if station_input.upper() in ['CS', 'EX', 'EY', 'ALL']:
        station = station_input.upper()
    else:
        print('\nError: Argument "station" must be one of "CS", "EX", "EY", or "ALL" (not case-sensitive).')
        sys.exit()
    # Choose config file based on injection type
    if injection_type.lower() in ['vib', 'vibrational']:
        config_name = 'config_files/config_vibrational.txt'
    elif injection_type.lower() in ['mag', 'magnetic']:
        config_name = 'config_files/config_magnetic.txt'
    #### PARSE CONFIG FILE ####
    config = ConfigParser.ConfigParser()
    try:
        config.read(config_name)
    except:
        print('\nError: Configuration file ' + config_name + ' not found.\n')
        sys.exit()
    # CHECK FOR MISSING OR EMPTY CONFIG FILE
    if len(config.sections()) == 0:
        print('\nError: Configuration file ' + config_name + ' not found.\n')
        sys.exit()
    if all( [len(config.options(x)) == 0 for x in config.sections()] ):
        print('\nError: Configuration file ' + config_name + ' is empty.\n')
        sys.exit()
    # READ CONFIG INPUTS INTO DICTIONARY, SPLIT BY SECTION
    config_dict = {}
    for section in config.sections():
        for option in config.options(section):
            config_dict[option] = config.get(section, option)
    # Source and output directory
    if options.directory is not None:
        directory = options.directory
    elif comp_dict['directory'] is not None:
        directory = config_dict['directory']
    else:
        print('\nError: No directory provided in command line option nor config file.\n')
        sys.exit()
    if directory[-1] == '/':
        directory = directory[:-1]
    # Composite coupling function options
    comp_dict = {}
    float_options = [
        'local_max_width', 'coupling_function_binning',
        'comp_fig_height', 'comp_fig_width',
        'comp_freq_min', 'comp_freq_max',
        'comp_y_min', 'comp_y_max'
    ]
    bool_options = ['composite_coupling', 'upper_lim', 'comp_est_amb_plot']
    for option, value in config_dict.items():
        if option in float_options:
            try:
                comp_dict[option] = float(value)
            except:
                if option == 'local_max_width':
                    comp_dict[option] = 0
                else:
                    comp_dict[option] = None
        elif option in bool_options:
            comp_dict[option] = (value.lower() in ['on', 'true', 'yes'])
    #### GET CHANNEL LIST ####
    if options.channel_list is not None:
        channels = get_channel_list(options.channel_list, ifo, station, search=options.channel_search, verbose=verbose)
    elif '.txt' in config_dict['channels']:
        channels = get_channel_list(config_dict['channels'], ifo, station, search=options.channel_search, verbose=verbose)
    else:
        print('\nChannel list input required in config file ("channels") or command line option ("-C").\n')
        sys.exit()
    # Add in quad sum channels if tri-axial sensors found
    qsum_dict = quad_sum_names(channels)
    channels += list(qsum_dict.keys())
    channels.sort()
    # Report if no channels found
    if len(channels)==0:
        sys.exit()
    print('\n{} channels(s) found:'.format(len(channels)))
    for c in channels:
        print(c)
    print('')
    #### IMPORT GWINC DATA FOR AMBIENT PLOT ####
    gwinc_file = 'config_files/darm/gwinc_nomm_noises.mat'
    try:
        gwinc = get_gwinc(gwinc_file)
    except IOError:
        gwinc = None
    if gwinc is None:
        print('Composite estimated ambient will not show GWINC.')
        gwinc = None
    #### LOOP OVER CHANNELS AND PERFORM COMPOSITE COUPLING CALCULATION ON EACH ####
    for channel_name in sorted(channels):
        print('Creating composite coupling function for ' + channel_name + '.')
        channel_short_name = channel_name[channel_name.index('-')+1:].replace('_DQ','')
        try:
            file_names = subprocess.check_output(['ls ' + directory + '/*/' + channel_short_name + '_coupling_data.txt'],\
                                                 shell=True).splitlines()
        except:
            print('\nWarning: No coupling function data found for ' + channel_short_name + ' in source directory ' +\
                  directory + '.\n')
            continue
        cf_list = []
        inj_names = []
        injection_freqs = {}
        for file_name in file_names:
            if not ('composite' in file_name):
                cf = CoupFunc.load(file_name, channelname=channel_name.replace('_DQ', ''))
                cf_list.append(cf)
                name_inj = file_name.split('/')[-2]
                inj_names.append(name_inj)
                freq_search_result = freq_search(name_inj)
                if freq_search_result is not None:
                    injection_freqs[name_inj] = freq_search_result
        get_composite_coup_func(cf_list, inj_names, comp_dict, directory,\
                                injection_freqs=injection_freqs, gwinc=gwinc, smoothing_width=0.005, verbose=verbose)
    t2 = time.time() - t1
    print('Lowest (composite) coupling functions processed. (Runtime: {:.3f} s.)\n'.format(t2))