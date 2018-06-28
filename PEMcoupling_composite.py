from optparse import OptionParser
import ConfigParser
import numpy as np
import pandas as pd
from scipy.io import loadmat
import time
import datetime
import os
import sys
import subprocess
import logging
# pemcoupling modules
try:
    from coupling.coupfunc import CoupFunc
    from coupling.coupfunccomposite import CoupFuncComposite
    from coupling.getparams import get_channel_list, freq_search
    from coupling.loaddata import get_gwinc
    from coupling.utils import quad_sum_names
except ImportError:
    print('')
    logging.error('Failed to load PEM coupling modules. Make sure you have all of these in the right place!')
    print('')
    raise

def get_composite_coup_func(
    cf_list, injection_names, out_dir,
    freq_lines=None, gwinc=None, local_max_width=0,
    upper_lim=True, est_amb_plot=True,
    freq_min=None, freq_max=None,
    factor_min=None, factor_max=None,
    fig_w=9, fig_h=6,
    verbose=False):
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
    out_dir : str
        Output directory for composite coupling functions.
    inejction_freqs : dict, optional
        Fundamental frequency of each injection, for magnetic line injections.
    gwinc : list, optional
        Frequencies and ASD values of GWINC spectrum for plotting.
    smoothing_width : float, optional
        Gaussian smoothing width.
    verbose : {False, True}, optional
    """
    
    if not os.path.exists(str(out_dir)):
        os.makedirs(str(out_dir))
    channel_name = cf_list[0].name
    band_widths = [cf.df for cf in cf_list]
    column_len = [cf.freqs.shape for cf in cf_list]
    if (any(bw != band_widths[0] for bw in band_widths) or any([k != column_len[0] for k in column_len])):
        print('\nError: Coupling data objects have unequal data lengths.')
        print('If all the band_widths are the same, this should not be an issue.\n')
    #### COMPUTE COMPOSITE COUPLING FUNCTION ####
    local_max_window = int(local_max_width / band_widths[0]) # Convert from Hz to bins
    if freq_lines is None:
        freq_lines = {injection: None for injection in injection_names}
    comp_cf = CoupFuncComposite.compute(cf_list, injection_names, local_max_window=local_max_window, freq_lines=freq_lines)
    
    #### X-AXIS (FREQUENCY) LIMITS ####
    if verbose:
        print('\nDetermining axis limits for plots...')
    x_axis = comp_cf.freqs[comp_cf.values > 0]
    if (len(x_axis) == 0):
        print('No lowest coupling factors for ' + comp_cf.name + '.')
        print('Data export aborted for this channel.')
        return
    try:
        float(freq_min)
    except TypeError:
        freq_min = max( [min(x_axis) / 1.5, 6] )
    try:
        float(freq_max)
    except TypeError:
        freq_max = min( [max(x_axis) * 1.5, max(comp_cf.freqs)] )
    
    #### Y-AXIS (COUPLING FACTOR) LIMITS ####
    y_axis = comp_cf.values[(comp_cf.values > 0) & (comp_cf.freqs>=freq_min) & (comp_cf.freqs < freq_max)]
    y_axis_counts = comp_cf.values_in_counts[(comp_cf.values > 0) & (comp_cf.freqs>=freq_min) & (comp_cf.freqs < freq_max)]
    if (len(y_axis) == 0):
        print('No lowest coupling factors for ' + comp_cf.name)
        print('between ' + str(freq_min) + ' and ' + str(freq_max) + ' Hz.')
        print('Data export aborted for this channel.')
        return
    try:
        float(factor_min)
    except TypeError:
        factor_min = np.min(y_axis) / 3
        factor_counts_min = np.min(y_axis_counts) / 3
    else:
        factor_counts_min = factor_min
    try:
        float(factor_max)
    except TypeError:
        factor_max = np.max(y_axis) * 1.5
        factor_counts_max = np.max(y_axis_counts) * 1.5
    else:
        factor_counts_max = factor_max
        
    #### SORTED NAMES OF INJECTIONS ####
    sorted_names = sorted(set(comp_cf.injections))
    if None in sorted_names:
        sorted_names.remove(None)
    
    #### FILEPATH ####
    base_filename = comp_cf.name.replace('_DQ', '') + '_composite_'
    csv_filename = os.path.join(out_dir, base_filename + 'coupling_data.txt')
    multi_filename = os.path.join(out_dir, base_filename + 'coupling_multi_plot.png')
    single_filename = os.path.join(out_dir, base_filename + 'coupling_plot.png')
    single_counts_filename = os.path.join(out_dir, base_filename + 'coupling_counts_plot.png')
    est_amb_multi_filename = os.path.join(out_dir, base_filename + 'est_amb_multi_plot.png')
    est_amb_single_filename = os.path.join(out_dir, base_filename + 'est_amb_plot.png')
    
    #### LOWEST COUPLING FUNCTION PLOT ####
    # Split/multi-plot
    comp_cf.plot(
        multi_filename, in_counts=False, split_injections=True, upper_lim=upper_lim,
        freq_min=freq_min, freq_max=freq_max, factor_min=factor_min, factor_max=factor_max,
        fig_w=fig_w, fig_h=fig_h
    )
    # Merged plot
    comp_cf.plot(
        single_filename, in_counts=False, split_injections=False, upper_lim=upper_lim,
        freq_min=freq_min, freq_max=freq_max, factor_min=factor_min, factor_max=factor_max,
        fig_w=fig_w, fig_h=fig_h
    )
    # Merged plot in counts
    comp_cf.plot(
        single_counts_filename, in_counts=True, split_injections=False, upper_lim=upper_lim,
        freq_min=freq_min, freq_max=freq_max, factor_min=factor_counts_min, factor_max=factor_counts_max,
        fig_w=fig_w, fig_h=fig_h
    )    
    if verbose:
        print('Composite coupling function plots complete.')

    #### LOWEST ESTIMATED AMBIENT PLOT ####
    if est_amb_plot:
        mask_freq = (comp_cf.freqs >= freq_min) & (comp_cf.freqs < freq_max) # data lying within frequency plot range
        ambs_pos = comp_cf.ambients[(comp_cf.ambients > 0) & mask_freq]
        darm_pos = comp_cf.darm_bg[(comp_cf.darm_bg > 0) & mask_freq]
        values = np.concatenate((ambs_pos, darm_pos)) # all positive data within freq range
        amb_min = values.min() / 4
        amb_max = values.max() * 2
        if np.any(comp_cf.flags != 'No data'):
            freqs_raw = np.mean([cf.freqs for cf in cf_list], axis=0)
            darm_raw = np.mean([cf.darm_bg for cf in cf_list], axis=0)
            # Split/multi-plot
            comp_cf.ambientplot(
                est_amb_multi_filename,
                gw_signal='darm', split_injections=True, gwinc=gwinc, darm_data=[freqs_raw, darm_raw],
                freq_min=freq_min, freq_max=freq_max, amb_min=amb_min, amb_max=amb_max, fig_w=fig_w, fig_h=fig_h
            )
            # Merged plot
            comp_cf.ambientplot(
                est_amb_single_filename,
                gw_signal='strain', split_injections=False, gwinc=gwinc, darm_data=[freqs_raw, darm_raw],
                freq_min=freq_min, freq_max=freq_max, amb_min=amb_min/4000., amb_max=amb_max/4000., fig_w=fig_w, fig_h=fig_h
            )
            if verbose:
                print('Composite estimated ambient plots complete.')        
        else:
            print('No composite coupling data for this channel.')
            
    #### CSV OUTPUT ####
    comp_cf.to_csv(csv_filename)
    if verbose:
        print('CSV saved.')
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
        injection_names = []
        freq_lines = {}
        for file_name in file_names:
            if not ('composite' in file_name):
                cf = CoupFunc.load(file_name, channelname=channel_name.replace('_DQ', ''))
                cf_list.append(cf)
                injection_name = file_name.split('/')[-2]
                injection_names.append(injection_name)
                freq_search_result = freq_search(injection_name)
                if freq_search_result is not None:
                    freq_lines[injection_name] = freq_search_result
        out_dir = os.path.join(directory, 'CompositeCouplingFunctions')
        get_composite_coup_func(
            cf_list, injection_names, out_dir,
            freq_lines=freq_lines, gwinc=gwinc, local_max_width=comp_dict['local_max_width'],
            upper_lim=comp_dict['upper_lim'], est_amb_plot=comp_dict['comp_est_amb_plot'],
            freq_min=comp_dict['comp_freq_min'], freq_max=comp_dict['comp_freq_max'],
            factor_min=comp_dict['comp_y_min'], factor_max=comp_dict['comp_y_max'],
            fig_w=comp_dict['comp_fig_width'], fig_h=comp_dict['comp_fig_height'],
            verbose=verbose
        )
    t2 = time.time() - t1
    print('Lowest (composite) coupling functions processed. (Runtime: {:.3f} s.)\n'.format(t2))