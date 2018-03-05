from optparse import OptionParser
import ConfigParser
import numpy as np
import pandas as pd
from scipy.io import loadmat
import time
import datetime
import sys
import subprocess

from PEMcoupling import getdata, couplingfunction, analysis, savedata


t1 = time.time()


parser = OptionParser()

parser.add_option("-c", "--channel_search", dest= "channel_search",
                  help = "Channel search keys separated by commas ',' (for AND) and forward slashes '/' (for OR) "+\
                  "(AND takes precedence over OR). Use minus signs '-' to exclude a string (i.e. NOT).")

parser.add_option("-o", "--output", dest = "directory", 
                  help = "Custom name of the directory that will hold all output data.")

parser.add_option("-v", "--verbose", action = "store_true", dest = "verbose", default = False,
                  help = "The porgram will give additional information about its procedures and show runtime "+\
                  "for specifc executions.")

(options, args) = parser.parse_args()


# PARSE ARGUMENTS

config_name, ifo_input, station_input = args

# Make sure interferometer input is valid
if ifo_input.lower() in ['h1', 'lho']:
    ifo = 'H1'
elif ifo_input.lower() in ['l1', 'llo']:
    ifo = 'L1'
else:
    print('\nError: Second argument "ifo" must be one of "H1", "LHO", "L1", or "LHO" (not case-sensitive).')
    sys.exit()

# Make sure station input is valid
if station_input.upper() in ['CS', 'EX', 'EY', 'ALL']:
    station = station_input.upper()
else:
    print('\nError: Third argument "station" must be one of "CS", "EX", "EY", or "ALL" (not case-sensitive).')
    sys.exit()

verbose = options.verbose

    
# READ CONFIG FILE
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

channel_list = config_dict['channels']
directory = config_dict['directory']
        
comp_dict = {}

float_options = [
    'local_max_width',
    'coupling_function_binning',
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

print(comp_dict)

if options.directory is not None:
    if options.directory[-1]=='/':
        comp_dict['directory'] =  options.directory[:-1]
    else:
        comp_dict['directory'] = options.directory


# GET CHANNEL LIST

try:
    channels = getdata.get_channel_names(channel_list, ifo, station)
except:
    print('\nError: Channel list ' + channel_list + ' not found.\n')
    sys.exit()

# Report if no channels found
if len(channels)==0:
    print('\nError: No channels found from ' + file_chans)
    sys.exit()
else:
    channels.sort()
    
# CHANNEL SEARCH
if (options.channel_search is not None):
    channels = getdata.search_channels(channels, options.channel_search, verbose)
    
    if len(channels) == 0:
        print('\nError: No channels found matching search entry.\n')
        sys.exit()
        
print('\n{} channels(s) found:'.format(len(channels)))
for c in channels:
    print(c)

print('')



# IMPORT GWINC DATA FOR AMBIENT PLOT
gwinc_file = 'gwinc_nomm_noises.mat'
try:
    gwinc_mat = loadmat(gwinc_file)
    gwinc = [
        np.asarray(gwinc_mat['nnn']['Freq'][0][0][0]),
        np.sqrt(np.asarray(gwinc_mat['nnn']['Total'][0][0][0])) * 4000. # Convert strain PSD to DARM ASD
    ]
except:
    print('\nWarning: GWINC data file ' + gwinc_file + ' not found.\n')
    gwinc=None

# LOOP FOR IMPORTING SENSOR COUPLING FUNCTIONS
for channel_name in sorted(channels):
    print('Processing results for ' + channel_name + '.')
    channel_short_name = channel_name[channel_name.index('-')+1:].replace('_DQ','')
    try:
        lines = subprocess.check_output(['ls ' + directory + '/*/' + channel_short_name + '_coupling_data.txt'],\
                                        shell=True).splitlines()
    except:
        print('\nWarning: No coupling function data found for ' + channel_short_name + ' in source directory ' +\
              directory + '.\n')
        continue
    file_names = [l for l in lines if not ('composite' in l)]
    
    #### DATA IMPORT LOOP ####
    df_list = []
    df_binned_list = []
    inj_names = []
    freqs_raw = []
    darm_raw = []
    sens_BG = {}
    sens_BG_binned = {}
    for i in file_names:
        try:
            df = pd.read_csv(i)
        except:
            print('Warning: File not found:\n' + i + '\nMoving on.')
            continue
        
        if any([col not in df.columns for col in ['frequency', 'factor', 'factor_counts', 'flag', 'sensBG', 'darmBG']]):
            # Invalid column headers; skip this file
            print('\nWarning: Invalid column headers. Skipping this file.\n')
            continue
        
        df = df[['frequency', 'factor', 'factor_counts', 'flag', 'sensBG', 'darmBG']]
        df = df[df['frequency'] > 0] # Disregard 0 Hz
        
        # Use this part for directly imposing plot limits onto data
        if comp_dict['comp_freq_min'] is not None:
            df = df[df['frequency'] >= comp_dict['comp_freq_min']]
        if comp_dict['comp_freq_max'] is not None:
            df = df[df['frequency'] <= comp_dict['comp_freq_max']]
        
        inj = i.split('/')[-2]
        inj_names.append(inj)
        
        sens_BG[inj] = np.asarray(df['sensBG'])
        df_list.append(df)
        
        #### BINNING ####
        if comp_dict['coupling_function_binning'] is not None:
            if comp_dict['coupling_function_binning'] > 0:
                df_binned = analysis.coupling_function_binned(df, comp_dict['coupling_function_binning'])
                sens_BG_binned[inj] = np.asarray(df_binned['sensBG'])
                df_binned_list.append(df_binned)
        
    if len(inj_names) == 0:
        print('\nError: No injections found for channel ' + channel_name); continue
    
    freqs = np.mean([x['frequency'].astype(float) for x in df_list], axis=0)
    darm = np.mean([x['darmBG'].astype(float) for x in df_list], axis=0)
    if comp_dict['coupling_function_binning'] is not None:
        freqs_binned = np.mean([x['frequency'].astype(float) for x in df_binned_list], axis=0)
        darm_binned = np.mean([x['darmBG'].astype(float) for x in df_binned_list], axis=0)
        
    ########################
    
    # Skip this channel if no coupling factors found in any injection
    if (not any(['Real' in x['flag'] for x in df_list])) and (not comp_dict['upper_lim']):
        print('Warning: No real coupling factors found for ' + channel_name + \
              '. Calculation aborted for this channel.')
        continue

    # Check band width consistency b/w data
    band_widths = [np.diff(i['frequency'])[0] for i in df_list] # Frequency step sizes
    if not all(j == band_widths[0] for j in band_widths):
        print('Error: Each imported csv file must have the same frequency band width. '+\
              'Please reformat the data and try again.')
        sys.exit()

    #If band_width is the same then column lengths / number of rows should be the same too
    column_len = [df.shape[0] for df in df_list]
    if not all([k == column_len[0] for k in column_len]):
        print('Data tables have unequal lengths.')
        print('If all the band_widths are the same, this shouldnt be an issue. '+\
               'Please investigate.')
    
    # Average of all sensor backgrounds
    sens_BG_avg = np.zeros_like(freqs)
    for i in range(len(freqs)):
        sens_BG_avg[i] = np.mean([sens_BG[inj][i] for inj in inj_names])
        # Could do median instead just to avoid unexpected transients in background spectra.')




    #===========================================
    #### COMPOSITE COUPLING FUNCTIONS
    #===========================================
    
    if verbose:
        print('\nObtaining composite (lowest) coupling function for ' + channel_name)
        
    w = int(comp_dict['local_max_width'] / band_widths[0]) # Local max window size (Hz to bins)
    
    
    
    # GET COMPOSITE COUPLING FUNCTIONS -- UNBINNED DATA
    
    composite_results = analysis.composite_coupling_function(
        [df['factor'] for df in df_list],
        [df['factor_counts'] for df in df_list],
        [df['flag'] for df in df_list],
        inj_names, comp_dict['upper_lim'], w
    )
    comp_factors, comp_factors_counts, comp_flags, comp_injs = composite_results
    
    
    # TRUNCATE UPPER LIMIT DATA FROM OVERLAPPING COUPLING FUNCTIONS
    
    if 'MAG' in channel_name.upper():
        lowest, freq_lowest = analysis.lowest_composite_upper_limit(freqs, comp_factors, comp_flags, comp_injs)
        lowest_idx = inj_names.index(lowest)
        lowest_upperlim_cf = cf_data_list[lowest_idx]
        for i in range(len(freqs)):
            if (comp_flags[i] != 'No data') and (freqs[i] > freq_lowest):
                if (comp_injs[i] == lowest):
                    comp_factors[i] = lowest_upperlim_cf.factors[i]
                    comp_factors_counts[i] = lowest_upperlim_cf.factors_counts[i]
                    comp_flags[i] = 'Upper Limit'
                    comp_injs[i] = lowest
                else:
                    comp_factors[i] = 0
                    comp_factors_counts[i] = 0
                    comp_flags[i] = 'No data'
                    comp_injs[i] = None
    
    
    # GAUSSIAN SMOOTHING JUST BEFORE EXPORT; FOR SLIGHTLY SMOOTHER DATA
    
    smooth_chans = ['ACC', 'MIC' 'WFS']    # Apply this procedure only to these channels
    if any(x in channel_name for x in smooth_chans):
        width_gauss = 0.005
        
        # Apply smoothing to unbinned data
        comp_factors = analysis.gaussian_smooth(freqs, comp_factors, width_gauss, comp_flags)
        comp_factors_counts = analysis.gaussian_smooth(freqs, comp_factors_counts, width_gauss, comp_flags)
        sens_BG_avg = analysis.gaussian_smooth(freqs, sens_BG_avg, width_gauss, comp_flags)
        
        if comp_dict['coupling_function_binning'] is not None:

            sens_BG_binned_avg = np.zeros_like(freqs_binned)
            for i in range(len(freqs_binned)):
                sens_BG_binned_avg[i] = np.mean([sens_BG_binned[inj][i] for inj in inj_names])
            
            # Apply smoothing to binned data
            comp_factors_binned = analysis.gaussian_smooth(freqs_binned, comp_factors_binned, width_gauss, comp_flags_binned)
            coup_low_factors_counts_binned = analysis.gaussian_smooth(freqs_binned, comp_factors_counts_binned, width_gauss, comp_flags_binned)
            sens_BG_binned_avg = analysis.gaussian_smooth(freqs_binned, sens_BG_binned_avg, width_gauss, comp_flags_binned)
    
    
    # DATA EXPORT (COUPLING PLOT, AMBIENT PLOT, CSV)
    
    path = directory + '/CompositeCouplingFunctions'
    
    comp_data = couplingfunction.CompositeCouplingData(
        channel_name, freqs,
        comp_factors, comp_factors_counts,
        comp_flags, comp_injs,
        sens_BG_avg, darm
    )
    savedata.export_composite_coupling_data(comp_data, freqs, darm, gwinc, inj_names, path, comp_dict, verbose)
    
    
    # APPLY SAME STEPS TO BINNED DATA
    
    if comp_dict['coupling_function_binning'] is not None:
            
        sens_BG_binned_avg = np.zeros_like(freqs_binned)
        for i in range(len(freqs_binned)):
            sens_BG_binned_avg[i] = np.mean([sens_BG_binned[inj][i] for inj in inj_names])
        
        composite_results_binned = analysis.composite_coupling_function(
            [df['factor'] for df in df_binned_list],
            [df['factor_counts'] for df in df_binned_list],
            [df['flag'] for df in df_binned_list],
            inj_names, comp_dict['upper_lim'], w
        )
        comp_factors_binned, comp_factors_counts_binned, comp_flags_binned, comp_injs_binned = composite_results_binned
        
        if any(x in channel_name for x in smooth_chans):
            width_gauss = 0.005

            # Apply smoothing to binned data
            comp_factors_binned = analysis.gaussian_smooth(freqs_binned, comp_factors_binned, width_gauss, comp_flags_binned)
            coup_low_factors_counts_binned = analysis.gaussian_smooth(freqs_binned, comp_factors_counts_binned, width_gauss, comp_flags_binned)
            sens_BG_binned_avg = analysis.gaussian_smooth(freqs_binned, sens_BG_binned_avg, width_gauss, comp_flags_binned)
        
        path_binned = path + 'Binned'
        comp_data_binned = couplingfunction.CompositeCouplingData(
            channel_name, freqs_binned,
            comp_factors_binned, comp_factors_counts_binned,
            comp_flags_binned, comp_injs_binned,
            sens_BG_binned_avg, darm_binned
        )
        savedata.export_composite_coupling_data(comp_data_binned, freqs, darm, gwinc, inj_names, path_binned, comp_dict, verbose)
    
    if verbose:
        print('\nLowest (composite) coupling function complete for ' + channel_name)
        

t2 = time.time() - t1
print('Lowest (composite) coupling functions processed. (Runtime: {:.3f} s.)\n'.format(t2))