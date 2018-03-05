"""
PEM Injection Analysis

Started by Julia Kruk
Completed and maintained by Philippe Nguyen

This program performs an analysis of environmental coupling for a set of PEM sensors during different PEM injections.
It uses GWpy and PEM_coupling modules to:
    1) preprocess data from PEM sensors and DARM
    2) compute coupling functions for each sensor during each injection
    3) estimates DARM ambients for every coupling function
    4) aggregate data across multiple injections to produce a composite coupling function and estimated ambient for each sensor
    5) export data in the form of:
        a) CSV files containing single-injection coupling functions (in physical units and in counts) and estimated ambients
        b) plots of single-injection coupling functions (in physical units and in counts)
        c) spectrum plots showing sensor and DARM ASDs during and before injections, super-imposed with a estimated ambients
        d) CSV files containing composite coupling functions (in physical units and in counts) and estimated ambients
        e) plots of composite coupling functions (in physical units and in counts) and estimated ambients
        f) multi-plots of composite coupling functions and estimated ambients, labeling data by injection


Usage notes:
    gwpy:
        This code is written to run on gwpy 0.4.
        Newest version may not be installed on your LIGO machine.
        On cluster, activate the gwpy virtualenv via ". ~detchar/opt/gwpysoft/bin/activate"
    PEM_coupling:
        All preprocessing, calculations, and data exports are done via PEM_coupling.
"""

from optparse import OptionParser
import ConfigParser
from gwpy.detector import Channel
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from gwpy.time import *
import numpy as np
import pandas as pd
from scipy.io import loadmat
import time
import datetime
import sys
import subprocess
import re
# Call modules from the PEMcoupling package
import getconfig, getdata, preprocess, analysis, savedata
# Time keeping for verbose printing of progress
t1 = time.time()




#================================
#### OPTIONS PARSING
#================================

parser = OptionParser()
parser.add_option("-D", "--dtt_list", dest = "dtt_list",
                  help = "Use times found in txt file containing DTT (.xml) filenames.")
parser.add_option("-d", "--dtt", dest = "dtt",
                  help = "Use times found in provided DTT (.xml) file(s). If using wildcard '*', close search entry in quotes.")
parser.add_option("-C", "--channel_list", dest = "channel_list",
                  help = "Txt file containing full channel names.")
parser.add_option("-c", "--channel_search", dest= "channel_search",
                  help = "Channel search keys separated by commas ',' (for AND) and forward slashes '/' (for OR) "+\
                  "(AND takes precedence over OR). Use minus signs '-' to exclude a string (i.e. NOT).")
parser.add_option("-I", "--injection_list", dest = "injection_list",
                  help = "Txt file containing list of injections, quiet times, and injetion times,"+\
                  "for running multiple injections. No config file times needed if these are provided.")
parser.add_option("-i", "--injection_search", dest="injection_search",
                  help = "Search keys for selecting times from table or .xml files by matching injection names. "+\
                  "Search keys separated by commas ',' (for AND) and forward slashes '/' (for OR) "+\
                  "(AND takes precedence over OR). Use minus signs '-' to exclude a string (i.e. NOT). "+\
                  "A list of times specified by -T (--injection_list) is required for searching.")
parser.add_option("-t", "--times", nargs = 2, type = "int", dest = "times",
                  help = "Background time and injection time (2 args). No config file times needed if these are provided.")
parser.add_option("-r", "--ratio_plot", nargs = 2, type = "int", dest = "ratio_plot",
                  help = "Create ratio plot with z-axis (ratio) minimum and maximum (2 args).")
parser.add_option("-R", "--ratio_plot_only", nargs = 2, type = "int", dest = "ratio_plot_only",
                  help = "Create ratio plot only, with z-axis (ratio) minimum and maximum (2 args); "+\
                  "quit immediately after. Useful for checking effectiveness of injections on-the-fly, "+\
                  "without having to look at dozens of spectra.")
parser.add_option("-o", "--output", dest = "directory", 
                  help = "Custom name of the directory that will hold all output data.")
parser.add_option("-v", "--verbose", action = "store_true", dest = "verbose", default = False,
                  help = "The porgram will give additional information about its procedures and show runtime "+\
                  "for specifc executions.")
(options, args) = parser.parse_args()
if len(args) != 3:
    print('\nError: Exactly 3 arguments required (configuration file, interferometer, and station).\n')
    sys.exit()

# PARSE ARGUMENTS
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
verbose = options.verbose

#=====================================================
#### CONFIG PARSING
#=====================================================

# Read config file into dictionary
config_dict = getconfig.get_config(config_name)
# Assign converted sub-dictionaries to separate dictionaries
general_dict = config_dict['General']
asd_dict = config_dict['ASD']
calib_dict = config_dict['Calibration']
smooth_dict = config_dict['Smoothing']
cf_dict = config_dict['Coupling Function']
plot_dict = config_dict['Coupling Function Plot']
coher_dict = config_dict['Coherence']
ratio_dict = config_dict['Ratio Plot']
comp_dict = config_dict['Composite Coupling Function']
if verbose:
    print('Config file parsing complete.')
# Choose DARM calibration file based on IFO
if ifo == 'H1':
    darm_notch_data = cf_dict['darm_notch_lho']
    darm_calibration_file = calib_dict['darm_cal_lho']
else:
    darm_notch_data = cf_dict['darm_notch_llo']
    darm_calibration_file = calib_dict['darm_cal_llo']
# Create dictionary for smoothing paramers
smooth_params = {}
for option, value in smooth_dict.iteritems():
    if '_smoothing' in option:
        smooth_params[option[:option.index('_')].upper()] = value
# Set coupling factor local max width to 0 if not given in config
if cf_dict['local_max_width'] is None:
    cf_dict['local_max_width'] = 0

# OVERRIDE CONFIG OPTIONS WHERE APPLICABLE
# Channel list
if options.injection_list is not None:
    general_dict['channels'] = options.injection_list
# Ratio plot options
if options.ratio_plot_only is not None:
    ratio_dict['ratio_plot'] = True
    ratio_dict['ratio_z_min'], ratio_dict['ratio_z_max'] = options.ratio_plot_only
elif options.ratio_plot is not None:
    ratio_dict['ratio_plot'] = True
    ratio_dict['ratio_z_min'], ratio_dict['ratio_z_max'] = options.ratio_plot
# Directory
if options.directory is not None:
    if options.directory[-1]=='/':
        general_dict['directory'] =  options.directory[:-1]
    else:
        general_dict['directory'] = options.directory

#==========================================
#### CHANNEL NAMES
#==========================================

if options.channel_list is not None:
    channels = getdata.get_channel_list(options.channel_list, ifo, station, search=options.channel_search, verbose=verbose)
elif '.txt' in general_dict['channels']:
    channels = getdata.get_channel_list(general_dict['channels'], ifo, station, search=options.channel_search, verbose=verbose)
else:
    print('\nChannel list input required in config file ("channels") or command line option ("--channel_list").\n')
    sys.exit()
print('\n{} channels(s) found:'.format(len(channels)))
for c in channels:
    print(c)
# Channel list to be processed by lowest coupling function calculator
# This keeps track of any new quad-sum channels generated along the way
channels_final = []
# Dictionary for keeping track of fundamental frequencies of magnetic injections
injection_freqs = {}

#===============================================
#### GET TIMES
#===============================================

injection_table = getdata.get_times(
    ifo, station=station, times=options.times, injection_list=general_dict['times'], dtt=options.dtt,\
    dtt_list=options.dtt_list, injection_search=options.injection_search
)
# REPORT INJECTIONS FOUND
if (options.times is not None) or (len(injection_table) > 0):
    if options.injection_search is not None:
        print('\n' + str(len(injection_table)) + ' injection(s) found matching search entry:')
    else:
        print('\n' + str(len(injection_table)) + ' injection(s) found:')
    for injection_name, _, _ in injection_table:
        print(injection_name)
else:
    print('\nError: No injections found matching search entry.')
    sys.exit()

#=====================================================
#### FFT CALCULATIONS
#=====================================================

# Convert averages and overlap percentage into times for the asd function
# FFT time and overlap time are necessary for taking amplitude ffts in gwpy.
# Here, the duration parameter takes precedence over bandwidth,
# i.e. if both are provided, duration is used to compute FFT time.
fft_time, overlap_time, duration, band_width = preprocess.get_FFT_params(
    asd_dict['duration'],
    asd_dict['band_width'],
    asd_dict['fft_overlap_pct'],
    asd_dict['fft_avg'],
    asd_dict['fft_rounding'],
    verbose
)
# Convert smoothing parameters from Hz to freq bins based on bandwidth
for key in smooth_params.keys():
    smooth_params[key] = [int(v/band_width) for v in smooth_params[key]]
# This time stamp will be used for directory labeling and will be displayed on outputted graphs.
t1 = time.time()
# All coupling functions put into a list for each channel, saved in one dictionary
coup_func_all = {}

###############################################################################################




#=====================================================
#### LOOP OVER INJECTION NAMES/TIMES
#=====================================================

for injection in injection_table:
    name_inj, time_bg, time_inj = injection
    # Create subdirectory for this injection
    if name_inj != '':
        subdir = name_inj
        print('\n' + '*'*20 + '\n')
        print('Running coupling calculations for ' + subdir)
    else:
        subdir = datetime.datetime.fromtimestamp(t1).strftime('DATA_%Y-%m-%d_%H:%M:%S')
    subdirectory = (general_dict['directory'] + '/' + subdir if general_dict['directory'] is not None else subdir)
    # Get fundamental frequency if this is a magnetic injection
    freq_search = getdata.freq_search(name_inj)
    if freq_search is not None:
        injection_freqs[name_inj] = freq_search

    #=====================================================
    #### PREPARE SENSOR DATA ####
    #=====================================================

    t11 = time.time()
    if not verbose: print('Fetching and pre-processing data...')
    # Exclude isolated locations (e.g. tables) from channels for certain injections (e.g. shaker)
    if 'shake' in subdir.lower():
        chans = []
        for c in channels:
            omit_channels = ['IOT', 'ISCT', 'OPLEV', 'PSL_TABLE', 'MIC']
            if all(x not in c for x in omit_channels) or ( ('PSL_TABLE' in c) and ('psl' in subdir.lower()) ):
                chans.append(c)
    elif 'acou' in subdir.lower():
        if 'ebay' in subdir.lower():
            chans = [c for c in channels if ('EBAY' in c)]
        elif 'psl' in subdir.lower():
            chans = [c for c in channels if ('PSL' in c)]
        else:
            chans = [c for c in channels if ('PSL' not in c)]
    elif 'mag' in subdir.lower() and 'ebay' not in subdir.lower():
        chans = [c for c in channels if ('EBAY' not in c)]
    else:
        chans = channels
    if len(chans) == 0:
        print('\nWarning: Excluding all given channels based on injection type.')
        continue
    
    #### GET TIME SERIES ####
    if verbose: print('Fetching sensor (background) data...')
    TS_quiet, chans_failed = getdata.get_time_series(chans, time_bg, time_bg + duration, return_failed=True)
    chans_good = [c for c in chans if c not in chans_failed]
    if len(chans_good) == 0:
        continue # No channels were successfully imported; move onto next injection
    
    if verbose: print('Fetching sensor (injection) data...')
    TS_inject = getdata.get_time_series(chans_good, time_inj, time_inj + duration)
    
    if verbose: print('All channel time series extracted. (Runtime: {:.3f} s)\n'.format(time.time() - t11))
    if len(chans_failed)>0:
        with open(subdirectory + '/0_FAILED_CHANNELS.txt', 'wb') as f:
            f.write('\n'.join(chans_failed))
    
    #### REJECT SATURATED TIME SERIES ####
    TS_inject = preprocess.reject_saturated(TS_inject, verbose)
    if len(TS_inject) == 0:
        print('All extracted time series are saturated. Skipping this injection.')
        continue
    channels_unsaturated = [ts.channel.name for ts in TS_inject]
    TS_quiet = [ts for ts in TS_quiet if ts.channel.name in channels_unsaturated]
    
    #### GET AMPLITUDE SPECTRAL DENSITIES ####
    ASD_raw_quiet = preprocess.convert_to_ASD(TS_quiet, fft_time, overlap_time)
    ASD_raw_inject = preprocess.convert_to_ASD(TS_inject, fft_time, overlap_time)
    
    t12 = time.time() - t11
    if verbose: print('ASDs completed for sensors. (Runtime: '+str(t12)+' s.)\n')
    
    #### CALIBRATE PEM SENSORS ####
    cal_results = preprocess.calibrate_sensor(ASD_raw_quiet, ifo, calib_dict['sensor_calibration'], verbose)
    ASD_cal_quiet, cal_factors, chans_uncalibrated = cal_results
    ASD_cal_inject, _, _ = preprocess.calibrate_sensor(ASD_raw_inject, ifo, calib_dict['sensor_calibration'], verbose)
    del ASD_raw_quiet, ASD_raw_inject
    if len(chans_uncalibrated) > 0:
        print('\nWarning: Channel(s) not recognized for calibration:')
        for c in chans_uncalibrated:
            print(c)
    
    #### GENERATE QUAD SUM SENSORS FROM TRI-AXIAL SENSORS ####
    if calib_dict['quad_sum']:
        if verbose:
            print('Generating quadrature sum channel from single-component channels.\n')
            
        # Compute quadrature-sum spectra
        ASD_cal_qsum_quiet = preprocess.quad_sum_ASD(ASD_cal_quiet)
        ASD_cal_qsum_inject = preprocess.quad_sum_ASD(ASD_cal_inject)
        
        # Combine quad-sum spectra with original channels
        ASD_cal_quiet_combined = ASD_cal_quiet + ASD_cal_qsum_quiet
        ASD_cal_inject_combined = ASD_cal_inject + ASD_cal_qsum_inject
        names = [asd.name for asd in ASD_cal_inject_combined]
        names_sorted = sorted(names)
        ASD_cal_quiet, ASD_cal_inject = [], []
        while len(names_sorted)>0:
            m = names_sorted.pop(0)
            ASD_cal_quiet.append(ASD_cal_quiet_combined[names.index(m)])
            ASD_cal_inject.append(ASD_cal_inject_combined[names.index(m)])
        del ASD_cal_quiet_combined, ASD_cal_inject_combined
    
    #### RATIO PLOT ####
    if ratio_dict['ratio_plot']:
        print('Generating ratio table...') # Convert to ChannelASD objects first
        chans_ASD_quiet = [preprocess.ChannelASD(i.name, i.frequencies.value, i.value) for i in ASD_cal_quiet]
        chans_ASD_inject = [preprocess.ChannelASD(i.name, i.frequencies.value, i.value) for i in ASD_cal_inject]
        ratio_method = 'raw'
        if ratio_dict['ratio_max']:
            ratio_method = 'max'
        elif ratio_dict['ratio_avg']:
            ratio_method = 'avg'
        savedata.ratio_table(
            chans_ASD_inject, chans_ASD_quiet,
            ratio_dict['ratio_z_min'], ratio_dict['ratio_z_max'],
            method=ratio_method,
            minFreq=ratio_dict['ratio_min_frequency'],
            maxFreq=ratio_dict['ratio_max_frequency'],
            directory=subdirectory, ts=t1
        )
        # QUIT CALCULATIONS IF RATIO PLOT IS ALL WE WANT (-R option instead of -r)
        if options.ratio_plot_only is not None:
            continue
        # Using this program like this serves as an easy real-time check of how extensive
        # the effect of a set of injections are, without having to open up dozens of PEM spectra.
        del chans_ASD_quiet, chans_ASD_inject
        
    #=====================================================
    #### PREPARE DARM DATA
    #=====================================================

    t_darm = time.time()
    # Fetching data and calibrating it in one function:
    if verbose:
        print('\nFetching DARM (background) data...')
    ASD_cal_darm_quiet = getdata.get_calibrated_DARM(
        ifo, calib_dict['calibration_method'], time_bg, time_bg + duration,
        fft_time, overlap_time, darm_calibration_file
    )
    if verbose:
        print('Fetching DARM (injection) data...')
    ASD_cal_darm_inject = getdata.get_calibrated_DARM(
        ifo, calib_dict['calibration_method'], time_inj, time_inj + duration,
        fft_time, overlap_time, darm_calibration_file
    )
    t_darm2 = time.time() - t_darm
    if verbose:
        print('DARM ASDs calculated. (Runtime: '+str(t_darm2)+' s.)\n')
    
    #=======================================
    #### CROP ASDs ####
    #=======================================
    
    N_chans = len(ASD_cal_quiet)
    # Match frequency domains of DARM and sensor spectra
    # Domain set either by config options (data_freq_min/max)
    # or by smallest domain shared by all spectra
    if verbose:
        print('Cropping ASDs to match frequencies.\n')
    ASD_quiet = []
    ASD_inject = []
    ASD_darm_quiet = []
    ASD_darm_inject = []
    for i in range(N_chans):
        sens_bg = ASD_cal_quiet[i]
        sens_inj = ASD_cal_inject[i]
        darm_bg = ASD_cal_darm_quiet.copy()
        darm_inj = ASD_cal_darm_inject.copy()
        sens_name = sens_bg.name
        darm_name = darm_bg.name
        sens_freqs = sens_bg.frequencies.value
        darm_freqs = darm_bg.frequencies.value
        # Create ChannelASD objects from each FrequencySeries
        asd_quiet = preprocess.ChannelASD(sens_name, sens_freqs, sens_bg.value, t0=time_bg)
        asd_inject = preprocess.ChannelASD(sens_name, sens_freqs, sens_inj.value, t0=time_inj)
        asd_darm_quiet = preprocess.ChannelASD(darm_name, darm_freqs, darm_bg.value, t0=time_bg)
        asd_darm_inject = preprocess.ChannelASD(darm_name, darm_freqs, darm_inj.value, t0=time_inj)
        # Determine frequency limits
        fmin = max([sens_freqs[0], darm_freqs[0]])
        fmax = min([sens_freqs[-1], darm_freqs[-1]])
        if asd_dict['data_freq_min'] is not None:
            fmin = max([fmin, asd_dict['data_freq_min']])
        if asd_dict['data_freq_max'] is not None:
            fmax = min([fmax, asd_dict['data_freq_max']])
        # Crop ASDs
        asd_quiet.crop(fmin, fmax)
        asd_inject.crop(fmin, fmax)
        asd_darm_quiet.crop(fmin, fmax)
        asd_darm_inject.crop(fmin, fmax)
        # Append to output list
        ASD_quiet.append(asd_quiet)
        ASD_inject.append(asd_inject)
        ASD_darm_quiet.append(asd_darm_quiet)
        ASD_darm_inject.append(asd_darm_inject)
    
    del ASD_cal_quiet, ASD_cal_inject, ASD_cal_darm_quiet, ASD_cal_darm_inject
    if verbose:
        print('ASDs are ready for coupling function calculations.\n')
    
    #=====================================================
    #### CALCULATIONS // DATA EXPORT
    #=====================================================
    
    # GET SMOOTHING PARAMETERS
    # Smoothing is applied within the coupling function calculation.
    log_smooth = smooth_dict['smoothing_log']
    smooth_params = {}
    for i in range(N_chans):
        for option, values in smooth_dict.iteritems():
            if '_smoothing' in option:
                chan_type = option[:option.index('_')].upper()
                if (chan_type in ASD_quiet[i].name) or (chan_type == 'ALL'):
                    values_bins = [int(value/ASD_quiet[i].df) for value in values]
                    smooth_params[ASD_quiet[i].name] = values_bins + [log_smooth]
                    break
            elif option == 'smoothing':
                values_bins = [int(value/ASD_quiet[i].df) for value in values]
                smooth_params[ASD_quiet[i].name] = values_bins + [log_smooth]
                break
                
    #### COUPLING FUNCTION ####
    print('Calculating coupling functions...')
    ts1 = time.time()
    coup_func_list = [] # List of CouplingData objects
    for i in range(N_chans):
        cf = analysis.coupling_function(
            ASD_quiet[i], ASD_inject[i], ASD_darm_quiet[i], ASD_darm_inject[i],
            darm_factor=cf_dict['darm_factor_threshold'], sens_factor=cf_dict['sens_factor_threshold'],
            local_max_width=cf_dict['local_max_width'], smooth_params=smooth_params[ASD_quiet[i].name],
            calibration_factors=cal_factors, notch_windows=darm_notch_data, fsearch=freq_search,
            verbose=verbose
        )
        coup_func_list.append(cf)
    ts2 = time.time() - ts1
    if verbose:
        print('Coupling functions calculated. (Runtime: {:4f} s)\n'.format(ts2))
    # Save results to dictionary for composite coupling calculations later
    coup_func_all[name_inj] = coup_func_list
    
    #### COHERENCE ####
    if coher_dict['coherence_calculator']:
        t21 = time.time()
        print('Calculating coherences...')
        coherence_results = analysis.coherence(
            calib_dict['calibration_method'], ifo,
            TS_inject, time_inj, time_inj + dur, fft_time, overlap_time,
            coher_dict['coherence_spectrum_plot'], coher_dict['coherence_threshold'], coher_dict['percent_data_threshold'],
            subdirectory, t1
        )
        t22 = time.time() - t21
        if verbose:
            print('Coherence data computed. (Runtime: '+str(t22)+' s.)\n')
    else:
        coherence_results = {}
    
    #### DATA EXPORT ####
    if (len(injection_table) > 0) and (name_inj != ''):
        print('Exporting data for ' + name_inj)
    else:
        print('Exporting data...')
    savedata.export_coup_data(
        coup_func_list,
        plot_dict['spectrum_plot'], plot_dict['darm/10'], plot_dict['upper_lim'], plot_dict['est_amb_plot'],
        plot_dict['plot_freq_min'], plot_dict['plot_freq_max'],
        plot_dict['coup_y_min'], plot_dict['coup_y_max'],
        plot_dict['spec_y_min'], plot_dict['spec_y_max'],
        plot_dict['coup_fig_height'], plot_dict['coup_fig_width'],
        plot_dict['spec_fig_height'], plot_dict['spec_fig_width'],
        subdirectory, t1, coherence_results, verbose
    )    
    if (not verbose) and ((options.injection_list is not None) or (options.injection_search is not None) or (options.dtt is not None)):
        print('\n' + subdir + ' complete.')
    
    # Update final channel list
    channels_final += [cf_data.name for cf_data in coup_func_list]
    channels_final = sorted(set(channels_final))
    
del TS_quiet, TS_inject, ASD_quiet, ASD_inject, ASD_darm_quiet, ASD_darm_inject
print('\n' + '*'*20 + '\n')
t2 = time.time()
if not options.ratio_plot_only:
    print('\nAll coupling functions finished. (Runtime: {:.3f} s.)\n'.format(t2 - t1))
# STOP HERE IF COMPOSITE COUPLING OPTION IS OFF OR RATIO-PLOT-ONLY IS CHOSEN
if (not comp_dict['composite_coupling']) or (options.ratio_plot_only):
    print('Program is finished. (Runtime: {:.3f} s.)\n'.format(t2 - t1))
    sys.exit()

##########################################################################################




#=======================================================
#### PREPARE DATA FOR COMPOSITE COUPLING FUNCTIONS
#=======================================================

print('Obtaining composite coupling functions.\n')
# IMPORT GWINC DATA FOR AMBIENT PLOT
gwinc = None
if comp_dict['gwinc_file'] is not None:
    try:
        gwinc_mat = loadmat(comp_dict['gwinc_file'])
        gwinc = [
            np.asarray(gwinc_mat['nnn']['Freq'][0][0][0]),
            np.sqrt(np.asarray(gwinc_mat['nnn']['Total'][0][0][0])) * 4000. # Convert strain PSD to DARM ASD
        ]
    except:
        print('\nWarning: GWINC data file ' + comp_dict['gwinc_file'] + ' not found. '+\
              'Composite estimated ambient will not show GWINC.\n')

# LOOP OVER CHANNELS AND PERFORM COMPOSITE COUPLING CALCULATION ON EACH
for channel_name in sorted(channels_final):
    print('Processing results for ' + channel_name + '.')
    # PARSE RESULTS FOR COUPLING FUNCTION DATA AND INJECTION NAMES
    cf_data_list = []
    inj_names = []
    for inj_name, coup_func_list in coup_func_all.items():
        for cf_data in coup_func_list:
            if cf_data.name == channel_name:
                cf_data_list.append(cf_data)
                inj_names.append(inj_name)
    freqs = np.mean([cf_data.freqs for cf_data in cf_data_list], axis=0)
    darm = np.mean([cf_data.darm_bg for cf_data in cf_data_list], axis=0)
    # CHECK BANDWIDTHS AND COLUMN LENGTHS
    band_widths = [cf_data.df for cf_data in cf_data_list]
    column_len = [cf_data.freqs.shape for cf_data in cf_data_list]
    if (
        any(bw != band_widths[0] for bw in band_widths) or \
        any([k != column_len[0] for k in column_len])
    ):
        print('\nError: Coupling data objects have unequal data lengths.')
        print('If all the band_widths are the same, this should not be an issue.\n')
    # COMPUTE COMPOSITE COUPLING FUNCTION
    local_max_window = int(cf_dict['local_max_width'] / band_widths[0]) # Convert from Hz to bins
    comp_data = analysis.composite_coupling_function(
        cf_data_list, inj_names,
        local_max_window=local_max_window,
        freq_lines=injection_freqs
    )    
    # GAUSSIAN SMOOTHING JUST BEFORE EXPORT; FOR SLIGHTLY SMOOTHER DATA
    smooth_chans = ['ACC', 'MIC' 'WFS']    # Apply this procedure only to these channels
    width = 0.005                          # stdev of Gaussian kernel
    if any(x in channel_name for x in smooth_chans):
        comp_data = analysis.smooth_comp_data(comp_data, width)
    # EXPORT RESULTS
    path = general_dict['directory'] + '/CompositeCouplingFunctions'
    savedata.export_composite_coupling_data(comp_data, freqs, darm, gwinc, inj_names, path, comp_dict, verbose)
    ####
    # APPLY ALL THE ABOVE STEPS TO BINNED DATA
    cf_binning = comp_dict['coupling_function_binning']
    if cf_binning is not None:
        # Bin original coupling function data
        cf_data_binned_list = [analysis.bin_coupling_data(cf_data, cf_binning) for cf_data in cf_data_list]
        # Keep track of binned frequencies and DARM separately for plotting an unbinned DARM
        freqs_binned = np.mean([cf_data.freqs for cf_data in cf_data_binned_list], axis=0)
        darm_binned = np.mean([cf_data.darm_bg for cf_data in cf_data_binned_list], axis=0)
        # Composite coupling function
        comp_data_binned = analysis.composite_coupling_function(
            cf_data_binned_list, inj_names,
            local_max_window=local_max_window
        )
        # Final smoothing
        if any(x in channel_name for x in smooth_chans):
            comp_data_binned = analysis.smooth_comp_data(comp_data_binned, width)
        # Data export
        path_binned = path + 'Binned'
        savedata.export_composite_coupling_data(comp_data_binned, freqs, darm, gwinc, inj_names, path_binned, comp_dict, verbose)
    if verbose:
        print('\nLowest (composite) coupling function complete for ' + channel_name)
t3 = time.time()
print('Lowest (composite) coupling functions processed. (Runtime: {:.3f} s.)\n'.format(t3 - t2))
print('Program is finished. (Runtime: {:.3f} s.)\n'.format(t3 - t1))

sys.exit()