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
"""

from optparse import OptionParser
import ConfigParser
import numpy as np
import sys
import os
import subprocess
import time
import datetime
import logging
 # Global time-stamp, for data exports
t1 = time.time()
# Configure event logger
if not os.path.exists('logging/'):
    os.makedirs('logging/')
logging_filename = 'logging/' + datetime.datetime.fromtimestamp(t1).strftime('%Y_%b_%d_%H:%M:%S') + '.log'
logging.basicConfig(filename=logging_filename,
                    level=logging.DEBUG,
                    format='%(asctime)s %(filename)s (%(funcName)s) %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')
stderrLogger = logging.StreamHandler()
stderrLogger.setLevel(logging.WARNING)
stderr_formatter = logging.Formatter('%(levelname)s: %(message)s')
stderrLogger.setFormatter(stderr_formatter)
logging.getLogger().addHandler(stderrLogger)
logging.info('Importing PEM coupling packages.')
# pemcoupling modules
try:
    from coupling import getparams, loaddata, preprocess
    from coupling.pemchannel import PEMChannelASD
    from coupling.coupfunc import CoupFunc
    from coupling.coherence import coherence
    from coupling.savedata import ratio_table
    from PEMcoupling_composite import get_composite_coup_func
except ImportError:
    print('')
    logging.error('Failed to load PEM coupling modules. Make sure you have all of these in the right place!')
    print('')
    raise

#================================
#### OPTIONS PARSING
#================================

parser = OptionParser()
parser.add_option("-d", "--dtt", dest = "dtt",
                  help = "Use times found in provided DTT (.xml) file(s). Input can be one or more .xml files separated by commas"+\
                  "or a directory containing .xml files. If using wildcard '*', close search entry in quotes.")
parser.add_option("-D", "--dtt_list", dest = "dtt_list",
                  help = "Use times found in txt file containing DTT (.xml) filenames.")
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
verbose = options.verbose
ifo, station, config_name = getparams.get_arg_params(args)

#=====================================================
#### CONFIG PARSING
#=====================================================

# Read config file into dictionary
config_dict = getparams.get_config_params(config_name)
if len(config_dict) == 0:
    print('')
    logging.error('Config parsing failed.')
    print('')
    sys.exit()
# Assign converted sub-dictionaries to separate dictionaries
logging.info('Separating config dictionary by section.')
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
logging.info('Handling conflicts between config options and command line options.')
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
    general_dict['times'] = options.injection_list
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
    logging.info('Reading in channel list given by command-line option.')
    channels = getparams.get_channel_list(options.channel_list, ifo, station, search=options.channel_search, verbose=verbose)
elif '.txt' in general_dict['channels']:
    logging.info('Reading in channel list given in config file.')
    channels = getparams.get_channel_list(general_dict['channels'], ifo, station, search=options.channel_search, verbose=verbose)
else:
    print('')
    logging.error('Channel list input required in config file ("channels") or command line option ("--channel_list").')
    print('')
    sys.exit()
if len(channels) == 0:
    print('')
    logging.error('No channels to process.')
    print('')
    sys.exit()
print('\n{} channels(s) found:'.format(len(channels)))
for c in channels:
    print(c)

#===============================================
#### GET TIMES
#===============================================

logging.info('Creating injection table.')
injection_table = getparams.get_times(ifo, station=station, times=options.times, dtt=options.dtt, dtt_list=options.dtt_list,\
                                      injection_list=general_dict['times'], injection_search=options.injection_search)
# REPORT INJECTIONS FOUND
logging.info('Reporting injection results.')
if (options.times is not None) or (len(injection_table) > 0):
    if options.injection_search is not None:
        print('\n' + str(len(injection_table)) + ' injection(s) found matching search entry:')
    else:
        print('\n' + str(len(injection_table)) + ' injection(s) found:')
    for injection_name, _, _ in injection_table:
        print(injection_name)
else:
    print('')
    logging.error('No injections to process.')
    print('')
    sys.exit()
# Dictionary for keeping track of fundamental frequencies of magnetic injections
freq_lines = {}

#=====================================================
#### FFT CALCULATIONS
#=====================================================

# Convert averages and overlap percentage into times for the asd function
# FFT time and overlap time are necessary for taking amplitude ffts in gwpy.
# Here, the duration parameter takes precedence over bandwidth,
# i.e. if both are provided, duration is used to compute FFT time.
logging.info('Calculating FFT parameters.')
fft_time, overlap_time, duration, band_width = getparams.get_FFT_params(
    asd_dict['duration'], asd_dict['band_width'], asd_dict['fft_overlap_pct'],
    asd_dict['fft_avg'], asd_dict['fft_rounding'], verbose
)
# Convert smoothing parameters from Hz to freq bins based on bandwidth
logging.info('Converting smoothing parameters to frequency bins based on bandwidth.')
for key in smooth_params.keys():
    smooth_params[key] = [int(v/band_width) for v in smooth_params[key]]
# This time stamp will be used for directory labeling and will be displayed on all plots.
t1 = time.time()

###############################################################################################

#=====================================================
#### LOOP OVER INJECTION NAMES/TIMES
#=====================================================

# All coupling functions put into a list for each channel, saved in one dictionary
coup_func_results = {}
logging.info('Beginning to loop over injections.')
for injection in injection_table:
    injection_name, time_bg, time_inj = injection
    # Create subdirectory for this injection
    if injection_name != '':
        print('\n' + '*'*20 + '\n')
        print('Analyzing injection: ' + injection_name)
        new_dir = injection_name
    else:
        new_dir = datetime.datetime.fromtimestamp(t1).strftime('DATA_%Y-%m-%d_%H:%M:%S')
    if general_dict['directory'] is not None:
        out_dir = os.path.join(general_dict['directory'], new_dir)
    else:
        out_dir = new_dir
    print('Output directory for this injection:\n' + out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        logging.info('Subdirectory ' + out_dir + ' created.')
    # Get fundamental frequency if this is a magnetic injection
    logging.info('Searching for fundamental frequency if applicable.')
    freq_search = getparams.freq_search(injection_name, verbose=True)
    if freq_search is not None:
        freq_lines[injection_name] = freq_search

    #=====================================================
    #### PREPARE SENSOR DATA
    #=====================================================

    t11 = time.time()
    if not verbose:
        print('Fetching and pre-processing data...')
        
    #### BE PICKY ABOUT CHANNELS ####
    # Exclude isolated locations (e.g. tables) from channels for certain injections (e.g. shaker)
    logging.info('Being picky about channels.')
    if 'shake' in injection_name.lower():
        chans = []
        for c in channels:
            omit_channels = ['IOT', 'ISCT', 'OPLEV', 'PSL_TABLE', 'MIC']
            if all(x not in c for x in omit_channels) or ( ('PSL_TABLE' in c) and ('psl' in injection_name.lower()) ):
                chans.append(c)
    elif 'acou' in injection_name.lower():
        if 'ebay' in injection_name.lower():
            chans = [c for c in channels if ('EBAY' in c)]
        elif 'psl' in injection_name.lower():
            chans = [c for c in channels if ('PSL' in c)]
        else:
            chans = [c for c in channels if ('PSL' not in c)]
    elif 'mag' in injection_name.lower() and 'ebay' not in injection_name.lower():
        chans = [c for c in channels if ('EBAY' not in c)]
    else:
        chans = channels
    if len(chans) == 0:
        print('')
        logging.warning('All given channels were excluded for this injection type. Moving on to the next injection.')
        print('')
        continue
    
    #### GET TIME SERIES ####
    if verbose: print('Fetching sensor (background) data...')
    TS_bg_list, chans_failed = loaddata.get_time_series(chans, time_bg, time_bg + duration, return_failed=True)
    chans_good = [c for c in chans if c not in chans_failed]
    if len(chans_good) == 0:
        continue # No channels were successfully imported; move onto next injection
    
    if verbose: print('Fetching sensor (injection) data...')
    TS_inj_list = loaddata.get_time_series(chans_good, time_inj, time_inj + duration)
    
    if verbose: print('All channel time series extracted. (Runtime: {:.3f} s)\n'.format(time.time() - t11))
    if len(chans_failed)>0:
        with open(out_dir + '/0_FAILED_CHANNELS.txt', 'wb') as f:
            f.write('\n'.join(chans_failed))
    
    #### REJECT SATURATED TIME SERIES ####
    TS_inj_list = preprocess.reject_saturated(TS_inj_list, verbose)
    if len(TS_inj_list) == 0:
        print('All extracted time series are saturated. Skipping this injection.')
        continue
    channels_unsaturated = [ts.channel.name for ts in TS_inj_list]
    TS_bg_list = [ts for ts in TS_bg_list if ts.channel.name in channels_unsaturated]
    
    #### GET AMPLITUDE SPECTRAL DENSITIES ####
    ASD_raw_bg_list = preprocess.convert_to_ASD(TS_bg_list, fft_time, overlap_time)
    ASD_raw_inj_list = preprocess.convert_to_ASD(TS_inj_list, fft_time, overlap_time)
    t12 = time.time() - t11
    if verbose:
        print('ASDs completed for sensors. (Runtime: '+str(t12)+' s.)\n')
    
    #### CALIBRATE PEM SENSORS ####
    cal_results = preprocess.calibrate_sensors(ASD_raw_bg_list, calib_dict['sensor_calibration'], verbose)
    ASD_bg_list, cal_factors, chans_uncalibrated = cal_results
    ASD_inj_list, _, _ = preprocess.calibrate_sensors(ASD_raw_inj_list, calib_dict['sensor_calibration'], verbose)
    del ASD_raw_bg_list, ASD_raw_inj_list
    N_uncal = len(chans_uncalibrated)
    if N_uncal > 0:
        print(str(N_uncal) + ' channels were not calibrated:')
        for c in chans_uncalibrated:
            print(c)
            logging.info('Channel not calibrated: ' + c)
    N_chans = len(ASD_bg_list)
    
    #### GENERATE QUAD SUM SENSORS FROM TRI-AXIAL SENSORS ####
    if calib_dict['quad_sum']:
        if verbose:
            print('Generating quadrature sum channel from single-component channels.\n')
        # Compute quadrature-sum spectra
        ASD_qsum_bg_list = preprocess.quad_sum_ASD(ASD_bg_list)
        ASD_qsum_inj_list = preprocess.quad_sum_ASD(ASD_inj_list)
        N_chans += len(ASD_qsum_bg_list)
        # Combine quad-sum spectra with original channels and sort the resulting list
        ASD_bg_combined = ASD_bg_list + ASD_qsum_bg_list
        ASD_inj_combined = ASD_inj_list + ASD_qsum_inj_list
        names = [asd.name for asd in ASD_inj_combined]
        names_sorted = sorted(names)
        ASD_bg_list, ASD_inj_list = [], []
        while len(names_sorted)>0:
            m = names_sorted.pop(0)
            ASD_bg_list.append(ASD_bg_combined[names.index(m)])
            ASD_inj_list.append(ASD_inj_combined[names.index(m)])
        del ASD_bg_combined, ASD_inj_combined
    
    #### RATIO PLOT ####
    if ratio_dict['ratio_plot']:
        print('Generating ratio table...') # Convert to ChannelASD objects first
        ratio_method = 'raw'
        if ratio_dict['ratio_max']:
            ratio_method = 'max'
        elif ratio_dict['ratio_avg']:
            ratio_method = 'avg'
        ratio_table(
            ASD_bg_list, ASD_inj_list,
            ratio_dict['ratio_z_min'], ratio_dict['ratio_z_max'],
            method=ratio_method, minFreq=ratio_dict['ratio_min_frequency'], maxFreq=ratio_dict['ratio_max_frequency'], 
            directory=out_dir, ts=t1
        )
        # QUIT CALCULATIONS IF RATIO PLOT IS ALL WE WANT (-R option instead of -r)
        if options.ratio_plot_only is not None:
            continue
        # Using this program like this serves as an easy real-time check of how extensive
        # the effect of a set of injections are, without having to open up dozens of PEM spectra.
        
    #======================================
    #### PREPARE DARM DATA
    #======================================

    t_darm = time.time()
    # Fetching data and calibrating it in one function:
    if verbose:
        print('\nFetching DARM (background) data...')
    ASD_darm_bg_single = loaddata.get_calibrated_DARM(
        ifo, calib_dict['calibration_method'], time_bg, time_bg + duration,
        fft_time, overlap_time, darm_calibration_file
    )
    if verbose:
        print('Fetching DARM (injection) data...')
    ASD_darm_inj_single = loaddata.get_calibrated_DARM(
        ifo, calib_dict['calibration_method'], time_inj, time_inj + duration,
        fft_time, overlap_time, darm_calibration_file
    )
    # Clone DARM ASDs; each DARM ASD pairs with a sensor ASD
    ASD_darm_bg_list = []
    ASD_darm_inj_list = []
    for i in range(N_chans):
        darm_bg = ASD_darm_bg_single.copy()
        darm_inj = ASD_darm_inj_single.copy()
        ASD_darm_bg = PEMChannelASD(darm_bg.name, darm_bg.frequencies.value, darm_bg.value, t0=time_bg)
        ASD_darm_inj = PEMChannelASD(darm_inj.name, darm_inj.frequencies.value, darm_inj.value, t0=time_inj)
        ASD_darm_bg_list.append(ASD_darm_bg)
        ASD_darm_inj_list.append(ASD_darm_inj)
    t_darm2 = time.time() - t_darm
    if verbose:
        print('DARM ASDs calculated and calibrated. (Runtime: '+str(t_darm2)+' s.)\n')
    
    #=======================================
    #### CROP ASDs
    #=======================================
    
    # Match frequency domains of DARM and sensor spectra
    # Domain set either by config options (data_freq_min/max)
    # or by smallest domain shared by all spectra
    if verbose:
        print('Cropping ASDs to match frequencies.\n')
    for i in range(N_chans):
        sens_freqs = ASD_bg_list[i].freqs
        darm_freqs = ASD_darm_bg_list[i].freqs
        fmin = max([sens_freqs[0], darm_freqs[0]])
        fmax = min([sens_freqs[-1], darm_freqs[-1]])
        if asd_dict['data_freq_min'] is not None:
            fmin = max([fmin, asd_dict['data_freq_min']])
        if asd_dict['data_freq_max'] is not None:
            fmax = min([fmax, asd_dict['data_freq_max']])
        logging.info('Cropping ASDs to match frequencies: ' + ASD_bg_list[i].name + '.')
        ASD_bg_list[i].crop(fmin, fmax)
        ASD_inj_list[i].crop(fmin, fmax)
        ASD_darm_bg_list[i].crop(fmin, fmax)
        ASD_darm_inj_list[i].crop(fmin, fmax)
    if verbose:
        print('ASDs are ready for coupling function calculations.\n')
    
    #======================================
    #### ANALYSIS
    #======================================
    
    #### GET SMOOTHING PARAMETERS ####
    # Smoothing is applied within the coupling function calculation.
    log_smooth = smooth_dict['smoothing_log']
    smooth_params = {}
    for i in range(N_chans):
        chan_name = ASD_bg_list[i].name
        logging.info('Acquiring smoothing parameters for channel ' + chan_name + '.')
        smooth_params[chan_name] = None
        for option, values in smooth_dict.iteritems():
            if 'shake' in injection_name:
                values[0] = values[1]
            if '_smoothing' in option:
                chan_type = option[:option.index('_')].upper()    # Channel type from smoothing dict
                if (chan_type in chan_name) or (chan_type == 'ALL'):
                    values_bins = [int(value/ASD_bg_list[i].df) for value in values]
                    smooth_params[chan_name] = values_bins + [log_smooth]
                    break
            elif option == 'smoothing':
                values_bins = [int(value/ASD_bg_list[i].df) for value in values]
                smooth_params[chan_name] = values_bins + [log_smooth]
                break
    #### COUPLING FUNCTION ####
    print('Calculating coupling functions...')
    ts1 = time.time()
    coup_func_list = []
    for i in range(N_chans):
        channel_name = ASD_bg_list[i].name
        cf = CoupFunc.compute(
            ASD_bg_list[i], ASD_inj_list[i], ASD_darm_bg_list[i], ASD_darm_inj_list[i],
            darm_factor=cf_dict['darm_factor_threshold'], sens_factor=cf_dict['sens_factor_threshold'],
            local_max_width=cf_dict['local_max_width'], smooth_params=smooth_params[channel_name],
            notch_windows=darm_notch_data, fsearch=freq_search,
            injection_name=injection_name,
            verbose=verbose
        )
        coup_func_list.append(cf)
        logging.info('Coupling function complete: ' + channel_name + '.')
        if channel_name not in coup_func_results.keys():
            coup_func_results[channel_name] = []
        coup_func_results[channel_name].append((injection_name, cf))
    ts2 = time.time() - ts1
    if verbose:
        print('Coupling functions calculated. (Runtime: {:4f} s)\n'.format(ts2))
    #### COHERENCE ####
    if coher_dict['coherence_calculator']:
        t21 = time.time()
        print('Calculating coherences...')
        coherence_results = coherence(
            calib_dict['calibration_method'], ifo,
            TS_inj, time_inj, time_inj + dur, fft_time, overlap_time,
            coher_dict['coherence_spectrum_plot'], coher_dict['coherence_threshold'], coher_dict['percent_data_threshold'],
            out_dir, t1
        )
        t22 = time.time() - t21
        if verbose:
            print('Coherence data computed. (Runtime: '+str(t22)+' s.)\n')
    else:
        coherence_results = {}
        
    #======================================
    #### DATA EXPORT
    #======================================
    
    print('Saving results...')
    for cf in coup_func_list:        
        if verbose:
            print('Exporting data for {}'.format(cf.name))
        base_filename = cf.name[cf.name.index('-')+1:].replace('_DQ','')
        cf_plot_filename = os.path.join(out_dir, base_filename + '_coupling_plot')
        cf_counts_plot_filename = os.path.join(out_dir, base_filename + '_coupling_counts_plot')
        spec_plot_filename = os.path.join(out_dir, base_filename + '_spectrum.png')
        csv_filename = os.path.join(out_dir, base_filename + '_coupling_data.txt')
        #### COUPLING FUNCTION PLOT ####
        # Coupling function in physical sensor units
        cf.plot(
            cf_plot_filename,
            in_counts=False, ts=t1, upper_lim=plot_dict['upper_lim'],
            freq_min=plot_dict['plot_freq_min'], freq_max=plot_dict['plot_freq_max'],
            factor_min=plot_dict['coup_y_min'], factor_max=plot_dict['coup_y_max'],
            fig_w=plot_dict['coup_fig_width'], fig_h=plot_dict['coup_fig_height']
        )
        # Coupling function in raw sensor counts
        cf.plot(
            cf_counts_plot_filename,
            in_counts=True, ts=t1, upper_lim=plot_dict['upper_lim'],
            freq_min=plot_dict['plot_freq_min'], freq_max=plot_dict['plot_freq_max'],
            factor_min=plot_dict['coup_y_min'], factor_max=plot_dict['coup_y_max'],
            fig_w=plot_dict['coup_fig_width'], fig_h=plot_dict['coup_fig_height']
        )
        # ASD PLOT WITH ESTIMATED AMBIENTS
        if plot_dict['spectrum_plot']:
            cf.specplot(
                spec_plot_filename,
                ts=t1, est_amb=plot_dict['est_amb_plot'],
                show_darm_threshold=plot_dict['darm/10'], upper_lim=plot_dict['upper_lim'],
                freq_min=plot_dict['plot_freq_min'], freq_max=plot_dict['plot_freq_max'],
                spec_min=plot_dict['spec_y_min'], spec_max=plot_dict['spec_y_max'],
                fig_w=plot_dict['spec_fig_width'], fig_h=plot_dict['spec_fig_height'],
            )
        # CSV DATA FILE
        if any(cf.flags != 'No data'):
            # Look for coherence data
            try:
                coherence_data = coherence_results[cf.name]
            except KeyError:
                coherence_data = None
            cf.to_csv(csv_filename, coherence_data=coherence_data)
    logging.info('Coupling functions saved for injection ' + injection_name + '.')
    if (not verbose) and ((options.injection_list is not None) or (options.injection_search is not None) or (options.dtt is not None)):
        print('\n' + new_dir + ' complete.')
    del TS_bg_list, TS_inj_list, ASD_bg_list, ASD_inj_list, ASD_darm_bg_list, ASD_darm_inj_list

#===========================

print('\n' + '*'*20 + '\n')
t2 = time.time()
if not options.ratio_plot_only:
    print('\nAll coupling functions finished. (Runtime: {:.3f} s.)\n'.format(t2 - t1))
# STOP HERE IF NOT PERFORMING COMPOSITE COUPLING OPTION
if (not comp_dict['composite_coupling']) or (options.ratio_plot_only) or (len(coup_func_results) == 0) or (options.times):
    if len(coup_func_results) == 0:
        print('')
        logging.warning('No coupling functions found.')
        print('')
    print('Program is finished. (Runtime: {:.3f} s.)\n'.format(t2 - t1))
    sys.exit()

##########################################################################################

#==============================================
#### COMPOSITE COUPLING FUNCTIONS
#==============================================

print('Obtaining composite coupling functions.\n')
# Import GWINC data for plotting
try:
    gwinc = loaddata.get_gwinc(comp_dict['gwinc_file'])
except IOError:
    gwinc = None
if gwinc is None:
    print('')
    logging.warning('Composite estimated ambient will not show GWINC.')
    print('')
final_channel_list = sorted(coup_func_results.keys())
for channel_name in final_channel_list:
    print('Creating composite coupling function for ' + channel_name + '.')
    cf_list = []
    injection_names = []
    for inj_name, coup_func in coup_func_results[channel_name]:
        injection_names.append(inj_name)
        cf_list.append(coup_func)
    out_dir = os.path.join(general_dict['directory'], 'CompositeCouplingFunctions')
    get_composite_coup_func(
        cf_list, injection_names, out_dir,
        freq_lines=freq_lines, gwinc=gwinc, local_max_width=cf_dict['local_max_width'],
        upper_lim=comp_dict['upper_lim'], est_amb_plot=comp_dict['comp_est_amb_plot'],
        freq_min=comp_dict['comp_freq_min'], freq_max=comp_dict['comp_freq_max'],
        factor_min=comp_dict['comp_y_min'], factor_max=comp_dict['comp_y_max'],
        fig_w=comp_dict['comp_fig_width'], fig_h=comp_dict['comp_fig_height'],
        verbose=verbose
    )
t3 = time.time()
print('Lowest (composite) coupling functions processed. (Runtime: {:.3f} s.)\n'.format(t3 - t2))
print('Program is finished. (Runtime: {:.3f} s.)\n'.format(t3 - t1))