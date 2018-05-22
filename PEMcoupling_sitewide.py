import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
plt.switch_backend('Agg')
from scipy.io import loadmat
from textwrap import wrap
import datetime
import time
import sys
import os
import subprocess
import re
from optparse import OptionParser
import logging
 # Global time-stamp, for data exports
t0 = time.time()
# Configure event logger
if not os.path.exists('logging/'):
    os.makedirs('logging/')
logging_filename = 'logging/' + datetime.datetime.fromtimestamp(t0).strftime('%Y_%b_%d_%H:%M:%S') + '.log'
logging.basicConfig(filename=logging_filename,
                    level=logging.DEBUG,
                    format='%(asctime)s %(pathname)s (%(funcName)s) %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S')
stderrLogger = logging.StreamHandler()
stderrLogger.setLevel(logging.WARNING)
stderr_formatter = logging.Formatter('%(levelname)s: %(message)s')
stderrLogger.setFormatter(stderr_formatter)
logging.getLogger().addHandler(stderrLogger)
logging.info('Importing PEM coupling packages.')
# pemcoupling modules
try:
    from utils import pem_sort
    from analysis import get_gwinc, max_coupling_function, max_estimated_ambient
    from summaryplots import plot_summary_coupfunc, plot_summary_ambient
except ImportError:
    print('')
    logging.error('Failed to load PEM coupling modules. Make sure you have all of these in the right place!')
    print('')
    raise

#=========================================
#### OPTIONS/ARGUMENTS PARSING
#=========================================

parser = OptionParser()
parser.add_option("-o","--omit", type = str, dest = "omit",
                  help = "Omit channels containing specified string.")
parser.add_option("-s","--search", type = str, dest = "search",
                  help = "Search for channels containing specified string.")
parser.add_option("-C","--channel_list", type = str, dest = "channel_list",
                  help = "Search for channels listed in given file.")
parser.add_option("-n","--no_upperlim", action = "store_false", dest = "upper_lim", default = True,
                  help = "Hide upper limits from plots.")
(options, args) = parser.parse_args()
if len(args) < 4:
    print('')
    logging.error('Four args required:\n[composite coupling function directory] [interferometer] [station] [injection type]')
    print('')
    sys.exit()
directory, ifo_input, station_input, injection_type = args
# Make sure interferometer input is valid
if ifo_input.lower() in ['h1', 'lho']:
    ifo = 'H1'
elif ifo_input.lower() in ['l1', 'llo']:
    ifo = 'L1'
else:
    print('')
    logging.error('Argument "ifo" must be one of "H1", "LHO", "L1", or "LHO" (not case-sensitive).')
    print('')
    sys.exit()
# Make sure station input is valid
if station_input.upper() in ['CS', 'EX', 'EY', 'ALL']:
    station = station_input.upper()
else:
    print('')
    logging.error('Argument "station" must be one of "CS", "EX", "EY", or "ALL" (not case-sensitive).')
    print('')
    sys.exit()
# Make sure injection input is valid
if injection_type.lower() in ['vib', 'vibrational']:
    injection_type = 'Vibrational'
elif injection_type.lower() in ['mag', 'magnetic']:
    injection_type = 'Magnetic'
else:
    print('')
    logging.error('Argument "injection_type" must be one of "vibrational" or "magnetic" (not case-sensitive).')
    print('')
    sys.exit()
# OTHER OPTIONS
upper_lim = options.upper_lim
freq_range = None
w_Hz = (0.5 if 'mag' in injection_type.lower() else 0) # Local max window width
# Marker properties
ms_real = (8. if 'MAG' in injection_type else 4.)
ms_upper = ms_real * (.8 if 'MAG' in injection_type else .6)
edgew_real = (.7 if 'MAG' in injection_type else .5)
edgew_upper = (1.5 if 'MAG' in injection_type else .7)
# PARSE SEARCH/OMIT ENTRIES
if options.omit is None:
    omit = ''
else:
    omit = options.omit.upper()
if options.search is None:
    search = ''
else:
    search = options.search.upper()
# GENERATE EXPORT FILE NAMES
if directory[-1] != '/':
    directory += '/'
subdir = directory + 'SummaryPlots/{}/'.format(str(time.ctime(t0))[4:].replace(' ','_'))
filepath = subdir + ifo + '_' + injection_type + '_' + station.replace(' ','')
if len(search) > 0:
    filepath += '_' + search.replace('(','').replace(')','').replace('|','and').replace('+','')
if len(omit) > 0:
    filepath += '_no' + omit.replace('(','').replace(')','').replace('|','and').replace('+','')
if freq_range is not None:
    filepath += '_' + str(freq_range[0]) + 'to' + str(freq_range[1])
csv_filepath = filepath + '_max_coupling_data.txt'
plot_filepath1 = filepath + '_max_coupling_plot.png'
plot_filepath2 = filepath + '_max_ambient.png'
plot_filepath1_no_upperlim = filepath + '_max_coupling_plot.png'
plot_filepath2_no_upperlim = filepath + '_max_ambient.png'
# PLOT TITLE
title_dict = {'Site-Wide': 'Site-Wide', 'CS': 'Corner Station', 'EX': 'End Station X', 'EY': 'End Station Y'}
title = ifo + ' ' + injection_type + ' - ' + title_dict[station]
if search != '' and 'XYZ' not in search:
    title += ' (' + search.replace('(','').replace(')','').replace('|','and').replace('+','') + ')'
subtitle1 = '(Highest coupling factor at each frequency across all channels)'
subtitle2 = '(Obtained from highest coupling factor at each frequency across all channels)'
subtitle3 = '(Highest estimated ambient at each frequency across all channels)'

#=================================
#### DATA IMPORT
#=================================

# Find composite coupling functions in given directory
print('Importing composite coupling functions...')
try:
    file_names = subprocess.check_output(['ls ' + directory + '/' + '*_composite_coupling_data.txt'], shell=True).splitlines()
except subprocess.CalledProcessError:
    print('')
    logging.error('No coupling data found in directory ' + directory + '.')
    print('')
    raise
# Search for correct ifo, station
if station in ['CS', 'EX', 'EY']:
    station_chans = []
    for f in file_names:
        chan = f.split('/')[-1]
        if '-' in chan:
            if chan[ chan.index('-')+1 : chan.index('-')+3 ] == station:
                station_chans.append(f)
                logging.info('Including data from ' + f)
        elif chan[:2] == station:
            station_chans.append(f)
            logging.info('Including data from ' + f)
    file_names = station_chans
channel_names = [f.split('/')[-1].replace('_composite_coupling_data.txt', '') for f in file_names]
# Search with channel list
if options.channel_list is not None:
    logging.info('Using channel list ' + options.channel_list + ' to narrow down channels.')
    with open(options.channel_list) as cl:
        lines = [line.replace('_DQ', '').replace('\n','') for line in cl.readlines()]
        channel_names = [c for c in channel_names if c in lines]
# Search with command line option
if search != '':
    channel_names = [c for c in channel_names if search != '' and re.search(search, c)]
if omit != '':
    channel_names = [c for c in channel_names if omit != '' and not re.search(omit, c)]
file_names = [f for f in file_names if f.split('/')[-1].replace('_composite_coupling_data.txt', '') in channel_names]
# Import files as Pandas Dataframes
bw_list = []
data_list = []
channels = []
freqs_raw = []
for f in file_names:
    c = f.split('/')[-1].replace('_composite_coupling_data.txt', '')
    # Import file
    try:
        df = pd.read_csv(f)
    except:
        print('No lowest coupling data found for ' + c)
        continue
    # Crop to desired frequency range
    if freq_range is not None:
        df = df[(df['frequency']>=freq_range[0]) & (df['frequency']<freq_range[1])]
    # Save raw raw frequencies and darm for plotting later
    if df.shape[0] > len(freqs_raw):
        freqs_raw = np.asarray(df['frequency'])#[~pd.isnull(df['darm'])])
    bw_list.append( df['frequency'].iloc[10] - df['frequency'].iloc[9] )
    data_list.append(df)
    if '-' in c:
        c = c[ c.index('-')+1 : ]    # Short channel name (No ifo and subsystem)
    channels.append(c)
print('Channels successfully imported:')
for c in channels:
    print(c)
# CHECK BANDWIDTHS
if not all([bw == bw_list[0] for bw in bw_list]):
    print(bw_list)
    print('\nError: Unequal bandwidths.\n')
    sys.exit()
# CHECK DATA-FRAME SIZES
if not all([i.shape[0] == data_list[0].shape[0] for i in data_list]):
    for i, df in enumerate(data_list):
        if df.shape[0] < len(freqs_raw):
            df_zeros = pd.DataFrame(index=range(df.shape[0], len(freqs_raw)), columns = df.columns)
            df = df.append(df_zeros)
            df['frequency'] = freqs_raw
            data_list[i] = df
# RE-ORGANIZE DATAFRAMES
# One dataframe per variable, with channels as columns
df_index = np.array(range(data_list[0].shape[0])) # Index for ALL dataframes
factor_df = pd.DataFrame(index=df_index)
flag_df = pd.DataFrame(index=df_index)
amb_df = pd.DataFrame(index=df_index)
darm_df = pd.DataFrame(index=df_index)
for c, df in zip(*(channels, data_list)):
    factor_df[c] = df['factor']
    flag_df[c] = df['flag']
    amb_df[c] = df['ambient']
    darm_df[c] = df['darm']
# FREQUENCIES AND AVERAGE DARM FOR PLOTTING DARM BACKGROUND
freqs = np.asarray(data_list[0]['frequency'])
darm_avg = np.asarray([np.mean(row[~np.isnan(row)]) for i,row in darm_df.iterrows()])
darm_freqs = freqs[~np.isnan(darm_avg)]
darm_values = darm_avg[~np.isnan(darm_avg)]
darm_data = [darm_freqs, darm_values]
# UNITS FOR PLOT LABEL
units_dict = {'MIC': 'Pa', 'MAG': 'T', 'RADIO': 'ADC', 'SEIS': 'm', 'ISI': 'm', 'ACC': 'm', 'HPI': 'm'}
units = []
for key,val in units_dict.iteritems():
    if any(key in c for c in channels):
        units.append(val)
print('Coupling functions imported.')
# GWINC (GW Inteferometer Noise Calculator) DARM ASD
try:
    gwinc = get_gwinc('config_files/darm/gwinc_nomm_noises.mat')
except IOError:
    gwinc = None
if gwinc is None:
    print('')
    logging.error('Composite estimated ambient will not show GWINC.')
    print('')
    sys.exit()

#================================
#### ANALYSIS // DATA EXPORT
#================================

#### COMPUTE SUMMARY DATA ####
print('Determining maximum coupling function...')
max_factor_df = max_coupling_function(freqs, factor_df, flag_df)
print('Determining maximum estimated ambient...')
max_amb_df = max_estimated_ambient(freqs, amb_df, flag_df)
#### CREATE OUTPUT DIRECTORY ####
if not os.path.exists(subdir):
    os.makedirs(subdir)
#### EXPORT CSV ####
print('Saving to CSV...')
max_factor_df.to_csv(csv_filepath, index=False)
max_amb_df.to_csv(csv_filepath.replace('coupling','ambient'), index=False)
#### COUPLING FUNCTION PLOT ####
print('Plotting coupling function...')
plot_summary_coupfunc(
    max_factor_df, plot_filepath1,
    upper_lim=upper_lim,
    injection_info=(ifo, station, injection_type),
    freq_range=freq_range,
    units=[],
    markersize_real=ms_real,
    markersize_upper=ms_upper,
    edgewidth_real=edgew_real,
    edgewidth_upper=edgew_upper
)
#### ESTIMATED AMBIENT PLOT ####
print('Plotting estimated ambient...')
plot_summary_ambient(
    max_amb_df, plot_filepath2,
    upper_lim=upper_lim,
    darm_data=darm_data,
    gwinc=gwinc,
    injection_info=(ifo, station, injection_type),
    freq_range=freq_range,
    units=[],
    markersize_real=ms_real,
    markersize_upper=ms_upper,
    edgewidth_real=edgew_real,
    edgewidth_upper=edgew_upper
)
print('Program has finished. (Runtime: {:.1f} s)'.format(time.time() - t0))