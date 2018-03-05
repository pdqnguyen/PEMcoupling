import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib as mpl
mpl.rc('text',usetex=True)
plt.switch_backend('Agg')
from scipy.io import loadmat
from textwrap import wrap
import datetime
import time
import sys
import os
import subprocess
from optparse import OptionParser

t0 = time.time()


#### SMOOTHING FUNCTIONS ####

gaussian = lambda x, mu, sig: np.exp(-(x-mu)**2/(2*sig**2))
normGaussian = lambda x, mu, sig: gaussian(x,mu,sig)/(np.sum(gaussian(x,mu,sig)))
gSmooth = lambda x, y, width: [np.sum(y*normGaussian(x, i, width)) for i in x]

def boxSmooth(y, width):
    w = int(width/2)
    y = np.asarray(y)
    rtn = np.zeros_like(y)
    for i in range(len(y)):
        wndw = y[ max([ i-w, 0 ]) : min([ i+w, len(y) ]) ]
        rtn[i] = np.median(wndw)
    return rtn


#### SORTING FUNCTION ####

def chan_sort(chans):
    
    sort_key = [
        'PSL',
        'ISCT1',
        'IOT',
        'HAM1',
        'HAM2',
        'INPUT',
        'MCTUBE',
        'HAM3',
        'LVEA_BS',
        'CS_ACC_BSC',
        'VERTEX',
        'OPLEV_ITMY',
        'YMAN',
        'YCRYO',
        'OPLEV_ITMX',
        'XMAN',
        'XCRYO',
        'HAM4',
        'HAM5',
        'HAM6',
        'OUTPUTOPTICS',
        'ISCT6',
        'CS_ACC_EBAY',
        'CS_MIC_EBAY',
        'CS_',
        'EX_',
        'EY_'
    ]
    rtn = []
    for s in sort_key:
        match = []
        for c in chans:
            if (s in c) and (c not in rtn):
                match.append(c)
        rtn += sorted(match)
    rtn += sorted([c for c in chans if (c not in rtn)])
    return rtn




#=========================================
#### OPTIONS/ARGUMENTS PARSING
#=========================================

parser = OptionParser()

parser.add_option("-o","--omit", type = str, dest = "omit",
                  help = "Omit channels containing specified string.")
parser.add_option("-s","--search", type = str, dest = "search",
                  help = "Search for channels containing specified string.")

(options, args) = parser.parse_args()

if len(args) < 2:
    print('\nError: Two args required, directory and coupling/injection type.\n'); sys.exit()

directory = args[0] #'/home/philippe.nguyen/public_html/test/LLO/SiteWide/SiteWideMag'
ifo = args[1]
injection_type = args[2] # 'Mag'
station = args[3]
if ifo not in ['H1', 'L1']:
    print('\nError: Arg 2 (interferometer) must be "H1" or "L1".\n'); sys.exit()


# OTHER OPTIONS
#binning = 0
upper_lim = True
lower_upper_lim = False
x_range = None # (6, 500)
w_Hz = (0.5 if 'mag' in injection_type.lower() else 0) # Local max window width


# PARSE SEARCH/OMIT ENTRIES
omit = options.omit # e.g. 'YMAN,PSL'
search = options.search # e.g. 'YMAN,PSL'

if omit is not None:
    omit = omit.replace(' ','').split(',')
else:
    omit = []
    
if search is not None:
    search = search.replace(' ','').split(',')
else:
    search = []
    
    
# GENERATE EXPORT FILE NAMES
if directory[-1] == '/':
    directory = directory[:-1]

filepath = directory + '/'+ ifo + '_' + injection_type + '_' + station.replace(' ','')
if len(search) > 0:
    for s in search:
        filepath += '_' + s
    
if len(omit) > 0:
    for o in omit:
        filepath += '_-' + o
if not upper_lim:
    filepath += '_real'
elif lower_upper_lim:
    filepath += '_lowestupperlim'
if x_range is not None:
    filepath += '_' + str(x_range[0]) + 'to' + str(x_range[1])
    
csv_filepath = filepath + '_max_coupling_data.txt'
plot_filepath1 = filepath + '_max_coupling_plot.png'
plot_filepath2 = filepath + '_max_coupling_ambient.png'
plot_filepath3 = filepath + '_max_ambient.png'

# PLOT TITLE
title_dict = {
    'Site-Wide': 'Site-Wide',
    'CS': 'Corner Station',
    'EX': 'End Station X',
    'EY': 'End Station Y'
}

title = ifo + ' ' + injection_type + ' - ' + title_dict[station]
if len(search) > 0 and 'XYZ' not in search:
    title += ' (' + ' '.join(search) + ')'
subtitle1 = '(Highest coupling factor at each frequency across all channels)'
subtitle2 = '(Obtained from highest coupling factor at each frequency across all channels)'
subtitle3 = '(Highest estimated ambient at each frequency across all channels)'




#=================================
#### DATA IMPORT
#=================================

# GWINC (GW Inteferometer Noise Calculator) DARM ASD
gwinc_mat = loadmat('gwinc_nomm_noises.mat')
gwinc = [
    np.asarray(gwinc_mat['nnn']['Freq'][0][0][0]),
    np.sqrt(np.asarray(gwinc_mat['nnn']['Total'][0][0][0])) * 4000. # Convert strain PSD to DARM ASD
]

file_names = subprocess.check_output(['ls ' + directory + '/' + '*_composite_coupling_data.txt'], shell=True).splitlines()

if station in ['CS', 'EX', 'EY']:
    station_chans = []
    for f in file_names:
        chan = f.split('/')[-1]
        if '-' in chan:
            if chan[ chan.index('-')+1 : chan.index('-')+3 ] == station:
                 station_chans.append(f)
        elif chan[:2] == station:
            station_chans.append(f)
    file_names = station_chans

bw_list = []
data_list = []
good_chans = []
freqs_raw = []
for f in file_names:
    c = f.split('/')[-1].replace('_composite_coupling_data.txt', '')
    
    # Apply search/omit options if provided
    omit_channel = False
    search_channel = True
    for s in search:
        if s not in c:
            search_channel = False
            break
    for o in omit:
        if o in c:
            omit_channel = True
            break
    if omit_channel or not search_channel:
        continue
    
    # Import file
    try:
        df = pd.read_csv(f)
    except:
        print('No lowest coupling data found for ' + c)
        continue
    
    # Crop to desired frequency range
    if x_range is not None:
        df = df[(df['frequency']>=x_range[0]) & (df['frequency']<x_range[1])]
    
    # Skip this channel if it has too little useful data
    if ('MAG' not in c) and (df[df['flag'] != 'Thresholds not met'].shape[0] < 0.25 * df.shape[0]):
        continue
    
    
    scale = 1.005
    bin_edges = [df['frequency'].min()]
    while bin_edges[-1] <= df['frequency'].max():
        bin_edges.append(bin_edges[-1] * scale)

    df_binned = pd.DataFrame(index=range(len(bin_edges)-1), columns=df.columns)

    for j in range(len(bin_edges)-1):
        f_min, f_max = (bin_edges[j], bin_edges[j+1])
        df_binned.set_value(j, 'frequency', (f_max + f_min) / 2)

        subset = df[(df['frequency'] >= f_min) & (df['frequency'] < f_max)]
        
        if subset.shape[0] > 0:
            factor_max_idx = np.asarray(subset['factor']).argmax()
            for col in ['flag', 'factor', 'factor_counts', 'darm', 'ambient']:
                df_binned.set_value(j, col, np.asarray(subset[col])[factor_max_idx])

        else:
            df_binned.set_value(j, 'flag', 'No data')
            df_binned.set_value(j, 'factor', 0)
            df_binned.set_value(j, 'factor_counts', 0)
            df_binned.set_value(j, 'ambient', 0)
            df_binned.set_value(j, 'darm', subset['darm'].astype(float).mean())
    
    df = df_binned
    
    
    # Save raw raw frequencies and darm for plotting later
    if df.shape[0] > len(freqs_raw):
        freqs_raw = np.asarray(df['frequency'])#[~pd.isnull(df['darm'])])
    
    bw_list.append( df['frequency'].iloc[10] - df['frequency'].iloc[9] )
    data_list.append(df)
    if '-' in c:
        c = c[ c.index('-')+1 : ]
    good_chans.append(c)

    
# CHECK BANDWIDTHS

if not all([bw == bw_list[0] for bw in bw_list]):
    print(bw_list)
    print('\nError: Unequal bandwidths.\n')
    sys.exit()


# CHECK DATA-FRAME SIZES

if not all([i.shape[0] == data_list[0].shape[0] for i in data_list]):
    print('Unequal frequency data. Filling in shorter data tables with zeroes.')
    for i, df in enumerate(data_list):
        if df.shape[0] < len(freqs_raw):
            df_zeros = pd.DataFrame(index=range(df.shape[0], len(freqs_raw)), columns = df.columns)
            df = df.append(df_zeros)
            df['frequency'] = freqs_raw
            data_list[i] = df


# CREATE DATA-FRAME FOR EACH TYPE OF DATA

factor_dict = {}
flag_dict = {}
real_amb_dict = {}
upper_amb_dict = {}
null_amb_dict = {}
darm_dict = {}

for i, df in enumerate(data_list):
    
    c = good_chans[i]
    if '-' in c:
        c = c[c.index('-')+1 : ]
    
    factor_dict[c] = np.asarray(df['factor']).astype(float)
    flag_dict[c] = np.asarray(df['flag'])
    darm_dict[c]  = np.asarray(df['darm']).astype(float)
    amb_arr = np.asarray(df['ambient']).astype(float)
    
    real_amb = np.zeros_like(amb_arr)
    upper_amb = np.zeros_like(amb_arr)
    null_amb = np.zeros_like(amb_arr)
    
    for j, amb in enumerate(amb_arr):
        flag = flag_dict[c][j]
        if flag == 'Real':
            real_amb[j] = amb
        elif flag == 'Upper Limit':
            upper_amb[j] = amb
        elif flag == 'Thresholds not met':
            null_amb[j] = amb
            
    real_amb_dict[c] = real_amb
    upper_amb_dict[c] = upper_amb
    null_amb_dict[c] = null_amb

factor_df = pd.DataFrame(factor_dict, index=range(data_list[0].shape[0]))
flag_df = pd.DataFrame(flag_dict, index=range(data_list[0].shape[0]))
real_amb_df = pd.DataFrame(real_amb_dict, index=range(data_list[0].shape[0]))
upper_amb_df = pd.DataFrame(upper_amb_dict, index=range(data_list[0].shape[0]))
null_amb_df = pd.DataFrame(null_amb_dict, index=range(data_list[0].shape[0]))
darm_df = pd.DataFrame(darm_dict, index=range(data_list[0].shape[0]))


# FREQUENCIES AND AVERAGE DARM FOR PLOTTING DARM BACKGROUND

freqs = np.asarray(data_list[0]['frequency'])
darm_avg = np.asarray([np.mean(row[~np.isnan(row)]) for i,row in darm_df.iterrows()])
freqs_darm = freqs[~np.isnan(darm_avg)]
darm_avg = darm_avg[~np.isnan(darm_avg)]


# FIGURE OUT UNITS FOR PLOT LABEL

units_dict = {'MIC': 'Pa', 'MAG': 'T', 'RADIO': 'ADC', 'SEIS': 'm', \
              'ISI': 'm', 'ACC': 'm', 'HPI': 'm'}
units = []
for key,val in units_dict.iteritems():
    if any(key in c for c in good_chans):
        units.append(val)





#===================================
#### PROCESS DATA
#===================================


#### MAX COUPLING FUNCTION ####

max_factor_df = pd.DataFrame(columns=['frequency','factor','flag','channel','amb'])

for i,factor_row in factor_df.iterrows():
    flag_row = flag_df.iloc[i]
    real_amb_row = real_amb_df.iloc[i]
    upper_amb_row = upper_amb_df.iloc[i]
    null_amb_row = null_amb_df.iloc[i]
    
    if ('Real' in flag_row.values) or ('Upper Limit' in flag_row.values):
        factor_row = factor_row[(flag_row=='Real') | (flag_row=='Upper Limit')]
        factor_row_max, max_chan = (factor_row.max(), factor_row.idxmax())
        max_factor_new_row = {
            'frequency':freqs[i], 
            'factor': factor_row_max, 
            'flag': flag_row[max_chan], 
            'channel': max_chan, 
            'amb': max([ real_amb_row[max_chan], upper_amb_row[max_chan] ])
        }
        max_factor_df = max_factor_df.append(max_factor_new_row, ignore_index=True)
        
    elif ('Thresholds not met' in flag_row.values):
        factor_row = factor_row[flag_row=='Thresholds not met']
        factor_row_max, max_chan = (factor_row.max(), factor_row.idxmax())
        max_factor_new_row = {
            'frequency':freqs[i], 
            'factor': factor_row_max, 
            'flag': flag_row[max_chan],
            'channel': max_chan,  
            'amb': null_amb_row[max_chan]
        }
        max_factor_df = max_factor_df.append(max_factor_new_row, ignore_index=True)

        
# Local maximum requirement
if w_Hz>0:
    for i, row in max_factor_df.iterrows():
        freq_lo = row['frequency']-w_Hz
        freq_hi = row['frequency']+w_Hz
        wndw = max_factor_df[(max_factor_df['frequency'] >= freq_lo) & (max_factor_df['frequency'] < freq_hi)]
        if 'mag' in injection_type.lower():
            wndw_max = wndw['factor'].max()
            if (row['factor'] < wndw_max):
                max_factor_df = max_factor_df.drop(i)
        else:
            wndw_max_real = wndw[wndw['flag']=='Real']['factor'].max()
            wndw_max_upper = wndw[wndw['flag']=='Upper Limit']['factor'].max()
            wndw_max_null = wndw[wndw['flag']=='Thresholds not met']['factor'].max()
            if (
                (row['flag'] == 'Real' and row['factor'] < wndw_max_real) or 
                (row['flag'] == 'Upper Limit' and row['factor'] < wndw_max_upper) or 
                (row['flag'] == 'Upper Limit' and 'Real' in wndw['flag'].values) or 
                (row['flag'] == 'Thresholds not met' and ('Upper Limit' in wndw['flag'].values or 'Real' in wndw['flag'].values))
            ):
                max_factor_df = max_factor_df.drop(i)

# Sorted lists of names for max coupling data
max_real_factor_chans = sorted(max_factor_df[max_factor_df['flag']=='Real']['channel'].drop_duplicates())
max_upper_factor_chans = sorted(max_factor_df[max_factor_df['flag']=='Upper Limit']['channel'].drop_duplicates())
max_null_factor_chans = sorted(max_factor_df[max_factor_df['flag']=='Thresholds not met']['channel'].drop_duplicates())
max_factor_chans = sorted(set(max_real_factor_chans + max_upper_factor_chans + max_null_factor_chans))
if len(max_factor_chans)>10:
    max_factor_chans = chan_sort(max_factor_chans)
else:
    max_factor_chans = sorted(max_factor_chans)

# Final data for max coupling function
max_real_factor = {}
for c in max_real_factor_chans:
    arr = max_factor_df[(max_factor_df['flag']=='Real') & (max_factor_df['channel']==c)][['frequency','factor','amb']]
    max_real_factor[c] = np.asarray(arr).T
max_upper_factor = {}
for c in max_upper_factor_chans:
    arr = max_factor_df[(max_factor_df['flag']=='Upper Limit') & (max_factor_df['channel']==c)][['frequency','factor','amb']]
    max_upper_factor[c] = np.asarray(arr).T
max_null_factor = {}
for c in max_null_factor_chans:
    arr = max_factor_df[(max_factor_df['flag']=='Thresholds not met') & (max_factor_df['channel']==c)][['frequency','factor','amb']]
    max_null_factor[c] = np.asarray(arr).T
        

#### MAX AMBIENT ####

max_amb_df = pd.DataFrame(columns=['frequency','flag','amb','channel'])

for i,flag_row in flag_df.iterrows():
    real_amb_row = real_amb_df.iloc[i]
    upper_amb_row = upper_amb_df.iloc[i]
    null_amb_row = null_amb_df.iloc[i]
    
    if ('Real' in flag_row.values) or ('Upper Limit' in flag_row.values):
        real_amb_row_max, real_max_chan = (real_amb_row.max(), real_amb_row.idxmax())
        upper_amb_row_max, upper_max_chan = (upper_amb_row.max(), upper_amb_row.idxmax())
        amb_row_max = max([ real_amb_row_max, upper_amb_row_max ])
        
        if upper_amb_row_max > real_amb_row_max:
            max_chan = upper_max_chan
            flag = 'Upper Limit'
            amb = upper_amb_row_max
        else:
            max_chan = real_max_chan
            flag = 'Real'
            amb = real_amb_row_max
            
        max_amb_new_row = {
            'frequency':freqs[i], 
            'factor': amb_row_max, 
            'flag': flag, 
            'channel': max_chan, 
            'amb': amb
        }
        max_amb_df = max_amb_df.append(max_amb_new_row, ignore_index=True)
        
    elif 'Thresholds not met' in flag_row.values:
        amb_row_max, max_chan = (null_amb_row.max(), null_amb_row.idxmax())
        flag = 'Thresholds not met'
        
        max_amb_new_row = {
            'frequency':freqs[i], 
            'factor': amb_row_max, 
            'flag': flag, 
            'channel': max_chan,
            'amb': amb_row_max
        }
        max_amb_df = max_amb_df.append(max_amb_new_row, ignore_index=True)

        
# Local maximum requirement
if w_Hz>0:
    for i, row in max_amb_df.iterrows():
        freq_lo = row['frequency']-w_Hz
        freq_hi = row['frequency']+w_Hz
        wndw = max_amb_df[(max_amb_df['frequency'] >= freq_lo) & (max_amb_df['frequency'] < freq_hi)]
        if 'mag' in injection_type.lower():
            wndw_max = wndw['amb'].max()
            if (row['amb'] < wndw_max):
                max_amb_df = max_amb_df.drop(i)
        else:
            wndw_max_real = wndw[wndw['flag']=='Real']['amb'].max()
            wndw_max_upper = wndw[wndw['flag']=='Upper Limit']['amb'].max()
            wndw_max_null = wndw[wndw['flag']=='Thesholds not met']['amb'].max()
            if (
                (row['flag'] == 'Real' and row['amb'] < wndw_max_real) or 
                (row['flag'] == 'Upper Limit' and row['amb'] < wndw_max_upper) or 
                (row['flag'] == 'Upper Limit' and 'Real' in wndw['amb'].values) or
                (row['flag'] == 'Thresholds not met' and \
                 ('Upper Limit' in wndw['flag'].values or 'Real' in wndw['flag'].values))
            ):
                max_amb_df = max_amb_df.drop(i)

# Sorted lists of names for max ambient data
max_real_amb_chans = sorted(max_amb_df[max_amb_df['flag']=='Real']['channel'].drop_duplicates())
max_upper_amb_chans = sorted(max_amb_df[max_amb_df['flag']=='Upper Limit']['channel'].drop_duplicates())
max_null_amb_chans = sorted(max_amb_df[max_amb_df['flag']=='Thresholds not met']['channel'].drop_duplicates())
max_amb_chans = sorted(set(max_real_amb_chans + max_upper_amb_chans + max_null_amb_chans))
if len(max_amb_chans)>10:
    max_amb_chans = chan_sort(max_amb_chans)
else:
    max_amb_chans = sorted(max_amb_chans)

# Final data for max ambient
max_real_amb = {}
for c in max_real_amb_chans:
    arr = max_amb_df[(max_amb_df['flag']=='Real') & (max_amb_df['channel']==c)][['frequency','amb']]
    max_real_amb[c] = np.asarray(arr).T
max_upper_amb = {}
for c in max_upper_amb_chans:
    arr = max_amb_df[(max_amb_df['flag']=='Upper Limit') & (max_amb_df['channel']==c)][['frequency','amb']]
    max_upper_amb[c] = np.asarray(arr).T
max_null_amb = {}
for c in max_null_amb_chans:
    arr = max_amb_df[(max_amb_df['flag']=='Thresholds not met') & (max_amb_df['channel']==c)][['frequency','amb']]
    max_null_amb[c] = np.asarray(arr).T

    
    
    
########################################################################################################
########################################################################################################
########################################################################################################




#### EXPORT CSV ####

max_factor_df[['frequency','factor','flag','channel']].to_csv(csv_filepath, index=False)
max_amb_df.to_csv(csv_filepath.replace('coupling','ambient'), index=False)


#### GLOBAL PLOT OPTIONS ####

# Marker properties
ms = (8. if 'MAG' in c else 4.)
edgew_circle = (.7 if 'MAG' in c else .5)
edgew_triangle = (1.5 if 'MAG' in c else .7)
ms_triangle = ms * (.8 if 'MAG' in c else .6)

# Colormap limits
c_min, c_max = 0.05, 0.95




#====================================
#### COUPLING FUNCTION PLOT
#====================================

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

all_factors = np.asarray(max_factor_df['factor'])
fact_min, fact_max = all_factors[all_factors>1e-99].min()/10, all_factors.max()*10


#### COLOR MAP ####

# Generate & discretize color map for distinguishing injections
# colors = ['b','lime','r','c','orange','purple','darkgreen','saddlebrown']
# colorsDict = {c: colors[i] for i,c in enumerate(max_factor_chans)}
colors = cm.jet(np.linspace(c_min, c_max, len(max_factor_chans)))
colorsDict = {c: tuple(colors[i][:-1]) for i,c in enumerate(max_factor_chans)}


#### PLOTTING LOOP ####

lgd_patches = []
for i,c in enumerate(max_factor_chans):
    
    # Create legend color patch for this channel
    if 'QUAD_SUM' in c:
        c_lgd = c.replace('QUAD_SUM', '').replace('_',' ')
    else:
        c_lgd = c.replace('_',' ')
    if c in max_real_factor.keys()+max_upper_factor.keys()+max_null_factor.keys():
        lgd_patches.append(mpatches.Patch(color=colorsDict[c], label=c_lgd))
    
    # Plot coupling function
    
    if c in max_real_factor.keys():
        plt.plot(
            max_real_factor[c][0], 
            max_real_factor[c][1], 
            'o', 
            markersize=ms, 
            color=colorsDict[c], 
            markeredgewidth=edgew_circle, 
            label=c, 
            zorder=2
        )
    
    if upper_lim and c in max_upper_factor.keys():
        plt.plot(
            max_upper_factor[c][0], 
            max_upper_factor[c][1], 
            '^', 
            markersize=ms_triangle, 
            markeredgewidth=edgew_triangle, 
            color='none', 
            markeredgecolor=colorsDict[c], 
            label=c, 
            zorder=1
        )
    
    if upper_lim and c in max_null_factor.keys():
        plt.plot(
            max_null_factor[c][0], 
            max_null_factor[c][1], 
            '^', 
            markersize=ms_triangle, 
            markeredgewidth=edgew_triangle, 
            color='none', 
            markeredgecolor=colorsDict[c], 
            label=c, 
            zorder=1
        )


#### SET AXIS STYLE ####

plt.ylim([fact_min, fact_max])
if x_range is not None:
    ax.set_xlim(x_range)
else:
    ax.set_xlim(freqs[freqs>0].min(), freqs[freqs>0].max())
ax.set_yscale('log', nonposy = 'clip')
ax.set_xscale('log', nonposx = 'clip')
ax.autoscale(False)
plt.grid(b=True, which='major',color='0.0',linestyle=':',zorder = 1)
plt.minorticks_on()
plt.grid(b=True, which='minor',color='0.6',linestyle=':', zorder = 1)


#### SET AXIS LABELS ####

# AXIS NAME LABELS
if len(units) > 1:
    units_str = ' or '.join(['[m/{}]'.format(u) for u in units])
    plt.ylabel('Coupling Function \n' + units_str, size=20)
else:
    plt.ylabel('Coupling Function [m/{}]'.format(units[0]), size=20)
plt.xlabel('Frequency [Hz]', size=20)
#plt.ylabel(r'Coupling Function $\left[ \mathrm{{{0}}} / \mathrm{{{1}}} \right]$'.format('m',units_str), size=20)
#plt.xlabel(r'Frequency $\left[ \mathrm{Hz} \right]$', size=20)

# AXIS TICK LABELS
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(25)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(25)

# TITLE
ttl = plt.title(title + ' Coupling Function\n' + subtitle1, size=20, x=.45, y=1.05)


#### CREATE LEGEND ####

lgd_cols = 3 if (len(lgd_patches) > 16) else 1

# Options based on legend position
if lgd_cols > 1:
    lgd_pos = (.45, -.18)
    lgd_loc = 'upper center'
    lgd_fs = 12
    pad = 0.03
    text_pos = (0., 1.)
    fig.subplots_adjust(left=0, right=1)
else:
    ttl.set_position((.7,1.05))
    lgd_pos = (1.025,1)
    lgd_loc = 'upper left'
    lgd_fs = 14
    pad = 0.01
    text_pos = (0.02, 0.98)

lgd = plt.legend(
    handles=lgd_patches,
    prop={'size':lgd_fs},
    bbox_to_anchor=lgd_pos,
    loc=lgd_loc,
    borderaxespad=0,
    ncol=lgd_cols
)
fig.canvas.draw()



#### TEXT BELOW LEGEND ####

caption1 = r'$\textbf{Circles}$ represent measured coupling factors, i.e. where a signal was seen in both sensor and DARM.'
if 'mag' in injection_type.lower():
    caption2 = r'$\textbf{Triangles}$ represent upper limit coupling factors, i.e. where a signal was seen in the sensor only.'
else:
    caption2 = r'$\textbf{Triangles}$ represent upper limit coupling factors, i.e. where a signal was not seen in DARM.'
text_width = 40 * lgd_cols
caption1 = '\n'.join(wrap(caption1, text_width))
caption2 = '\n'.join(wrap(caption2, text_width))
caption = caption1 + '\n' + caption2

# Create area for text box
leg_pxls = lgd.get_window_extent()
ax_pxls = ax.get_window_extent()
fig_pxls = fig.get_window_extent()

# Convert back to figure normalized coordinates to create new axis
ax2 = fig.add_axes([leg_pxls.x0/fig_pxls.width, ax_pxls.y0/fig_pxls.height,\
                    leg_pxls.width/fig_pxls.width, (leg_pxls.y0-ax_pxls.y0)/fig_pxls.height-pad])

# Hide tick marks
ax2.tick_params(axis='both', left='off', top='off', right='off',
                bottom='off', labelleft='off', labeltop='off',
                labelright='off', labelbottom='off')
ax2.axis('off')

# Add text (finally)
ax2_pxls = ax2.get_window_extent()
ax2.text(text_pos[0], text_pos[1], caption, size=lgd_fs*.9, va='top')


#### EXPORT PLOT ####

my_dpi = fig.get_dpi()
plt.savefig(plot_filepath1, bbox_inches='tight', dpi=2*my_dpi)
plt.close()




#============================
#### MAX ESTIMATED AMBIENT
#============================

fig3 = plt.figure(figsize=(8,6))
ax = fig3.add_subplot(111)

all_ambs = np.asarray(max_amb_df['amb'])
amb_min = 10 ** np.floor( np.log10( all_ambs[all_ambs>1e-99].min() ) )
amb_max = 10 ** np.ceil( np.log10( darm_avg.max() ) )
#amb_min, amb_max = all_ambs[all_ambs>1e-99].min(), darm_avg.max()*2



#### COLOR MAP ####

# Generate & discretize color map for distinguishing injections
# colors = ['b','lime','r','c','orange','purple','darkgreen','saddlebrown']
# colorsDict = {c: colors[i] for i,c in enumerate(max_factor_chans)}
colors = cm.jet(np.linspace(c_min, c_max, len(max_amb_chans)))
colorsDict = {c: tuple(colors[i][:-1]) for i,c in enumerate(max_amb_chans)}



#### PLOTTING LOOP ####

gwincline, = plt.plot(gwinc[0], gwinc[1], color='0.5', lw=3, label='GWINC, 125 W, No Squeezing', zorder=1)
darmline, = plt.plot(freqs_darm, darm_avg, 'k-', lw=1, label='DARM background', zorder=2)

lgd_patches = [darmline, gwincline]
for i,c in enumerate(max_amb_chans):
    
    if 'QUAD_SUM' in c:
        c_lgd = c.replace('QUAD_SUM', '').replace('_',' ')
    else:
        c_lgd = c.replace('_',' ')
    if c in max_real_amb.keys() + max_upper_amb.keys() + max_null_amb.keys():
        lgd_patches.append(mpatches.Patch(color=colorsDict[c], label=c_lgd))
        
    if c in max_real_amb.keys():
        plt.plot(
            max_real_amb[c][0],
            max_real_amb[c][1],
            'o', 
            markersize=ms, 
            color=colorsDict[c], 
            markeredgewidth=edgew_circle,
            label=c, 
            zorder=3
        )
        
    if upper_lim and c in max_upper_amb.keys():
        plt.plot(
            max_upper_amb[c][0],
            max_upper_amb[c][1],
            '^', 
            markersize=ms_triangle, 
            color='none', 
            markeredgecolor=colorsDict[c], 
            markeredgewidth=edgew_triangle,
            label=c, 
            zorder=3
        )
        
    if upper_lim and c in max_null_amb.keys():
        plt.plot(
            max_null_amb[c][0],
            max_null_amb[c][1],
            '^', 
            markersize=ms_triangle, 
            color='none', 
            markeredgecolor=colorsDict[c], 
            markeredgewidth=edgew_triangle,
            label=c, 
            zorder=3
        )


        
#### SET AXIS STYLE ####

plt.ylim([amb_min, amb_max])
if x_range is not None:
    ax.set_xlim(x_range)
else:
    ax.set_xlim(freqs[freqs>0].min(), freqs[freqs>0].max())
ax.set_yscale('log', nonposy = 'clip')
ax.set_xscale('log', nonposx = 'clip')
ax.autoscale(False)
plt.grid(b=True, which='major',color='0.0',linestyle=':',zorder = 1)
plt.minorticks_on()
plt.grid(b=True, which='minor',color='0.6',linestyle=':', zorder = 1)



#### SET AXIS LABELS ####

# AXIS NAME LABELS
plt.ylabel('DARM ASD [m/Hz$^{1/2}$]', size=20)
plt.xlabel('Frequency [Hz]', size=20)
#plt.ylabel(r'DARM ASD $\left[\mathrm{m}/\mathrm{Hz}^{1/2}\right]$', size=20)
#plt.xlabel(r'Frequency $\left[\mathrm{Hz}\right]$', size=20)

# AXIS TICK LABELS
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(25)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(25)

# TITLE
ttl = plt.title(title + ' Estimated Ambient\n' + subtitle3, size=20, x=.45, y=1.05)



#### CREATE LEGEND ####

lgd_cols = 3 if (len(lgd_patches) > 16) else 1

# Options based on legend position
if lgd_cols > 1:
    lgd_pos = (.45, -.18)
    lgd_loc = 'upper center'
    lgd_fs = 12
    pad = 0.03
    text_pos = (0, 1.)
    fig3.subplots_adjust(left=0, right=1)
else:
    ttl.set_position((.7,1.05))
    lgd_pos = (1.025,1)
    lgd_loc = 'upper left'
    lgd_fs = 14
    pad = 0.01
    text_pos = (0.02, 0.98)

lgd = plt.legend(
    handles=lgd_patches,
    prop={'size':lgd_fs},
    bbox_to_anchor=lgd_pos,
    loc=lgd_loc,
    borderaxespad=0,
    ncol=lgd_cols
)
fig3.canvas.draw()



#### TEXT BELOW LEGEND ####

caption = 'Ambient estimates are made by multiplying coupling factors by injection-free sensor levels. ' +\
r'$\textbf{Circles}$ indicate estimates from measured coupling factors, i.e. where the injection signal was seen in ' +\
'the sensor and in DARM. '
if 'mag' in injection_type.lower():
    caption += r'$\textbf{Triangles}$ represent upper limit coupling factors, i.e. where a signal was seen in the sensor only. '
else:
    caption += r'$\textbf{Triangles}$ represent upper limit coupling factors, i.e. where a signal was not seen in DARM. '
caption += 'For some channels, at certain frequencies the ambient estimates are upper limits ' +\
'because the ambient level is below the sensor noise floor.'
text_width = 45 * lgd_cols
caption = '\n'.join(wrap(caption, text_width))

# Create area for text box
leg_pxls = lgd.get_window_extent()
ax_pxls = ax.get_window_extent()
fig_pxls = fig.get_window_extent()

# Convert back to figure normalized coordinates to create new axis
ax2 = fig3.add_axes([leg_pxls.x0/fig_pxls.width, ax_pxls.y0/fig_pxls.height,\
                    leg_pxls.width/fig_pxls.width, (leg_pxls.y0-ax_pxls.y0)/fig_pxls.height-pad])

# Hide tick marks
ax2.tick_params(axis='both', left='off', top='off', right='off',
                bottom='off', labelleft='off', labeltop='off',
                labelright='off', labelbottom='off')
ax2.axis('off')

# Add text (finally)
ax2_pxls = ax2.get_window_extent()
ax2.text(text_pos[0], text_pos[1], caption, size=lgd_fs*.9, va='top')


#### EXPORT PLOT ####

my_dpi = fig3.get_dpi()
plt.savefig(plot_filepath3, bbox_inches='tight', dpi=2*my_dpi)
plt.close()



print('Complete. (Runtime: {:.1f} s)'.format(time.time() - t0))
