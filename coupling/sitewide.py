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
import re
from optparse import OptionParser
import logging
# pemcoupling modules
try:
    from coupling.utils import pem_sort
except ImportError:
    print('')
    logging.error('Failed to load PEM coupling modules. Make sure you have all of these in the right place!')
    print('')
    raise

def max_coupling_function(freqs, factor_df, flag_df):
    """
    Determine maximum coupling function.
    
    Parameters
    ----------
    freqs : array-like
        Frequency array.
    factor_df : pandas.DataFrame
        Dataframe where each column is a coupling function for a channel.
    flag_df : pandas.DataFrame
        Corresponding flags for the coupling factors in factor_df.
        
    Returns
    -------
    max_factor_df : pandas.DataFrame
        Maximum coupling function with corresponding frequencies, flags, and channels.
    """
    
    above_thresh = (flag_df == 'Real') | (flag_df == 'Upper Limit')
    below_thresh = ~above_thresh
    row_above_thresh = np.any(above_thresh, axis=1)
    row_below_thresh = np.all(below_thresh, axis=1)
    factor_df_above_thresh = factor_df.copy()
    factor_df_below_thresh = factor_df.copy()
    factor_df_above_thresh[~above_thresh] = 0.
    factor_df_below_thresh[~below_thresh] = 0.
    max_channels = pd.Series(index=factor_df.index)
    max_channels[row_above_thresh] = factor_df_above_thresh.idxmax(axis=1)
    max_channels[row_below_thresh] = factor_df_below_thresh.idxmax(axis=1)
    has_data = ~pd.isnull(max_channels)
    max_channels = max_channels[has_data]
    max_factor_df = pd.DataFrame(index=factor_df.index[has_data])
    max_factor_df['frequency'] = freqs[has_data]
    max_factor_df['factor'] = factor_df[has_data].lookup(max_channels.index, max_channels.values)
    max_factor_df['flag'] = flag_df[has_data].lookup(max_channels.index, max_channels.values)
    max_factor_df['channel'] = max_channels.values
    return max_factor_df

def max_estimated_ambient(freqs, amb_df, flag_df):
    """
    Determine maximum estimated ambient.
    
    Parameters
    ----------
    freqs : array-like
        Frequency array.
    amb_df : pandas.DataFrame
        Dataframe where each column is an estimated ambient (ASD) for a channel.
    flag_df : pandas.DataFrame
        Corresponding flags for the estimated ambients in amb_df.
        
    Returns
    -------
    max_amb_df : pandas.DataFrame
        Maximum ambient with corresponding frequencies, flags, and channels.
    """
    
    above_thresh = (flag_df == 'Real') | (flag_df == 'Upper Limit')
    below_thresh = ~above_thresh
    row_above_thresh = np.any(above_thresh, axis=1)
    row_below_thresh = np.all(below_thresh, axis=1)
    amb_df_above_thresh = amb_df.copy()
    amb_df_below_thresh = amb_df.copy()
    amb_df_above_thresh[~above_thresh] = 0.
    amb_df_below_thresh[~below_thresh] = 0.
    max_channels = pd.Series(index=amb_df.index)
    max_channels[row_above_thresh] = amb_df_above_thresh.idxmax(axis=1)
    max_channels[row_below_thresh] = amb_df_below_thresh.idxmax(axis=1)
    has_data = ~pd.isnull(max_channels)
    max_channels = max_channels[has_data]
    max_amb_df = pd.DataFrame(index=amb_df.index[has_data])
    max_amb_df['frequency'] = freqs[has_data]
    max_amb_df['amb'] = amb_df[has_data].lookup(max_channels.index, max_channels.values)
    max_amb_df['flag'] = flag_df[has_data].lookup(max_channels.index, max_channels.values)
    max_amb_df['channel'] = max_channels.values
    return max_amb_df

def plot_summary_coupfunc(
    data, filepath, upper_lim=True, injection_info=None, freq_range=None, units=[],
    marker_real='o', marker_upper='^', markersize_real=4., markersize_upper=0.6,
    edgewidth_real=0.5, edgewidth_upper=0.7, fig_w=8, fig_h=6
):
    if injection_info is not None:
        ifo, station, injection_type = injection_info
    else:
        ifo, station, injection_type = '', '', ''
    if upper_lim:
        all_factors = np.asarray(data['factor'])
    else:
        all_factors = np.asarray(data['factor'][data['flag'] == 'Real'])
    channels = sorted(set(data['channel']))
    channels = pem_sort(channels)
    all_factors = np.asarray(data['factor'])
    real_factors = {}
    upper_factors = {}
    null_factors = {}
    for channel in channels:
        real_data = data[(data['flag']=='Real') & (data['channel']==channel)][['frequency','factor']]
        upper_data = data[(data['flag']=='Upper Limit') & (data['channel']==channel)][['frequency','factor']]
        null_data = data[(data['flag']=='Thresholds not met') & (data['channel']==channel)][['frequency','factor']]
        if real_data.shape[0] > 0:
            real_factors[channel] = np.asarray(real_data).T
        if upper_data.shape[0] > 0:
            upper_factors[channel] = np.asarray(upper_data).T
        if null_data.shape[0] > 0:
            null_factors[channel] = np.asarray(null_data).T
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)
    #### COLOR MAP ####
    # Generate & discretize color map for distinguishing injections
    c_min, c_max = 0.05, 0.95
    colors = cm.jet(np.linspace(c_min, c_max, len(channels)))
    colorsDict = {c: tuple(colors[i][:-1]) for i,c in enumerate(channels)}
    #### PLOTTING LOOP ####
    lgd_patches = []
    for channel in channels:
        if 'QUAD_SUM' in channel:
            c_lgd = channel.replace('QUAD_SUM', '').replace('_',' ')
        else:
            c_lgd = channel.replace('_',' ')
        # Create legend color patch for this channel
        color = colorsDict[channel]
        if channel in real_factors.keys()+upper_factors.keys()+null_factors.keys():
            lgd_patches.append(mpatches.Patch(color=color, label=c_lgd))
        # Plot coupling function
        if channel in real_factors.keys():
            plt.plot(
                real_factors[channel][0],
                real_factors[channel][1],
                marker_real,
                markersize=markersize_real,
                color=color,
                markeredgewidth=edgewidth_real,
                label=channel,
                zorder=2
            )
        if upper_lim and channel in upper_factors.keys():
            plt.plot(
                upper_factors[channel][0],
                upper_factors[channel][1],
                marker_upper,
                markersize=markersize_upper,
                markeredgewidth=edgewidth_upper,
                color='none',
                markeredgecolor=color,
                label=channel,
                zorder=1
            )
        if upper_lim and channel in null_factors.keys():
            plt.plot(
                null_factors[channel][0],
                null_factors[channel][1],
                marker_upper,
                markersize=markersize_upper,
                markeredgewidth=edgewidth_upper,
                color='none',
                markeredgecolor=color,
                label=channel,
                zorder=1
            )
    #### SET AXIS STYLE ####
    if upper_lim:
        y_min = all_factors[all_factors>1e-99].min()/10
        y_max = all_factors.max()*10
    else:
        real_y = data['factor'][data['flag'] == 'Real']
        y_min = real_y[real_y>1e-99].min()/10
        y_max = real_y.max()*10
    plt.ylim([y_min, y_max])
    if freq_range is not None:
        x_min, x_max = freq_range
    elif upper_lim:
        x_min = freqs[freqs>0].min()
        x_max = freqs[freqs>0].max()
    else:
        freqs_real = np.asarray(data['frequency'][data['flag'] == 'Real'])
        x_min = freqs_real[freqs_real>0].min() / 1.2
        x_max = freqs_real.max() * 1.2
    ax.set_xlim(x_min, x_max)
    ax.set_yscale('log', nonposy = 'clip')
    ax.set_xscale('log', nonposx = 'clip')
    ax.autoscale(False)
    plt.grid(b=True, which='major',color='0.0',linestyle=':',zorder = 1)
    plt.minorticks_on()
    plt.grid(b=True, which='minor',color='0.6',linestyle=':', zorder = 1)
    #### SET AXIS LABELS ####
    # AXIS NAME LABELS
    if type(units) != list:
        units = [units]
    if len(units) > 1:
        units_str = ' or '.join(['[m/{}]'.format(u) for u in units])
        plt.ylabel('Coupling Function \n' + units_str, size=20)
    elif len(units) == 1:
        plt.ylabel('Coupling Function [m/{}]'.format(units[0]), size=20)
    else:
        plt.ylabel('Coupling Function [m/Counts]', size=20)
    plt.xlabel('Frequency [Hz]', size=20)
    # AXIS TICK LABELS
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    # TITLE
    title_dict = {'Site-Wide': 'Site-Wide', 'CS': 'Corner Station', 'EX': 'End Station X', 'EY': 'End Station Y'}
    if injection_info is not None:
        title = ifo + ' ' + injection_type + ' - ' + title_dict[station]
    else:
        title = 'Summary'
    subtitle = '(Highest coupling factor at each frequency across all channels)'
    ttl = plt.title(title + ' Coupling Function\n' + subtitle, size=20, x=.45, y=1.05)
    #### CREATE LEGEND ####
    lgd_cols = 3 if (len(lgd_patches) > 16) else 1
    # Options based on legend position
    if lgd_cols > 1:
        # Multi-column format if there are too many channels
        lgd_pos = (.45, -.18)
        lgd_loc = 'upper center'
        lgd_fs = 12
        pad = 0.03
        text_pos = (0., 1.)
        fig.subplots_adjust(left=0, right=1)
    else:
        # Single-column format, legend sits on the right side of the figure
        ttl.set_position((.7,1.05))
        lgd_pos = (1.025,1)
        lgd_loc = 'upper left'
        lgd_fs = 14
        pad = 0.01
        text_pos = (0.02, 0.98)
    # Draw legend on figure
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
    plt.savefig(filepath, bbox_inches='tight', dpi=2*my_dpi)
    plt.close()
    return

def plot_summary_ambient(
    data, filepath, upper_lim=True, darm_data=None, gwinc=None, injection_info=None, freq_range=None, units=[],
    marker_real='o', marker_upper='^', markersize_real=4., markersize_upper=0.6,
    edgewidth_real=0.5, edgewidth_upper=0.7, fig_w=8, fig_h=6
):
    if injection_info is not None:
        ifo, station, injection_type = injection_info
    else:
        ifo, station, injection_type = '', '', ''
    if upper_lim:
        all_ambs = np.asarray(data['amb'])
    else:
        all_ambs = np.asarray(data['amb'][data['flag'] == 'Real'])
    channels = sorted(set(data['channel']))
    channels = pem_sort(channels)
    real_ambients = {}
    upper_ambients = {}
    null_ambients = {}
    for channel in channels:
        real_data = data[(data['flag']=='Real') & (data['channel']==channel)][['frequency','amb']]
        upper_data = data[(data['flag']=='Upper Limit') & (data['channel']==channel)][['frequency','amb']]
        null_data = data[(data['flag']=='Thresholds not met') & (data['channel']==channel)][['frequency','amb']]
        if real_data.shape[0] > 0:
            real_ambients[channel] = np.asarray(real_data).T
        if upper_data.shape[0] > 0:
            upper_ambients[channel] = np.asarray(upper_data).T
        if null_data.shape[0] > 0:
            null_ambients[channel] = np.asarray(null_data).T
    
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)
    #### COLOR MAP ####
    # Generate & discretize color map for distinguishing injections
    c_min, c_max = 0.05, 0.95
    colors = cm.jet(np.linspace(c_min, c_max, len(channels)))
    colorsDict = {c: tuple(colors[i][:-1]) for i,c in enumerate(channels)}
    #### PLOT DATA ####
    if gwinc is not None:
        gwincline, = plt.plot(gwinc[0], gwinc[1], color='0.5', lw=3, label='GWINC, 125 W, No Squeezing', zorder=1)
    if darm_data is not None:
        darm_freqs, darm_values = darm_data
        darmline, = plt.plot(darm_freqs, darm_values, 'k-', lw=1, label='DARM background', zorder=2)
    lgd_patches = [darmline, gwincline]
    for channel in channels:
        if 'QUAD_SUM' in channel:
            c_lgd = channel.replace('QUAD_SUM', '').replace('_',' ')
        else:
            c_lgd = channel.replace('_',' ')
        # Create legend color patch for this channel
        color = colorsDict[channel]
        if channel in real_ambients.keys() + upper_ambients.keys() + null_ambients.keys():
            lgd_patches.append(mpatches.Patch(color=color, label=c_lgd))
        if channel in real_ambients.keys():
            plt.plot(
                real_ambients[channel][0],
                real_ambients[channel][1],
                marker_real, 
                markersize=markersize_real, 
                color=color, 
                markeredgewidth=edgewidth_real,
                label=channel, 
                zorder=3
            )
        if upper_lim and channel in upper_ambients.keys():
            plt.plot(
                upper_ambients[channel][0],
                upper_ambients[channel][1],
                marker_upper, 
                markersize=markersize_upper, 
                color='none', 
                markeredgecolor=color, 
                markeredgewidth=edgewidth_upper,
                label=channel, 
                zorder=3
            )
        if upper_lim and channel in null_ambients.keys():
            plt.plot(
                null_ambients[channel][0],
                null_ambients[channel][1],
                marker_upper, 
                markersize=markersize_upper, 
                color='none', 
                markeredgecolor=color, 
                markeredgewidth=edgewidth_upper,
                label=channel, 
                zorder=3
            )
    #### SET AXIS STYLE ####
    y_min = 10 ** np.floor( np.log10( all_ambs[all_ambs>1e-99].min() ) )
    y_max = 10 ** np.ceil( np.log10( all_ambs[all_ambs>1e-99].max() ) )
    plt.ylim([y_min, y_max])
    if freq_range is not None:
        x_min, x_max = freq_range
    elif upper_lim:
        x_min = freqs[freqs>0].min()
        x_max = freqs[freqs>0].max()
    else:
        freqs_real = np.asarray(data['frequency'][data['flag'] == 'Real'])
        x_min = freqs_real[freqs_real>0].min() / 1.2
        x_max = freqs_real.max() * 1.2
    plt.xlim([x_min, x_max])
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
    # AXIS TICK LABELS
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    # TITLE
    title_dict = {'Site-Wide': 'Site-Wide', 'CS': 'Corner Station', 'EX': 'End Station X', 'EY': 'End Station Y'}
    if injection_info is not None:
        title = ifo + ' ' + injection_type + ' - ' + title_dict[station]
    else:
        title = 'Summary'
    subtitle = '(Highest coupling factor at each frequency across all channels)'
    ttl = plt.title(title + ' Estimated Ambient\n' + subtitle, size=20, x=.45, y=1.05)
    #### CREATE LEGEND ####
    lgd_cols = 3 if (len(lgd_patches) > 16) else 1
    # Options based on legend position
    if lgd_cols > 1:
        # Multi-column format if there are too many channels
        lgd_pos = (.45, -.18)
        lgd_loc = 'upper center'
        lgd_fs = 12
        pad = 0.03
        text_pos = (0, 1.)
        fig.subplots_adjust(left=0, right=1)
    else:
        # Single-column format, legend sits on the right side of the figure
        ttl.set_position((.7,1.05))
        lgd_pos = (1.025,1)
        lgd_loc = 'upper left'
        lgd_fs = 14
        pad = 0.01
        text_pos = (0.02, 0.98)
    # Draw legend on figure
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
    plt.savefig(filepath, bbox_inches='tight', dpi=2*my_dpi)
    plt.close()
    return