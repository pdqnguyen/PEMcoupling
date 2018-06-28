""""
Routines for exporting composite coupling function data

FUNCTIONS:
    export_composite_coup_func --- Generates desired data forms, whether its plots, csv files, or both.
    ratio_plot -- Plot of injection-to-background ratios of each channel's spectrum.
    LowestCouplingExport -- Lowest coupling function plot, lowest estimated ambient plot, and csv.
"""

from gwpy.time import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from textwrap import wrap
import os
import time
import datetime
import sys
import warnings
import logging

def export_coup_data(
    data_list,
    spec_plot, show_darm_threshold, upper_lim, est_amb,
    freq_min_dict, freq_max_dict, factor_min_dict, factor_max_dict, spec_min_dict, spec_max_dict,
    fig_h_coup, fig_w_coup, fig_h_spec, fig_w_spec,
    options_directory, ts, coherence_results, verbose
):
    """
    Generates desired data forms, whether its plots, csv files, or both. 

    Parameters
    ----------
    data_list : list
        CouplingData objects to be plotted.
    spec_plot : bool
        If True, generate spectrum plot.
    show_darm_threshold : bool
        If True, show DARM spectrum in spec plot.
    upper_lim : bool
        If True, show upper limit coupling factors.
    est_amb : bool,
        If True, show estimated ambient spectrum in spec plot.
    freq_min : float
        Minimum frequency for plots.
    freq_max : float
        Maximum frequency for plots.
    factor_min : float
        Y-axis minimum for coupling function plot.
    factor_max : float
        Y-axis maximum for coupling function plot.
    spec_min : float
        Y-axis minimum for sensor spectrum plot.
    spec_max : float
        Y-axis maximum for sensor spectrum plot.
    fig_h_coup : float, int
        Coupling function plot figure height.
    fig_w_coup : float, int
        Coupling function plot figure width.
    fig_h_spec : float, int
        Spectrum plot figure height.
    fig_w_spec : float, int
        Spectrum plot figure width.
    options_directory : str
        Output directory.
    ts : time.time object
        Time stamp of main calculation routine.
    coherence_results : dict
        Channel names and coherence data (gwpy FrequencySeries objects)
    verbose : bool
        If True, print progress.
    """
    t1 = time.time()
    for cf in data_list:        
        if verbose:
            print('Exporting data for {}'.format(cf.name))    
        # LOOK FOR COHERENCE DATA FOR THIS CHANNEL
        if cf.name in coherence_results.keys():
            coherence_data = coherence_results[cf.name]
        else:
            coherence_data = None
        # FREQUENCY DOMAIN - SAME FOR ALL PLOTS
        if freq_min_dict is None:
            freq_min = cf.freqs[0]
        else:
            freq_min = freq_min_dict
        if freq_max_dict is None:
            freq_max = cf.freqs[-1]
        else:
            freq_max = freq_max_dict
        # COUPLING FUNCTION PLOT
        # Coupling function in physical sensor units
        cf.plot(
            options_directory,
            in_counts=False,
            freq_min=freq_min, freq_max=freq_max,
            factor_min=factor_min_dict, factor_max=factor_max_dict,
            fig_w=fig_w_coup, fig_h=fig_h_coup,
            ts=ts, upper_lim=upper_lim
        )
        # Coupling function in raw sensor counts
        cf.plot(
            options_directory,
            in_counts=True,
            ts=ts, upper_lim=upper_lim,
            freq_min=freq_min, freq_max=freq_max,
            factor_min=factor_min_dict, factor_max=factor_max_dict,
            fig_w=fig_w_coup, fig_h=fig_h_coup
        )
        # ASD PLOT WITH ESTIMATED AMBIENTS
        if spec_plot:
            cf.specplot(
                path=options_directory, ts=ts, est_amb=est_amb, show_darm_threshold=show_darm_threshold, upper_lim=upper_lim,
                freq_min=freq_min, freq_max=freq_max,
                spec_min=spec_min_dict, spec_max=spec_max_dict,
                fig_w=fig_w_spec, fig_h=fig_h_spec,
            )
        # CSV DATA FILE
        if any(cf.flags != 'No data'):
            if options_directory is None:
                directory = datetime.datetime.fromtimestamp(ts).strftime('DATA_%Y-%m-%d_%H:%M:%S')
            else:
                directory = options_directory
            if not os.path.exists(str(directory)):
                os.makedirs(str(directory))
            strip_name = cf.name[7:].replace('_DQ','')
            filename = str(directory)+'/'+strip_name+'_coupling_data.txt'
            cf.to_csv(filename, coherence_data=coherence_data)
    t2 = time.time() - t1
    if verbose:
        print('\nData export procedures finished. (Runtime: {:3f} s)'.format(t2))
    return

def export_composite_coupling_data(comp_cf, freqs_raw, darm_raw, gwinc, inj_names, path, upper_lim=True, est_amb_plot=True,\
                                   freq_min=None, freq_max=None, factor_min=None, factor_max=None, fig_w=9, fig_h=6, verbose=False):
    """
    Produces a coupling function plot, estimated ambient plot, and csv from lowest coupling function data.
    
    Parameters
    ----------
    comp_cf : CompositeCouplingData object
        Composite coupling function, frequencies, and other data.
    freqs_raw : array
        Raw frequencies for plotting DARM spectrum.
    darm_raw : array
        Raw DARM ASD values for plotting DARM spectrum.
    gwinc : list
        Frequency array and spectrum values for plotting GWINC spectrum.
    inj_names : list
        Names of injections.
    path : str
        Output directory.
    refDict : dict
        Dictionary of plot options.
    verbose : bool
        If True, print progress.
    """
    
    # Create subdirectory if it does not exist yet
    if not os.path.exists(str(path)):
        os.makedirs(str(path))        
    if verbose:
        print('\nPreparing data for output.')
    
    #======================================
    #### ORGANIZE DATA FOR PLOTTING ####
    #======================================
    
    #### DETERMINE AXIS LIMITS ####
    if verbose:
        print('\nDetermining axis limits for plots...')
    mask_nonzero = (comp_cf.values > 0)    
    #### X-AXIS (FREQUENCY) LIMITS ####
    x_axis = comp_cf.freqs[mask_nonzero]    
    if (len(x_axis) == 0):
        print('No lowest coupling factors for ' + comp_cf.name + '.')
        print('Data export aborted for this channel.')
        return
    if freq_min is None:
        freq_min = max( [min(x_axis) / 1.5, 6] )
    if freq_max is None:
        freq_max = min( [max(x_axis) * 1.5, max(comp_cf.freqs)] )
    
    #### Y-AXIS (COUPLING FACTOR) LIMITS ####
    y_axis = comp_cf.values[mask_nonzero & (comp_cf.freqs>=freq_min) & (comp_cf.freqs < freq_max)]
    y_axis_counts = comp_cf.values_in_counts[mask_nonzero & (comp_cf.freqs>=freq_min) & (comp_cf.freqs < freq_max)]
    if (len(y_axis) == 0):
        print('No lowest coupling factors for ' + comp_cf.name)
        print('between ' + str(freq_min) + ' and ' + str(freq_max) + ' Hz.')
        print('Data export aborted for this channel.')
        return
    if factor_min is None:
        factor_min = np.min(y_axis) / 3
        factor_counts_min = np.min(y_axis_counts) / 3
    else:
        factor_counts_min = factor_min
            
    if factor_max is None:
        factor_max = np.max(y_axis) * 1.5
        factor_counts_max = np.max(y_axis_counts) * 1.5
    else:
        factor_counts_max = factor_max
        
    #### SORTED NAMES OF INJECTIONS ####
    sorted_names = sorted(set(comp_cf.injections))
    if None in sorted_names:
        sorted_names.remove(None)
    
    #### FILEPATH ####
    filename = path + '/' + comp_cf.name.replace('_DQ', '') + '_composite_'
    csv_filename = filename + 'coupling_data.txt'
    multi_filename = filename + 'coupling_multi_plot.png'
    single_filename = filename + 'coupling_plot.png'
    single_counts_filename = filename + 'coupling_counts_plot.png'
    est_amb_multi_filename = filename + 'est_amb_multi_plot.png'
    est_amb_single_filename = filename + 'est_amb_plot.png'    
    
    #===========================================
    #### LOWEST COUPLING FUNCTION PLOT ####
    #===========================================
    
    comp_cf.plot(
        multi_filename, in_counts=False, split_injections=True, upper_lim=upper_lim,
        freq_min=freq_min, freq_max=freq_max, factor_min=factor_min, factor_max=factor_max,
        fig_w=fig_w, fig_h=fig_h
    )    
    comp_cf.plot(
        single_filename, in_counts=False, split_injections=False, upper_lim=upper_lim,
        freq_min=freq_min, freq_max=freq_max, factor_min=factor_min, factor_max=factor_max,
        fig_w=fig_w, fig_h=fig_h
    )    
    comp_cf.plot(
        single_counts_filename, in_counts=True, split_injections=False, upper_lim=upper_lim,
        freq_min=freq_min, freq_max=freq_max, factor_min=factor_counts_min, factor_max=factor_counts_max,
        fig_w=fig_w, fig_h=fig_h
    )    
    if verbose:
        print('Composite coupling function plots complete.')

    #===========================================
    #### LOWEST ESTIMATED AMBIENT PLOT ####
    #===========================================

    if est_amb_plot:
        mask_freq = (comp_cf.freqs >= freq_min) & (comp_cf.freqs < freq_max) # data lying within frequency plot range
        ambs_pos = comp_cf.ambients[(comp_cf.ambients > 0) & mask_freq]
        darm_pos = comp_cf.darm_bg[(comp_cf.darm_bg > 0) & mask_freq]
        values = np.concatenate((ambs_pos, darm_pos)) # all positive data within freq range
        amb_min = values.min() / 4
        amb_max = values.max() * 2
        if np.any(comp_cf.flags != 'No data'):
            comp_cf.ambientplot(
                est_amb_multi_filename,
                gw_signal='darm', split_injections=True, gwinc=gwinc, darm_data=[freqs_raw, darm_raw],
                freq_min=freq_min, freq_max=freq_max, amb_min=amb_min, amb_max=amb_max, fig_w=fig_w, fig_h=fig_h
            )
            comp_cf.ambientplot(
                est_amb_single_filename,
                gw_signal='strain', split_injections=False, gwinc=gwinc, darm_data=[freqs_raw, darm_raw],
                freq_min=freq_min, freq_max=freq_max, amb_min=amb_min/4000., amb_max=amb_max/4000., fig_w=fig_w, fig_h=fig_h
            )
            if verbose:
                print('Composite estimated ambient plots complete.')        
        else:
            print('No composite coupling data for this channel.')
            
    #===================================================
    #### CSV OUTPUT ####
    #===================================================
    
    comp_cf.to_csv(csv_filename)
    if verbose:
        print('CSV saved.')
    return

def ratio_table(chansI, chansQ, z_min, z_max, method='raw', minFreq=5, maxFreq=None, directory=None, ts=None):
    """
    Plot of injection-to-background ratios of each channel's spectrum, combined into a single table-plot.
    
    Parameters
    ----------
    chansI : list
        Injection sensor ASDs.
    chansQ : list
        Background sensor ASDs.
    z_min : float, int
        Minimum ratio for colorbar.
    z_max : float, int
        Maximum raito for colorbar.
    ts : time.time object
        Time stamp of main calculation routine.
    method : {'raw', 'max', 'avg}
        Show raw ('raw'), average ('avg'), or maximum ('max') ratio per frequency bin.
    minFreq : float, int, optional
        Minimum frequency for ratio plot.
    maxFreq : float, int, optional
        Maximum frequency for ratio plot.
    directory : {None, str}, optional
        Output directory.
    """
    
    if ts is None:
        ts = time.time()
    
    #Get the maximum frequency for each channel's subplot.
    maxFreqList = []
    for i in chansI:
        maxFreqList.append(i.freqs[-1])
    #If maxFreq is not specified, then use the biggest maximum frequency across channels
    if not maxFreq:
        maxFreq = max(maxFreqList)
    #Number of points between 1 Hz frequencies.  I use the half window instead of the full number of entries between 1Hz.
    HzIndex = int(1. / (chansI[0].freqs[1] - chansI[0].freqs[0]))

    # The following fills the ratios list with numpy arrays that contain the ratios between each
    # channel's quiet and injected spectra. If avg is on, a moving average is performed and then
    # the ratio matrix is filled by striding through the average matrix so that the value is
    # only taken at integer Hz values. This ensures that each value is the average of a bin centered
    # at integer Hz frequencies. The plots are made using the pcolor function, which requires a
    # two-dimensional z-matrix. So, the ratio has two rows even though they are the same values.
    # This is just a stupid thing to get the plot to work.
    nChans = len(chansI)
    ratios = []
    freqs = [] # List of frequency ranges for each channel
    maxLen = 30
    for i in range(0, len(chansI)):
        maxValue = np.amax(chansQ[i].values)
        eps = 1e-8
        minCriteria = eps * maxValue        
        if (method == 'max'):
            freqs.append(chansI[i].freqs[HzIndex: -1: HzIndex]) # Frequencies for this channel
            ratio = np.zeros([2, freqs[i].shape[0]]) # Empty ratios array            
            # Computes ratio of raw channel values
            tratio = np.divide(chansI[i].values, chansQ[i].values)
            for j in range(len(tratio)):
                if tratio[j] < z_min:
                    tratio[j] = 0
            # Previously, this method was giving buggy outputs:
            #tratio = np.divide(chansI[i].values, chansQ[i].values, where=chansQ[i].values > minCriteria)            
            # Get max value for each frequency bin:
            for j in range(0,ratio.shape[1]):
                li = max((j * HzIndex), 0)
                up = min((j + 1) * HzIndex, tratio.shape[0])
                ratio[0, j] = np.amax(tratio[li : up])
            ratio[1, :] = ratio[0, :]
            ratio = ratio.astype(int)
        elif (method == 'avg'):
            freqs.append(chansI[i].freqs[HzIndex: -1: HzIndex]) # Frequencies for this channel
            ratio = np.zeros([2, freqs[i].shape[0]]) # Empty ratios array            
            # This performs the moving average, and gets the stride of the moving average
            avgQ = np.convolve(chansQ[i].values, np.ones(HzIndex), mode = 'valid')[HzIndex: -1: HzIndex] / HzIndex
            avgI = np.convolve(chansI[i].values, np.ones(HzIndex), mode = 'valid')[HzIndex: -1: HzIndex] / HzIndex            
            # Computes the ratio of the averages
            tratio = np.divide(avgI, avgQ)
            for j in range(len(tratio)):
                if tratio[j] < z_min:
                    tratio[j] = 0
            ratio[0, :] = tratios.astype(int)
            ratio[1, :] = ratio[0, :]
        else:
            ratio = np.zeros([2, chansI[i].freqs.shape[0]])
            # No binning, just report raw ratios
            ratio[0, :] = (np.divide(chansI[i].values, chansQ[i].values, where=chansQ[i].values > minCriteria)).astype(int)
            ratio[1, :] = ratio[0, :]
            freqs.append(chansI[i].freqs)
        ratios.append(ratio)
        # Align the channel names so that they are not overlapping with the plots.
        if len(chansI[i].name) > maxLen:
            maxLen = len(chansI[i].name)
    #These are used to ensure that the color bars, channel names, and plots are displayed in an aesthetically pleasing way.
    offset = 0.
    figHt = 1.1*nChans
    fontsize = 16
    fontHt = (fontsize / 72.272) / figHt
    
    #### MAIN PLOT LOOP ####
    # Performs the plotting. Runs through each channel, and plots the ratio using pcolor.
    # Then, some options are added/changed to make the plots look better
    for i in range(nChans):
        ax = plt.subplot2grid((nChans+2, 20), (i+1, 6), colspan=13)
        im = ax.pcolor(freqs[i], np.array([0, 1]), ratios[i], vmin = z_min, vmax = z_max)
        #Don't want yticks, as rows are meaningless
        plt.yticks([], [])
        plt.xlim([minFreq, maxFreq])
        ax.set_xscale('log', nonposy = 'clip')
        #Alter the xticks
        ax.xaxis.set_tick_params(which = 'major', width=3, colors = 'white')
        ax.xaxis.set_tick_params(which = 'minor', width=2, colors = 'white')
        ax.xaxis.grid(True)
        if not (i == nChans - 1):
            for xtick in ax.get_xticklabels():
                xtick.set_label('')
            #plt.xticks([], [])
        #Shift the ratio plots.  This doesn't seem to work without having the color bar overwrite the plot in the same row as it.
        pos = ax.get_position()
        #ax.set_position([pos.x0 + offset, pos.y0, 0.9 * pos.width, pos.height])
        plt.figtext(
            0.02, pos.y0 + pos.height / 2.0 - fontHt / 2.0, chansI[i].name,
            fontsize=fontsize, color='k', fontweight='bold'
        )
    #Adds black tick labels to bottom plot
    for xtick in ax.get_xticklabels():
        xtick.set_color('k')
        
    #### COLORBAR ####    
    cax = plt.subplot2grid((nChans, 20), (int(nChans/2)-1, 19), colspan= 2)
    cbar = plt.colorbar(im, cax = cax)#, ticks = range(10, 110, 10))
    pos = cax.get_position()    
    # Labels
    cbar_min = ( int(np.ceil(z_min/10.)*10) if (z_min%10!=0) else z_min+10)
    cbar_max = ( int(np.floor(z_max/10.)*10) + 10 if (z_max%10!=0) else z_max)
    tick_values = [z_min] + range(cbar_min, cbar_max, 10) + [z_max]
    tick_labels = ['$<${}'.format(z_min)] + list(np.arange(cbar_min, cbar_max, 10).astype(str)) + ['$>${}'.format(z_max)]
    cbar.set_ticks(tick_values)
    cbar.ax.set_yticklabels(tick_labels, fontsize=(nChans/30. + 1)*fontsize)
    # Positioning
    cbHeight = (0.23*nChans + 2) / float(figHt)
    cax.set_position([0.9, 0.5 - cbHeight/2.0, 0.01, cbHeight])    
    
    #### EXPORT ####    
    plt.suptitle("Ratio of Injected Spectra to Quiet Spectra", fontsize=fontsize*2, color='k')    
    if directory:
        plt.figtext(
            0.5, 0.02, "Injection name: " + directory.split('/')[-1],
            fontsize=fontsize*2, color ='b',
            horizontalalignment='center', verticalalignment='bottom'
        )
    else:
        directory = datetime.datetime.fromtimestamp(ts).strftime('DATA_%Y-%m-%d_%H:%M:%S')
    # Change the size of the figure
    fig = plt.gcf()
    fig.set_figwidth(14)
    fig.set_figheight(figHt)
    if not os.path.exists(str(directory)):
        os.makedirs(str(directory))
    if directory[-1] == '/':
        subdir = directory.split('/')[-2]
    else:
        subdir = directory.split('/')[-1]
    fn = directory + "/" + subdir + "RatioTable.png"
    plt.savefig(fn)
    plt.close()
    dt = time.time() - ts
    print('Ratio table complete. (Runtime: {:.3f} s)'.format(dt))
    return