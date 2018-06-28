import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import os
import time
import datetime

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