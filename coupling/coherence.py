import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
plt.switch_backend('Agg')
from textwrap import wrap
import time
import datetime
import os
from gwpy.timeseries import TimeSeries

def coherence(
    gw_channel, ifo, sensor_time_series,
    t_start, t_end, FFT_time, overlap_time,
    cohere_plot, thresh1, thresh2, path, ts
):
    """
    Coherence between DARM signal and sensor signal. Notifies user if a certain amount of the data exhibits poor coherence.
    This coherence data will be reported in the csv output file.

    Parameters
    ----------
    gw_channel : str
        Either 'strain_channel' or 'deltal_channel'.
    ifo : str
        Interferometer name, 'H1' or 'L1'.
    sensor_time_series : list
        Sensor TimeSeries objects.
    t_start : float, int
        Start time of time series segment.
    t_end : float, int
        End time of time series segment.
    FFT_time :
        Length (seconds) of FFT averaging windows.
    overlap_time : float, int
        Overlap time (seconds) of FFT averaging windows.
    cohere_plot : bool
        If True, save coherence plot to path.
    thresh1 : float, int
        Coherence threshold. Used only for notifications.
    thresh2 : float, int
        Threshold for how much of coherence data falls below thresh1. Used only for notifications.
    path : str
        Output directory
    ts : time.time object
        Time stamp of main routine; important for output plots and directory name.
    
    Returns
    -------
    coherence_dict : dict
        Dictionary of channel names and coherences (gwpy FrequencySeries object).
    """
    
    if type(sensor_TS) != list:
        sensor_TS = [sensor_TS]
    ts1 = time.time()
    thresh_single = thresh1*(1e-2)
    coherence_dict = {}
    for i, sensor_timeseries in enumerate(sensor_TS):
        # COMPUTE COHERENCE
        if cal_from == 'strain_channel':
            darm = TimeSeries.fetch(interfer+':GDS-CALIB_STRAIN', iT, fT)
        elif cal_from == 'deltal_channel':
            darm = TimeSeries.fetch(interfer+':CAL-DELTAL_EXTERNAL_DQ', iT, fT)
        coherence = sensor_timeseries.coherence(darm, FFT_t, overlap)
        coherences = np.asarray(coherence.value)
        freqs = np.asarray(coherence.frequencies.value)
        coherence_dict[x.name] = coherences    # Save to output dictionary
        # PLOT COHERENCE DATA
        if cohere_plot:            
            # Create plot
            ax = plt.subplot(111)
            plt.plot(freqs, coherences, '-', color = 'b')
            plt.ylabel('Coherence') 
            plt.xlabel('Frequency [Hz]')
            plt.ylim([0, 1])
            plt.xlim([10, max(freqs)])
            ax.set_xscale('log', nonposx = 'clip')
            ax.autoscale(False)
            plt.grid(b=True, which='major',color='0.75',linestyle=':',zorder = 1)
            plt.minorticks_on()
            plt.grid(b=True, which='minor',color='0.65',linestyle=':', zorder = 1)
            plt.title(sensor_timeseries.name[7:].replace('_DQ','') + ' and ' + darm.name[7:].replace('_DQ',''))
            plt.suptitle(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), fontsize = 10)
            # Export plot
            if path is None:
                path = datetime.datetime.fromtimestamp(ts).strftime('DATA_%Y-%m-%d_%H:%M:%S')
            if not os.path.exists(path):
                os.makedirs(path)
            path = path + '/' + sensor_timeseries.name[7:].replace('_DQ', '') + '_coherence_plot'
            plt.savefig(path)
            plt.close()
        # COHERENCE REPORT
        # Reports coherence results if overall coherence falls below threshold
        pts_above_threshold = np.count_nonzero(coherences > thresh_single) # How many points are above thresh1%
        pts_total = float(len(coherences))
        percent_above_threshold = (pts_above_threshold / pts_total) * (10**2)
        if percent_above_threshold < thresh2:
            print('\n')
            print('Less than {:.1f}% of {} data has a coherence of {:.1f}% or greater.'.format(thresh2, sensor_timeseries.name, thresh1))
            print('Only {:.3f}% of the data fit this criteria.'.format(percent_cohere))
        else:
            print('\n')
            print('Coherence thresholds passed for channel {}.'.format(sensor_timeseries.name))
    ts2 = time.time() - ts1
    print('Coherence(s) calculated. (Runtime: {:.3f} s)'.format(ts2))    
    return coherence_dict