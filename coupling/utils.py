import numpy as np

def pem_sort(chans):
    """
    Sort PEM channels by location.
    
    Parameters
    ----------
    chans : list
        Channel names to be sorted.
        
    Returns
    -------
    chans_sorted : list
        Sorted channel names.
    """
    
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
        'SRTUBE',
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
    chans_sorted = []
    for key in sort_key:
        # Add channels that match this key and are not yet added
        chans_sorted += sorted([c for c in chans if (key in c) and (c not in chans_sorted)])
    # Add remaining channels at the end
    chans_sorted += sorted([c for c in chans if (c not in chans_sorted)])
    return chans_sorted

def quad_sum_names(channel_names):
    """
    Find tri-axial channel names and make corresponding quadrature-summed channel names.
    
    Parameters
    ----------
    channel_names : list of strings
    
    Returns
    -------
    qsum_dict : dict
        Keys are quad-sum channel names; values are lists of X, Y, and Z channel names.
    """
    qsum_dict = {}
    for name in channel_names:
        short_name = name.replace('_DQ', '')
        if short_name.split('_')[-1] in ['X', 'Y', 'Z']:
            qsum_name = short_name[:-2] + '_XYZ'
            if qsum_name not in qsum_dict.keys():
                qsum_dict[qsum_name] = [name]
            else:
                qsum_dict[qsum_name].append(name)
    qsum_dict = {name: axes for name, axes in qsum_dict.items() if len(axes) == 3}
    return qsum_dict

def smooth_ASD(x, y, width, smoothing_log=False):
    """
    Sliding-average smoothing function for cleaning up noisiness in a spectrum.
    Example of logarithmic smoothing:
    If width = 5 and smoothing_log = True, smoothing windows are defined such that the window is
    5 frequency bins wide (i.e. 5 Hz wide if bandwidth = 1 Hz) at 100 Hz, and 50 bins at 1000 Hz.
    ...
    A bit about smoothing: the underlying motivation is to minimize random noise which artificially yields coupling
    factors in the calculations later, but smoothing is also crucial in eliminating point-source features
    (e.g. drops in microphone spectrum due to the point-like microphone sitting at an anti-node of the injected
    sound waves). It is not so justifiable to smooth the sensor background, or DARM, since background noise and
    overall coupling to DARM are diffuse, so smoothing of these should be very limited.
    The extra objects ASD_new_bg_smoother is the background ASD smoothed as much as the injection.
    ASD_bg and ASD_bg_smoother are both used in the CouplingFunction routine; the latter for determining
    coupling regions in frequency domain, the former for actual coupling computation.
    
    Parameters
    ----------
    x : array 
        Frequencies.
    y : array
        ASD values to be smoothed.
    width : int
        Size of window for sliding average (measured in frequency bins).
    smoothing_log : {False, True}, optional
        If True, smoothing window width grows proportional to frequency (see example above).
    
    Returns
    -------
    y_smooth : array
        Smoothed ASD values.
    """
    y_smooth = np.zeros_like(y)
    if smoothing_log:
        # Num of bins at frequency f:  ~ width * f / 100
        widths = np.round(x * width / 100).astype(int)
        for i,w in enumerate(widths):
            lower_ = max([0, i-int(w/2)])
            upper_ = min([len(y), i+w-int(w/2)+1])
            y_smooth[i] = np.mean(y[lower_:upper_])
    else:
        for i in range(len(y)):
            lower_ = max([0, i-int(width/2)])
            upper_ = min([len(y), i+width-int(width/2)])
            y_smooth[i] = np.mean(y[lower_:upper_])
    return y_smooth