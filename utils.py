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