import ConfigParser
import sys

def get_config(config_name):
    """
    Parse config file and return a dictionary of the config options and values.
    
    Parameters
    ----------
    config_name: str
        Name of config file.
    
    Returns
    -------
    config_dict: dict
        Dictionary containing a sub-dictionary of config options/values for each section in the config file.
    """
    
    # Read config file
    config = ConfigParser.ConfigParser()
    try:
        config.read(config_name)
    except:
        print('\nError: Configuration file ' + config_name + ' not found.\n')
        sys.exit()
    # Check for missing/empty config file
    if len(config.sections()) == 0:
        print('\nError: Configuration file ' + config_name + ' not found.\n')
        sys.exit()
    if all( [len(config.options(x)) == 0 for x in config.sections()] ):
        print('\nError: Configuration file ' + config_name + ' is empty.\n')
        sys.exit()
    # Read config inputs into sub-dictionaries of config_dict (separated by config section)
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {option: config.get(section, option) for option in config.options(section)}
    # Categories for converting numerical config options to appropriate types
    float_options = [
        'darm_factor_threshold', 'sens_factor_threshold', 'local_max_width',\
        'duration', 'fft_overlap_pct', 'band_width',\
        'coup_fig_height', 'coup_fig_width', 'spec_fig_height', 'spec_fig_width',\
        'data_freq_min', 'data_freq_max', 'plot_freq_min', 'plot_freq_max',\
        'coup_y_min', 'coup_y_max', 'spec_y_min', 'spec_y_max',\
        'coherence_threshold', 'percent_data_threshold',\
        'ratio_min_frequency', 'ratio_max_frequency',\
        'coupling_function_binning',\
        'comp_fig_height', 'comp_fig_width',\
        'comp_freq_min', 'comp_freq_max',\
        'comp_y_min', 'comp_y_max'
    ]
    int_options = ['fft_avg']
    bool_options = [
        'fft_rounding', 'smoothing_log', 'quad_sum',\
        'darm/10', 'spec_plot', 'spec_plot_name', 'est_amb_plot',\
        'coherence_calculator', 'coherence_spectrum_plot',\
        'ratio_plot', 'ratio_avg', 'ratio_max',\
        'composite_coupling', 'upper_lim', 'comp_est_amb_plot'\
    ]
    float_required = ['darm_factor_threshold', 'sens_factor_threshold' ]
    int_required = ['fft_avg']
    # Convert numerical config options to appropriate types (floats and ints)
    for sub_dict in config_dict.values():
        for option, value in sub_dict.items():
            if option in float_options:
                try:
                    sub_dict[option] = float(value)
                except:
                    sub_dict[option] = None
                    if option in float_required:
                        print('\nError: float input required for option ' + option + '.\n')
                        sys.exit()
            elif option in int_options:
                try:
                    sub_dict[option] = int(value)
                except:
                    sub_dict[option] = None
                    if option in int_required:
                        print('\nError: int input required for option ' + option + '.\n')
                        sys.exit()
            elif option in bool_options:
                sub_dict[option] = (value.lower() in ['on', 'true', 'yes'])
            elif '_smoothing' in option:
                try:
                    sub_dict[option] = [float(v) for v in value.split(',')]
                except:
                    print('\nError: Input for ' + option + ' should be list of three floats (e.g. ACC_smothing: 5, 0.5, 0.5).\n')
                    sys.exit()
            elif 'notch' in option:
                if value is not None:
                    try:
                        notch_file = open(value)
                    except:
                        print('\nError: Cannot open file ' + value + ' provided in config file.\n')
                        sys.exit()
                    notch_data = notch_file.read().split('\n')
                    notch_list = []
                    for row in notch_data:
                        try:
                            notch_list.append(map(float, row.split(',')))
                        except:
                            print('\nError: Data in DARM notch files must be floats.\n')
                            sys.exit()
                    sub_dict[option] = notch_list
                    notch_file.close()
                else:
                    sub_dict[option] = []
    return config_dict