import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
plt.switch_backend('Agg')
from textwrap import wrap
import sys
import logging
try:
    from pemchannel import PEMChannelASD
except ImportError:
    print('')
    logging.error('Failed to load PEM coupling modules. Make sure you have all of these in the right place!')
    print('')
    raise

class CoupFuncComposite(PEMChannelASD):
    """
    Composite coupling function data for a single sensor
    
    Attributes
    ----------
    name : str
        Name of PEM sensor.
    freqs : array
        Frequency array.
    values : array
        Coupling factors in physical units.
    values_in_counts : array
        Coupling factors in raw counts.
    flags : array
        Coupling factor flags ('Real', 'Upper limit', 'Thresholds not met', 'No data').
    injections : array
        Names of injections corresponding to each coupling factor.
    sens_bg : array
        ASD of sensor background.
    darm_bg : array
        ASD of DARM background.
    ambients : array
        Estimated ambient ASD for this sensor.
    df : float
        Frequency bandwidth in Hz
    unit : str
        Sensor unit of measurement
    qty : str
        Quantity measured by sensor
    type : str
        Coupling type associated with sensor.
    
    Methods
    -------
    get_channel_info
        Use channel name to get unit of measurement, name of quantity measured by sensor, and coupling type.
    plot
        Plot coupling function from the data.
    ambientplot
        Plot DARM spectrum with estimated ambient of the sensor.
    to_csv
        Save data to csv file.
    """
    
    def __init__(self, name, freqs, values, values_in_counts, flags, injections, sens_bg=None, darm_bg=None, ambients=None, unit=''):
        """
        Parameters
        ----------
        name : str
            Name of PEM sensor.
        freqs : array
            Frequency array.
        values : array
            Coupling factors in physical units.
        values_in_counts : array
            Coupling factors in raw counts.
        flags : array
            Coupling factor flags ('Real', 'Upper limit', 'Thresholds not met', 'No data').
        injections : array
            Names of injections corresponding to each coupling factor.
        sens_bg : array
            ASD of sensor background.
        darm_bg : array
            ASD of DARM background.
        """
        # Coupling function data
        super(CoupFuncComposite, self).__init__(name, freqs, values)
        self.values_in_counts = np.asarray(values_in_counts)
        self.flags = np.asarray(flags)
        self.injections = np.asarray(injections)
        # ASDs
        self.sens_bg = sens_bg
        self.darm_bg = darm_bg
        if ambients is None and self.sens_bg is not None:
            self.ambients = self.values * self.sens_bg
        else:
            self.ambients = ambients
        # Metadata
        self.df = self.freqs[1] - self.freqs[0]
       
    @classmethod
    def compute(cls, cf_list, injection_names, local_max_window=0, freq_lines=None):
        """
        Selects for each frequency bin the "nearest" coupling factor, i.e. the lowest across multiple injection locations.

        Parameters
        ----------
        cf_list : list
            CoupFunc objects to be processed.
        injection_names : list
            Names of injections.
        local_max_window : int, optional
            Local max window in number of frequency bins.
        freq_lines : dict, optional
            Frequency lines for each injection name.

        Returns
        -------
        comp_cf : CompositeCoupFunc object
            Contains composite coupling function tagged with flags and injection names
        """

        if type(cf_list) != list or type(injection_names) != list:
            print('\nError: Parameters "cf_list" and "injection_names" must be lists.\n')
            sys.exit()
        if len(cf_list) != len(injection_names):
            print('\nError: Parameters "cf_list" and "injection_names" must be equal length.\n')
            sys.exit()
        sens_bg = np.mean([cf.sens_bg for cf in cf_list], axis=0)
        darm_bg = np.mean([cf.darm_bg for cf in cf_list], axis=0)
        freqs = cf_list[0].freqs
        N_rows = len(freqs)
        channel_name = cf_list[0].name
        factor_cols = [cf.values for cf in cf_list]
        factor_counts_cols = [cf.values_in_counts for cf in cf_list]
        flag_cols = [cf.flags for cf in cf_list]
        ratio_cols = []
        df = cf_list[0].df
        for cf in cf_list:
            ratio = np.zeros(freqs.size)
            if local_max_window == 0:
                for i in range(ratio.size):
                    idx1 = max([0, i - int(5./df)])
                    idx2 = min([freqs.size, i + int(5./df)])
                    mean_inj = cf.sens_inj[idx1:idx2].mean()
                    mean_bg = cf.sens_bg[idx1:idx2].mean()
                    if mean_inj > mean_bg:
                        ratio[i] = mean_inj / mean_bg
                    else:
                        ratio[i] = 0
            else:
                for i in range(ratio.size):
                    ratio[i] = max([0, cf.sens_inj[i] / cf.sens_bg[i]])
            ratio_cols.append(ratio)
        injection_cols = [[n]*N_rows for n in injection_names]

        if len(cf_list) == 1:
            # Only one injection; trivial
            cf = cf_list[0]
            factors = cf.values
            factors_counts = cf.values_in_counts
            flags = cf.flags
            injs = np.asarray(injection_cols[0])
            comp_cf = cls(channel_name, freqs, factors, factors_counts, flags, injs, sens_bg, darm_bg)
            return comp_cf
        # Stack columns in matrices
        matrix_fact = np.column_stack(tuple(factor_cols))
        matrix_fact_counts = np.column_stack(tuple(factor_counts_cols))
        matrix_flag = np.column_stack(tuple(flag_cols))
        matrix_ratio = np.column_stack(tuple(ratio_cols))
        matrix_inj = np.column_stack(tuple(injection_cols))    
        # Output lists
        factors = np.zeros(N_rows)
        factors_counts = np.zeros(N_rows)
        flags = np.array(['No data'] * N_rows, dtype=object)
        injs = np.array([None] * N_rows)
        for i in range(N_rows):

            # Find loudest injections at this frequency
            ratio_row = np.asarray(matrix_ratio[i,:])
#             loud_injs = np.ones(ratio_row.shape, dtype=bool)
            loud_injs = np.argmax(ratio_row)
#             loud_injs = (ratio_row > 0.5 * ratio_row.max()) & (matrix_flag[i,:] != 'No data')
            # Assign rows to arrays
            factor_row = np.asarray(matrix_fact[i,loud_injs])
            factor_counts_row = np.asarray(matrix_fact_counts[i,loud_injs])
            flag_row = np.asarray(matrix_flag[i,loud_injs])
            ratio_row = np.asarray(matrix_ratio[i,loud_injs])
            inj_row = np.asarray(matrix_inj[i,loud_injs])

            # Local data, relevant if local minimum search is applied
            i1 = max([0, i - local_max_window])
            i2 = min([i + local_max_window + 1, N_rows])
            local_factors = np.ravel(matrix_fact[i1:i2, loud_injs])
            local_flags = np.ravel(matrix_flag[i1:i2, loud_injs])
            mask_zero = (local_flags != 'No data')
            local_factors_nonzero = local_factors[mask_zero]
            local_flags_nonzero = local_flags[mask_zero]
            # Separate each column into 'Real', 'Upper Limit', and 'Thresholds not met' lists
            flag_types = ['Real', 'Upper Limit', 'Thresholds not met']
            factors_dict, factors_counts_dict, injs_dict = [{}, {}, {}]
            for f in flag_types:
                factors_dict[f] = factor_row[flag_row == f]
                factors_counts_dict[f] = factor_counts_row[flag_row == f]
                injs_dict[f] = inj_row[flag_row == f]
            # Conditions for each type of coupling factor
            flag = 'No data'
            for f in flag_types:
                if len(factors_dict[f]) > 0:
                    if min(factors_dict[f]) == min(local_factors_nonzero):
                        flag = f
            # Assign lowest coupling factor and injection; assign flag based on above conditions
            if flag != 'No data':
                idx = np.argmin(factors_dict[flag])
                factors[i] = factors_dict[flag][idx]
                factors_counts[i] = factors_counts_dict[flag][idx]
                flags[i] = flag
                injs[i] = injs_dict[flag][idx]
        # Truncate low-freq injections when higher-freq injections are present
        # This fixes the issue of overlapping upper limits at high-freq in magnetic coupling functions
        if type(freq_lines) is dict:
            if all(name in list(freq_lines.keys()) for name in injection_names):
                freq_lines[None] = 0.
                flines = sorted(set(freq_lines.values()))    # List of injection fundamental frequencies
                freq_ranges = [(flines[i], flines[i+1]) for i in range(1, len(flines) - 1)]
                freq_ranges.append((flines[-1], 1e10))   # Between highest injection freq and "infinity"
                # Loop over frequency range pairs
                for fmin, fmax in freq_ranges:
                    # Find upper limits within this freq range
                    idx = np.where((flags == 'Upper Limit') & (freqs > fmin) & (freqs <= fmax))[0]
                    # Find first instance of an injection whose fundamental freq is fmin
                    if len(idx) > 0:
                        f0_arr = np.array([freq_lines[inj] for inj in injs])
                        try:
                            start_idx = np.where(f0_arr >= fmin)[0][0]
                        except IndexError:
                            continue
                        idx = idx[idx >= start_idx]
                        if len(idx) > 0:
                            for i in idx:
                                if f0_arr[i] < fmin:
                                    # Zero out this data pt if it's from an injection whose f0 is below this freq range
                                    factors[i] = 0.
                                    factors_counts[i] = 0.
                                    flags[i] = 'No data'
                                    injs[i] = None
        comp_cf = cls(channel_name, freqs, factors, factors_counts, flags, injs, sens_bg=sens_bg, darm_bg=darm_bg)
        return comp_cf
    
    def plot(self, filename, in_counts=False, split_injections=True, upper_lim=True,\
             freq_min=None, freq_max=None, factor_min=None, factor_max=None, fig_w=9, fig_h=6):
        """
        Plot composite coupling function from the data.
        
        Parameters
        ----------
        filename : str
            Target file name.
        in_counts : bool, optional
            If True, convert coupling function to counts, and treat data as such
        split_injections : bool, optional
            If True, split data by injection, using different colors for each.
        upper_lim : bool, optional
            If True, include upper limits in the plot. Otherwise just plot measured (real) values.
        freq_min : float, optional
            Minimum frequency (x-axis).
        freq_max : float, optional
            Maximum frequency (x_axis).
        factor_min : float, optional
            Minimum value of coupling factor axis (y-axis).
        factor_max : float, optional
            Maximum value of coupling factor axis (y-axis).
        fig_w : float or int, optional
            Figure width.
        fig_h : float or int, optional
            Figure height.
        """
        # Use factors and sensor units in counts
        if in_counts:
            factors = self.values_in_counts
            coupling_type = ''
            unit = 'Counts'
        else:
            factors = self.values
            coupling_type = self.coupling
            unit = str(self.unit)
            if unit == 'm/s2':
                unit = '(m/s^2)'
            elif unit == 'm/s':
                unit = '(m/s)'
        # Plot limits if not provided
        if freq_min is None:
            freq_min = self.freqs[0]
        if freq_max is None:
            freq_max = self.freqs[-1]
        if factor_min is None:
            factor_min = factors.min() / 1.2
        if factor_max is None:
            factor_max = factors.max() * 1.2
        # Different plot marker sizes for magnetometers
        if 'MAG' in self.name:
            ms = 10.
            edgew_circle = 0.7
            edgew_triangle = 1.
            ms_triangle = 8.
            lgd_size = 16
        else:
            ms = 5.
            edgew_circle = 0.4
            edgew_triangle = 0.7
            ms_triangle = 3.
            lgd_size = 12

        #### CREATE FIGURE ####
        if fig_w is None:
            fig_w = 9
        if fig_h is None:
            fig_h = 6
        fig = plt.figure(figsize=(fig_w,fig_h))
        ax = fig.add_subplot(111)
        if split_injections:
            #### COLOR MAP ####
            # Generate & discretize color map for distinguishing injections
            injection_names = sorted(set(self.injections))
            if None in injection_names:
                injection_names.remove(None)
            colors = cm.jet(np.linspace(0.05, 0.95, len(injection_names)))
            colorsDict = {injection: colors[i] for i, injection in enumerate(injection_names)}
            #### PLOTTING LOOP ####
            lgd_patches = []
            for injection in injection_names:
                # Color by injection
                c = colorsDict[injection]
                lgd_patches.append(mpatches.Patch(color=c, label=injection))
                mask_injection = (self.injections == injection) & (self.values > 0)
                mask_real = mask_injection & (self.flags == 'Real')
                if in_counts:
                    mask_upper = mask_injection & ((self.flags == 'Upper Limit') | (self.flags == 'Thresholds not met'))
                else:
                    mask_upper = mask_injection & (self.flags == 'Upper Limit')
                plt.plot(
                    self.freqs[mask_real],
                    factors[mask_real],
                    'o',
                    markersize=ms,
                    color=tuple(c[:-1]),
                    markeredgewidth=edgew_circle,
                    zorder=2
                )
                if upper_lim:
                    plt.plot(
                        self.freqs[mask_upper],
                        factors[mask_upper],
                        '^',
                        markersize=ms_triangle,
                        markeredgewidth=edgew_triangle,
                        color='none',
                        markeredgecolor=tuple(c[:-1]),
                        zorder=1
                    )
        else:
            lgd_patches = [mpatches.Patch(color='r',label='Measured Value\n(Signal seen in both\nsensor and DARM)')]

            mask_real = self.flags == 'Real'
            mask_upper = self.flags == 'Upper Limit'
            mask_null = self.flags == 'Thresholds not met'

            ms = 7 if 'MAG' in self.name else 3
            plt.plot(self.freqs[mask_real], factors[mask_real], 'r.', ms=ms, zorder=3)
            if upper_lim:
                lgd_patches.append(mpatches.Patch(color='b',label='Upper Limit\n(Signal seen in\nsensor but not DARM)'))
                plt.plot(self.freqs[mask_upper], factors[mask_upper], '.', color='b', ms=ms, zorder=2)
                if in_counts:
                    plt.plot(self.freqs[mask_null], factors[mask_null], '.', color='c', ms=ms, zorder=1)
                    lgd_patches.append(mpatches.Patch(color='c',label='Thresholds Not Met\n(Signal not seen\nin sensor nor DARM)'))
        #### SET AXIS STYLE ####
        plt.ylim([factor_min, factor_max])
        plt.xlim([freq_min, freq_max])
        ax.set_yscale('log', nonposy = 'clip')
        ax.set_xscale('log', nonposx = 'clip')
        ax.autoscale(False)
        plt.grid(b=True, which='major', color='0.0', linestyle=':', linewidth=1, zorder=0)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='0.6', linestyle=':', zorder=0)
        #### SET AXIS LABELS ####
        # AXIS NAME LABELS
        plt.ylabel('{0} Coupling [m/{1}]'.format(coupling_type, unit), size=22)
        plt.xlabel('Frequency [Hz]', size=22)
        # AXIS TICK LABELS
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(25)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(25)
        # TITLE
        if 'XYZ' in self.name:
            title = self.name[:-4].replace('_',' ') + ' (Quadrature sum of X, Y, and Z components)\nComposite Coupling Function ' +\
            '(Lowest at each frequency over multiple injection locations)'
        else:
            title = self.name.replace('_DQ','').replace('_',' ') + ' - Composite Coupling Function' + \
            '\n(Lowest at each frequency over multiple injection locations)'
        ttl = plt.title(title, size=22)
        ttl.set_position([.7,1.05])
        #### CREATE LEGEND ####
        lgd = plt.legend(handles=lgd_patches, prop={'size':lgd_size}, bbox_to_anchor=(1.025,1), loc=2, borderaxespad=0.)
        fig.canvas.draw()
        if split_injections:
            # TEXT BELOW LEGEND
            leg_pxls = lgd.get_window_extent()
            ax_pxls = ax.get_window_extent()
            fig_pxls = fig.get_window_extent()
            # Convert back to figure normalized coordinates to create new axis
            pad = 0.01
            ax2 = fig.add_axes([leg_pxls.x0/fig_pxls.width, ax_pxls.y0/fig_pxls.height,\
                                leg_pxls.width/fig_pxls.width, (leg_pxls.y0-ax_pxls.y0)/fig_pxls.height-pad])
            # Eliminate all the tick marks
            ax2.tick_params(axis='both', left='off', top='off', right='off',
                            bottom='off', labelleft='off', labeltop='off',
                            labelright='off', labelbottom='off')
            ax2.axis('off')
            # Add text (finally)
            ax2_pxls = ax2.get_window_extent()
            ax2_pxls = ax2.get_window_extent()
            caption1 = 'CIRCLES represent measured coupling factors, i.e. where a signal was seen in both sensor and DARM.'
            if 'MAG' in self.name:
                caption2 = 'TRIANGLES represent upper limit coupling factors, i.e. where a signal was seen in the sensor only.'
            else:
                caption2 = 'TRIANGLES represent upper limit coupling factors, i.e. where a signal was not seen in DARM.'
            text_width = 10 + max([len(name) for name in injection_names])
            caption1 = '\n'.join(wrap(caption1, text_width))
            caption2 = '\n'.join(wrap(caption2, text_width))
            ax2.text(0.02, .98, caption1 + '\n' + caption2, size=lgd_size, va='top')
        #### EXPORT PLOT ####
        mydpi = fig.get_dpi()
        plt.savefig(filename, bbox_inches='tight', dpi=mydpi*2)
        plt.close()
        return

    def ambientplot(self, filename, gw_signal='darm', split_injections=True, gwinc=None, darm_data=None,\
                    freq_min=None, freq_max=None, amb_min=None, amb_max=None, fig_w=9, fig_h=6):
        """
        Plot DARM spectrum and composite estimated ambient from the data.
        
        Parameters
        ----------
        filename : str
            Target file name.
        gw_signal : 'darm' or 'strain', optional
            Treat GW channel as DARM (meters/Hz^(1/2)) or strain (1/Hz^(1/2)).
        split_injections : bool, optional
            If True, split data by injection, using different colors for each.
        gwinc : tuple of arrays, optional
            Frequency and ASD arrays of GWINC estimate, for plotting.
        darm_data: tuple of arrays, optional
            Frequency and ASD arrays of DARM spectrum, for plotting.
        freq_min : float, optional
            Minimum frequency (x-axis).
        freq_max : float, optional
            Maximum frequency (x_axis).
        factor_min : float, optional
            Minimum value of coupling factor axis (y-axis).
        factor_max : float, optional
            Maximum value of coupling factor axis (y-axis).
        fig_w : float or int, optional
            Figure width.
        fig_h : float or int, optional
            Figure height.
        """
        # Different plot marker sizes for magnetometers
        if 'MAG' in self.name:
            ms = 10.
            edgew_circle = 0.7
            edgew_triangle = 1.
            ms_triangle = 8.
            lgd_size = 16
        else:
            ms = 5.
            edgew_circle = 0.4
            edgew_triangle = 0.7
            ms_triangle = 3.
            lgd_size = 12
        #### CREATE PLOT ####
        # CREATE FIGURE
        if fig_w is None:
            fig_w = 9
        if fig_h is None:
            fig_h = 6
        fig = plt.figure(figsize=(fig_w,fig_h))
        ax = fig.add_subplot(111)
        # COLORMAP FOR DIFFERENT INJECTIONS
        injection_names = sorted(set(self.injections))
        if None in injection_names:
            injection_names.remove(None)
        colors = cm.jet(np.linspace(0.05, .95, len(injection_names)))
        colorsDict = {injection: colors[i] for i, injection in enumerate(injection_names)}
        # DARM DATA
        if darm_data is None:
            darm_freqs, darm_bg = [self.freqs, self.darm_bg]
        else:
            darm_freqs, darm_bg = darm_data
        # DARM VS STRAIN
        if gw_signal.lower() == 'strain':
            darm_bg = darm_bg / 4000.
            ambients = self.ambients / 4000.
        else:
            ambients = self.ambients
        # PLOT LIMITS
        if freq_min is None:
            freq_min = self.freqs[(self.flags == 'Real') | (self.flags == 'Upper Limit')][0]
        if freq_max is None:
            freq_max = self.freqs[(self.flags == 'Real') | (self.flags == 'Upper Limit')][-1]
        if amb_min is None:
            amb_min = ambients.min() / 1.2
        if amb_max is None:
            amb_max = ambients.max() * 1.2
        # PLOT DARM
        darmline1, = plt.plot(
            darm_freqs,
            darm_bg,
            'k-',
            lw=.7,
            label='DARM background',
            zorder=3
        )
        lgd_patches = [darmline1]
        if gwinc is not None:
            gwinc_freqs = gwinc[0]
            if gw_signal.lower() == 'strain':
                gwinc_amp = gwinc[1] / 4000
            else:
                gwinc_amp = gwinc[1]
            gwincline, = plt.plot(
                gwinc_freqs,
                gwinc_amp,
                color='0.5',
                lw=3,
                label='GWINC, 125 W, No Squeezing',
                zorder=1)
            lgd_patches.append(gwincline)
        # MAIN LOOP FOR PLOTTING AMBIENTS
        if split_injections:
            for injection in injection_names:
                c = colorsDict[injection]
                lgd_patches.append(mpatches.Patch(color=tuple(c[:-1]), label=injection))
                mask_injection = (self.injections == injection) & (ambients > 0)
                mask_real = mask_injection & (self.flags == 'Real')
                mask_upper = mask_injection & (self.flags == 'Upper Limit')
                if np.any(mask_real):
                    plt.plot(
                        self.freqs[mask_real],
                        ambients[mask_real],
                        'o',
                        markersize=ms,
                        color=tuple(c[:-1]),
                        markeredgewidth=edgew_circle,
                        label=injection + ' (Above Threshold)',
                        zorder = 4
                    )
                if np.any(mask_upper):
                    plt.plot(
                        self.freqs[mask_upper],
                        ambients[mask_upper],
                        '^',
                        markersize=ms_triangle,
                        color='none',
                        markeredgewidth=edgew_triangle,
                        markeredgecolor=tuple(c[:-1]),
                        label=injection + ' (Below Threshold)',
                        zorder=3
                    )
        else:
            lgd_patches.append(mpatches.Patch(color='r',label='Measured Value\n(Signal seen in both sensor and DARM)'))
            lgd_patches.append(mpatches.Patch(color='b',label='Upper Limit\n(Signal seen in sensor but not DARM)'))
            lgd_patches.append(mpatches.Patch(color='c',label='Thresholds Not Met\n(Signal not seen in sensor nor DARM)'))
            mask_real = self.flags == 'Real'
            mask_upper = self.flags == 'Upper Limit'
            ms = 7 if 'MAG' in self.name else 3
            plt.plot(self.freqs[mask_real], ambients[mask_real], 'r.', ms=ms, zorder=3)
            plt.plot(self.freqs[mask_upper], ambients[mask_upper], '.', color='b', ms=ms, zorder=2)
        #### SET AXES STYLE ####
        plt.ylim([amb_min, amb_max])
        plt.xlim([freq_min, freq_max])
        ax.set_yscale('log', nonposy = 'clip')
        ax.set_xscale('log', nonposx = 'clip')
        ax.autoscale(False)
        plt.grid(b=True, which='major', color='0.0', linestyle=':', linewidth=1, zorder=0)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='0.6', linestyle=':', zorder=0)
        #### SET LABELS ####
        # AXIS NAME LABELS
        if gw_signal.lower() == 'darm':
            plt.ylabel('DARM ASD [m/Hz$^{1/2}$]', size=22)
        else:
            plt.ylabel('Strain [Hz$^{-1/2}$]', size=22)
        plt.xlabel('Frequency [Hz]', size=22)
        # AXIS TICK LABELS
        for tick, label in zip(ax.xaxis.get_major_ticks(), ax.get_xticklabels()):
            tick.label.set_fontsize(25)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(25)
        # TITLE
        if 'XYZ' in self.name:
            title = self.name[:-4].replace('_',' ') + ' (Quadrature sum of X, Y, and Z components)' +\
            '\nComposite Estimated Ambient'
        else:
            title = self.name.replace('DQ','').replace('_',' ') + ' - Composite Estimated Ambient'
        title += '\n(Obtained from lowest coupling function over multiple injection locations)'
        ttl = plt.title(title, size=22)
        ttl.set_position([.7,1.05])
        #### CREATE LEGEND ####
        lgd = plt.legend(handles=lgd_patches, prop={'size':lgd_size}, bbox_to_anchor=(1.025,1), loc=2, borderaxespad=0.)
        fig.canvas.draw()
        # TEXT BELOW LEGEND
        leg_pxls = lgd.get_window_extent()
        ax_pxls = ax.get_window_extent()
        fig_pxls = fig.get_window_extent()
        # Convert back to figure normalized coordinates to create new axis
        pad = 0.01
        ax2 = fig.add_axes([leg_pxls.x0/fig_pxls.width, ax_pxls.y0/fig_pxls.height,\
                            leg_pxls.width/fig_pxls.width, (leg_pxls.y0-ax_pxls.y0)/fig_pxls.height-pad])
        # Eliminate all the tick marks
        ax2.tick_params(axis='both', left='off', top='off', right='off',
                        bottom='off', labelleft='off', labeltop='off',
                        labelright='off', labelbottom='off')
        ax2.axis('off')
        # Add text (finally)
        ax2_pxls = ax2.get_window_extent()
        text_width = (15 + max([len(name) for name in injection_names])) / .9
        if split_injections:
            caption = 'Ambient estimates are made by multiplying coupling factors by injection-free sensor levels. ' +\
            'CIRCLES indicate estimates from measured coupling factors, i.e. where the injection signal was seen in ' +\
            'the sensor and in DARM. '
            if 'MAG' in self.name:
                caption += 'TRIANGLES represent upper limit coupling factors, i.e. where a signal was seen in the sensor only. '
            else:
                caption += 'TRIANGLES represent upper limit coupling factors, i.e. where a signal was not seen in DARM. '
            caption += 'For some channels, at certain frequencies the ambient estimates are upper limits ' +\
            'because the ambient level is below the sensor noise floor.'
        else:
            caption = 'Ambient estimates are made by multiplying coupling factors by injection-free sensor levels. ' +\
            'For some channels, at certain frequencies the ambient estimates are upper limits because the ambient ' +\
            'level is below the sensor noise floor.'
        caption = '\n'.join(wrap(caption, text_width))
        ax2.text(0.02, .98, caption, size=lgd_size*.9, va='top')
        #### EXPORT PLOT ####
        my_dpi = fig.get_dpi()
        plt.savefig(filename, bbox_inches='tight', dpi=2*my_dpi)
        plt.close()
        return
        
    def to_csv(self, filename):
        """
        Save data to a csv file.
        
        Parameters
        ----------
        filename : str
            Target file name.
        """
        with open(filename, 'w') as file:
            header = 'frequency,factor,factor_counts,flag,ambient,darm\n'
            file.write(header)
            for i in range(len(self.freqs)):
                row = [
                    '{:.2f}'.format(self.freqs[i]),
                    '{:.2e}'.format(self.values[i]),
                    '{:.2e}'.format(self.values_in_counts[i]),
                    self.flags[i],
                    '{:.2e}'.format(self.ambients[i]),
                    '{:.2e}'.format(self.darm_bg[i])
                    ]
                line = ','.join(row) + '\n'
                file.write(line)
        return