import numpy as np
import re
import logging
from gwpy.detector import Channel

class PEMChannel(Channel):
    def __init__(self, name, unit=''):
        super(PEMChannel, self).__init__(name)
        try:
            sensor_match = re.search('(?P<sensor>[a-zA-Z0-9]+)', self.signal)
            if sensor_match is None:
                raise ValueError("Cannot parse channel name according to LIGO "
                                 "channel-naming convention T990033")
            parts = sensor_match.groupdict()
        except (TypeError, ValueError):
            self.sensor = ''
        else:
            for key, val in parts.items():
                try:
                    setattr(self, key, val)
                except AttributeError:
                    setattr(self, '_%s' % key, val)
        self._init_pem_info(unit=unit)
    
    def _init_pem_info(self, unit=''):
        units_dict = {'MIC': 'Pa', 'MAG': 'T', 'RADIO': 'ADC', 'SEIS': 'm', \
                      'ISI': 'm', 'ACC': 'm', 'HPI': 'm', 'ADC': 'm'}
        qty_dict = {'Pa': 'Pressure', 'T': 'Magnetic Field', 'm': 'Displacement', 'm/s': 'Velocity', 'm/s2': 'Acceleration'}
        coupling_dict = {'MIC': 'Acoustic', 'MAG': 'Magnetic', 'RADIO': 'RF', 'SEIS': 'Seismic', \
                     'ISI': 'Seismic', 'ACC': 'Vibrational', 'HPI': 'Seismic', 'ADC': 'Vibrational'}
        if unit == '':
            try:
                self.unit = units_dict[self.sensor]
            except KeyError:
                try:
                    self.unit = units_dict[self.system]
                except KeyError:
                    self.unit = 'Counts'
        else:
            self.unit = unit
        try:
            self.qty = qty_dict[self.unit.name]
        except KeyError:
            self.qty = ''
        try:
            self.coupling = coupling_dict[self.sensor]
        except KeyError:
            try:
                self.coupling = coupling_dict[self.system]
            except KeyError:
                self.coupling = ''

class PEMChannelASD(PEMChannel):
    """
    Calibrated amplitude spectral density of a single channel.
    
    Attributes
    ----------
    name : str
        Channel name.
    freqs : array
        Frequencies.
    values : array
        Calibrated ASD values.
    t0 : float, int
        Start time of channel.
    unit : str
        Unit of measurement for 'values'
    calibration : array-like
        Calibration factors across all frequencies.
    """
    
    def __init__(self, name, freqs, values, t0=None, unit='', calibration=None):
        """
        Parameters
        ----------
        name : str
        freqs : array
        values : array
        t0 : float, int, optional
        unit : str, optional
        calibration : array-like, optional
        """
        super(PEMChannelASD, self).__init__(name, unit=unit)
        freqs = np.asarray(freqs)
        values = np.asarray(values)
        self.freqs = freqs[freqs > 1e-3]
        self.values = values[freqs > 1e-3]
        self.t0 = t0
        try:
            self.df = self.freqs[1] - self.freqs[0]
        except IndexError:
            self.df = None
        self.calibration = calibration
    
    def calibrate(self, calibration_factor, new_unit=None):
        """
        Calibrate sensor based on sensor type.
        
        Parameters
        ----------
        calibration_factor : float, int
            Conversion factor from ADC counts to physical units.
        new_unit : str, optional
            Physical unit to assign to calibrated data.
        """
        
        sensor_units = {'SEI': 'm/s', 'ACC': 'm/s2', 'MIC': 'Pa', 'MAG': 'T'}
        system_units = {'HPI': 'm/s', 'ISI': 'm/s'}
        if self.sensor in sensor_units.keys():
            new_unit = sensor_units[self.sensor]
        elif self.system in system_units.keys():
            new_unit = system_units[self.system]
        else:
            new_unit = ''
        self.calibration = np.ones_like(self.values) * calibration_factor
        # Convert m/s and m/s2 into displacement
        if new_unit == 'm/s':
            self.calibration /= self.freqs * (2*np.pi) # Divide by omega
            new_unit = 'm'
        elif new_unit == 'm/s2':
            self.calibration /= (self.freqs * (2 * np.pi))**2 # Divide by omega^2
            new_unit = 'm'
        # Apply calibration and assign new (physical units)
        self.values *= self.calibration
        self.unit = new_unit
    
    def crop(self, fmin, fmax):
        """
        Crop ASD between fmin and fmax.
        
        Parameters
        ----------
        fmin : float, int
            Minimum frequency (Hz).
        fmax : float, int
            Maximum frequency (Hz).
        """
        
        try:
            fmin = float(fmin)
            fmax = float(fmax)
        except:
            print('')
            logging.warning('.crop method for ChannelASD object requires float inputs for fmin, fmax.')
            print('')
            return
        # Determine start and end indices from fmin and fmax
        if fmin > self.freqs[0]:
            start = int(float(fmin - self.freqs[0]) / float(self.df))
        else:
            start = None
        if fmax <= self.freqs[-1]:
            end = int(float(fmax - self.freqs[0]) / float(self.df))
        else:
            end = None
        # Crop data to start and end
        self.freqs = self.freqs[start:end]
        self.values = self.values[start:end]
        if self.calibration is not None:
            self.calibration = self.calibration[start:end]