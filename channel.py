import numpy as np
import logging

class ChannelInfoBase(object):
    """
    Channel information for a PEM sensor; used in ChannelASD, CoupFunc, and CompCoupFunc.
    
    Attributes
    ----------
    name : str
        Channel name.
    unit : str
        Units in which channel data is given.
    system : str
        System, usual PEM.
    station : str
        'CS', 'EX', or 'EY'.
    sensor : str
        Type of sensor, e.g. ACC or MAG.
    qty : str
        Physical quantity measured by this sensor, e.g. 'Magnetic Field'.
    coupling : str
        Coupling type, e.g. 'Magnetic'.
    """
    
    def __init__(self, name, unit):
        """
        Parameters
        ----------
        name : str
        unit : str
        """
        self.name = name
        self.ifo = self._ifo()
        self.system = self._system()
        self.station = self._station()
        self.sensor = self._sensor()
        if unit == '':
            self.unit = self._get_unit_from_sensor()
        else:
            self.unit = unit
        self.qty, self.coupling = self._get_measurement_info()
    
    def _ifo(self):
        try:
            idx = self.name.index(':')
            if self.name[ : idx] in ['H1', 'L1']:
                return self.name[ : self.name.index(':')]
            else:
                return ''
        except:
            return ''
    
    def _system(self):
        try:
            idx1 = self.name.index(':') + 1
            idx2 = self.name.index('-')
            return self.name[idx1:idx2]
        except:
            return ''
    
    def _station(self):
        try:
            idx1 = self.name.index('-') + 1
            idx2 = self.name.index('_')
            return self.name[idx1:idx2]
        except:
            return ''

    def _sensor(self):
        try:
            idx1 = self.name.index('_') + 1
            idx2 = self.name[idx1:].index('_') + idx1
            return self.name[idx1:idx2]
        except:
            return ''
    
    def _get_unit_from_sensor(self):
        units_dict = {'MIC': 'Pa', 'MAG': 'T', 'RADIO': 'ADC', 'SEIS': 'm', \
                      'ISI': 'm', 'ACC': 'm', 'HPI': 'm', 'ADC': 'm'}
        if self.sensor in units_dict.keys():
            return units_dict[self.sensor]
        elif self.system in units_dict.keys():
            return units_dict[self.system]
        else:
            return 'Counts'
    
    def _get_measurement_info(self):
        qty_dict = {'Pa': 'Pressure', 'T': 'Magnetic Field', 'm': 'Displacement', 'm/s': 'Velocity', 'm/s2': 'Acceleration'}
        coupling_dict = {'MIC': 'Acoustic', 'MAG': 'Magnetic', 'RADIO': 'RF', 'SEIS': 'Seismic', \
                     'ISI': 'Seismic', 'ACC': 'Vibrational', 'HPI': 'Seismic', 'ADC': 'Vibrational'}
        qty, coupling = ('', '')
        if self.unit in qty_dict.keys():
            qty = qty_dict[self.unit]
        if self.sensor in coupling_dict.keys():
            coupling = coupling_dict[self.sensor]
        elif self.system in coupling_dict.keys():
            coupling = coupling_dict[self.system]
        return qty, coupling

class ChannelASD(ChannelInfoBase):
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
        self.name = name
        freqs = np.asarray(freqs)
        values = np.asarray(values)
        self.freqs = freqs[freqs > 1e-3]
        self.values = values[freqs > 1e-3]
        self.t0 = t0
        self.unit = unit
        self.df = self.freqs[1] - self.freqs[0]
        self.channel = ChannelInfoBase(self.name, unit)
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
        if self.channel.sensor in sensor_units.keys():
            new_unit = sensor_units[self.channel.sensor]
        elif self.channel.system in system_units.keys():
            new_unit = system_units[self.channel.system]
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