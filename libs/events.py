import signals
from matplotlib import pyplot
import math
import numpy as np
import faults

class Event:
    """An Event in the style of the data stored in the dostream.txt file

    An Event stores all the information relevant for plotting an event"""
    
    def __init__(self,df,signal,**kwargs):
        # at a minimum and event is given by a signal
        # and a dataframe that conveys a window of time
        self.signal = signal
        self.window_start_t = df['DateTime'].iloc[0]
        self.window_end_t = df['DateTime'].iloc[-1]
        if kwargs.has_key('faultinst'):    
            self.faultinst = kwargs['faultinst']
        else:
            self.faultinst = None

        if kwargs.has_key('phase_data'):
            assert kwargs.has_key('abc_phases')
            p=kwargs['abc_phases']

            self.phase_a = p['A']
            self.phase_b = p['B']
            self.phase_c = p['C']

            if kwargs['phase_data'][p['A']].has_key('v_ss'):
                self.v_ss_a = kwargs['phase_data'][p['A']]['v_ss']
                self.v_ss_b = kwargs['phase_data'][p['B']]['v_ss']
                self.v_ss_c = kwargs['phase_data'][p['C']]['v_ss']

                self.v_ss_start_t_a = kwargs['phase_data'][p['A']]['ss_start_t']
                self.v_ss_start_t_b = kwargs['phase_data'][p['B']]['ss_start_t']
                self.v_ss_start_t_c = kwargs['phase_data'][p['C']]['ss_start_t']

                self.t_ss = min(self.v_ss_start_t_a,
                                self.v_ss_start_t_b,
                                self.v_ss_start_t_c)
                
                # this also informs the start window
                self.window_start_t = self.t_ss

                self.v_ss_end_t_a = kwargs['phase_data'][p['A']]['ss_end_t']
                self.v_ss_end_t_b = kwargs['phase_data'][p['B']]['ss_end_t']
                self.v_ss_end_t_c = kwargs['phase_data'][p['C']]['ss_end_t']

                if self.faultinst:
                    self.bpa_cycles_from_t_ss = signals.timedelta_to_cycles(self.faultinst['bpa_exact_time'] - self.t_ss)
            
            if kwargs['phase_data'][p['A']].has_key('v_sag'):
                self.v_sag_a = kwargs['phase_data'][p['A']]['v_sag']
                self.v_sag_b = kwargs['phase_data'][p['B']]['v_sag']
                self.v_sag_c = kwargs['phase_data'][p['C']]['v_sag']
                
                self.v_sagt_a = kwargs['phase_data'][p['A']]['v_sag_t']
                self.v_sagt_b = kwargs['phase_data'][p['B']]['v_sag_t']
                self.v_sagt_c = kwargs['phase_data'][p['C']]['v_sag_t']

           
        if kwargs.has_key('relativewindow'):
            fsi,fei = kwargs['relativewindow']
            self.delta_t_cycles = fei - fsi
            self.window_start_t = df['DateTime'].iloc[fsi]
            self.window_end_t = df['DateTime'].iloc[fei]

            if kwargs.has_key('phase_data'):
                self._vsi_a = kwargs['phase_data'][p['A']]['v_sag_i'] - fsi
                self._vsi_b = kwargs['phase_data'][p['B']]['v_sag_i'] - fsi
                self._vsi_c = kwargs['phase_data'][p['C']]['v_sag_i'] - fsi

            if self.faultinst:
                dft0 = df['DateTime'].iloc[0]
                self.cycles_from_bpa = fsi - signals.timedelta_to_cycles(self.faultinst['bpa_exact_time']-dft0)
    
    def dv_t(self):
        if self.has('v_sag_a'):
            dv_t = math.sqrt((1.0-self.v_sag_a)**2 + (1.0-self.v_sag_b)**2 + (1.0-self.v_sag_c)**2)/3.0
            return dv_t
        else:
            raise ValueError("Cannot calculate dv_t")
    def has(self, key):
        return self.__dict__.has_key(key)

    def smallest_sag(self):
        return min(self.v_sag_a, self.v_sag_b, self.v_sag_c)
        
    def dataline(self):
        if not (self.has('faultinst') and self.has('v_ss_a') and self.has('v_sag_a')):
            return "? Insufficient info to produce a dataline"

        return '%d "%s" %f %f %f %f %f %f %f %d %f %d %d %d %d'%(self.faultinst['id'],
                    self.signal, self.v_ss_a, self.v_sag_a,
                    self.v_ss_b, self.v_sag_b, self.v_ss_c, self.v_sag_c,
                    (self.delta_t_cycles)/60.0,self.delta_t_cycles,
                    (self.cycles_from_bpa)/60.0,self.cycles_from_bpa,
                    self._vsi_a, self._vsi_b, self._vsi_c)


    
def plot(h5minutestore, event, title=None, faultregiononly=False):
    """Plot an event."""
    #df_duration = (event.bpa_cycles_from_t_ss + event.cycles_from_bpa +
    #               event.delta_t_cycles + 60)
    
    
    if title == None:
        title = "Fault: %s\nSignal: %s"%(str((event.faultinst.index, event.faultinst.subindex)),event.signal)

    duration = (event.window_end_t - event.t_ss).total_seconds() + 1
    df = faults.get_window_as_dataframe(event.t_ss, 0,
                                          duration, #math.ceil(df_duration/60.0),
                                          h5minutestore)

    # set x = 0 at t_bpa
    xrng = np.arange(df.shape[0])
    xrng -= event.bpa_cycles_from_t_ss

    pyplot.figure(figsize=(10, 5), dpi=100)

    pyplot.plot(xrng, df[event.phase_a]/event.v_ss_a, 'r-')
    pyplot.plot(xrng, df[event.phase_b]/event.v_ss_b, 'g-')
    pyplot.plot(xrng, df[event.phase_c]/event.v_ss_c, 'b-')
    
    if event.has('cycles_from_bpa') and event.has('delta_t_cycles'):
        pyplot.axvspan(event.cycles_from_bpa,
                       event.cycles_from_bpa + event.delta_t_cycles,
                       **{'color':'orange', 'alpha':.5})
    
    pyplot.axvline(0, c='r', linestyle='--')
    pyplot.xlabel('Time (cycles since fault.exact_time: %s)'%(str(event.faultinst.exact_time)))
    pyplot.ylabel('Voltage (pu)\nA,B,C Phases in Red,Green,Blue')
    
    if event.has('delta_t_cycles'):
        pyplot.figtext(.65,.15,
                   "$\Delta t=%d \/ \mathrm{cycles}$"%(event.delta_t_cycles),
                   fontsize=14)
            
    if faultregiononly:

        pyplot.xlim(event.cycles_from_bpa - 30,
                    event.cycles_from_bpa + event.delta_t_cycles + 30)
                    
    pyplot.title(title)
    pyplot.show()

def plot(h5minutestore, event, xd_class, dc_class, title=None, faultregiononly=False):
    """Plot an event."""
    #df_duration = (event.bpa_cycles_from_t_ss + event.cycles_from_bpa +
    #               event.delta_t_cycles + 60)
    
    
    if title == None:
        title = "Fault: %s\nSignal: %s"%(str((event.faultinst.index, event.faultinst.subindex)),event.signal)
	title  += '\nXD decision tree: %s'% xd_class
	title  += '\nMy decision tree: %s'% dc_class

    duration = (event.window_end_t - event.t_ss).total_seconds() + 1
    df = faults.get_window_as_dataframe(event.t_ss, 0,
                                          duration, #math.ceil(df_duration/60.0),
                                          h5minutestore)

    # set x = 0 at t_bpa
    xrng = np.arange(df.shape[0])
    xrng -= event.bpa_cycles_from_t_ss

    pyplot.figure(figsize=(10, 5), dpi=100)

    pyplot.plot(xrng, df[event.phase_a]/event.v_ss_a, 'r-')
    pyplot.plot(xrng, df[event.phase_b]/event.v_ss_b, 'g-')
    pyplot.plot(xrng, df[event.phase_c]/event.v_ss_c, 'b-')
    
    if event.has('cycles_from_bpa') and event.has('delta_t_cycles'):
        pyplot.axvspan(event.cycles_from_bpa,
                       event.cycles_from_bpa + event.delta_t_cycles,
                       **{'color':'orange', 'alpha':.5})
    
    pyplot.axvline(0, c='r', linestyle='--')
    pyplot.xlabel('Time (cycles since fault.exact_time: %s)'%(str(event.faultinst.exact_time)))
    pyplot.ylabel('Voltage (pu)\nA,B,C Phases in Red,Green,Blue')
    
    if event.has('delta_t_cycles'):
        pyplot.figtext(.65,.15,
                   "$\Delta t=%d \/ \mathrm{cycles}$"%(event.delta_t_cycles),
                   fontsize=14)
            
    if faultregiononly:

        pyplot.xlim(event.cycles_from_bpa - 30,
                    event.cycles_from_bpa + event.delta_t_cycles + 30)
                    
    pyplot.title(title)
    pyplot.show()
    
