# Derived from pinpointfault_v1.py with tweak to limit the range of fault search
# to a subset of the data frame

import signals
import faults
import datetime as dt
import numpy as np
import logging
import events

class SignalError(RuntimeError):
    """A generic Signal Error -- such as not being able to find the signal"""
    def __init__(self, shortmsg, details=''):
        RuntimeError.__init__(self, shortmsg + (': %s'%details if details else ''))
        self.shortmsg = shortmsg

    def short(self):
        return "%s(%s)"%(self.__class__.__name__, self.shortmsg)
    
class SteadyStateError(SignalError):
    """An error regarding a steady state value"""
    pass

class NominalVoltageError(SignalError):
    """An error regarding a signals nominal voltage"""
    pass



def find_steadystate_for_signal_cluster(df, rightedge_i, clustered_ss_map, signal):
    """Find the steady state prior to a known fault
    
    For each phase in the signal cluster, compute the steady state
    signal prior to the fault and sanity check its value.
    
    returns (first_phase, phase_data)
    """
    outofservice_threshold = .8
    phase_data = {}
    for c in clustered_ss_map[signal]:
        try:
            ss_info = signals.find_steadystate(df, 60, c, rightedge_i=rightedge_i)            
            
        except KeyError:
            raise SignalError("No signal", c)
            
        if ss_info == None:
            raise SteadyStateError("Can't compute steady state", c)
        if '500' in c: 
            nominal = 500
        elif '230' in c:
            nominal = 230
        elif '34.5' in c:
            nominal = 34.5
        else:
            raise NominalVoltageError('Unexpected/no stated nominal voltage')

        si, ei, mean = ss_info 

        if (mean < outofservice_threshold*nominal):
            raise NominalVoltageError('OutOfService','(mean kV=%.f) %s'%(mean, c))

        phase_data[c] = {'ss_start_t':df['DateTime'].iloc[si],
                         'ss_end_t':df['DateTime'].iloc[ei-1],
                         'v_ss':mean}

        
    # Find the column where the v_ss is obtained earliest...
    first_phase = clustered_ss_map[signal][0]
    for c in clustered_ss_map[signal]:
        if phase_data[c]['ss_start_t'] < phase_data[first_phase]['ss_start_t']:
            first_phase = c
        
    return (first_phase, phase_data)


def find_vsag(df, i0, clustered_ss_map, signal, phase_data, i1=None):
    """Find v_sag for a given signal cluster; UPDATES phase_data.
    
    Given:
      phase_data containing 'v_ss' values for each phased signal
   
    Finds:
      v_sag (pu) -- the lowest value in the dataframe for the phased signal.
      v_sag_i -- the index at which v_sag occurs
      v_sag_t -- the time at which v_sag occurs
    Updates phase_data with v_sag (pu) and v_sag_i,v_sag_t for each phased signal
    
    """
    
    dominant_sag_v = None
    dominant_sag = None
    if i1 == None:
        df_fault = df.iloc[i0:] 
    else:
        df_fault = df.iloc[i0:i1]
    #print "i0, i1=", i0, i1
    for column in clustered_ss_map[signal]:
        assert not 'v_sag' in phase_data[column]
        assert not 'v_sag_i' in phase_data[column]
        
        # careful here. idxmin() gives a pandas index which here is
        # a number that corresponds to the integer location since the beginning
        # of the dataframe.  However, that assumption could be violated in the future
        v_sag_i = df_fault[column].idxmin()  
        # check assumption
	if v_sag_i is np.nan: raise SignalError('The PMU data is unavailable.')
	assert df[column].iloc[v_sag_i] == df[column][v_sag_i]
        v_sag = df[column][v_sag_i] / phase_data[column]['v_ss']
        phase_data[column]['v_sag'] = v_sag
        phase_data[column]['v_sag_i'] = v_sag_i
        phase_data[column]['v_sag_t'] = df['DateTime'][v_sag_i]
        if dominant_sag_v == None or dominant_sag_v > v_sag:
            dominant_sag_v = v_sag
            dominant_sag = column

    return (dominant_sag, dominant_sag_v)


def find_fault_window(df, phase_data, signal, N=1):
    """Finds the start and end points of a fault
    
    Given:
    phase_data with v_sag_i, v_ss and v_sag
    
    walk forward and backward from v_sag_i until a
    steady state condition is maintained for N seconds
    
    Returns:
    (s,e,N) : the fault occurs in the window: df[signal][s:e] 
      steady state occurs at:
      df[signal][s-N*60:s]
      df[signal][e:e+N*60]
      e-s is the fault duration
      
    """
    v_sag_i = phase_data[signal]['v_sag_i']
    v_ss = phase_data[signal]['v_ss']
    v_fault_si = v_sag_i - int(N*60)
    while v_fault_si > 0:
        region = df[signal].iloc[v_fault_si:v_fault_si+int(N*60)]
        if region.iloc[-1] >= .99*v_ss and region.min() >= 0.95*v_ss and region.max() <= 1.05*v_ss:
            break
        v_fault_si -= 1
    else:
        logging.critical("Didn't recover steady state condition at front: %s"%(signal))
        v_fault_si = 0

    v_fault_ei = v_sag_i + int(N*60)
    while v_fault_ei < len(df[signal]):
        region = df[signal].iloc[(v_fault_ei-int(N*60)):v_fault_ei]

        if region.iloc[0] >= .99*v_ss and region.min() >= 0.95*v_ss and region.max() <= 1.05*v_ss:
            break
        v_fault_ei += 1
    else:
        logging.critical("Didn't recover steady state condition at tail: %s"%(signal))
        v_fault_ei = len(df[signal])-1

    return (v_fault_si+int(N*60), v_fault_ei-int(N*60), N)


def clean_dataframe(df):
    return df.replace(0.0, np.nan)

def pinpoint_fault(h5minutestore, faultinst, max_duration_after_fault=120, N=1):
   
    """Given a faultinstance and a maximum duration,
    find the salient signals for the fault and return an Event for each signal.
    
    The Event is constructed by looking from the nominal (bpa) time forward
    until the event's duration or the max_duration_after_fault is exceeded. 
    Within this window, the sag voltage for each phase is obtained.
    Given the smallest sag value, the event's start and end time are determined.
    """
    results = {}
    eventlst = []
    # Step 1. Find all salient signals
    _ = signals.salient_signals(faultinst, lambda x: x.endswith('Voltage Mag'))
    clustered_ss_signals, mps = signals.cluster_phase_signals(_)
   
    dominant_signal_sag = (None,None,None)
    
    # Step 2. Get the data frame (including a window for finding steady state)
    # note that this deviates slightly from the original implementation in PlotsOfAllFaults...
    t_prior = 10  # number of seconds prior to the fault for finding v_ss
    max_window = min( faultinst.duration.seconds + t_prior, 
                      max_duration_after_fault)

    df = faults.get_window_as_dataframe(faultinst.exact_time, t_prior, max_window, h5minutestore)
    df = clean_dataframe(df)
   
    for signal in mps:

        # Step 3. Find v_ss
        signal_states = {}
        try:
            (firstphase, phase_data) = find_steadystate_for_signal_cluster(df, t_prior*60,
                                                                          clustered_ss_signals,
                                                                          signal)
            signal_states[signal] = "[OK]"
        except SignalError as e:
            signal_states[signal] = "[%s]"%(e.short())
            print "%s %s"%(signal,signal_states[signal])
            continue


        # Step 4. Find the v_sag (assumption here that there is only one)
        (ds, dsv) = find_vsag(df, t_prior*60-60, clustered_ss_signals, signal, phase_data)
        if dominant_signal_sag[0] == None or dsv < dominant_signal_sag[2]:
            dominant_signal_sag = (signal, ds, dsv)
        v_fault_si,v_fault_ei,_ = find_fault_window(df, phase_data, ds, N=N)

        print "%s %s"%(signal,signal_states[signal])
        
        p = signals.phases(clustered_ss_signals[signal])
        event = events.Event(df, signal, phase_data=phase_data, abc_phases=p, 
                             relativewindow=(v_fault_si,v_fault_ei),faultinst=faultinst)

        eventlst.append(event)

    return eventlst
