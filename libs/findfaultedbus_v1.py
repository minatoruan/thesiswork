
import faults
import math
import pinpointfault_v1 as ppf
import pprint
import signals
import logging
import events

def find_faulted_bus(h5minutestore, faultinst, generic_signals, signalmap):
    """Given a fault instance, use it's approximate region of time and determine
    the bus that is most likely nearest to the fault. 
    
    The approach is:
    1) scan each generic signal in the approximate time region for a minimum sag value
    2) using per phase sag values, compute delta V, either with time independent sags 
        (method 1) or time dependent sags (method 2)
    3) find the bus with the maximum delta V using method 1 and method 2, and return 
        the bus and delta v for the two methods respectively"""
    
    
    t_prior = 10  # number of seconds prior to the fault for finding v_ss
    max_duration_after_fault = 120
    max_window = min( faultinst.duration.seconds + t_prior, 
                      max_duration_after_fault)

    df = faults.get_window_as_dataframe(faultinst.exact_time, t_prior, max_window, h5minutestore)
    df = ppf.clean_dataframe(df)

    # for each clustered vphase, find v_ss
    all_phase_data = {}
    bad_signal_outcomes = set([])
    for signal in generic_signals:
        try:
            (_, phase_data) = ppf.find_steadystate_for_signal_cluster(df, t_prior*60,
                                                                          signalmap,
                                                                          signal)
            all_phase_data.update(phase_data)
        except ppf.SignalError as e:
            bad_signal_outcomes.add(signal)
            continue

       
   
    pruned_signals = set(generic_signals) - bad_signal_outcomes
    dominant_signal_dvt = (None, None, None) # generic signal, phased signal, dv_t
    dominant_signal_dvt2 = (None, None, None) # generic signal, phased signal, dv_t
    for signal in pruned_signals:
        (ds, dsv) = ppf.find_vsag(df, t_prior*60-60, signalmap, signal, all_phase_data)
        # dv_t method 1: the same as the ground truth: sag's don't occur at the same time (necessarily)        dv_t = 0.0
        dv_t = 0.0
        for p in signalmap[signal]:
            dv_t += (1.0-all_phase_data[p]['v_sag'])**2
        dv_t = math.sqrt(dv_t)/3.0
        # dv_t method 2: sag's are forced to occur at the same moment in time
        dv_t2 = 0.0
        sagi = all_phase_data[ds]['v_sag_i']
        for p in signalmap[signal]:
            dv_t2 += (1.0 - (df[p].iloc[sagi]/all_phase_data[p]['v_ss']))**2
        dv_t2 = math.sqrt(dv_t2)/3.0
        
        if dominant_signal_dvt[0] == None or dv_t > dominant_signal_dvt[2]:
            dominant_signal_dvt = (signal, ds, dv_t)
        if dominant_signal_dvt2[0] == None or dv_t2 > dominant_signal_dvt2[2]:
            dominant_signal_dvt2 = (signal, ds, dv_t2)
    
    # below use the phased dominant signal...
    (st,en,_) = ppf.find_fault_window(df, all_phase_data, dominant_signal_dvt[1])
    e1 = events.Event(df,dominant_signal_dvt[0],faultinst=faultinst, 
                      phase_data=all_phase_data, relativewindow=(st,en),
                       abc_phases=signals.phases(signalmap[dominant_signal_dvt[0]]))

    (st,en,_) = ppf.find_fault_window(df, all_phase_data, dominant_signal_dvt2[1])
    sagi = all_phase_data[dominant_signal_dvt2[1]]['v_sag_i']
    for p in signalmap[dominant_signal_dvt2[0]]:
        all_phase_data[p]['v_sag'] = df[p].iloc[sagi]/all_phase_data[p]['v_ss']
    e2 = events.Event(df,dominant_signal_dvt2[0],faultinst=faultinst, 
                      phase_data=all_phase_data, relativewindow=(st,en),
                       abc_phases=signals.phases(signalmap[dominant_signal_dvt2[0]]))
    
    return (e1, e2, df)