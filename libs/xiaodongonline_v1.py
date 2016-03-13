
import pinpointfault_v1 as ppf 
import classifyfault_v1 as cf
import faults
import signals
import events
import random
import numpy as np


def all_fault_intervals(all_fault_sig):
    x = []
    for k in all_fault_sig:
        x.extend([(k,interval) for interval in all_fault_sig[k] if interval[0] != 'F'])
    x.sort(key=lambda x: x[1][2])
    return x


def scan_dataframe(df, tp_signals, signalmap, startrow=None, 
                   endrow=None, ss_n=20):
    """Scan a dataframe, incrementally, as though it were a stream and examine
    each timestep with respect to Xiaodong's fault decision tree.
    
    df         - the data frame to scan
    tp_signals - a list of 'three phase' generic/pos sequence signals  
    signalmap  - a map from generic signal name to a list of phased-names
    startrow   - the row from which to begin the scan (None, starts at ss_n)
    endrow     - the row at which to end the scan (None will scan to the end 
                 of the frame)
    ss_n       - the size of the window used to compute steady state
    """

    # initialize some data structures and validate that the signals all exist.
    bad_signals = set([])
    rlfault_per_sig = {}
    phase_data = {}
    ss_per_sig = {}
    for sig in tp_signals:
        try:
            a = df[sig]
            for phase in signalmap[sig]:
                phase_data[phase] = {}
                ss_per_sig[phase] = []
        except KeyError:
            bad_signals.add(sig)
    generic_signals = list( set(tp_signals) - bad_signals )

    if endrow == None:
        endrow = df.shape[0]
    if startrow == None:
        startrow = ss_n
        
    row = startrow
    ss_s = row - ss_n  # steady state start point
    ss_sv = row - ss_n # steady state VALID start point (may lag ss_s)

    # walk the rows...
    # this is suboptimal and could/should be parallelized and/or vectorized
    rowtime = df['DateTime'].iloc[row]
    while row < endrow:
        # note that prowtime = rowtime for first row... that's fine.
        prowtime = rowtime  # the time of the prior row...
        rowtime = df['DateTime'].iloc[row]
    
        for sig in generic_signals:
            # get A/B/C phases
            # p = signals.phases(signalmap[sig])
            for phase in signalmap[sig]:
                # for each phase, compute the steadystate
                ss_segment = df[phase].iloc[ss_s:row]
                ss = np.median(ss_segment)
                mn = np.amin(ss_segment)
                mx = np.amax(ss_segment)
                # subject the calculation to a quick sanity check for 
                # outliers. If the sanity check fails, use the last
                # valid steady state if it exists.
                if mn < .95*ss or mx > 1.05*ss:
                    a = ss_per_sig[phase]
                    if len(a) == 0:
                        ss = np.nan
                        ss_sv = ss_s
                    else:
                        ss = a[-1]
                        # don't update ss_sv, we're reusing the old ss value!
                else:
                    ss_sv = ss_s
                    
                ss_per_sig[phase].append(ss)
                phase_data[phase]['v_ss'] = ss
                phase_data[phase]['ss_start_t'] = df['DateTime'].iloc[ss_sv]
                phase_data[phase]['ss_end_t'] = df['DateTime'].iloc[ss_sv+ss_n]
                phase_data[phase]['v_sag'] = df[phase].iloc[row]/ss
                phase_data[phase]['v_sag_t'] = df['DateTime'].iloc[row]
                    
            e = events.Event(df, sig, phase_data=phase_data, 
                             abc_phases=signals.phases(signalmap[sig]))

            r = 'F'
            if e.smallest_sag() < .95:
                r = cf.classify_event(e, quiet=True)
        
            # NOTE that these regions are INCLUSIVE of both endpoints.
            # Be careful is slicing with iloc!
            regions = rlfault_per_sig.setdefault(sig,[[r,rowtime,rowtime]])
            if r != regions[-1][0]:
                regions[-1][2] = rowtime #row-startrow
                regions.append([r,rowtime,prowtime])
                #regions.append([r,row-startrow,row-startrow])
        row += 1
        ss_s += 1
        
    # when we exit the loop, rowtime is the last valid row time
    # (i.e., prowtime has not been updated) so, we use rowtime here
    # instead of prowtime as above.
    for sig in generic_signals:
        rlfault_per_sig[sig][-1][2] = rowtime
    
    return (rlfault_per_sig, ss_per_sig)