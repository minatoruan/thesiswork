
"""Xiaodong's approach for determining fault type"""

import math
import numpy as np
import faults
import signals

def is_slg(e, sags, checktime=False, quiet=False):
    """is the fault a single line to ground?
    
    checktime -- True iff fault duration should be used for classification
    quiet -- True to minimize diagnostic output
    
    returns one of: {'AG','BG','CG','F'}

    """
    if sags[0] <= 0.95:
        if not quiet: print " - smallest sag is below .95 pu (testing for SLG)"

        if sags[1] >= .93 and sags[2] >= .93:
            if not quiet: print " - two sags are above .95 pu (testing for SLG)"
            if checktime and (e.delta_t_cycles < 6 or e.delta_t_cycles > 10):
                if not quiet: print " - bad fault length for SLG and LL %d"%(e.delta_t_cycles)
                return "F"
            if sags[0] == e.v_sag_a: return "AG"
            if sags[0] == e.v_sag_b: return "BG"
            return "CG"
        else:
            if not quiet: print " - fails SLG tests for sags[1] and sags[2] %s"%(str(sags))
    else:
        if not quiet: print " - fails SLG test for sag[0] %.f"%(sags[0])
    return "F"

def is_ll(e, sags, checktime=False, quiet=False):
    """is the fault a line to line?
    
    checktime -- True iff fault duration should be used for classification
    quiet -- True to minimize diagnostic output
    
    returns one of: {'AB','BC','AC','F'}

    """
    if sags[0] >= 0.35 and sags[1] >= 0.35:
        if not quiet: print " - two sags are above .52 pu (testing for LL)"
        if sags[0] <= 0.93 and sags[1] <= 0.93:
            if not quiet: print " - two sags are below .90 pu (testing for LL)"
            if sags[2] > 0.88:
                if not quiet: print " - one sags are above .90 pu (testing for LL)"
                
                if checktime and (e.delta_t_cycles < 6 or e.delta_t_cycles > 10):
                    if not quiet: print " - bad fault length for SLG and LL %d"%(e.delta_t_cycles)
                    return "F"

                if sags[2] == e.v_sag_a: return "BC"
                if sags[2] == e.v_sag_b: return "AC"
                return "AB"
            else:
                if not quiet: print " - fails sags[2] upper bound: %.f"%(sags[2])
        else:
            if not quiet: print " - fails LL sags[0]/sags[1] upper bound %s"%(str(sags))
    else:
        if not quiet: print " - fails LL sags[0]/sags[1] lower bound %s"%(str(sags))
    return "F"

def is_tp(e, sags, checktime=False, quiet=False):
    """is the fault three phase?
    
    checktime -- True iff fault duration should be used for classification
    quiet -- True to minimize diagnostic output
    
    returns one of: {'ABC','F'}

    """
    if np.all([s >= 0 for s in sags]):
        if not quiet: print " - sags are all above 0 (testing for 3P)"

        if np.all([s <= .8 for s in sags]): 
            if not quiet: print " - sags are all <= .8 (testing for 3P)"

            if sags.std() < 0.05 * sags.mean():
                if not quiet: print " - sags standard deviation is reasonable (testing for 3P)"
                if checktime and (e.delta_t_cycles < 2 or e.delta_t_cycles > 10):
                    if not quiet: print " - bad fault length for 3P %d"%(e.delta_t_cycles)
                    return "F"

                return "ABC"
            else:
                if not quiet: print " - fails 3P std test [std: %f, mean: %f]"%(sags.std(), sags.mean())
                return "F"
        else:
            if not quiet: print " - fails 3P upper bound test %s"%(str(sags))
            return "F"
    else:
        if not quiet: print " - fails 3P lower bound test %s"%(str(sags))
        return "F"

def classify_event(e, quiet=False):
    """classify an Event instance based on Xiaodong's hand-built decision treee
    
    quiet -- True to minimize diagonostic output
    
    returns one of: {'AG', 'BG', 'CG', 'AB', 'BC', 'AC', 'ABC', 'F'}
    also, modifies the event's classification field to contain the returned value
    
    """
    if not quiet: print " -- classifying fault --"
    sags = np.array(sorted([e.v_sag_a, e.v_sag_b, e.v_sag_c])) # ascending order
    
    slg = is_slg(e, sags, quiet=quiet)
    if slg != 'F': 
        e.classification = slg
        return slg
    ll = is_ll(e, sags, quiet=quiet)
    if ll != 'F': 
        e.classification = ll
        return ll
    tp = is_tp(e, sags, quiet=quiet)
    e.classification = tp
    return tp