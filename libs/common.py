import os
import pickle
import csv
import numpy as np
import datetime as dt
import math
import re
from collections import namedtuple
from pytz import timezone

TreeEvent = namedtuple('TreeEvent', 'faultId v_sag_a v_sag_b v_sag_c v_sag_1 v_sag_2 v_sag_3 std mean delta_v distance pcc generic_class class_name xd_decision')

def date_time_utc_converter(isoformat):
    if isoformat == '': return np.NaN
    return timezone('US/Pacific').localize(dt.datetime.strptime(isoformat, '%m/%d/%y %H:%M')).astimezone(timezone('UTC')).replace(tzinfo=None)

def date_time_converter(isoformat):
    if isoformat == '': return np.NaN
    return dt.datetime.strptime(isoformat, '%y-%m-%d %H:%M:%S.%f')

def chunk_date_range(cycle, before_seconds, after_seconds):
    starting_date = cycle - dt.timedelta(0, before_seconds, 0)
    return [starting_date + dt.timedelta(0, i, 0) for i in range(before_seconds + after_seconds)]   

def get_precise_cycle_fault_signature():
    return 'txt/precise_cycle_fault_signature.pickle'

def get_svm_classifier():
    return 'txt/svm_classifier.pickle'

def get_fn_sag_mode_find_faulted_bus():
    return 'txt/sag_vote_selector_events_find_faulted_bus_v1a.pickle'

def get_fn_normal_minutes():
    return 'txt/normal_array_4.pickle', 'txt/directed_normal_array_4.pickle'
"""
PROBLEM_FAULTS = [(10,0), # There is no data for this fault
                  (7,0),  # Duplicate spike of (7,1) & (7,0)'s signal is offline
                  (16,1), # Duplicate spike of (16,0)
                  (29,1),(29,2), # Duplicates of (29,0) spike
                  (51,1),(51,2), # Duplicates of (51,0) spike
                  (52,1),(52,2), # Duplicate of (52,0) spike
                  (60,1), # Duplicate of (60,0) spike
                  (100,0), # "Exact" time is incorrect, no data for this fault,
                  (1,0), (6,0), (9,0), (25,0), (33,0), (60,0), (60,1), (92,0), (98,0), (100,0), (3,0)
                  ]
"""
def savepickle(fn, obj):
    with open(fn, 'wb') as fin:
        pickle.dump(obj, fin)

def loadpickle(events_fn, predicate = None):
    with open(events_fn, 'rb') as fin:
        if (not predicate): return pickle.load(fin)
        return select(pickle.load(fin), predicate)

def get_generic_class(classfication):
    if (classfication in ['AG', 'BG', 'CG']): return 'SLG'
    if (classfication in ['AB', 'BC', 'AC']): return 'LtL'
    if (classfication in ['ABC']): return '3P'
    return 'noFault'