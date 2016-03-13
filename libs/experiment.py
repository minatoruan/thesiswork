import matplotlib.pyplot as pyplot
from numpy import array, isnan, nan, append, delete, mean
from libs.arff import *
from libs.classifiers import *
from libs.common import *
from libs.features import *
from bpati import signals

def experimental_plot(df, results, start_i, signal, smags, xline, xlim, filename=''):
    if results == None or all([isnan(r) for r in results]): 
        print 'No signals'
        return
    
    result_diff_i = []
    for idx, _class in enumerate(results):
        if len(result_diff_i) == 0 or idx == len(results) - 1: 
            result_diff_i.append(idx)
            continue
        if results[idx-1] == _class: continue
        result_diff_i.append(idx)
    
    xrng = np.arange(df.shape[0] - start_i)
    phase_a, phase_b, phase_c = smags
    
    pyplot.figure(figsize=(12, 4), dpi=100)   
    #pyplot.subplot(211)
    pyplot.title('%s\nStarting At: %s\nOrange: SLG, Yellow: LtL, Magenta: 3P, Purple: N/A' % (signal, str(df['DateTime'][0])))
    pyplot.xlabel('Cycle')
    pyplot.ylabel('Voltage\nA,B,C Phases in Red,Green,Blue')
    pyplot.plot(xrng, df[phase_a][start_i:], 'r-')
    pyplot.plot(xrng, df[phase_b][start_i:], 'g-')
    pyplot.plot(xrng, df[phase_c][start_i:], 'b-')
    if xlim is not None: pyplot.xlim((xlim[0]+start_i, xlim[1]+start_i))
    
    if xline:
        for idx in xrng: pyplot.axvline(idx, c='r', linestyle='--')
    
    for idx, start_span in enumerate(result_diff_i[:-1]):
        #if start_span > 38: continue
        if (results[start_span] == 0): color = {'color':'orange', 'alpha':.5}
        elif (results[start_span] == 1): color = {'color':'yellow', 'alpha':.5}
        elif (results[start_span] == 2): color = {'color':'magenta', 'alpha':.5}
        elif (np.isnan(results[start_span])): color = {'color':'purple', 'alpha':.5}
        else: continue
        pyplot.axvspan(start_span, result_diff_i[idx+1], **color)
    if filename == '': pyplot.show()   
    else: pyplot.savefig(filename)
        
def experimental_classifier_plot(df, classifier, xline=True, filename='', xlim=None):
    ce, results, start_i, v_ss = classifier.scan_df(df)
    experimental_plot(df, results, start_i, classifier.signal, classifier.smags, xline, xlim, filename=filename)
    
    """
    ce, results, start_i, v_ss = classifier.scan_df(df)
    
    if results == None or all([isnan(r) for r in results]): 
        print 'No signals'
        return
    
    result_diff_i = []
    for idx, _class in enumerate(results):
        if len(result_diff_i) == 0 or idx == len(results) - 1: 
            result_diff_i.append(idx)
            continue
        if results[idx-1] == _class: continue
        result_diff_i.append(idx)
    
    xrng = np.arange(df.shape[0] - start_i)
    phase_a, phase_b, phase_c = classifier.smags
    
    pyplot.figure(figsize=(12, 4), dpi=100)   
    #pyplot.subplot(211)
    pyplot.title('%s\nStarting At: %s\nOrange: SLG, Yellow: LtL, Magenta: 3P, Purple: N/A' % (classifier.signal, str(df['DateTime'][0])))
    pyplot.xlabel('Cycle')
    pyplot.ylabel('Voltage\nA,B,C Phases in Red,Green,Blue')
    pyplot.plot(xrng, df[phase_a][start_i:], 'r-')
    pyplot.plot(xrng, df[phase_b][start_i:], 'g-')
    pyplot.plot(xrng, df[phase_c][start_i:], 'b-')
    if xlim is not None: pyplot.xlim((xlim[0]+start_i, xlim[1]+start_i))
    
    if xline:
        for idx in xrng: pyplot.axvline(idx, c='r', linestyle='--')
    
    for idx, start_span in enumerate(result_diff_i[:-1]):
        #if start_span > 38: continue
        if (results[start_span] == 0): color = {'color':'orange', 'alpha':.5}
        elif (results[start_span] == 1): color = {'color':'yellow', 'alpha':.5}
        elif (results[start_span] == 2): color = {'color':'magenta', 'alpha':.5}
        elif (np.isnan(results[start_span])): color = {'color':'purple', 'alpha':.5}
        else: continue
        pyplot.axvspan(start_span, result_diff_i[idx+1], **color)
    
    pyplot.subplot(212)
    pyplot.xlabel('Cycle')
    pyplot.ylabel('Voltage (pu)\nA,B,C Phases in Red,Green,Blue')
    
    pyplot.plot(xrng, df[phase_a][start_i:]/array([v_a for v_a, v_b, v_c in v_ss]), 'r-')
    pyplot.plot(xrng, df[phase_b][start_i:]/array([v_b for v_a, v_b, v_c in v_ss]), 'g-')
    pyplot.plot(xrng, df[phase_c][start_i:]/array([v_c for v_a, v_b, v_c in v_ss]), 'b-')
    if xlim is not None: pyplot.xlim(xlim)
    
    if xline:
        for idx in xrng: pyplot.axvline(idx, c='r', linestyle='--')
    
    for idx, start_span in enumerate(result_diff_i[:-1]):
        if start_span > 38: continue
        if (results[start_span] == 0): color = {'color':'orange', 'alpha':.5}
        elif (results[start_span] == 1): color = {'color':'yellow', 'alpha':.5}
        elif (results[start_span] == 2): color = {'color':'magenta', 'alpha':.5}
        elif (np.isnan(results[start_span])): color = {'color':'purple', 'alpha':.5}
        else: continue
        pyplot.axvspan(start_span, result_diff_i[idx+1], **color)
    
    if filename == '': pyplot.show()   
    else: pyplot.savefig(filename)
    """
    
class ClassifyingExperimenter(object):
    def __init__(self, signal, smags, sangles, steady_state_checker_for_detection, classifier_pickle = get_svm_classifier(), exportingeventobj=ExportingEvent):
        self.signal = signal
        self.smags = smags[signal]
        self.sangles = sangles[signal.replace('Mag', 'Ang')]
        self.nominal = signals.nominalvoltage(signal)
        self.exportingeventobj = exportingeventobj

        self.last_ss_scan_datetime = None       
        self.signaldata = None
        self.datapoint = None
        
        self.classifier = EventClassifier.LoadPickle(classifier_pickle)
        self.steady_state_checker_for_detection = steady_state_checker_for_detection
        pass
        
    #the function return the tuple: success, normal_voltages, starting point and ending point
    def steady_state_(self, arr):
        extent = 30
        assert self.signaldata is not None or len(arr) > extent, 'Scan window is too narrow'
        
        for starti in range(0, len(arr) - extent):
            window = arr[starti:starti + extent]
            #print 'check nan or window <=25', any(isnan(window)) or any(window <= 25)
            #print 'come heh', any(isnan(window)), any(window <= 25)
            if any(isnan(window)) or any(window <= 25): continue
            if self.is_steady_state_(window): return True, mean(window), starti, starti + extent
        return False, nan, nan, nan
    
    def is_steady_state_(self, voltages):
        sagthresh=.9
        edgethresh=.01
        outofservice_threshold = 0.8
        
        w_mean, w_min = mean(voltages), min(voltages)
        edgemin = (1.0 - edgethresh) * w_mean
        edgemax = (1.0 + edgethresh) * w_mean
        return w_min > sagthresh*w_mean and voltages[0] > edgemin and\
                voltages[-1] > edgemin and w_mean >= outofservice_threshold * self.nominal
        
    def update_first_window_(self, df, steady_states, starti):
        extent_data_point = 3
        self.signaldata = {key: array(df[key][v_ss[2]:v_ss[3]]) for key, v_ss in zip(self.smags, steady_states)}
        self.signaldata.update({key: array(df[key][v_ss[2]:v_ss[3]]) for key, v_ss in zip(self.sangles, steady_states)})
        
        self.datapoint = {key: array(df[key][starti-extent_data_point:starti]) for key in self.signaldata}
    
    def compute_starting_point_(self, df):
        # test if the scan is continuous
        not_reset = self.last_ss_scan_datetime is not None and \
                        0 <= (df['DateTime'][0] - self.last_ss_scan_datetime).total_seconds() < 0.02
        #if self.last_ss_scan_datetime is not None:
        #    print 'Different from last scan', (df['DateTime'][0] - self.last_ss_scan_datetime).total_seconds()
        #print 'reset', not not_reset
        if any([col not in df for col in self.smags+self.sangles]): return -1
        if not_reset: return 0
        
        # no scan before
        v_ss_info = [self.steady_state_(array(df[s])) for s in self.smags]
        if not all([v_ss[0] for v_ss in v_ss_info]): return -1
        starti = max([v_ss[3] for v_ss in v_ss_info])
        self.update_first_window_(df, v_ss_info, starti)
        return starti
        
    def get_exporting_event_(self, df, idx, verbose=False):       
        for s in self.signaldata:
            self.datapoint[s] = append(delete(self.datapoint[s], 0), df[s][idx])
        if verbose:
            print '\tDatapoint'
            for k in datapoint: print '\t\t', k, datapoint[k]        
        return self.exportingeventobj(self.signal, self.smags, self.sangles, self.datapoint, self.signaldata)        
    
    # the function is a combination of logics to determine that the moment of time is appropriate to move
    # in this experiment, I would let the movement forward if the voltage is not less than 90% steady state.
    # the sag threshod is equal to the one used in steady_state_
    def is_safe_to_slide_(self, predict, v_a, v_b, v_c, verbose):
        ss_flags = [self.is_steady_state_(append(delete(self.signaldata[s], 0), v)) for v, s in zip([v_a, v_b, v_c], self.smags)]
        if verbose:
            print '\tSteady state flags:'
            for s, f in zip(self.smags, ss_flags): print '\t\t', s, f
        if self.steady_state_checker_for_detection: # if the last outcome is noFault, or is steady_state check is ok
            return predict == 3.0 or all(ss_flags)
        else: # if the last outcome is noFault and is steady_state check is ok
            return predict != 3.0 and all(ss_flags)        
    
    # return a tuple as
    #    self: the object experiment which contains up to date signal data
    #    results: the predict outcome 
    #    start_i: the start index of dataframe in which corespondents to the first predict outcome element in results 
    def scan_df(self, df, verbose = False):
        results = []
        v_ss = []
        start_i = self.compute_starting_point_(df)
               
        if start_i < 0: 
            if verbose: print '\tFailing steady state computation'
            return self, None, nan, None

        for idx in range(start_i, df.shape[0]):
            self.last_ss_scan_datetime = df['DateTime'][idx]
            
            ss_a, ss_b, ss_c = [mean(self.signaldata[s]) for s in self.smags]
                           
            if verbose:
                print 'At cycle', idx
                print '\tSteady state phase A', ss_a
                print '\tSteady state phase B', ss_b
                print '\tSteady state phase C', ss_c             
            
            ee = self.get_exporting_event_(df, idx, verbose)
            if not ee.error: # all success for ss computation
                x = {field.split()[0]: lambda_fn(ee)
                          for field, lambda_fn 
                          in defaultfields if field.split()[0] in self.classifier.features()}                
                if verbose:
                    print '\tComputed features:'
                    for k in x: print '\t\t', k, x[k]
                outcome_predict = self.classifier.predict_instance(x)
                # check to move
                if self.is_safe_to_slide_(outcome_predict, # recently predict outcome, test to bypass the ss
                                              df[self.smags[0]][idx], # v_a
                                              df[self.smags[1]][idx], # v_b
                                              df[self.smags[2]][idx],
                                              verbose = verbose): # v_c
                    if verbose: print '\tUpdating steady state', idx 
                    if self.steady_state_checker_for_detection:
                        if verbose and outcome_predict != 3.0: 
                            print '\tForcing the prediction', outcome_predict, 'to 3.0'
                        outcome_predict = 3.0 # can be seen as v_ss then force to nofault
                    for key in self.signaldata:
                        if verbose: print '\t\t', key, df[key][idx]
                        self.signaldata[key] = append(delete(self.signaldata[key], 0), df[key][idx])
                else:
                    if verbose: print '\tSkipping updating steady state', idx
                        
                results.append(outcome_predict)
                if verbose: print '\tPrediction outcome:', results[-1]
            else:
                if verbose: print '\tSkip since the signal is unvailable'
                results.append(nan)

        v_ss.append((ss_a, ss_b, ss_c))  
        return self, results, start_i, v_ss