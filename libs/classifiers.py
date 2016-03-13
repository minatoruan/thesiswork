import numpy as np
import common
from sklearn.svm import SVC

class BaseClassifier(object):
    def get_generic_class_(self, classfication):
        if (classfication in ['AG', 'BG', 'CG']): return 'SLG'
        if (classfication in ['AB', 'BA', 'BC', 'BC', 'AC', 'CA']): return 'LtL'
        if (classfication in ['ABC']): return '3P'
        return 'noFault'
    
    def translate_(self, classfication):
        if (classfication in ['AG', 'BG', 'CG', 'SLG']): return 0.0
        if (classfication in ['AB', 'BA', 'BC', 'BC', 'AC', 'CA', 'LtL']): return 1.0
        if (classfication in ['ABC', '3P']): return 2.0
        return 3.0
    
    def vote_predict_by_fault_(self, faults, X):
        results = {}
        classification = []
        for x, y_p in zip(X, self.predict(X)):
            fid = x['faultId'], x['subfaultId']
            if fid not in results: results[fid] = []
            results[fid].append(y_p)
    
        for fid in faults:
            if fid not in results: raise Exception('No data for (%d, %d)' % (fid[0], fid[1]))
            if 3 in results[fid] and len(results[fid]) > 1: results[fid].remove(3)
            classification.append([_c for _c in results[fid]])
        return classification
    
    def predict(self, X):
        return [self.predict_instance(e) for e in X]
    
    def decision_function(self, X):
        return [self.decision_function_instance(e) for e in X]
    
    def predict_instance(self, X): pass
    
    def decision_function_instance(self, X): pass
    
    def features(self): pass
    
    def __init__(self): 
        pass
    
    def __str__(self): pass

class XLVoltageSagClassifier(BaseClassifier):
    def vote_predict_by_fault(self, faults, X):
        return BaseClassifier.vote_predict_by_fault_(self, faults, X)        
    
    def is_slg_(self, sags):
        if sags[0] <= 0.95 and sags[1] >= .93 and sags[2] >= .93: return True
        return False

    def is_ll_(self, sags):
        if sags[0] >= 0.35 and sags[1] >= 0.35:
            if sags[0] <= 0.93 and sags[1] <= 0.93:
                if sags[2] > 0.88: return True
        return False

    def is_tp_(self, sags):
        if np.all([s >= 0 for s in sags]):
            if np.all([s <= .8 for s in sags]): 
                if sags.std() < 0.05 * sags.mean():
                    return True
        return False
    
    def predict_instance(self, X):
        sags = np.array([X[f] for f in self.features()]) # ascending order
        if self.is_slg_(sags): return self.translate_('SLG')
        elif self.is_ll_(sags): return self.translate_('LtL')
        elif self.is_tp_(sags): return self.translate_('3P')
        return self.translate_('noFault')
    
    def features(self):
        return ['v_sag_1', 'v_sag_2', 'v_sag_3']
    
    def __init__(self):
        super(XLVoltageSagClassifier, self).__init__()
        
class XLRevisedVoltageSagClassifier(XLVoltageSagClassifier):
    def is_slg_(self, sags):
        if sags[0] <= 0.95 and sags[1] >= .95 and sags[2] >= .95: return True
        return False

    def is_ll_(self, sags):
        if sags[0] >= 0.35 and sags[1] >= 0.35:
            if sags[0] <= 0.95 and sags[1] <= 0.95:
                if sags[2] > 0.95: return True
        return False

    def is_tp_(self, sags):
        if np.all([s >= 0 for s in sags]):
            if np.all([s <= .8 for s in sags]):
                sag_mean = sags.mean() 
                dsag = [(s - sag_mean) for s in sags]
                dsag_abs = [abs(d) for d in dsag]
                dsag_absmax = max(dsag_abs)
                if (dsag_absmax/sag_mean) <= .1:
                    return True
        return False
    
    def __init__(self):
        super(XLRevisedVoltageSagClassifier, self).__init__()

class XLRuleBasedClassifier(BaseClassifier):
    def vote_predict_by_fault(self, faults, X):
        return BaseClassifier.vote_predict_by_fault_(self, faults, X)    
               
    def predict_instance(self, x):
        _F = ['F']
        #3P
        if x['r1'] > 50:
            if x['change_neg_seq_neg_peak_a_change'] > 100 or x['change_zero_seq_neg_peak_a_change'] > 100:
                _F.append('ABC')

        #LtL
        LLVotes = []
        per_phase_sags = [np.abs(x['adiff']), np.abs(x['bdiff']), np.abs(x['cdiff'])]
        sper_phase_sags = sorted(per_phase_sags[:])
        if x['zdiff'] < 0.004: LLVotes.append('R1.1')
        else:
            if sper_phase_sags[1] > .1:
                if np.abs(x['zdiff']) < np.abs(x['ndiff']):
                    LLVotes.append('R1.2')
        if x['change_neg_seq_neg_peak_a_change'] > 100 or x['change_zero_seq_neg_peak_a_change'] > 100:
            LLVotes.append('R2')
        if x['r1'] < 50:
            LLVotes.append('R3')
        if np.abs(x['pdiff']/x['ndiff']) < 3.0:
            LLVotes.append('R4')
        if len(LLVotes) == 4:
            if sper_phase_sags[1] > .1:
                _fault = ""
                if sper_phase_sags[2] == per_phase_sags[0]:
                    _fault = "A"
                elif sper_phase_sags[2] == per_phase_sags[1]:
                    _fault = "B"
                elif sper_phase_sags[2] == per_phase_sags[2]:
                    _fault = "C"
                if sper_phase_sags[1] == per_phase_sags[0]:
                    _fault = _fault + "A"
                elif sper_phase_sags[1] == per_phase_sags[1]:
                    _fault = _fault + "B"
                elif sper_phase_sags[1] == per_phase_sags[2]:
                    _fault = _fault + "C"
                _F.append(_fault)

        # XL SLG Fault:
        SLGVotes = []
        if np.abs(x['zdiff']) > np.abs(x['ndiff']): 
            SLGVotes.append('R1.1')
        else:
            if sper_phase_sags[2] > .1 and sper_phase_sags[1] < .1:
                if x['nm_nmdiff']/x['zm_nmdiff'] < 4:
                    SLGVotes.append('R1.2')
        if x['change_neg_seq_neg_peak_a_change'] > 100 or x['change_zero_seq_neg_peak_a_change'] > 100:
            SLGVotes.append('R2')
        if x['r1'] < 50:
            SLGVotes.append('R3')
        if np.abs(x['pdiff']/x['ndiff']) < 3.0:
            SLGVotes.append('R4')
        if len(SLGVotes) == 4:
            if sper_phase_sags[2] > .1 and sper_phase_sags[1] < .1:
                if sper_phase_sags[2] == per_phase_sags[0]:
                    _F.append('AG')
                elif sper_phase_sags[2] == per_phase_sags[1]:
                    _F.append('BG')
                elif sper_phase_sags[2] == per_phase_sags[2]:
                    _F.append('CG')

        # Look at the classification results -- if we've added a classification 
        # to the default value 'F', then we can remove the 'F'... Since the rules
        # are not mutually exclusive, we may have more than one classification
        if len(_F) > 1 and "F" in _F: _F.remove('F')
        return self.translate_(_F[0])

class XLRevisedRuleBasedClassifier(XLRuleBasedClassifier):
    
    def predict_instance(self, x):
        _F = ['F']
        #3P
        if x['r1'] > 50:
            if x['change_neg_seq_max_a_change'] > 100 or x['change_zero_seq_max_a_change'] > 100:
                _F.append('ABC')

        #LtL
        LLVotes = []
        per_phase_sags = [np.abs(x['adiff']), np.abs(x['bdiff']), np.abs(x['cdiff'])]
        sper_phase_sags = sorted(per_phase_sags[:])
        if x['zdiff'] < 0.004: LLVotes.append('R1.1')
        else:
            if sper_phase_sags[1] > .1:
                if np.abs(x['zdiff']) < np.abs(x['ndiff']):
                    LLVotes.append('R1.2')
        if x['change_neg_seq_max_a_change'] > 100 or x['change_zero_seq_max_a_change'] > 100:
            LLVotes.append('R2')
        if x['r1'] < 50:
            LLVotes.append('R3')
        if np.abs(x['pdiff']/x['ndiff']) < 3.0:
            LLVotes.append('R4')
        if (len(LLVotes) == 4) or ('R1.1' in LLVotes):
            if sper_phase_sags[1] > .1:
                _fault = ""
                if sper_phase_sags[2] == per_phase_sags[0]:
                    _fault = "A"
                elif sper_phase_sags[2] == per_phase_sags[1]:
                    _fault = "B"
                elif sper_phase_sags[2] == per_phase_sags[2]:
                    _fault = "C"
                if sper_phase_sags[1] == per_phase_sags[0]:
                    _fault = _fault + "A"
                elif sper_phase_sags[1] == per_phase_sags[1]:
                    _fault = _fault + "B"
                elif sper_phase_sags[1] == per_phase_sags[2]:
                    _fault = _fault + "C"
                _F.append(_fault)

        # XL SLG Fault:
        SLGVotes = []
        if np.abs(x['zdiff']) > np.abs(x['ndiff']): 
            SLGVotes.append('R1.1')
        else:
            if sper_phase_sags[2] > .1 and sper_phase_sags[1] < .1:
                if x['nm_nmdiff']/x['zm_nmdiff'] < 4:
                    SLGVotes.append('R1.2')
        if x['change_neg_seq_max_a_change'] > 100 or x['change_zero_seq_max_a_change'] > 100:
            SLGVotes.append('R2')
        if x['r1'] < 50:
            SLGVotes.append('R3')
        if np.abs(x['pdiff']/x['ndiff']) < 3.0:
            SLGVotes.append('R4')
        if (len(SLGVotes) == 4) or ('R1.1' in SLGVotes):
            if sper_phase_sags[2] > .1 and sper_phase_sags[1] < .1:
                if sper_phase_sags[2] == per_phase_sags[0]:
                    _F.append('AG')
                elif sper_phase_sags[2] == per_phase_sags[1]:
                    _F.append('BG')
                elif sper_phase_sags[2] == per_phase_sags[2]:
                    _F.append('CG')

        # Look at the classification results -- if we've added a classification 
        # to the default value 'F', then we can remove the 'F'... Since the rules
        # are not mutually exclusive, we may have more than one classification
        if len(_F) > 1 and "F" in _F: _F.remove('F')
        return self.translate_(_F[0])
    
# please do not change since it for stage gate
class EventClassifier(BaseClassifier):
    @classmethod
    def LoadPickle(cls, pickle_fn):
        return common.loadpickle(pickle_fn)
    
    def __init__(self):
        super(EventClassifier, self).__init__()
        self.nofaultclassifier = SVC(kernel='poly', degree=8, C = 1, coef0=1.5) # can be set to 8-10 with ss check  
        self.phaseclassifier = SVC(kernel='linear', C=0.1, coef0=1)    
        self.slgclassifier = SVC(kernel='poly', degree=3, C=0.1, coef0=1)
        self.faultfeatures = 'ID v_sag_1 a_change_1 nofaultvsfault'
        self.phasefeatures = 'ID rel_phase_3_1 rel_phase_3_2 phasevsnophase'
        self.slgfeatures = 'ID rel_phase_2_1 sub_phase_2_1 v_sag_2 groundvsnoground'
        pass
    
    def features(self):
        return self.faultfeatures.split()[1:-1] + self.phasefeatures.split()[1:-1] + self.slgfeatures.split()[1:-1]
    
    def fit(self, interpretor):
        X, Y = interpretor.select(self.faultfeatures)
        self.nofaultclassifier = self.nofaultclassifier.fit(X[:,1:], Y)
        ids = [x[0]for x, y, y_p in zip(X, Y, self.nofaultclassifier.predict(X[:,1:])) if y == y_p == 1.0]

        X, Y = interpretor.clone().id(ids).select(self.phasefeatures)
        self.phaseclassifier = self.phaseclassifier.fit(X[:,1:], Y)
        ids = [x[0]for x, y, y_p in zip(X, Y, self.phaseclassifier.predict(X[:,1:])) if y == y_p == 1.0]
        
        X, Y = interpretor.clone().id(ids).select(self.slgfeatures)
        self.slgclassifier = self.slgclassifier.fit(X[:,1:], Y)
        
    def get_decision_(self, y_fault, y_phase, y_slg):
        if y_fault[0] == 0: return 3.0
        if y_phase[0] == 0: return 2.0
        return y_slg[0] 
       
    def predict_instance(self, X):
        y_f_pred = self.nofaultclassifier.predict([X[key] for key in self.faultfeatures.split()[1:-1]])
        y_p_pred = self.phaseclassifier.predict([X[key] for key in self.phasefeatures.split()[1:-1]])
        y_s_pred = self.slgclassifier.predict([X[key] for key in self.slgfeatures.split()[1:-1]])
        return self.get_decision_(y_f_pred, y_p_pred, y_s_pred)
    
    def __str__(self):
        s = []
        
        s.append('EventClassifier')
        
        s.append('classifier.nofaultclassifier: %s' % classifier.nofaultclassifier)
        s.append('\t%s' % classifier.faultfeatures)
        
        s.append('classifier.phaseclassifier: %s' % classifier.phaseclassifier)
        s.append('\t%s' % classifier.phasefeatures)
        
        s.append('classifier.slgclassifier: %s' % classifier.slgclassifier)
        s.append('\t%s' % classifier.slgfeatures)        
    
        return '\n'.join(s)

class ProxyEventClassifier(BaseClassifier):
    def __init__(self, cls, features):
        super(ProxyEventClassifier, self).__init__()
        self.cls = cls
        self._f = features
    
    def features(self): return self._f
    
    def fit(self, interpretor): 
        X, y = interpretor.select(self._f.replace('ID ',''))
        self.cls = self.cls.fit(X, y)
    
    def predict_instance(self, X):
        return self.cls.predict([X[key] for key in self._f.split()[1:-1]])[0]
    
    def decision_function_instance(self, X):
        return self.cls.decision_function([X[key] for key in self._f.split()[1:-1]])[0]    
    
    def __str__(self):
        s = []
        
        s.append('ProxyEventClassifier')
        s.append('classifier: %s' % self.cls)
        s.append('\t%s' % self._f)

        return '\n'.join(s)        
    
class FilteredEventClassifier(BaseClassifier):  
    def __init__(self, cls, signal_ids):
        super(FilteredEventClassifier, self).__init__()
        self.signal_ids = signal_ids
        self.cls = cls
        pass
    
    def features(self): return self.cls.features

    def fit(self, interpretor):
        self.cls.fit(interpretor.signal(self.signal_ids))
        
    def predict_instance(self, X):
        return self.cls.predict_instance(X)
    
    def __str__(self):
        s = []
        s.append('FilteredEventClassifier')
        s.append(self.cls)
        return '\n'.join(s)
    
class EventClassifier_20150827(EventClassifier):
    """
    Classifier is using 'arff/sag_mode_0_98_series datasets'
    """
    def __init__(self):
        super(EventClassifier_20150827, self).__init__()
        self.nofaultclassifier = SVC(kernel='poly', degree=7.5, C = 1., coef0=1.) # can be set to 8-10 with ss check  
        self.phaseclassifier = SVC(kernel='poly', degree=2, C = 1., coef0=1.)    
        self.slgclassifier = SVC(kernel='poly', degree=1.5, C = 1., coef0=1.)
        self.faultfeatures = 'ID v_sag_1 a_change_1 nofaultvsfault'
        self.phasefeatures = 'ID rel_phase_3_1 rel_phase_3_2 phasevsnophase'
        self.slgfeatures = 'ID a_change_2 rel_phase_2_1 groundvsnoground'
    
# please do not change since it for stage gate
class SeriesEventClassifier(EventClassifier):
    """
    Classifier is using 'arff/sag_mode_0_98_series datasets'
    """
    def __init__(self):
        super(SeriesEventClassifier, self).__init__()
        self.nofaultclassifier = SVC(kernel='poly', degree=14, C=1, coef0=1.5)
        self.faultfeatures = 'ID v_sag_1 avg_sag_1 nofaultvsfault' 
        
        self.slgclassifier = SVC(kernel='poly', degree=19, C=1, coef0=1.2)
        self.slgfeatures = 'ID rel_phase_2_1 avg_sag_2 groundvsnoground'
        pass
    
class SeriesEventClassifier1(EventClassifier):
    """
    Classifier is using 'arff/sag_mode_0_98_series datasets'
    """
    def __init__(self):
        super(SeriesEventClassifier1, self).__init__()
        self.nofaultclassifier = SVC(kernel='poly', degree=16, C=1, coef0=1.5)
        self.faultfeatures = 'ID avg_sag_1 delta_sag_1 nofaultvsfault' 
        
        self.slgclassifier = SVC(kernel='poly', degree=19, C=1, coef0=1.2)
        self.slgfeatures = 'ID rel_phase_2_1 avg_sag_2 groundvsnoground'
        pass