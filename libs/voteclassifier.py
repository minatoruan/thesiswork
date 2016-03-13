import numpy as np

class BaseClassifier:
    def get_generic_class(self, classfication):
        if (classfication in ['AG', 'BG', 'CG']): return 'SLG'
        if (classfication in ['AB', 'BC', 'AC']): return 'LtL'
        if (classfication in ['ABC', '3P']): return '3P'
        return 'noFault'

class XLOriginalClassifier(BaseClassifier):
    def is_slg(self, sags):
        if sags[0] <= 0.95 and if sags[1] >= .93 and sags[2] >= .93: return True
        return False

    def is_ll(self, sags):
        if sags[0] >= 0.35 and sags[1] >= 0.35:
            if sags[0] <= 0.93 and sags[1] <= 0.93:
                if sags[2] > 0.88: return True
        return False

    def is_tp(self, sags):
        if np.all([s >= 0 for s in sags]):
            if np.all([s <= .8 for s in sags]): 
                if sags.std() < 0.05 * sags.mean():
                    return True
        return False
    
    def predict(self, X):
        result = []
        for v1, v2, v3 in X:
            sags = np.array(sorted([v1, v2, v3])) # ascending order
            if self.is_slg(sags): result.append('SLG')
            elif self.is_ll(sags): result.append('LtL')
            elif self.is_tp(sags): result.append('3P')
            else: result.append('noFault')
        return result

class XLRuleBasedClassifier:   
    def generic_predict(self, X):
        return [BaseClassifier.get_generic_class(_c) for _c in self.predict(X)]
            
    def predict(self, X):
        result = []
        for x in X:
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
            if len(_F) > 1 and "F" in _F:
                _F.remove('F')
            _F = ','.join(_F)
            result.append(_F)
        return result