from subprocess import Popen, PIPE
from sklearn.datasets import load_svmlight_file
from collections import namedtuple
from operator import itemgetter, attrgetter
from bpati import signals
from numpy import array, isnan, isinf, nan, append, delete, mean, min, average
from libs import feature, common
from scipy import stats
import random
import numpy as np

Field = namedtuple('Field', 'description expr')
weka_java = 'libs/weka.jar'

defaultfields = [
            Field('faultId numeric', lambda event: event.eventid[0]),
            Field('subfaultId numeric', lambda event: event.eventid[1]),
            Field('isfaultedbus numeric', lambda event: 1 if event.isfaultedbus else 0),
            Field('signal numeric', lambda event: signals.SIGNALS_2013.index(event.signal)),
            Field('v_sag_1 numeric', lambda event: event.get_vsags()[0]),
            Field('v_sag_2 numeric', lambda event: event.get_vsags()[1]),
            Field('v_sag_3 numeric', lambda event: event.get_vsags()[2]),
            Field('avg_sag_1 numeric', lambda event: event.get_avg_sags()[0]),
            Field('avg_sag_2 numeric', lambda event: event.get_avg_sags()[1]),
            Field('avg_sag_3 numeric', lambda event: event.get_avg_sags()[2]),
            Field('u_delta_sag_1 numeric', lambda event: event.get_u_delta_sags()[0]),
            Field('u_delta_sag_2 numeric', lambda event: event.get_u_delta_sags()[1]),
            Field('u_delta_sag_3 numeric', lambda event: event.get_u_delta_sags()[2]),
            Field('delta_sag_1 numeric', lambda event: event.get_delta_sags()[0]),
            Field('delta_sag_2 numeric', lambda event: event.get_delta_sags()[1]),
            Field('delta_sag_3 numeric', lambda event: event.get_delta_sags()[2]),
            Field('delta_v numeric', lambda event: event.dv()),
            Field('a_change_1 numeric', lambda event: event.get_a_changes()[0]),
            Field('a_change_2 numeric', lambda event: event.get_a_changes()[1]),
            Field('a_change_3 numeric', lambda event: event.get_a_changes()[2]),
            Field('sub_a_change_2_1 numeric', lambda event: event.get_sub_a_change_2_1()),
            Field('sub_a_change_3_1 numeric', lambda event: event.get_sub_a_change_3_1()),
            Field('sub_a_change_3_2 numeric', lambda event: event.get_sub_a_change_3_2()),
            Field('rel_a_change_2_1 numeric', lambda event: event.get_relative_a_change_2_1()),
            Field('rel_a_change_3_1 numeric', lambda event: event.get_relative_a_change_3_1()),
            Field('rel_a_change_3_2 numeric', lambda event: event.get_relative_a_change_3_2()),
            Field('rel_phase_2_1 numeric', lambda event: event.get_relative_phase_2_1()),
            Field('rel_phase_3_1 numeric', lambda event: event.get_relative_phase_3_1()),
            Field('rel_phase_3_2 numeric', lambda event: event.get_relative_phase_3_2()),
            Field('rel_sagged_1_3 numeric', lambda event: event.get_relative_sagged_phase_1_3()),
            Field('rel_sagged_1_2 numeric', lambda event: event.get_relative_sagged_phase_1_2()),
            Field('rel_sagged_2_3 numeric', lambda event: event.get_relative_sagged_phase_2_3()),
            Field('sub_phase_2_1 numeric', lambda event: event.get_sub_phase_2_1()),
            Field('sub_phase_3_1 numeric', lambda event: event.get_sub_phase_3_1()),
            Field('sub_phase_3_2 numeric', lambda event: event.get_sub_phase_3_2()),
            Field('rel_sub_phase_32_21 numeric', lambda event: event.get_sub_phase_3_2()/event.get_sub_phase_2_1()),
            Field('sub_phase_32_21 numeric', lambda event: event.get_sub_phase_3_2()-event.get_sub_phase_2_1()),
            Field('sag_neg_seq_1 numeric', lambda event: event.get_neg_seq()[0]/mean(event.ss_mags)),
            Field('sag_neg_seq_angle_1 numeric', lambda event: event.get_neg_seq()[1]/120),
            Field('sag_pos_seq_1 numeric', lambda event: event.get_pos_seq()[0]/mean(event.ss_mags)),
            Field('sag_pos_seq_angle_1 numeric', lambda event: event.get_pos_seq()[1]/120),
            Field('sag_zero_seq_1 numeric', lambda event: event.get_zero_seq()[0]/mean(event.ss_mags)),
            Field('sub_phase_21_32 numeric', lambda event: event.get_relative_phase_2_1()-event.get_relative_phase_3_2()),
            Field('ed numeric', lambda event: event.electrical_distance),
        ]

defaultneighbourfields = [
            Field('v_sag_n_most numeric', lambda event: min([neighbour.get_vsags()[0] for neighbour in event.neighbours])),
            Field('v_sag_n_median numeric', lambda event: mean([neighbour.get_vsags()[0] for neighbour in event.neighbours])),
        ]

defaultlabels = [            
            Field('nofaultvsfault {noFault, fault}', lambda event: 'noFault' if event.get_sw_annotation() == 'noFault' else 'fault'),
            Field('groundvsnoground {SLG, noSLG}', lambda event: 'SLG' if event.get_sw_annotation() == 'SLG' else 'noSLG'),
            Field('linevsnoline {LtL, noLtL}', lambda event: 'LtL' if event.get_sw_annotation() == 'LtL' else 'noLtL'),
            Field('phasevsnophase {3P, no3P}', lambda event: '3P' if event.get_sw_annotation() == '3P' else 'no3P'),
            Field('xl_annotation {SLG, LtL, 3P, noFault}', lambda event: event.get_xl_annotation()),
            Field('sw_annotation {SLG, LtL, 3P, noFault}', lambda event: event.get_sw_annotation())
        ]

def execute(cmd):
    p = Popen(cmd, shell = True, stdout=PIPE)
    (output, err) = p.communicate()
    return p.wait(), output

def j48fit(input_arff, settings, verbose = False):
    command = 'java -Xmx1000M -classpath libs/weka.jar weka.classifiers.trees.J48 -t ' + input_arff + ' ' + settings
    if verbose: print command
    return execute(command)

def buildArff(fn, dsname, fields, generators):
    with open(fn, 'w') as writer:
        writer.write('@relation '+dsname+'\n\n')
        writer.write('@attribute ID numeric\n')
        for field in fields:
            writer.write('@attribute ' + field.description + '\n')
        writer.write('\n@data\n')
        index = 1
        for _generator in generators:
            for event in _generator:
                arr = [str(index)]
                for field in fields:
                    v = field.expr(event)
                    if isinstance(v, float): 
                        v = round(v, 4)
                        if isnan(v):
                            print 'Existing nan', event.eventid, event.signal, field.description
                            break
                        if isinf(v): 
                            print 'Existing inf', event.eventid, event.signal, field.description
                            break
                    arr.append(str(v))
                if len(fields) > len(arr): continue
                writer.write(', '.join(arr)+'\n')
                index += 1            

class ExportingEvent(object):
    def __init__(self, signal, phases, angles, datapoint, signaldata):
        self.error = any([_v == 0 for s in phases for _v in datapoint[s]])
        if self.error: pass

        self.signal = signal
        self.phases = phases
        self.angles = angles
        self.datapoint = datapoint
        
        self.init_ss_mags_(phases, signaldata)
        self.init_ss_angle_(angles, signaldata)
           
    #init steady state voltage mags A B C
    def init_ss_mags_(self, phases, signaldata):
        self.ss_mags = array([mean(signaldata[phases[0]]),\
                                  mean(signaldata[phases[1]]),\
                                  mean(signaldata[phases[2]])\
                                  ])  
    #init steady state of AOB, AOC and BOC
    def init_ss_angle_(self, angles, signaldata):
        angle_ab = mean([feature.relativeAngle(x, y) for x, y in zip(signaldata[angles[0]], signaldata[angles[1]])])
        angle_ac = mean([feature.relativeAngle(x, y) for x, y in zip(signaldata[angles[0]], signaldata[angles[2]])])
        angle_bc = mean([feature.relativeAngle(x, y) for x, y in zip(signaldata[angles[1]], signaldata[angles[2]])])
        self.ss_angles = array([angle_ab, angle_ac, angle_bc])
    
    def add_neighbour(self, event):
        if not hasattr(self, 'neighbours'): self.neighbours = []
        self.neighbours.append(event)
        
    def get_vmags_(self, idx = -1):
        return array([self.datapoint[self.phases[0]][idx], 
                          self.datapoint[self.phases[1]][idx], 
                          self.datapoint[self.phases[2]][idx]])         
    
    def get_angles_(self, idx = -1):
        return array([self.datapoint[self.angles[0]][idx], 
                          self.datapoint[self.angles[1]][idx], 
                          self.datapoint[self.angles[2]][idx]]) 
    
    def get_phase_angles_(self, idx = -1):
        angles = self.get_angles_(idx = idx)
        return array([feature.relativeAngle(angles[0], angles[1]),
                       feature.relativeAngle(angles[0], angles[2]),
                       feature.relativeAngle(angles[1], angles[2])])
    
    def dv(self, idx = -1):
        vmags = self.get_vmags_(idx=idx)
        return ((sum((1 - (vmags * 1.0 / self.ss_mags)) ** 2))**0.5)/3.0
    
    def get_dipped_phases_(self, idx = -1):
        vmags = self.get_vmags_(idx = idx)
        dipped_phases = abs(1 - vmags * 1.0 / self.ss_mags)
        return sorted([max(x, 0.0001) for x in dipped_phases], reverse = True)
        
    def get_normalized_a_changes(self, idx = -1):
        phase_angles = self.get_phase_angles_(idx = idx)
        dipped_a_change = abs(1 - phase_angles * 1.0 / self.ss_angles)
        return sorted([max(x, 0.0001) for x in dipped_a_change], reverse = True)
        
    def get_a_changes(self, idx = -1):
        phase_angles = self.get_phase_angles_(idx = idx)
        return sorted(phase_angles * 1.0 / self.ss_angles)
        
    def get_vsags(self, idx = -1):
        vmags = self.get_vmags_(idx = idx)
        return sorted(vmags * 1.0 / self.ss_mags)
    
    def get_sub_a_change_2_1(self, idx = -1):
        phase_angles = self.get_phase_angles_(idx = idx)
        achanges = sorted(phase_angles * 1.0 / self.ss_angles)
        return achanges[1] - achanges[0]
    
    def get_sub_a_change_3_1(self, idx = -1):
        phase_angles = self.get_phase_angles_(idx = idx)
        achanges = sorted(phase_angles * 1.0 / self.ss_angles)
        return achanges[2] - achanges[0]
    
    def get_sub_a_change_3_2(self, idx = -1):
        phase_angles = self.get_phase_angles_(idx = idx)
        achanges = sorted(phase_angles * 1.0 / self.ss_angles)
        return achanges[2] - achanges[1]
    
    def get_relative_a_change_2_1(self, idx = -1):
        achanges = self.get_normalized_a_changes(idx = idx)
        return achanges[1] / achanges[0]
    
    def get_relative_a_change_3_1(self, idx = -1):
        achanges = self.get_normalized_a_changes(idx = idx)
        return achanges[2] / achanges[0]
    
    def get_relative_a_change_3_2(self, idx = -1):
        achanges = self.get_normalized_a_changes(idx = idx)
        return achanges[2] / achanges[1]    
      
    def get_relative_sagged_phase_1_3(self, idx = -1):
        v_sags = self.get_vsags(idx = idx)
        return v_sags[0] / v_sags[2]
    
    def get_relative_sagged_phase_1_2(self, idx = -1):
        v_sags = self.get_vsags(idx = idx)
        return v_sags[0] / v_sags[1]
    
    def get_relative_sagged_phase_2_3(self, idx = -1):
        v_sags = self.get_vsags(idx = idx)
        return v_sags[1] / v_sags[2]
    
    def get_relative_phase_2_1(self, idx = -1):
        dipped_phase = self.get_dipped_phases_(idx = idx)
        return dipped_phase[1]/dipped_phase[0]    
    
    def get_relative_phase_3_1(self, idx = -1):
        dipped_phase = self.get_dipped_phases_(idx = idx)
        return dipped_phase[2]/dipped_phase[0]
    
    def get_relative_phase_3_2(self, idx = -1):
        dipped_phase = self.get_dipped_phases_(idx = idx)
        return dipped_phase[2]/dipped_phase[1]
    
    def get_sub_phase_2_1(self, idx = -1):
        v_sags = self.get_vsags(idx = idx)
        return v_sags[1] - v_sags[0]
    
    def get_sub_phase_3_1(self, idx = -1):
        v_sags = self.get_vsags(idx = idx)
        return v_sags[2] - v_sags[0]    
    
    def get_sub_phase_3_2(self, idx = -1):
        v_sags = self.get_vsags(idx = idx)
        return v_sags[2] - v_sags[1]
        
    def get_neg_seq(self, idx = -1):
        angles = self.get_angles_(idx = idx)
        vmags = self.get_vmags_(idx = idx) 
        return feature.neg_seq(vmags[0], angles[0], vmags[1], angles[1], vmags[2], angles[2])
    
    def get_pos_seq(self, idx = -1):
        angles = self.get_angles_(idx = idx)
        vmags = self.get_vmags_(idx = idx) 
        return feature.pos_seq(vmags[0], angles[0], vmags[1], angles[1], vmags[2], angles[2])    
    
    def get_zero_seq(self, idx = -1):
        angles = self.get_angles_(idx = idx)
        vmags = self.get_vmags_(idx = idx) 
        return feature.zero_seq(vmags[0], angles[0], vmags[1], angles[1], vmags[2], angles[2])
    
    #avg sag last three datapoint
    def get_avg_sags(self):
        v_sag_a = average([i / self.ss_mags[0] for i in self.datapoint[self.phases[0]][-3:]])
        v_sag_b = average([i / self.ss_mags[1] for i in self.datapoint[self.phases[1]][-3:]])
        v_sag_c = average([i / self.ss_mags[2] for i in self.datapoint[self.phases[2]][-3:]])
        return sorted([v_sag_a, v_sag_b, v_sag_c])
    
    def get_splot(self):
        stats.linregress(df[phase][ss_end-4:ss_end+1],df[phase][idx-4:idx+1],)[0]
        
        
    def get_u_delta_sags(self):
        vmags = self.get_vmags_(idx = -1)/self.ss_mags
        prior_vmags = self.get_vmags_(idx = -2)/self.ss_mags
        return sorted(abs(vmags-prior_vmags))
    
    def get_delta_sags(self):
        vmags = self.get_vmags_(idx = -1)/self.ss_mags
        prior_vmags = self.get_vmags_(idx = -2)/self.ss_mags
        return sorted(vmags-prior_vmags)

#angle in 2014 represents in radians
class RadExportingEvent(ExportingEvent): 
    def __init__(self, signal, phases, angles, datapoint, signaldata):
        #copy datapoint and signal data
        #convert rad 2 deg
        
        coverted_datapoint = {k: array([k in angles and np.rad2deg(v) or v * np.sqrt(3) for v in datapoint[k]]) for k in datapoint}
        coverted_signaldata = {k: array([k in angles and np.rad2deg(v) or v * np.sqrt(3) for v in signaldata[k]]) for k in signaldata}
        
        super(RadExportingEvent, self).__init__(signal, phases, angles, coverted_datapoint, coverted_signaldata)  
    
class MetaExportingEvent(ExportingEvent):
    def __init__(self, eventid, signal, phases, angles, datapoint, signaldata, electrical_distance, isfaultedbus, sw_annotation, xl_annotation):
        super(MetaExportingEvent, self).__init__(signal, phases, angles, datapoint, signaldata)
        self.eventid = eventid
        self.electrical_distance = electrical_distance
        self.isfaultedbus = isfaultedbus
        self.sw_annotation = sw_annotation
        self.xl_annotation = xl_annotation
        
    def get_sw_annotation(self):
        return common.get_generic_class(self.sw_annotation)
    
    def get_xl_annotation(self):
        return common.get_generic_class(self.xl_annotation)
    
class Interpreter(object):
    def __init__(self, fn, meta, data, fieldparser):
        self.fn = fn
        self.meta = meta
        self.data = data
        self.predicate = []
        self.orderfield = ''
        self.isdesc = False
        self.fieldparser = fieldparser
        self.top = -1

    def __iter__(self):
        for data in self.__select__(None):
            yield data

    def where(self, depricate):
        self.predicate.append(self.fieldparser.lambdaparse(depricate))
        return self
    
    def signal(self, signalIds):
        exp = 'signal in [%s]' % ','.join([str(x) for x in signalIds])
        return self.where(exp)
    
    def fid(self, ids):
        exp = '( faultId , subfaultId ) in [%s]' % ','.join([str(x) for x in ids])
        return self.where(exp)
    
    def notfid(self, ids):
        exp = '( faultId , subfaultId ) not in [%s]' % ','.join([str(x) for x in ids])
        return self.where(exp)
    
    def id(self, ids):
        exp = 'ID in [%s]' % ','.join([str(x) for x in ids])
        return self.where(exp)

    def notid(self, ids):
        exp = 'ID not in [%s]' % ','.join([str(x) for x in ids])
        return self.where(exp)

    def sort(self, fieldname, reverse = False):
        self.orderfield = fieldname
        self.isdesc = reverse
        return self

    def max(self, fieldname):
        result = self.__select__(fieldname)
        return max(self.__select__(fieldname)) if len(result) > 0 else nan

    def min(self, fieldname):
        result = self.__select__(fieldname)
        return min(self.__select__(fieldname)) if len(result) > 0 else nan

    def fieldDesc(self):
        arr = []
        arrfieldnames = self.__selectfieldname__(None)
        arrfields = self.fieldparser.translatefield(arrfieldnames)
        for attribute in self.fieldparser.getarffattributes(arrfieldnames):
            arr.append(attribute)
        return arr

    def __len__(self):
        result = self.get('ID')[:,0]
        return len(result)

    def __selectfieldname__(self, fieldnames):
        if fieldnames == None: return self.fieldparser.fieldnames
        return fieldnames.split()

    def __select__(self, fieldnames):
        X = []
        predicate = [eval(p) for p in self.predicate]
        for e in self.data:
            if len(predicate) != 0 and any([p(e)==False for p in predicate]): continue
            X.append(e)

        if len(X) == 0: return []
        if self.orderfield != '':  
            X = sorted(X, key = itemgetter(self.fieldparser.translatefield([self.orderfield])[0]), reverse = self.isdesc)

        X = array(X)

        if fieldnames == None: return X

        arrfieldnames = fieldnames.split()
        arrfields = self.fieldparser.translatefield(arrfieldnames)
        return X[:, arrfields]

    def kfold(self, numFold, shuffle = False, random = 0):
        return KFold(self, numFold, shuffle, random)

    def split(self, testsize=0.2, numFold=10, shuffle = False):
        return Splitter(self, testsize, numFold, shuffle)
    
    def build(self):
        return DatasetBuilder(self)

    def select(self, fieldnames=None):
        result = self.__select__(fieldnames)
        return result[:,:-1], result[:,-1] if len(result) > 0 else []
    
    def todict(self, fieldnames=None):
        result = self.__select__(fieldnames)
        if len(result) == 0: return []
        fieldnames = fieldnames.split() if fieldnames != None else self.fieldparser.fieldnames
        fieldnames = fieldnames[:-1]
        arr = []
        for sample in result:
            arr.append({field:sample[idx] for idx, field in enumerate(fieldnames)})
        return arr, result[:,-1]
        
    def get(self, fieldnames=None):
        return self.__select__(fieldnames)

    def save(self, fn, fieldnames=None):
        arrfieldnames = self.__selectfieldname__(fieldnames)
        arrfields = self.fieldparser.translatefield(arrfieldnames)
        result = self.__select__(fieldnames)
        with open(fn, 'w') as writer:
            writer.write('@relation ' + self.meta + '\n\n')
            for attribute in self.fieldparser.getarffattributes(arrfieldnames):
                writer.write('@attribute %s\n' % attribute)
            writer.write('@data\n\n');
            for e in result:
                writer.write(', '.join([self.fieldparser.getarffvalue(e[i], index) for i, index in enumerate(arrfields)]) + '\n')

    def clone(self):
        interpreter = Interpreter(self.fn, self.meta, self.data, self.fieldparser)
        interpreter.predicate = list(self.predicate)
        interpreter.orderfield = self.orderfield
        interpreter.isdesc = self.isdesc
        return interpreter            

class FieldParser(object):
    def __init__(self, fieldnames, datatypes):
        self.datatypes = datatypes
        self.fieldnames = fieldnames

    def lambdaparse(self, expstr):
        arr = expstr.split()
        for index in range(len(arr)):
            if arr[index] in self.fieldnames:
                arr[index] = 'e[' + str(self.fieldnames.index(arr[index])) + ']'
        return 'lambda e: ' + ' '.join(arr)

    def translatefield(self, fieldnames):
        return [self.fieldnames.index(fieldname) for fieldname in fieldnames]

    def translatevalue(self, fieldname, value):
        index = self.fieldnames.index(fieldname)
        return self.datatypes[index][int(value)]

    def getValue(self, fieldname, rawValue):
        index = self.fieldnames.index(fieldname)
        return self.datatypes[index].index(rawValue)

    def getarffvalue(self, value, index):
        if self.datatypes[index] == 1:
            return str(value)
        else:
            return self.datatypes[index][int(value)]

    def getarffattributes(self, fieldnames):           
        arrfields = self.translatefield(fieldnames)

        for field, dtindex in zip(fieldnames, arrfields):
            if self.datatypes[dtindex] == 1:
                yield '%s numeric' % field
            else:
                yield '%s {%s}' % (field, ', '.join(self.datatypes[dtindex]))    

class DatasetBuilder(object):
    def __init__(self, interpreter):
        self.fielnames = []
        self.datatypes = []
        self.expr = []
        self.interpreter = interpreter

    def addfield(self, fielname, datatype, exp):
        self.fielnames.append(fielname)
        self.expr.append(exp)
        if datatype == 'numeric': self.datatypes.append(1)
        else: self.datatypes.append(datatype.split())
        return self

    def get(self):
        fieldparser = FieldParser(self.fielnames, self.datatypes)
        exp = [self.interpreter.fieldparser.lambdaparse(s) for s in self.expr]
        data = []
        for event in self.interpreter.__select__(None):
            data.append([round(eval(lam)(event), 4) for lam in exp])
        return Interpreter(self.interpreter.fn, self.interpreter.meta, array(data), fieldparser)
    
class Splitter():
    def __init__(self, interpretor, testsize=0.1, numFold=10, shuffle = False):
        self.has_test_set = testsize > 0
        self.interpretor = interpretor
        self.stat_(testsize, numFold, shuffle, random)
    
    def getinterpretor(self):
        return self.interpretor.clone()
        
    def stat_(self, testsize, numFold, shuffle, random):
        raw_ids = self.interpretor.get('ID sw_annotation')
        faultids = [[_id for _id, fault in raw_ids if fault == faulttype] for faulttype in range(4)]
        if shuffle: 
            for subfault in faultids: random.shuffle(subfault)
        
        #Pop testing data with size
        print 'total size of each classification', [int(len(subfaults)) for subfaults in faultids]
        
        if self.has_test_set:
            print 'pop of testing sample', [int(len(subfaults)*testsize + 0.4) for subfaults in faultids]
            self.testingids, faultids = self.pop_(faultids, [int(len(subfaults)*testsize + 0.4) for subfaults in faultids])
        else: self.testingids = []
        
        #Pop k-fold
        size_folds = [int(len(subfaults)/numFold*1.0 + 0.4) for subfaults in faultids]
        self.fold_data = []
        for idx in range(numFold-1):
            sample, faultids = self.pop_(faultids, size_folds)
            self.fold_data.append(sample)
        
        self.fold_data.append(reduce(lambda x, y: list(x) + list(y), faultids))
    
    def kfold_data(self):
        return self.interpretor.clone().id(reduce(lambda x, y: list(x) + list(y), list(self.fold_data)))

    def pop_(self, faultids, n_samples):
        samples = reduce(lambda x, y: x + y, [random.sample(subsample, n_sample) for subsample, n_sample in zip(faultids, n_samples)])
        faultids = [set(subsample) - set(samples) for subsample in faultids]
        return samples, faultids
    
    def get(self):
        arr = []
        for sample in self.fold_data:
            arr.append((self.interpretor.clone().id(sample), self.interpretor.clone().notid(sample).notid(self.testingids))) #testing , training                
        return arr
    
    def get_testing(self):
        if self.has_test_set:
            return self.interpretor.clone().id(self.testingids)
        return None 
        
class KFold(object):
    def __init__(self, interpretor, numFold, shuffle = False, random = 0):
        self.interpretor = interpretor
        self.numFold = numFold
        self.shuffle =  shuffle
        self.random = random
        pass

    def __fold__(self):
        raw_ids = self.interpretor.get('ID sw_annotation')
        id_faulttypes = {faulttype: [_id for _id, fault in raw_ids if fault == faulttype] for faulttype in range(3)}
        
        random.seed(self.random)
        
        if self.shuffle: 
            for faultype in id_faulttypes: random.shuffle(id_faulttypes[faultype])
        
        size_folds = {faultype: len(id_faulttypes[faultype])/self.numFold for faultype in id_faulttypes}
        id_faulttypes = {faultype: set(id_faulttypes[faultype]) for faultype in id_faulttypes}
        #fold
        self.fold_data = []
        for n in range(self.numFold - 1):
            sample = []
            for faultype in id_faulttypes:
                subsample = random.sample(id_faulttypes[faultype], size_folds[faultype])
                id_faulttypes[faultype] -= set(subsample)
                sample += subsample
            self.fold_data.append(sample)
        self.fold_data.append([item for sublist in id_faulttypes.values() for item in sublist])

    def getinterpretor(self):
        return self.interpretor.clone()

    def get(self):
        self.__fold__()
        arr = []
        for sample in self.fold_data:
            arr.append((self.interpretor.clone().id(sample), self.interpretor.clone().notid(sample))) #testing , training                
        return arr
                    
class Dataset(object):
    def __parsedata__(self, fn):
        metadata, fields, datatypes, data = '', [], [], []
        with open(fn, 'r') as fhandler:
            while (True):
                try:
                    line = fhandler.next()
                    if line.startswith('@relation'): 
                        metadata = line.replace('@relation', '').replace('\n', '')
                        continue
                    if line.startswith('@attribute'): 
                        arr = line.replace(',','').replace('{','').replace('}','').replace('\n','').split()
                        fields.append(arr[1])
                        if arr[2] != 'numeric': datatypes.append(arr[2:])
                        else: datatypes.append(1)
                        continue
                    if line == '@data\n':
                        continue
                    if line =='\n': continue
                    arr = line.replace(', ', ' ').replace('\n','').split(' ')
                    data.append([float(value) if datatypes[index]==1 else datatypes[index].index(value) for index, value in enumerate(arr)])
                except StopIteration:
                    break
        return metadata, fields, datatypes, array(data)
    
    def __init__(self, datafiles):
        self.datafiles = []
        for index, datafile in enumerate(datafiles):
            metadata, fields, datatypes, data = self.__parsedata__(datafile)
            self.datafiles.append([datafile, metadata, fields, datatypes, data])
    
    def __getitem__(self, index):
        fieldparser = FieldParser(self.datafiles[index][2], self.datafiles[index][3])
        return Interpreter(self.datafiles[index][0], self.datafiles[index][1], self.datafiles[index][4], fieldparser)