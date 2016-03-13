import numpy as np
import pandas as pd
import classifyfault_v1
import csv, re

from libs import common
from bpati import pmu
from bpati import signals

from os import listdir
from os.path import join, isfile

class DataFiles:
    # option indicate how pmu object is initiated
    # 0: local using minute20.h5
    # 1: 2013_zeta using minutes.h5
    # 2: 2014_qnap using 2014 access method
    def __init__(self, option):
        self.valid_faults = set(['AG', 'BG', 'CG', 'AB', 'BC', 'AC', 'ABC'])
        if option == 0:
            BPA_DATA = "pmu-data/"
            self.pmudata = pmu.PMUData(pmusearchpath=BPA_DATA,
                               pmudata=["minutes20.h5"],
                               )
            self.hdf5 = BPA_DATA+"minutes20.h5"
        elif option == 1:
            BPA_DATA = "../../../../BPA-PMU/data/"
            self.pmudata = pmu.PMUData(pmusearchpath=BPA_DATA+"hdf5/",
                               pmudata=["minutes.h5"],
                               eventdata=BPA_DATA+"events/precise-event-signatures.csv")
            self.hdf5 = BPA_DATA+"hdf5/minutes.h5"
        else:
            BPA_DATA14 = "/mnt/qnap/BPATI_2014"
            self.pmudata = pmu.PMUData(pmusearchpath=BPA_DATA14,
                               pmudata=['2014.cmap'],
                               configcachepath='new-configs',
                               configcachesuffix='-.ncfg')
        
        if option != 2:
            self.electrical_sites, self.electrical_distances = self.init_distance_matrix_('txt/electrical_dist_500kv_sites.csv')
            self.hop_sites, self.hop_distances = self.init_distance_matrix_('txt/site_matrix_distance.csv')
            self.filterhandler = lambda x: any([x.startswith(s) for s in signals.SITES_2013])
            self.sitetranslator = lambda x: [site for site in signals.SITES_2013 if x.startswith(site)][0]            
            self.get_distance = self.get_distance_
            self.get_electrical_distance = self.get_electrical_distance_
            self.get_hop_distance = self.get_hop_distance_
            self.get_faults_signature = self.get_faults_signature_
            self.get_sw_faults = self.get_sw_faults_
            self.get_xl_faults = self.get_xl_faults_
            self.get_normal_minutes = self.get_normal_minutes_
            self.get_smag = self.get_smag_
            self.get_signal_id_by = self.get_signal_id_by_
        else:
            self.get_faults_signature = self.get_faults_signature_2014_
            self.get_smag = self.get_smag_2014_
            
        pass
    
    def init_distance_matrix_(self, fn):
        distances = {}
        sites = []
        with open(fn, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
            for rowIndex, row in enumerate(csvreader):
                if rowIndex == 0:
                    sites = row[1:]
                    continue
                for colIndex in range(1,len(row)):
                    if sites[rowIndex-1] == sites[colIndex-1]: continue
                    distances[(sites[rowIndex-1], sites[colIndex-1])] = round(float(row[colIndex]), 8)
        return sites, distances
    
    #option0: electrical distance
    #option1: hop distance
    def get_distance_(self, signal1, signal2, option=0):
        if signal1 == signal2: return 0
        if self.sitetranslator(signal1) == self.sitetranslator(signal2): return 0
        
        if option == 0 and (self.sitetranslator(signal1) not in self.electrical_sites or \
                                self.sitetranslator(signal2) not in self.electrical_sites): return -1
        if option == 1 and (self.sitetranslator(signal1) not in self.hop_sites or \
                                self.sitetranslator(signal2) not in self.hop_sites): return -1
        
        if option == 0: return self.electrical_distances[self.sitetranslator(signal1), self.sitetranslator(signal2)]
        return self.hop_distances[self.sitetranslator(signal1), self.sitetranslator(signal2)]
        
    def get_electrical_distance_(self, signal1, signal2):
        return self.get_distance_(signal1, signal2, option=0)

    def get_hop_distance_(self, signal1, signal2):
        return self.get_distance_(signal1, signal2, option=1)

    def get_faults_signature_(self):
        faults = {}
        for faultdata in self.pmudata.events():
            faults[faultdata['id']] = [faultdata['xl_annotation'], faultdata['sw_annotation']]
        return faults
    
    def get_faults_signature_2014_(self, month, year):
        return pd.read_csv('/mnt/qnap/dataframes/%d/events%d%d.csv' % (month,year,month), 
                           converters = {'precise start time': common.date_time_converter,
                                         'precise end time': common.date_time_converter,
                                         'OutDatetime': common.date_time_utc_converter,
                                         'InDatetime': common.date_time_utc_converter}) 
    
    def get_sw_faults_(self):
        return {faultdata['id']: faultdata for faultdata in self.pmudata.events() if faultdata['sw_annotation'] in self.valid_faults}
    
    def get_xl_faults_(self):
        return {faultdata['id']:faultdata for faultdata in self.pmudata.events() if faultdata['xl_annotation'] in self.valid_faults}
    
    def get_normal_minutes_(self):
        return [x for x in self.pmudata.normalminutes().atoms()]
    
    def get_smag_(self):
        phased_signals,_ = signals.cluster_phase_signals(signals.SIGNALS_2013)
        phased_angles = {key: phased_signals[key] for key in phased_signals 
                             if len(phased_signals[key]) > 0 and key.endswith('Voltage Ang')}
        phased_mags = {key: phased_signals[key] for key in phased_signals 
                             if len(phased_signals[key]) > 0 and key.endswith('Voltage Mag')}
        return phased_mags.keys(), phased_mags, phased_angles
    
    def get_smag_2014_(self):
        columns = signals.SIGNALS_2014
        _PHASEANGRE = re.compile('\|V\|[ABC]\|ANG')
        _PHASEMAGRE = re.compile('\|V\|[ABC]\|MAG')
        _GENRE = re.compile('\|VP\|\|MAG')
        
        gen_signals = [x for x in columns if _GENRE.search(x)]
        phased_angles = {key: [column for column in columns if _GENRE.sub('', key) == _PHASEANGRE.sub('', column)] 
                                 for key in gen_signals }
        phased_mags = {key: [column for column in columns if _GENRE.sub('', key) == _PHASEMAGRE.sub('', column)] 
                                 for key in gen_signals }
        return gen_signals, phased_mags, phased_angles
    
    def get_smag_(self):
        phased_signals,_ = signals.cluster_phase_signals(signals.SIGNALS_2013)
        phased_angles = {key: phased_signals[key] for key in phased_signals 
                             if len(phased_signals[key]) > 0 and key.endswith('Voltage Ang')}
        phased_mags = {key: phased_signals[key] for key in phased_signals 
                             if len(phased_signals[key]) > 0 and key.endswith('Voltage Mag')}
        return phased_mags.keys(), phased_mags, phased_angles
    
    def get_signal_id_by_(self, depricate):
        return [idx for idx, signal in enumerate(signals.SIGNALS_2013) if depricate(signal)]

    def get_neigbour_signals_(self, signal, n_neigbour=2, predicator=None):
        neigbour_signals = []
        signals, _, _ = self.get_smag()
        neigbour_bus = []
        neigbour_sites = []
        
        for x in signals:
            if x == signal or self.sitetranslator(x) == self.sitetranslator(signal): continue
            if self.sitetranslator(x) in neigbour_sites: continue
            ed, hop = self.get_distance_(x, signal, 0), self.get_distance_(x, signal, 1)       
            if (not predicator(x, ed, hop)): continue
            neigbour_bus.append((hop, x))
            neigbour_sites.append(self.sitetranslator(x))
            if len(neigbour_bus) == n_neigbour: break
        
        return sorted(neigbour_bus)
    
    def repopulate_result(self, length, fault_cycles): 
        cycles = [0.0 for i in range(length)]
        for c, f in fault_cycles:
           cycles[c] = f
        return cycles
    
    def print_scan_detail(self, directory, minutes, prefix='nofault'):
        data = {}
        sites = []
        for filename in sorted(filter(lambda x: isfile(join(directory, x)) and x.startswith(prefix), listdir(directory))):
            #print 'Reading', filename
            subdata = common.loadpickle(join(directory, filename))
            idx = int(filename.split('_')[1])
            #print '-- Reading subdata starting at', idx,len(subdata)

            for minute in subdata:
                target_idx = minute+idx
                if target_idx not in data:
                    data[target_idx] = subdata[minute]
                    #print '-- Appending minute', minute, 'to', target_idx
                    continue

                for site in subdata[minute]: data[target_idx][site] = subdata[minute][site]
                #print '-- Insert minute', minute, 'to', target_idx

        print len(data)
        total_scan = 0
        total_scan_classification = 0
        for minute in data:
            #if minute not in minutes: continue
            print minute, minutes[minute]
            for site in data[minute]:
                if data[minute][site][0] is (None): continue
                n_classified_cycles, inclassified_cycles = data[minute][site]
                #total_scan += n_classified_cycles
                #total_scan_classification += n_classified_cycles
                if len(inclassified_cycles) > 0:
                    print '\t',site, 'number of inclassified cycles', len([fault_classification[0] for fault_classification in inclassified_cycles])
                    print '\t', [fault_classification[0] for fault_classification in inclassified_cycles]
                    sites.append(site)

        #print 'total cycles:', len(data) * 3600 * len(generics)
        #print 'total scanned cycles every sites', total_scan
        #print 'Number of corrected classifier', total_scan - total_scan_classification
        #print 'Number of incorrected classifier', total_scan_classification
        #print 'PMUs with no faults:', len(set(generics) - set(sites))
        #for site in set(generics) - set(sites):
        #    print '\t', site      
        
        return data
    
    #PAML papers 
    #data from common.get_fn_normal_minutes()[0]
    def get_training_and_testing_normal_data_key(self):
        arr = [(-22, 23), (-21, 20), (-21, 21), (-21, 22), (-21, 23), (-20, 30), (-20, 31), (-20, 32), (-20, 33), (-19, 10), (-19, 11), (-19, 12), (-19, 30), (-19, 31), (-19, 32), (-19, 33), (-18, 10), (-18, 11), (-18, 12), (-18, 13), (-17, 0), (-17, 1), (-17, 3), (-17, 10), (-17, 11), (-17, 12), (-17, 13), (-17, 30), (-17, 31), (-17, 32), (-17, 33), (-17, 50), (-17, 51), (-17, 52), (-17, 53), (-14, 30), (-14, 31), (-14, 32), (-14, 33), (-14, 50), (-14, 51), (-14, 52), (-14, 53), (-13, 10), (-13, 11), (-13, 12), (-13, 13), (-12, 20), (-12, 21), (-12, 22), (-12, 23), (-12, 50), (-12, 51), (-12, 52), (-12, 53), (-10, 10), (-10, 11), (-10, 12), (-10, 13), (-8, 30), (-8, 31), (-8, 32), (-8, 33), (-7, 30), (-7, 31), (-7, 32), (-7, 33), (-6, 30), (-6, 31), (-6, 32), (-6, 33), (-6, 50), (-6, 51), (-6, 52), (-6, 53), (-5, 31), (-5, 32), (-5, 33), (-2, 10), (-2, 11), (-2, 12), (-2, 13), (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-31, 0), (-31, 1), (-31, 2), (-31, 3), (-31, 30), (-31, 31), (-31, 32), (-31, 33), (-30, 0), (-30, 1), (-30, 2), (-30, 3), (-30, 40), (-30, 41), (-30, 42), (-30, 43), (-28, 10), (-28, 11), (-28, 12), (-28, 13), (-28, 40), (-28, 41), (-28, 43), (-27, 0), (-27, 1), (-27, 2), (-27, 3), (-27, 10), (-27, 11), (-27, 12), (-27, 13), (-26, 0), (-26, 1), (-26, 2), (-26, 3), (-24, 0), (-24, 1), (-24, 2), (-22, 10), (-22, 11), (-22, 12), (-22, 13), (-22, 50), (-22, 51), (-22, 52), (-22, 53), (-21, 10), (-21, 11), (-21, 12), (-21, 13), (-20, 0), (-20, 1), (-20, 2), (-20, 3), (-18, 0), (-18, 1), (-18, 2), (-18, 3), (-18, 30), (-18, 31), (-18, 32), (-18, 33), (-18, 50), (-18, 51), (-18, 52), (-18, 53), (-16, 0), (-16, 1), (-16, 2), (-16, 3), (-16, 40), (-16, 41), (-16, 42), (-16, 43), (-14, 0), (-14, 1), (-14, 3), (-14, 40), (-14, 41), (-14, 42), (-14, 43), (-13, 30), (-13, 31), (-13, 32), (-13, 33), (-11, 10), (-11, 11), (-11, 12), (-11, 13), (-11, 20), (-11, 21), (-11, 23), (-10, 40), (-10, 41), (-10, 42), (-10, 43), (-8, 40), (-8, 41), (-8, 42), (-8, 43), (-6, 0), (-6, 1), (-6, 2), (-6, 3), (-5, 10), (-5, 11), (-5, 12), (-5, 13), (-4, 0), (-4, 1)]
        return [eid for index, eid in enumerate(arr) if index % 2 ==0] , [eid for index, eid in enumerate(arr) if index % 2 > 0]
    
    #the function will return a list of data point for xl original classification and cascade function
    def load_faulted_array(self, events_fn):
        faulted_array = []
        id_events, id_phaseAngles, id_phaseAngles_ss = common.loadpickle(events_fn)
        for faultdata in self.pmudata.events():
            faultid = faultdata['id']
            if faultid not in id_events: continue
            if faultid not in id_phaseAngles_ss or len(id_phaseAngles_ss[faultid]) ==  0: 
                print 'skip', faultid 
                continue

            fault_comment = faultdata['sw_annotation']

            event0 = id_events[faultid][0]
            classification0 = classifyfault_v1.classify_event(event0, quiet=True)

            if (faultdata['sw_annotation'] not in self.valid_faults): 
                print 'skip', faultid
                continue

            #print 'add', faultid
            for i in range(len(id_phaseAngles_ss[faultid])):
                event = id_events[faultid][i]
                angles = id_phaseAngles[faultid][i]
                angles_ss = id_phaseAngles_ss[faultid][i]

                faulted_array.append((faultid, event.signal, event.v_sag_a, event.v_sag_b, event.v_sag_c, event.dv_t(),
                                      event.v_sag_a * event.v_ss_a, event.v_sag_b * event.v_ss_b, event.v_sag_c * event.v_ss_c,
                                      angles[0], angles[1], angles[2], angles_ss[0], angles_ss[1], angles_ss[2], i == 0,
                                      faultdata['sw_annotation'], classifyfault_v1.classify_event(event, quiet=True)))
        return faulted_array
    
    #for IEEE papers
    def get_data_points_for_v_sags_classifier(self):
        _, _, normal_data = common.get_fn_normal_minutes()
        sag_mode_fn1, sag_mode_fn2 = common.get_fn_sag_mode()
        
        sag_mode_array_1, sag_mode_array_2 = self.load_faulted_array(sag_mode_fn1), self.load_faulted_array(sag_mode_fn2)
        normal_array_1, normal_array_2 = common.loadpickle(normal_data[0]), common.loadpickle(normal_data[1])
        return sag_mode_array_1 + normal_array_1, sag_mode_array_2 + normal_array_2
        
    def get_data_points_for_rule_based_classifier(self):
        _, _, normal_data = common.get_fn_normal_minutes()
        fault_fn1, fault_fn2 = common.get_fn_new_xl_features()
        
        fault_array_1, fault_array_2 = common.loadpickle(fault_fn1), common.loadpickle(fault_fn2)
        normal_array_1, normal_array_2 = common.loadpickle(normal_data[0]), common.loadpickle(normal_data[1])
        return fault_array_1 + normal_array_1, fault_array_2 + normal_array_2
    
    def get_data_points_sag_mode_for_rule_based_classifier(self):
        _, _, normal_data = common.get_fn_normal_minutes()
        fault_fn1, fault_fn2 = common.get_fn_new_xl_features_sag_mode()
        
        fault_array_1, fault_array_2 = common.loadpickle(fault_fn1), common.loadpickle(fault_fn2)
        normal_array_1, normal_array_2 = common.loadpickle(normal_data[0]), common.loadpickle(normal_data[1])
        return fault_array_1 + normal_array_1, fault_array_2 + normal_array_2