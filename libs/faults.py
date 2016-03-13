"""faults.py - processing the faults data set"""
import datetime as dt
import math
import csv
import os
import fnmatch
import numpy as np
import pandas as pd
import pickle
import re
import logging

class FaultMap:
   """This is essentially a fault database... perhaps we should
   use a databae ;)"""

   def __init__(self, faultlist, minuteslist):
      """Construct the FaultMap using:
      faultlist - a list of Fault instances
      minuteslist - a list of (path,file,[minutes covered])

      upon construction we obtain the following maps:
       pf_m:           map (path,file) -> [minute]
       faultpath_pf:   map faultpath -> [(path,file)]
       m_pf        :   map minute -> [(path,file)]
       d_m         :   map date -> [minute]
       m_fault     :   map minute -> [fault]
       fault_m     :   map fault -> [minute]
      """
 
      self.has_broken = False
      self.pf_m = {}         # map (path,file) -> [minute]
      self.faultpath_pf = {} # map faultpath -> [(path,file)]
      self.m_pf = {}         # map minute -> [(path,file)]
      self.d_m = {}          # map date -> [minute]

      self.m_faultid = {}         # map minute -> [faultid]
      self.faultid_m = {}         # map faultid -> [minute]
      self.faultid_faultinst = {} # map faultid -> faultinstance
      self.nonfaultminutes = None
      self.faultminutes = None
      self.h5minutestore = None   # set this for fast access to minutes
      
      for m in minuteslist:
         # m is (path, file, [minutes as datetime objects] | ['Broken'])
         assert m[2] != ['Broken'], "The data is a bit broken."
         #if m[2] == ['Broken']:
         #   self.has_broken = True
         #   continue

         pftuple = (m[0],m[1])
         assert not (pftuple in self.pf_m), "Data file duplicate!"
         
         self.pf_m[pftuple] = m[2]

         pflist = self.faultpath_pf.setdefault(m[0],[])
         pflist.append(pftuple)
        
         for dtm in m[2]:
            flist = self.m_pf.setdefault(dtm, [])
            flist.append(pftuple)
            mlist = self.d_m.setdefault(dtm.date(), [])
            mlist.append(dtm)

 

      # initialize m_faultid ... faster than using setdefault in the inner loop
      for m in self.m_pf:
         self.m_faultid[m] = []
         
      for f in faultlist:
         # fault begins at first_minute and continues to last_minute (inclusive)
         fi = (f.index, f.subindex)
         assert (not fi in self.faultid_m), "Fault id (%d,%d) is not unique!"%fi
         min_in_fault = []
         # iterate over all known minutes ... slow ... ;)
         for m in self.m_pf:
            if m >= f.first_minute and m <= f.last_minute:
               min_in_fault.append(m)
               filist = self.m_faultid[m]
               filist.append(fi)
         self.faultid_m[fi] = min_in_fault
         self.faultid_faultinst[fi] = f

      self._map_time_rawpod_to_faultid()
      

   def _map_time_rawpod_to_faultid(self):
      self.timepod_faultid = {}
      for f in self.faultid_faultinst:
         finst = self.faultid_faultinst[f]
         key = (finst.exact_time,finst.rawpod)
         assert not key in self.timepod_faultid, "Oops. time,rawpod is not unique."
         self.timepod_faultid[key] = f
      
   def dump_core_representation(self, fname):
      if os.path.exists(fname):
         print "%s exists, move away if you want to rewrite it."%(fname)
      else:
         with open(fname, 'wb') as fmc:
            pickle.dump(1,fmc) # faultmap core dump version
            pickle.dump(self.has_broken, fmc)
            pickle.dump(self.pf_m, fmc)
            pickle.dump(self.faultpath_pf, fmc)
            pickle.dump(self.m_pf, fmc)
            pickle.dump(self.d_m, fmc)
            pickle.dump(self.m_faultid, fmc)
            pickle.dump(self.faultid_m, fmc)
            pickle.dump(self.faultid_faultinst, fmc)

   def msig(self,minute):
      """DEPRECIATED:  use faults.msig()"""

      return msig(minute)

   def sigm(self,sig):
      """DEPRECIATED: use faults.sigm()"""
      return sigm(sig)

   def revive_core(self,fname):
      with open(fname, 'rb') as fmc:
         version = pickle.load(fmc)
         assert version == 1, "Unexpected version!"

         self.has_broken = pickle.load(fmc)
         self.pf_m = pickle.load(fmc)
         self.faultpath_pf = pickle.load(fmc)
         self.m_pf = pickle.load(fmc)
         self.d_m = pickle.load(fmc)
         self.m_faultid = pickle.load(fmc)
         self.faultid_m = pickle.load(fmc)
         self.faultid_faultinst = pickle.load(fmc)
         self.faultminutes = None
         self.nonfaultminutes = None

      self._map_time_rawpod_to_faultid()
         
   def normal_minutes(self):
      """return a list of minutes for which we have data, 
      and for which there is no fault condition"""
      
      if self.nonfaultminutes != None: 
         return self.nonfaultminutes
      # otherwise, we need to figure this out and cache it
      self.nonfaultminutes = []
      self.faultminutes = []
      
      for m in self.m_faultid:
         if not self.m_faultid[m]:
            self.nonfaultminutes.append(m)
         else:
            self.faultminutes.append(m)
            
      self.nonfaultminutes.sort()
      self.faultminutes.sort()
      return self.nonfaultminutes

   def fault_minutes(self):
      if self.faultminutes == None:
         self.normal_minutes()
      return self.faultminutes

   def get_minute_as_dataframe(self,minute):
      (header,minarray) = self.get_minute(minute)
      df = pd.DataFrame(minarray,columns=header,dtype=np.float64)
      df.insert(0,'DateTime',[dt.datetime(minute.year,minute.month,minute.day,
                                          minute.hour,minute.minute,s,int(ms/60.*1E6))
                                          for s in range(60) for ms in range(60)])
      return df
      
   def get_minute(self, minute):
      (p,f) = self.m_pf[minute][0]
      with open(os.path.join(self.CSV_DATA_ROOT,p,f), 'r') as minf:
         minr = csv.reader(minf)
         timeheader = minr.next()
         assert timeheader[5] == '0'
         timestamp = dt.datetime(int(timeheader[0]),int(timeheader[1]),
                                 int(timeheader[2]),int(timeheader[3]),
                                 int(timeheader[4]), 0)
         assert minute >= timestamp, "Requested minute, not found in target file!"

         header = minr.next()
         data = np.loadtxt(minf, delimiter=',')
         
         offset = minute - timestamp
         startrow = offset.seconds*60
         endrow = startrow+3600
         requested_data = data[startrow:endrow,:]
         if len(header) != requested_data.shape[1]:
            # check for the '2013-02/130222_230647', 'BPAe_1302222301.csv'
            # workaround situation...
            assert len(header) == requested_data.shape[1]+4, "Header is mismatched from data shape: no known workaround"
            JDFN = "John Day 230 A1 SA - Reactor Group #2"
            assert header.count(JDFN) == 4, "Header does not contain the expected duplicate fields for known workaround"

            # find the first occurance of the problematic field.
            keyloc = header.index(JDFN)
            assert header[keyloc] == header[keyloc+2] == header[keyloc+4] == header[keyloc+6], "Header does not contain the expected field orderings for known workaround"
            assert header[keyloc+1] == ' section 1 Current Mag'
            assert header[keyloc+3] == ' section 1 Current Ang'
            assert header[keyloc+5] == ' section 2 Current Mag'
            assert header[keyloc+7] == ' section 2 Current Ang'
            header = header[:keyloc] + [header[keyloc+2*i]+header[keyloc+2*i+1] for i in range(4)] + header[keyloc+8:]
            
               
         assert len(header) == len(set(header))
         assert len(header) == requested_data.shape[1]
         assert requested_data.shape[0] == 3600
         return (header, requested_data)

   def get_window_as_dataframe(self, kpt, sbefore, safter):
      """DEPRECIATED: use faults.get_window_as_dataframe()"""
      return get_window_as_dataframe(kpt, sbefore, safter, 
         self.h5minutestore)

def get_window_as_dataframe(keypoint, sbefore, safter, h5minutestore):
   """Get a window of data that starts sbefore seconds prior
   to the keypoint, and ends safter seconds after the keypoint"""

   begin = keypoint - dt.timedelta(seconds=sbefore)
   after = keypoint + dt.timedelta(seconds=safter)

   #print "Window: (%s, %s, %s)"%(str(begin),str(keypoint), str(after))
   begin0 = dt.datetime(begin.year,begin.month,begin.day,
                        begin.hour,begin.minute,0)
   after0 = dt.datetime(after.year,after.month,after.day,
                        after.hour,after.minute,0)
   # how many seconds into that first minute do we start?
   start = begin - begin0
   end = after - after0
   oneminute = dt.timedelta(seconds=60)

   # modf returns fractional,integer parts
   # so, find look at fractional part after changing to cycles....
   # add 1 to microseconds to force a round up
   cruft, s_cycles = math.modf(start.seconds*60+(start.microseconds+1)*60/1E6)
   assert abs(cruft-.5) > .4998, "start time not 1/60 second multiple? %s"%(str(cruft))
   s_cycles = int(s_cycles)  # this is safe.
   cruft, e_cycles = math.modf(end.seconds*60+(end.microseconds+1)*60/1E6)
   assert abs(cruft-.5) > .4998, "end time not 1/60 second multiple? %s"%(str(cruft))
   e_cycles = int(e_cycles)  # this is safe.

   # if begin0 and after0 are the same minute, life is easy...
   if begin0 == after0:
      t = h5minutestore[msig(begin0)].iloc[s_cycles:e_cycles,:]
      t.index = pd.Index(range(t.shape[0]))
      return t
   # otherwise, we need to glue some minutes together   
   try:
      # get the first minute
      m = begin0
      t = h5minutestore[msig(m)].iloc[s_cycles:,:]
      m = m + oneminute
      while m < after0:
         #print m
         n = h5minutestore[msig(m)]
         t = t.append(n, ignore_index=True)
         m = m + oneminute
      if m == after0:
         n = h5minutestore[msig(after0)].iloc[:e_cycles,:]
         t = t.append(n, ignore_index=True)
      return t
   except KeyError:
      print "KEY ERROR. Sorry"
      return None

def msig(minute):
   """Get the signature of a minute, suitable to acquire the minute table
   from our HDF5 store"""

   return minute.strftime('M%Y%m%d_%H%M%S')

def sigm(sig):
   """Get a minute as a datatime from a minute signature."""
   return dt.datetime.strptime(sig, 'M%Y%m%d_%H%M%S')

FAULT_POD_FORMAT = re.compile('([^-]+)-([^-]+) No (\d+) (\d+)kV line')

class Fault:
   def __init__(self, csvrow, countid, subindex):
      """create a Fault object from a list of data from
      the fault.csv file. A Fault contains the following
      fields:
        index - the index of the fault from the fault.csv file
        time - a datetime object indicating the fault onset

      """
      
      self.index = int(countid)
      self.subindex = int(subindex)
      self.exact_time = dt.datetime.strptime(csvrow[1]+'-'+csvrow[2],
                                       "%m/%d/%y-%H:%M:%S")
      # interpret duration as the maximum possible length
      # so, 0 min duration here is 59 seconds....
      self.duration = dt.timedelta(seconds = 60*int(csvrow[3])+59) 

      self.first_minute = dt.datetime(self.exact_time.year,
                                      self.exact_time.month,
                                      self.exact_time.day,
                                      self.exact_time.hour,
                                      self.exact_time.minute)

      endtime = self.exact_time + self.duration
      self.last_minute = dt.datetime(endtime.year, endtime.month, endtime.day,
                                     endtime.hour, endtime.minute)
                                     
                                     
      self.rawpod = csvrow[4]
      assert FAULT_POD_FORMAT.match(self.rawpod), "Raw pod '%s' fails to validate"%(self.rawpod)
      
   def __str__(self):
      return "(Fault: #%d.%d begins at %s for %d seconds)"%(self.index,self.subindex,
                                                  str(self.exact_time),
                                                  self.duration.seconds)
      
def load_faults(faultfile):
   faults = []
   with open(faultfile, 'r') as faultf:
      faultr = csv.reader(faultf)

      # advance to proper header
      header = faultr.next()
      while header[0] != 'Count':
         header = faultr.next()

      last_count = '0'
      subindex = 0
      # read the data
      while True:
         data = faultr.next()
         if len(data) > 4 and data[1] != '' and data[2] != '':
            if data[0] != '': 
               last_count = data[0]
               subindex = 0
            else: 
               subindex += 1
            f = Fault(data, last_count, subindex)
            faults.append(f)
         else:
            break
   return faults
 

