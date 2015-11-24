# coding:utf-8
'''
Created on 2015/07/15

@author: takuya-hv2
'''
import argparse, sys, ontology_reader, dataset_walker, time, json

from pybrain.datasets import SequentialDataSet
from itertools import cycle
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer
from pybrain.structure.modules import SigmoidLayer
from pybrain.structure.connections import FullConnection
from pybrain.supervised import RPropMinusTrainer
from sys import stdout
import re
import cPickle as pickle
from pybrain.tools.customxml import NetworkWriter, NetworkReader
import random
from fuzzywuzzy import fuzz
import gzip
import copy
import codecs
import nltk
from nltk.corpus import wordnet
from gensim.models.doc2vec import *
import multiprocessing
import numpy
from LSTMWithBOW import LSTMWithBOWTracker
import shutil, glob
import string
from NaiveEnsembler import NaiveEnsembleBasedTracker

#12空の修正点
#提出時に不正なslot valueを推定時に除外する NEnsemblerのみ

#13空の修正点
#Schedule 2 対策型の作成(NaiveEnsemblerNBest)
#-slotのvalueの最大数をそれぞれ計算して、推定の際にそれを超えると上位の信頼値をもつvalueのみが選択される
#-dev データに対するエラー分析コードの作成 (NaiveEnsemblerNBestのみ)
#-TODO fix fuzzy matching (notmalize with string.lower) part in LSTMWithBOW
#-TODO #Calculate average maximum fuzzy matching result score in training data, and make additional baseline tracker with this information

#15空の修正点
#-NaiveEmsemblerからクラス継承
#-test用にdictMaxを修正

class NaiveEnsembleBasedTrackerWithNBest(NaiveEnsembleBasedTracker):    
    #Max number of value of each slot in N-best (from analysis training data)
#     dictMaxSlotValue={u'INFO': 2, u'CUISINE': 3, u'FROM': 2, u'TYPE_OF_PLACE': 2, 
#                       u'DRINK': 2, u'TIME': 3, u'TO': 2, u'STATION': 3, u'PLACE': 6, 
#                       u'MEAL_TIME': 2, u'ACTIVITY': 4, u'DISH': 5, u'TICKET': 2, 
#                       u'TYPE': 2, u'LINE': 3, u'NEIGHBOURHOOD': 3, 'AverageNumber': 3}
    #Max number of value of each slot in N-best (from analysis training+dev data)
    dictMaxSlotValue={u'INFO': 3, u'CUISINE': 7, u'FROM': 2, u'TYPE_OF_PLACE': 3, 
                      u'DRINK': 2, u'TIME': 3, u'TO': 3, u'STATION': 3, u'PLACE': 6, 
                      u'MEAL_TIME': 2, u'ACTIVITY': 4, u'DISH': 5, u'TICKET': 2, 
                      u'TYPE': 2, u'LINE': 3, u'NEIGHBOURHOOD': 3, 'AverageNumber': 3}


    def __init__(self,tagsets=None,nameOfODictPickle=None):
        NaiveEnsembleBasedTracker.__init__(self,tagsets,nameOfODictPickle)
        print "The output is selected according to N-best score"
    
    #Overrided
    def addUtter(self, utter, call):
        #Pre-processes 
        output = {'utter_index': utter['utter_index']}
        if not self.isStoreFrameOutputedByLSTMAtEachTurnInSubDialog:
            self.frame = {}#As LSTM take the dialogue history consideration, output frame is completely updated by LSTM output at each turn
        if self.isIgnoreUtterancesNotRelatedToMainTask:
            if utter['segment_info']['target_bio'] == "O":
                output['frame_label'] = {}
                return output
        #-mae shori2
        if utter['segment_info']['target_bio'] == 'B':
            self.frame = {}
            if self.isSeparateDialogIntoSubDialog:
                for lstm in self.dictLSTMBaseTrackers.keys():
                    self.dictLSTMBaseTrackers[lstm].reset()

        #Convert input into vector representation, and store sequential data
        convInputs={}
        for lstm in self.dictLSTMBaseTrackers.keys():
            convInputs[lstm]=self.dictLSTMBaseTrackers[lstm]._translateUtteranceIntoInputVector(utter, call)
            
        #Estimate frame label of input
        #calc.
        avrOutputs=numpy.zeros(len(self.dictOut.keys()))
        for lstm in self.dictLSTMBaseTrackers.keys():
            avrOutputs+=numpy.array(self.dictLSTMBaseTrackers[lstm].LSTM.activate(convInputs[lstm]))
        avrOutputs/=float(len(self.dictLSTMBaseTrackers.keys()))
        ravrOuputs=avrOutputs.tolist()
        #print str(len(outputVec))
        #Interpret output vector of LSTM into frame_label
        topic = utter['segment_info']['topic']
        if topic in self.tagsets:
            for slot in self.tagsets[topic]:
                for value in self.tagsets[topic][slot]:
                    if ravrOuputs[self.dictOut[topic+"_"+slot+"_"+value]] > 0.50:
                        if value not in self.listIlligalSlotValues:
                            if slot not in self.frame:
                                self.frame[slot] = {}
                            if value not in self.frame[slot]:
                                self.frame[slot][value]=ravrOuputs[self.dictOut[topic+"_"+slot+"_"+value]]
        
        #Integrate with Baseline
        if self.isCombineResultWithBaseline:
            if topic in self.tagsets:
                for slot in self.tagsets[topic]:
                    for value in self.tagsets[topic][slot]:
                        ratio = convInputs[self.dictLSTMBaseTrackers.keys()[0]][self.dictLSTMBaseTrackers[self.dictLSTMBaseTrackers.keys()[0]].dictIn["CLASS_"+value]]
                        if ratio > 80:
                            if value not in self.listIlligalSlotValues:
                                if slot not in self.frame:
                                    self.frame[slot] = {}
                                #In orijinal tracker, string are not normalized with lower
                                ratio=fuzz.partial_ratio(string.lower(value), string.lower(utter['transcript']))
                                temp=(float)(len(str(value))) * (float)(ratio)/(100.0)
                                if (value not in self.frame[slot]) or (self.frame[slot][value] < temp):
                                    self.frame[slot][value]=temp

                if topic == 'ATTRACTION' and 'PLACE' in self.frame and 'NEIGHBOURHOOD' in self.frame and self.frame['PLACE'].keys() == self.frame['NEIGHBOURHOOD'].keys():
                    del self.frame['PLACE']
        #N-best candidate selection
        print "Frame :"
        print self.frame
        #
        nbestFrame={}
        for slot in self.frame:
            if slot not in nbestFrame:
                sInDict=None
                if slot in self.dictMaxSlotValue:
                    sInDict=slot
                else:
                    sInDict='AverageNumber'
                
                if len(self.frame[slot].keys()) > self.dictMaxSlotValue[sInDict]:
                    temp=[]
                    temp2=[]
                    #Partial ratio has priority than expected matching length in candidate selection
                    for value in self.frame[slot].keys():
                        if (self.frame[slot][value]/(float)(len(str(value)))) >=1.0:
                            temp.append((value,self.frame[slot][value]))
                        else:
                            temp2.append((value,self.frame[slot][value]))
                    sk=None
                    if len(temp) >= self.dictMaxSlotValue[sInDict]:
                        sk=sorted(temp, key=lambda x:x[1],reverse=True)
                    elif len(temp) < self.dictMaxSlotValue[sInDict]:
                        sk=temp
                        sk=sk+sorted(temp2, key=lambda x:x[1],reverse=True)
                    print "The number of value of " + slot + " is more than sleshold"
                    print "Before list: ",
                    print sk
                    nbestFrame[slot]=[]
                    for ind in range(self.dictMaxSlotValue[sInDict]):
                        nbestFrame[slot].append(sk[ind][0])
                    print "After list: ",
                    print nbestFrame[slot]
                else:
                    nbestFrame[slot]=self.frame[slot].keys()
        
        #return
        output['frame_label'] = nbestFrame
        return output

    
    
    
    
    
    
    
    
    
def errorAnalysis(argv):
    print "ERROR ANALYSIS OF NAIVEENSEMBLER"
    print argv
    
    parser = argparse.ArgumentParser(description='Simple hand-crafted dialog state tracker baseline.')
    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True, help='The dataset to analyze')
    parser.add_argument('--dataroot',dest='dataroot',action='store',required=True,metavar='PATH', help='Will look for corpus in <destroot>/<dataset>/...')
    parser.add_argument('--trackfile',dest='trackfile',action='store',required=True,metavar='JSON_FILE', help='File to write with tracker output')
    parser.add_argument('--ontology',dest='ontology',action='store',metavar='JSON_FILE',required=True,help='JSON Ontology file')

    #args = parser.parse_args()
    args = parser.parse_args(argv)
    dataset = dataset_walker.dataset_walker(args.dataset,dataroot=args.dataroot,labels=True)
    tagsets = ontology_reader.OntologyReader(args.ontology).get_tagsets()
    
    track = {"sessions":[]}
    track["dataset"]  = args.dataset
    start_time = time.time()
    
    tracker = NaiveEnsembleBasedTrackerWithNBest(tagsets,nameOfODictPickle="dictOutput.pic")
    for call in dataset:
        this_session = {"session_id":call.log["session_id"], "utterances":[]}
        tracker.reset()
        for (utter,label) in call:
            #-mae shori2
            if utter['segment_info']['target_bio'] == 'B':
                print "\n -----New sub-dialogue----------------------------------------------------"
            print "s:"+str(call.log['session_id'])+ " u:"+str(utter['utter_index'])
            print "Input=" + utter["transcript"]
            tracker_result = tracker.addUtter(utter,call)
            if tracker_result is not None:
                this_session["utterances"].append(tracker_result)
                #
                print "Tracker's output:"
                print tracker_result
                if "frame_label" in label:
                    for slot in label["frame_label"].keys():
                        if (slot not in tracker_result["frame_label"]):
                            print "-slot [" + slot + "] is not exsisted in output"
                            for value in label["frame_label"][slot]:
                                print "-value [" + value + "] of slot [" + slot +"] is not exsisted in output"
                        else:
                            if len(label["frame_label"][slot]) != len(tracker_result["frame_label"][slot]):
                                #In case value in output, but repudant
                                print "-slot [" + slot + "] include repudant values"
                            for value in label["frame_label"][slot]:
                                #In case value not in output
                                if (value not in tracker_result["frame_label"][slot]):
                                    print "-value [" + value + "] of slot [" + slot +"] is not exsisted in output"
        track["sessions"].append(this_session)
    end_time = time.time()
    elapsed_time = end_time - start_time
    track['wall_time'] = elapsed_time
    
    
    
    
    
    
    
    
    
    
        
def main(argv):
    print argv
    
    parser = argparse.ArgumentParser(description='Simple hand-crafted dialog state tracker baseline.')
    parser.add_argument('--dataset', dest='dataset', action='store', metavar='DATASET', required=True, help='The dataset to analyze')
    parser.add_argument('--dataroot',dest='dataroot',action='store',required=True,metavar='PATH', help='Will look for corpus in <destroot>/<dataset>/...')
    parser.add_argument('--trackfile',dest='trackfile',action='store',required=True,metavar='JSON_FILE', help='File to write with tracker output')
    parser.add_argument('--ontology',dest='ontology',action='store',metavar='JSON_FILE',required=True,help='JSON Ontology file')

    #args = parser.parse_args()
    args = parser.parse_args(argv)
    dataset = dataset_walker.dataset_walker(args.dataset,dataroot=args.dataroot,labels=False)
    tagsets = ontology_reader.OntologyReader(args.ontology).get_tagsets()
    
    track_file = open(args.trackfile, "wb")
    track = {"sessions":[]}
    track["dataset"]  = args.dataset
    start_time = time.time()
    
    tracker = NaiveEnsembleBasedTrackerWithNBest(tagsets,nameOfODictPickle="dictOutput.pic")
    for call in dataset:
        this_session = {"session_id":call.log["session_id"], "utterances":[]}
        tracker.reset()
        for (utter,_) in call:
            sys.stderr.write('%d:%d\n'%(call.log['session_id'], utter['utter_index']))
            tracker_result = tracker.addUtter(utter,call)
            if tracker_result is not None:
                this_session["utterances"].append(tracker_result)
        track["sessions"].append(this_session)
    end_time = time.time()
    elapsed_time = end_time - start_time
    track['wall_time'] = elapsed_time
    
    json.dump(track, track_file, indent=4)
    
    track_file.close()
    