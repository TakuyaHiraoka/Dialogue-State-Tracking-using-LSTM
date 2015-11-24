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

#12空の修正点
#提出時に不正なslot valueを推定時に除外する実験条件を追記 NEnsemblerのみ



class NaiveEnsembleBasedTracker(object):
    #Experimental condition 1
    isStoreFrameOutputedByLSTMAtEachTurnInSubDialog=True#各サブダイアログで、フレームに各ターン出力された結果を保存するか
    isIgnoreUtterancesNotRelatedToMainTask=True#問題の簡易化１　BIOがOとアノテーションされた、メインタスクと関係ない発話を除外して学習、トラッキングするか
    isSeparateDialogIntoSubDialog=True#問題の簡易化２　LSTMを副対話ごとに学習、トラッキングするか
    #Experimental condition 2
    isCombineResultWithBaseline=True#LSTMの出力結果をBaselineの出力結果を組合わせるか
    #Experimentl condition 5
    isIgnoreIlligalSlotValueInEstimation=True#不正なスロット値を除外するか
    listIlligalSlotValues=["Raffles City Shopping Centre","Vivo City"]#for test
#     listIlligalSlotValues=["Furama RiverFront Hotel",
#                            "Grand Park City Hall Hotel",
#                            "PARKROYAL on Beach Road Hotel",
#                            "Vivo City"]

    def __init__(self,tagsets=None,nameOfODictPickle=None):
        if self.isIgnoreIlligalSlotValueInEstimation:
            print "Naive Ensembler ignore following values:"
            print self.listIlligalSlotValues
        #Load
        if nameOfODictPickle is not None:
            print "Load output dictionary file from " + nameOfODictPickle
            self.dictOut=pickle.load(open(nameOfODictPickle,"r"))
            assert self.dictOut is not None, "Failed to read Output dictionary"
        if tagsets==None:
            self.tagsets = ontology_reader.OntologyReader("scripts/config/ontology_dstc4.json").get_tagsets()
        else:
            self.tagsets=tagsets
        #
        self.__initBaseTracker()
        #
        self.frame = {}
        self.reset()
    def __initBaseTracker(self):
        self.dictLSTMBaseTrackers={}
        #BT1
        nameBT="BaseTracker_BOWWithoutM1V"
        lstm=None
        if os.path.exists(nameBT):
            lstm=LSTMWithBOWTracker(self.tagsets,nameOfODictPickle="dictOutput.pic",nameOfIDictPickle=(nameBT+"/dictInput.pic"),nameOfLSTMFile=(nameBT+"/LSTM.rnnw"))
        else:
            print "Directory " + nameBT + "is not found. This directory and any required files are needed before tracking. "
            os.mkdir(nameBT)
            lstm=LSTMWithBOWTracker(self.tagsets)
        lstm.isEnableToUseM1sFeature=False
        lstm.isUseSentenceRepresentationInsteadofBOW=False
        self.dictLSTMBaseTrackers[nameBT]=lstm

        #BT2
        nameBT="BaseTracker_BOWWithM1V"
        lstm=None
        if os.path.exists(nameBT):
            lstm=LSTMWithBOWTracker(self.tagsets,nameOfODictPickle="dictOutput.pic",nameOfIDictPickle=(nameBT+"/dictInput.pic"),nameOfLSTMFile=(nameBT+"/LSTM.rnnw"))
        else:
            print "Directory " + nameBT + "is not found. This directory and any required files are needed before tracking. "
            os.mkdir(nameBT)
            lstm=LSTMWithBOWTracker(self.tagsets)
        lstm.isEnableToUseM1sFeature=True
        lstm.isUseSentenceRepresentationInsteadofBOW=False
        self.dictLSTMBaseTrackers[nameBT]=lstm
        #BT3
        nameBT="BaseTracker_D2VWithoutM1V"
        lstm=None
        if os.path.exists(nameBT):
            lstm=LSTMWithBOWTracker(self.tagsets,nameOfODictPickle="dictOutput.pic",nameOfIDictPickle=(nameBT+"/dictInput.pic"),nameOfLSTMFile=(nameBT+"/LSTM.rnnw"))
        else:
            print "Directory " + nameBT + "is not found. This directory and any required files are needed before tracking. "
            os.mkdir(nameBT)
            lstm=LSTMWithBOWTracker(self.tagsets)
        lstm.isEnableToUseM1sFeature=False
        lstm.isUseSentenceRepresentationInsteadofBOW=True
        self.dictLSTMBaseTrackers[nameBT]=lstm
        #BT4
        nameBT="BaseTracker_D2VWitM1V"
        lstm=None
        if os.path.exists(nameBT):
            lstm=LSTMWithBOWTracker(self.tagsets,nameOfODictPickle="dictOutput.pic",nameOfIDictPickle=(nameBT+"/dictInput.pic"),nameOfLSTMFile=(nameBT+"/LSTM.rnnw"))
        else:
            print "Directory " + nameBT + "is not found. This directory and any required files are needed before tracking. "
            os.mkdir(nameBT)
            lstm=LSTMWithBOWTracker(self.tagsets)
        lstm.isEnableToUseM1sFeature=True
        lstm.isUseSentenceRepresentationInsteadofBOW=True
        self.dictLSTMBaseTrackers[nameBT]=lstm
            
        
    def learn(self,pathdataset=["dstc4_train"], Pathdataroot="data"):
        for lstm in self.dictLSTMBaseTrackers.keys():
            #self.dictLSTMBaseTrackers[lstm].learn(numberOfHiddenUnit=1, EPOCHS_PER_CYCLE = 1, CYCLES = 2)
            self.dictLSTMBaseTrackers[lstm].learn(pathdataset=pathdataset,numberOfHiddenUnit=20, EPOCHS_PER_CYCLE = 10, CYCLES = 10)
            rnns=glob.glob("*.rnnw")
            for rnn in rnns:
                shutil.move(rnn, lstm)
            shutil.move("dictInput.pic", lstm)
        print "Learning is finished you need manualy select good network file for each base classifier."
    
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
                        if self.isIgnoreIlligalSlotValueInEstimation:
                            if value not in self.listIlligalSlotValues:
                                if slot not in self.frame:
                                    self.frame[slot] = []
                                if value not in self.frame[slot]:
                                    self.frame[slot].append(value)
                            else:
                                print "found in LSTM"
        
        #Integrate with Baseline
        if self.isCombineResultWithBaseline:
            if topic in self.tagsets:
                for slot in self.tagsets[topic]:
                    for value in self.tagsets[topic][slot]:
                        ratio = convInputs[self.dictLSTMBaseTrackers.keys()[0]][self.dictLSTMBaseTrackers[self.dictLSTMBaseTrackers.keys()[0]].dictIn["CLASS_"+value]]
                        if ratio > 80:
                            if self.isIgnoreIlligalSlotValueInEstimation:
                                if value not in self.listIlligalSlotValues:
                                    if slot not in self.frame:
                                        self.frame[slot] = []
                                    if value not in self.frame[slot]:
                                        self.frame[slot].append(value)
                                else:
                                    print "found in Baseline"
                if topic == 'ATTRACTION' and 'PLACE' in self.frame and 'NEIGHBOURHOOD' in self.frame and self.frame['PLACE'] == self.frame['NEIGHBOURHOOD']:
                    del self.frame['PLACE']
        
        #return
        #print str(len(self.frame.keys()))
        output['frame_label'] = self.frame
        return output

    def reset(self):
        self.frame = {}
        for lstm in self.dictLSTMBaseTrackers.keys():
            self.dictLSTMBaseTrackers[lstm].reset()
    
        
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
    
    tracker = NaiveEnsembleBasedTracker(tagsets,nameOfODictPickle="dictOutput.pic")
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
    