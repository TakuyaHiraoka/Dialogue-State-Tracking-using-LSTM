'''
Created on 2015/07/25

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


def main(argv):
    #TODO implementation
    #Confirmation hypothesis about data
    tagsets = ontology_reader.OntologyReader("scripts/config/ontology_dstc4.json").get_tagsets()
    datasetTrain = dataset_walker.dataset_walker("dstc4_train",dataroot="data",labels=True)
    datasetDev = dataset_walker.dataset_walker("dstc4_dev",dataroot="data",labels=True)
    print "Calculate statics of dialog. "
    #-Is number of value in each slot is always 1 if it exist? i.e., it does not contain multiple value?
    #-There are many multiple value
    isEnumerateMultiValueCase=True
    isEnumerateMultiSlotCase=True       
    countMultipleValueInOneSlot=0
    #
    maxSlotValueTrain={}
    countMultipleSlot=0
    for call in datasetTrain:
        for (uttr,label) in call:
            if "frame_label" in label:
                if isEnumerateMultiSlotCase:
                    if len(label["frame_label"].keys()) > 1:
                        print label["frame_label"].keys()
                        countMultipleSlot+=1
                for slot in label["frame_label"].keys():
                    if isEnumerateMultiValueCase:
                        if slot not in maxSlotValueTrain:
                            maxSlotValueTrain[slot]=len(label["frame_label"][slot])
                        else:
                            if maxSlotValueTrain[slot] < len(label["frame_label"][slot]):
                                maxSlotValueTrain[slot] = len(label["frame_label"][slot])
                        if len(label["frame_label"][slot]) > 1:
                            print "slot=" + slot + ":",
                            print label["frame_label"][slot]
                            countMultipleValueInOneSlot+=1                            
    
    for call in datasetDev:
        for (uttr,label) in call:
            if "frame_label" in label:
                if isEnumerateMultiSlotCase:
                    if len(label["frame_label"].keys()) > 1:
                        print label["frame_label"].keys()
                        countMultipleSlot+=1
                for slot in label["frame_label"].keys():
                    if isEnumerateMultiValueCase:
                        if slot not in maxSlotValueTrain:
                            maxSlotValueTrain[slot]=len(label["frame_label"][slot])
                        else:
                            if maxSlotValueTrain[slot] < len(label["frame_label"][slot]):
                                maxSlotValueTrain[slot] = len(label["frame_label"][slot])
                        if len(label["frame_label"][slot]) > 1:
                            print "slot=" + slot + ":",
                            print label["frame_label"][slot]
                            countMultipleValueInOneSlot+=1
    if isEnumerateMultiValueCase:
        print "Number of multiple value situation = " + str(countMultipleValueInOneSlot)
        avr=0.0
        for slot in maxSlotValueTrain.keys():
            avr+=(float)(maxSlotValueTrain[slot])
        avr/=float(len(maxSlotValueTrain.keys()))
        maxSlotValueTrain["AverageNumber"]=int(round(avr))
        print "Number of max slot value per slot:"
        print maxSlotValueTrain
        
    if isEnumerateMultiSlotCase:
        print "Number of multiple slot situation = " + str(countMultipleSlot)        
    #-How many OOV case?
    #-Train -> dev: 1195, Dev->Train: 4789
    #-With additional text normalizing, Train -> dev: 937, Dev->Train: 3643
    #-With additional normalization Train -> dev: 831, Dev->Train: 3237
    isCountNumberofOOVCase=False
    dictVocabInTrain={}
    dictVocabInDev={}
    numberOfOOVCaseInTrain2Dev=0
    numberOfOOVCaseInDev2Train=0
    if isCountNumberofOOVCase:
        for call in datasetTrain:
            for (uttr,label) in call:
                trans=uttr["transcript"]
                transt=re.sub("\,","",trans)
                transt=re.sub("\?","",transt)
                transt=re.sub("\.","",transt)
                transt=re.sub("(%.+ )?","",transt)
                #Additional normalize
                transt=re.sub("(%.+$)?","",transt)
                transt=re.sub("%","",transt)
                transt=re.sub("(-|~)"," ",transt)
                transt=re.sub("\!","",transt)
                transt=re.sub("'"," ",transt)
                transt=re.sub("\"","",transt)
                #
                transt=re.sub("/","",transt)
                transt=re.sub("[1-9]+","Replacedval",transt)
                transt=transt.lower()
                                    
                words=transt.split(" ")
                for word in words:
                    #Additional normalization
                    lmtr=nltk.stem.wordnet.WordNetLemmatizer()
                    word=lmtr.lemmatize(word)

                    dictVocabInTrain[word]=0
        for call in datasetDev:
            for (uttr,label) in call:
                trans=uttr["transcript"]
                transt=re.sub("\,","",trans)
                transt=re.sub("\?","",transt)
                transt=re.sub("\.","",transt)
                transt=re.sub("(%.+ )?","",transt)
                #Additional normalize
                transt=re.sub("(%.+$)?","",transt)
                transt=re.sub("%","",transt)
                transt=re.sub("(-|~)"," ",transt)
                transt=re.sub("\!","",transt)
                transt=re.sub("'"," ",transt)
                transt=re.sub("\"","",transt)
                #
                transt=re.sub("/","",transt)
                transt=re.sub("[1-9]+","Replacedval",transt)
                transt=transt.lower()
                
                words=transt.split(" ")
                for word in words:
                    #Additional normalization
                    lmtr=nltk.stem.wordnet.WordNetLemmatizer()
                    word=lmtr.lemmatize(word)

                    if word not in dictVocabInTrain:
                        print word.encode("utf-8")
                        numberOfOOVCaseInTrain2Dev+=1
        print "Number of OOV case in Train -> Dev situation = " + str(numberOfOOVCaseInTrain2Dev)
        print "\n\n\n\n\n"
        for call in datasetDev:
            for (uttr,label) in call:
                trans=uttr["transcript"]
                transt=re.sub("\,","",trans)
                transt=re.sub("\?","",transt)
                transt=re.sub("\.","",transt)
                transt=re.sub("(%.+ )?","",transt)
                #Additional normalize
                transt=re.sub("(%.+$)?","",transt)
                transt=re.sub("%","",transt)
                transt=re.sub("(-|~)"," ",transt)
                transt=re.sub("\!","",transt)
                transt=re.sub("'"," ",transt)
                transt=re.sub("\"","",transt)
                #
                transt=re.sub("/","",transt)
                transt=re.sub("[1-9]+","Replacedval",transt)                    
                transt=transt.lower()

                words=transt.split(" ")
                for word in words:
                    #Additional normalization
                    lmtr=nltk.stem.wordnet.WordNetLemmatizer()
                    word=lmtr.lemmatize(word)
                    
                    dictVocabInDev[word]=0
        for call in datasetTrain:
            for (uttr,label) in call:
                trans=uttr["transcript"]
                transt=re.sub("\,","",trans)
                transt=re.sub("\?","",transt)
                transt=re.sub("\.","",transt)
                transt=re.sub("(%.+ )?","",transt)
                #Additional normalize
                transt=re.sub("(%.+$)?","",transt)
                transt=re.sub("%","",transt)
                transt=re.sub("(-|~)"," ",transt)
                transt=re.sub("\!","",transt)
                transt=re.sub("'"," ",transt)
                transt=re.sub("\"","",transt)
                #
                transt=re.sub("/","",transt)
                transt=re.sub("[1-9]+","Replacedval",transt)
                transt=transt.lower()

                words=transt.split(" ")
                for word in words:
                    #Additional normalization
                    lmtr=nltk.stem.wordnet.WordNetLemmatizer()
                    word=lmtr.lemmatize(word)

                    if word not in dictVocabInDev:
                        print word.encode("utf-8")
                        numberOfOOVCaseInDev2Train+=1            
        print "Number of OOV case in Dev -> Train situation = " + str(numberOfOOVCaseInDev2Train)
        
    #-How many frame_label are unseen between train and dev data?
    #-So many, train -> dev 96/313 (unseen/all in dev), dev -> train 346/563 (unseen/all in train)
    isCountUnseenframeLabel=False
    dictTopicSlotValueTrain=[]
    numUnseenframeLabel=0
    alreadychecked=[]
    dictTopicSlotValueDev={}
    if isCountUnseenframeLabel:
        for call in datasetTrain:
            for (uttr,label) in call:
                if "frame_label" in label:
                    for slot in label["frame_label"].keys():
                        for value in label["frame_label"][slot]:
                            dictTopicSlotValueTrain.append(slot+value)
        for call in datasetDev:
            for (uttr,label) in call:
                if "frame_label" in label:
                    for slot in label["frame_label"].keys():
                        for value in label["frame_label"][slot]:
                            dictTopicSlotValueDev[(slot+value)]=0
                            if (slot+value) not in dictTopicSlotValueTrain:
                                if (slot+value) not in alreadychecked:
                                    numUnseenframeLabel+=1
                                    alreadychecked.append((slot+value))
        print "Number of Unseen label train -> dev = " + str(numUnseenframeLabel)
        print "Ratio (unseen/all in dev) = " + str(numUnseenframeLabel) + "/" + str(len(dictTopicSlotValueDev.keys()))

        dictTopicSlotValueDev=[]
        numUnseenframeLabel=0
        alreadychecked=[]
        dictTopicSlotValueTrain={}
        for call in datasetDev:
            for (uttr,label) in call:
                if "frame_label" in label:
                    for slot in label["frame_label"].keys():
                        for value in label["frame_label"][slot]:
                            dictTopicSlotValueDev.append(slot+value)
        for call in datasetTrain:
            for (uttr,label) in call:
                if "frame_label" in label:
                    for slot in label["frame_label"].keys():
                        for value in label["frame_label"][slot]:
                            dictTopicSlotValueTrain[(slot+value)]=0
                            if (slot+value) not in dictTopicSlotValueDev:
                                if (slot+value) not in alreadychecked:
                                    numUnseenframeLabel+=1
                                    alreadychecked.append((slot+value))
        print "Number of Unseen label dev -> train = " + str(numUnseenframeLabel)
        print "Ratio (unseen/all in train) = " + str(numUnseenframeLabel) + "/" + str(len(dictTopicSlotValueTrain.keys()))
                            

