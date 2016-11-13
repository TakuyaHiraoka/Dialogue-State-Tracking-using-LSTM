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

#V2からの修正点
#1.出力を、線形回帰から、ロジスティック回帰にして、その出力科率を元にフレームの値を推定
#2.オンライン学習時にトレーニングエポックごとにデータの順を入れ替えること（局所解に落ちにくくする）

#V3空の修正点
#1.LSTMのネットワークトポロジーを参考文献（深層学習, Hendersonの論文）と似たように修正（入力層と出力層の間に結合を加える・中間層の数をslot数にする（Hendersonと同様））
#1.->結合数が多すぎるので中間層のユニット数を中間層の数を２０にする（Hendersonの論文と同じ(http://www.aclweb.org/anthology/W13-4073)）して、IO間の層を結合を削除
#2.特徴量追加(BIO), クラス変数(オントロジーslotとvalueにそれぞれマッチしたかどうか：ベースラインと同様),
#3.加重減衰を入れる0.01->学習が遅いので0.0001に変更

#V4からの修正点
#0.高速化 FuzyMatchin組成の辞書化　文字列から組成の辞書を保持
#1.自動評価　最新のエポックの学習結果をLSTM.rnnに保持
#2.LSTMのresetを各対話（副対話ではなく）に変更（すなわち、学習の際の条件と合わせた。（クローズドテストでのf2向上0.14->0.81）

#V5空の修正点
#1.問題の簡易化を行う実験条件を導入
#1.1.メインタスクと関係ない発話を学習と評価から除外
#1.2.対話を副対話ごとに分割（副対話以前の履歴を考慮しない）

#7空の修正点
#組成の追加をするコードを追記
#ルールベース素性抽出例と外部知識を用いた素性例を追記

#8空の修正点
#動作を高速化するため修正　入力ベクトルを複数の部分ベクトルにわけた
#M1sFeatureの計算結果の記録
#分析用のコードを作成。データのいくつかの統計を表示
#分析結果に基づいて、データ間のOOVを下げるように、BOWを計算する際の文字の標準化方法を更新。

#10空の修正点
#erroAnalysisのところをここからぶんり
#Doc2Vecを導入

#11空の修正点
#fuzzymatchingresultを一回目以降読み込ませないようにして、高速化
#実験結果の要約作成

class LSTMWithBOWTracker(object):
    #initiate class and instance variables simultaneously
    #Experimental condition 1
    isStoreFrameOutputedByLSTMAtEachTurnInSubDialog=True#各サブダイアログで、フレームに各ターン出力された結果を保存するか
    isIgnoreUtterancesNotRelatedToMainTask=True#問題の簡易化１　BIOがOとアノテーションされた、メインタスクと関係ない発話を除外して学習、トラッキングするか
    isSeparateDialogIntoSubDialog=True#問題の簡易化２　LSTMを副対話ごとに学習、トラッキングするか
    #Experimental condition 2
    isCombineResultWithBaseline=True#LSTMの出力結果をBaselineの出力結果を組合わせるか
    #Experimental condition 3
    isEnableToUseM1sFeature=True#M1が考えた特徴量を考慮するか
    isUtilizeM1VectorDictionary=True#計算結果を保存した辞書を使うか
    #Experimental condition 4
    isUseSentenceRepresentationInsteadofBOW=True#文の分散表現をBOWとメタ情報の代わりに使うか
    
    
    #
    FileNameofdictFuzzyMatchingResult="FuzzyMatchingResult.pic.gz"
    dictFuzzyMatchingResult=None#[{utterance}] -> RatioVector
    #
    FileNameofNumClassFeature="numClassFeature.pic"
    FileNameofNumSentenceFeature="numSentenceFeature.pic"
    FileNameofNumM1Feature="numM1Feature.pic"
    TOTALSIZEOFCLASSFeature=None#1611
    TOTALSIZEOFSENTENCEFeature=None
    TOTALSIZEOFM1DEFINEDFeature=None
    #
    FileNameofM1Vector="M1Vec.pic.gz"
    dictM1Vector=None#[indicator(see __calculateM1sInputFeature)] -> corresponding M1 vecor
    
    
    def __init__(self,tagsets=None,nameOfODictPickle=None,nameOfIDictPickle=None,nameOfLSTMFile=None,NameOfLearnedD2VFile="LearnedDoc2Vec.d2v"):
        #Print out Experimenal setup 
        print "In both of learning and tracking,"
        if self.isStoreFrameOutputedByLSTMAtEachTurnInSubDialog:
            print "the tracker store output of LSTM at each turn in Sub.dial,"
        else:
            print "the tracker doesn't store output of LSTM at each turn in Sub.dial,"
        if self.isIgnoreUtterancesNotRelatedToMainTask:
            print "and tracker ignore the utterance which is not related to main task."
        else:
            print "and tracker doesn't ignore the utterance which is not related to main task."            
        if self.isSeparateDialogIntoSubDialog:
            print "and tracker consider one subdialog at one sequence."
        else:
            print "and tracker consider one dialog at one sequence."
        if self.isCombineResultWithBaseline:
            print "and the output of LSTM is combined with that of baseline."
        else:
            print "and the output of LSTM is not combined with that of baseline."
        if self.isEnableToUseM1sFeature:
            print "and features made by M1s are used in input."
        else:
            print "and features made by M1s are not used in input."
        if self.isUseSentenceRepresentationInsteadofBOW:
            print "and distributed sentence reprentation is used instead of BOW and meta info."
        else:
            print "and BOW and meta indo is used for sentense feature, and Distributed sentence reprentation is not used."
            
        #Variables for tracking state
        self.LSTM=None
        self.dictOut=None
        self.dictIn=None
        if nameOfLSTMFile is not None:
            print "Load LSTM network file from " + nameOfLSTMFile
            self.LSTM=NetworkReader.readFrom(nameOfLSTMFile)
            assert self.LSTM is not None, "Failed to read LSTM"
        if nameOfIDictPickle is not None:
            print "Load input dictionary file from " + nameOfIDictPickle
            f=open(nameOfIDictPickle,"r")
            self.dictIn=pickle.load(f)
            f.close()
            assert self.dictIn is not None, "Failed to read Input dictionary"
            
        if nameOfODictPickle is not None:
            print "Load output dictionary file from " + nameOfODictPickle
            f=open(nameOfODictPickle,"r")
            self.dictOut=pickle.load(f)
            f.close()
            assert self.dictOut is not None, "Failed to read Output dictionary"
        if tagsets==None:
            self.tagsets = ontology_reader.OntologyReader("scripts/config/ontology_dstc4.json").get_tagsets()
        else:
            self.tagsets=tagsets
        #Variables for fast processing
        #-1
        if LSTMWithBOWTracker.dictFuzzyMatchingResult == None:
            try: 
                f=gzip.open(self.FileNameofdictFuzzyMatchingResult,"rb")
            except Exception:
                print "FuzzyMatchingResult.pic was not found. Dictionary are newly created. "
                LSTMWithBOWTracker.dictFuzzyMatchingResult={}
            else:
                print "FuzzyMatchingResult.pic Dictionary are loaded."
                LSTMWithBOWTracker.dictFuzzyMatchingResult=pickle.load(f)
                f.close()
        #-2
        if self.isEnableToUseM1sFeature and self.isUtilizeM1VectorDictionary:
            try: 
                f=gzip.open(self.FileNameofM1Vector,"rb")
            except Exception:
                print self.FileNameofM1Vector + "was not found. Dictionary are newly created. "
                self.dictM1Vector={}
            else:
                print self.FileNameofM1Vector + " Dictionary are loaded."
                self.dictM1Vector=pickle.load(f)
                f.close()

        #
        try:
            f=open(self.FileNameofNumClassFeature,"rb")
            self.TOTALSIZEOFCLASSFeature=pickle.load(f)
            f.close()
            print "TSizeClasssFeature=" + str(self.TOTALSIZEOFCLASSFeature)
        except Exception:
            print self.FileNameofNumClassFeature + " was not found. learn() is required before tracking. "
        try:
            f=open(self.FileNameofNumSentenceFeature,"rb")
            self.TOTALSIZEOFSENTENCEFeature=pickle.load(f)
            f.close()
            print "TSizeSentenceFeature=" + str(self.TOTALSIZEOFSENTENCEFeature)
        except Exception:
            print self.FileNameofNumSentenceFeature + " was not found. learn() is required before tracking. "
        try:
            f=open(self.FileNameofNumM1Feature,"rb")
            self.TOTALSIZEOFM1DEFINEDFeature=pickle.load(f)
            f.close()
            print "TSizeM1Feature=" + str(self.TOTALSIZEOFM1DEFINEDFeature)
        except Exception:
            print self.FileNameofNumM1Feature + " was not found. learn() is required before tracking. "

        #
        if self.isUseSentenceRepresentationInsteadofBOW:
            self.d2v=LSTMWithBOWTracker.loadDoc2VecAndCheckAppropriateness(NameOfLearnedD2VFile)
        #
        self.frame = {}
        self.reset()
    def __del__(self):
        #for test
        #Already covered dstc4_train and dstc4_dev
        print "Write fuzzymatching result to file."
        f=gzip.open(self.FileNameofdictFuzzyMatchingResult,"wb")
        pickle.dump(LSTMWithBOWTracker.dictFuzzyMatchingResult, f,pickle.HIGHEST_PROTOCOL)
        f.close()
        
        if self.isEnableToUseM1sFeature and self.isUtilizeM1VectorDictionary:
            print "Write M1 vector comp. result to file."
            f=gzip.open(self.FileNameofM1Vector,"wb")
            pickle.dump(self.dictM1Vector, f,pickle.HIGHEST_PROTOCOL)
            f.close()
        pass
        
        
    def learn(self,pathdataset=["dstc4_train"], Pathdataroot="data",numberOfHiddenUnit=20, EPOCHS_PER_CYCLE = 10, CYCLES = 40,weightdecayw=0.01):
        print "Start learning LSTM, and make dictionary file"
        #Construct dictionary: variable name -> corresponding index of element in i/o vector
        print "Star make dictionary: variable name -> corresponding index of element in i/o vector"
        self.dictOut={}#"TOPIC_SLOT_VALUE" -> corresponding index of element
        self.dictIn={}#"SPEAKER_{val}"or"TOPIC_{val}","WORD_{word}" "BIO_{BIO}", "CLASS_{slot,value}", ""{defined label}-> corresponding  index of element
        #-target vector dictionary 
        index=0
        totalNumSlot=0
        for topic in self.tagsets.keys():
            for slot in self.tagsets[topic].keys():
                totalNumSlot+=1
                for value in self.tagsets[topic][slot]:
                    self.dictOut[topic+"_"+slot+"_"+value]=index
                    index+=1
        print "totalNumSlot:" + str(totalNumSlot)
        print "outputSize:"+str(len(self.dictOut.keys()))
        #-input dictionry
        dataset=[]
        for pathdat in pathdataset:
            dataset.append(dataset_walker.dataset_walker(pathdat,dataroot=Pathdataroot,labels=True))#False))
        #--(sub input vector 1) Class features i.e., Slot and value ratio (Similar to base line)
        index=0
        for topic in self.tagsets.keys():
            for slot in self.tagsets[topic].keys():
                if ("CLASS_"+ slot) not in self.dictIn:
                    self.dictIn["CLASS_"+slot]=index
                    index+=1
                for value in self.tagsets[topic][slot]:
                    if ("CLASS_"+ value) not in self.dictIn:
                        self.dictIn["CLASS_"+value]=index
                        index+=1
        self.TOTALSIZEOFCLASSFeature=index
        f=open(self.FileNameofNumClassFeature,"wb")
        pickle.dump(self.TOTALSIZEOFCLASSFeature,f)
        f.close()
        #--(sub input vector 2) Sentence features
        if not self.isUseSentenceRepresentationInsteadofBOW:
            index=0
            for elemDataset in dataset:
                for call in elemDataset:
                    for (uttr,_) in call:
                        #General info1 (CLASS; this feature must be rejistered at first)
                        if ("SPEAKER_"+uttr["speaker"]) not in self.dictIn:
                            self.dictIn["SPEAKER_"+uttr["speaker"]]=index
                            index+=1 
                        if ("TOPIC_"+uttr["segment_info"]["topic"]) not in self.dictIn:
                            self.dictIn["TOPIC_"+uttr["segment_info"]["topic"]]=index
                            index+=1 
                        #General info2
                        #-BIO
                        if ("BIO_"+uttr['segment_info']['target_bio']) not in self.dictIn:
                            self.dictIn["BIO_"+uttr['segment_info']['target_bio']]=index
                            index+=1
        
                        #BOW
                        if LSTMWithBOWTracker.isIgnoreUtterancesNotRelatedToMainTask:
                            if not (uttr['segment_info']['target_bio'] == "O"):
                                #-BOW
                                splitedtrans=self.__getRegurelisedBOW(uttr["transcript"])
                                for word in splitedtrans:
                                    if ("WORD_"+word) not in self.dictIn:
                                        self.dictIn["WORD_"+word]=index
                                        index+=1
            self.TOTALSIZEOFSENTENCEFeature=index
            f=open(self.FileNameofNumSentenceFeature,"wb")
            pickle.dump(self.TOTALSIZEOFSENTENCEFeature,f)
            f.close()
        elif self.isUseSentenceRepresentationInsteadofBOW:
            index=0
            for i in range(0,LSTMWithBOWTracker.D2V_VECTORSIZE):
                self.dictIn[str(index)+"thElemPV"]=index
                index+=1
            index=0
            for i in range(0,LSTMWithBOWTracker.D2V_VECTORSIZE):
                self.dictIn[str(index)+"thAvrWord"]=index
                index+=1
            assert self.D2V_VECTORSIZE == LSTMWithBOWTracker.D2V_VECTORSIZE, "D2V_VECTORSIZE is restrected to be same over the class"
        else:
            assert False, "Unexpected block" 
        #--(sub input vector 3) Features M1s defined
        index=0
        if self.isEnableToUseM1sFeature:
            rejisteredFeatures=self.__rejisterM1sInputFeatureLabel(self.tagsets,dataset)
            for rFeature in rejisteredFeatures:
                assert rFeature not in self.dictIn, rFeature +" already registered in input vector. Use different label name. "
                self.dictIn[rFeature]=index
                index+=1
            self.TOTALSIZEOFM1DEFINEDFeature=index
            f=open(self.FileNameofNumM1Feature,"wb")
            pickle.dump(self.TOTALSIZEOFM1DEFINEDFeature,f)
            f.close()

        print "inputSize:"+str(len(self.dictIn.keys()))
        assert self.dictIn["CLASS_INFO"] == 0, "Unexpected index CLASS_INFO should has value 0"
        assert self.dictIn["CLASS_Fort Siloso"] == 334, "Unexpected index CLASS_Fort Siloso should has value 334"
        assert self.dictIn["CLASS_Yunnan"] == 1344, "Unexpected index CLASS_Yunnan should has value 1611"
        #--write 
        fileObject = open('dictInput.pic', 'w')
        pickle.dump(self.dictIn, fileObject)
        fileObject.close()
        fileObject = open('dictOutput.pic', 'w')
        pickle.dump(self.dictOut, fileObject)
        fileObject.close()
        
        #Build RNN frame work
        print "Start learning Network"
        #Capability of network is: (30 hidden units can represents 1048576 relations) wherease (10 hidden units can represents 1024)
        #Same to Henderson (http://www.aclweb.org/anthology/W13-4073)?
        net = buildNetwork(len(self.dictIn.keys()), numberOfHiddenUnit, len(self.dictOut.keys()), hiddenclass=LSTMLayer, outclass=SigmoidLayer, outputbias=False, recurrent=True)
        
        #Train network
        #-convert training data into sequence of vector 
        convDataset=[]#[call][uttr][input,targetvec]
        iuttr=0
        convCall=[]
        for elemDataset in dataset:
            for call in elemDataset:
                for (uttr,label) in call:
                    if self.isIgnoreUtterancesNotRelatedToMainTask:
                        if uttr['segment_info']['target_bio'] == "O":
                            continue
                    #-input
                    convInput=self._translateUtteranceIntoInputVector(uttr,call)
                    #-output
                    convOutput=[0.0]*len(self.dictOut.keys())#Occured:+1, Not occured:0
                    if "frame_label" in label:
                        for slot in label["frame_label"].keys():
                            for value in label["frame_label"][slot]:
                                convOutput[self.dictOut[uttr["segment_info"]["topic"]+"_"+slot+"_"+value]]=1
                    #-post proccess
                    if self.isSeparateDialogIntoSubDialog:
                        if uttr['segment_info']['target_bio'] == "B":
                            if len(convCall) > 0:
                                convDataset.append(convCall)
                            convCall=[]
                    convCall.append([convInput,convOutput])
                    #print "Converted utterance" + str(iuttr)
                    iuttr+=1
                if not self.isSeparateDialogIntoSubDialog:
                    if len(convCall) > 0:
                        convDataset.append(convCall)
                    convCall=[]
        #Online learning
        trainer = RPropMinusTrainer(net,weightdecay=weightdecayw)
        EPOCHS = EPOCHS_PER_CYCLE * CYCLES
        for i in xrange(CYCLES):
            #Shuffle order
            ds = SequentialDataSet(len(self.dictIn.keys()),len(self.dictOut.keys()))
            datInd=range(0,len(convDataset))
            random.shuffle(datInd)#Backpropergation already implemeted data shuffling, however though RpropMinus don't. 
            for ind in datInd:
                ds.newSequence()
                for convuttr in convDataset[ind]:
                    ds.addSample(convuttr[0],convuttr[1])
            #Evaluation and Train
            epoch = (i+1) * EPOCHS_PER_CYCLE
            print "\r epoch {}/{} Error={}".format(epoch, EPOCHS,trainer.testOnData(dataset=ds))
            stdout.flush()
            trainer.trainOnDataset(dataset=ds,epochs=EPOCHS_PER_CYCLE)
            NetworkWriter.writeToFile(trainer.module, "LSTM_"+"Epoche"+str(i+1)+".rnnw")
            NetworkWriter.writeToFile(trainer.module, "LSTM.rnnw")

    @staticmethod
    def __getRegurelisedBOW(trans):
        transt=re.sub("\,","",copy.copy(trans))
        transt=re.sub("\?","",transt)
        transt=re.sub("\.","",transt)
        transt=re.sub("(%.+ )?","",transt)        
        #-Additional normalization
        transt=re.sub("(%.+$)?","",transt)
        transt=re.sub("%","",transt)
        transt=re.sub("(-|~)"," ",transt)
        transt=re.sub("\!","",transt)
        transt=re.sub("'"," ",transt)
        transt=re.sub("\"","",transt)
        transt=re.sub("/","",transt)
        transt=re.sub("[1-9]+","replacedval",transt)
        transt=transt.lower()
                
        #print trans.encode("utf-8")
        splitedtrans=transt.split(" ")
        bow=[]
        #-Additional normalization
        lmtr=nltk.stem.wordnet.WordNetLemmatizer()
        for word in splitedtrans:
            bow.append(lmtr.lemmatize(word))
        return bow

    
    def _translateUtteranceIntoInputVector(self,utter, call):
        #Metainfo+BOW+SLOT/Value matching result
        #--CLASS
        convClassInput=None
        if (utter["transcript"] not in LSTMWithBOWTracker.dictFuzzyMatchingResult):
            convClassInput=[0.0]*self.TOTALSIZEOFCLASSFeature
            for topic in self.tagsets.keys():
                for slot in self.tagsets[topic].keys():
                    convClassInput[self.dictIn["CLASS_"+slot]]=fuzz.partial_ratio(slot, utter["transcript"])
                    for value in self.tagsets[topic][slot]:
                        convClassInput[self.dictIn["CLASS_"+value]]=fuzz.partial_ratio(value, utter["transcript"])
            LSTMWithBOWTracker.dictFuzzyMatchingResult[utter["transcript"]]=copy.deepcopy(convClassInput)
        else:
            convClassInput=LSTMWithBOWTracker.dictFuzzyMatchingResult[utter["transcript"]]
        
        #-input
        convSentenceInput=None
        if not self.isUseSentenceRepresentationInsteadofBOW:
            convSentenceInput=[0.0]*self.TOTALSIZEOFSENTENCEFeature
            convSentenceInput[self.dictIn["SPEAKER_"+utter["speaker"]]]=1.0
            convSentenceInput[self.dictIn["TOPIC_"+utter["segment_info"]["topic"]]]=1.0
            splitedtrans=self.__getRegurelisedBOW(utter["transcript"])
            for word in splitedtrans:
                if ("WORD_"+word) in self.dictIn:#IGNORING OOV
                    convSentenceInput[self.dictIn["WORD_"+word]]=1.0
            convSentenceInput[self.dictIn["BIO_"+utter['segment_info']['target_bio']]]=1.0
        elif self.isUseSentenceRepresentationInsteadofBOW:
            sid=call.log["session_id"]
            uid=utter["utter_index"]
            label="s"+str(sid)+"u"+str(uid)#
            docv=self.d2v.docvecs[label].tolist()
            assert self.D2V_VECTORSIZE == LSTMWithBOWTracker.D2V_VECTORSIZE, "D2V_VECTORSIZE is restrected to be same over the class"
            assert len(docv) == self.D2V_VECTORSIZE, "Illigal vectors size"
            words=LSTMWithBOWTracker.__getRegurelisedBOW(copy.copy(utter["transcript"]))
            avrV=None
            for word in words:
                if avrV==None:
                    avrV=self.d2v[word]
                else:
                    avrV+=self.d2v[word]
            avrV/=len(words)
            avrWV=avrV.tolist()
            assert len(avrWV) == self.D2V_VECTORSIZE, "Illigal vectors size"
            convSentenceInput=docv+avrWV
            assert len(convSentenceInput) == (2*self.D2V_VECTORSIZE), "Illigal vectors size"
        else:
            assert False, "Unexpected block" 
        
        #-Features M1 defined
        convM1Input=None
        if self.isEnableToUseM1sFeature:
            if self.isUtilizeM1VectorDictionary:
                #print str(utter["utter_index"]) + "th uttreance"
                if (utter["speaker"]+utter["transcript"]+str(utter["utter_index"])) not in self.dictM1Vector:
                    convM1Input=[0.0]*self.TOTALSIZEOFM1DEFINEDFeature
                    convM1Input=self.__calculateM1sInputFeature(self.tagsets,convM1Input, utter)
                    self.dictM1Vector[(utter["speaker"]+utter["transcript"]+str(utter["utter_index"]))]=copy.copy(convM1Input)
                else:
                    convM1Input=self.dictM1Vector[(utter["speaker"]+utter["transcript"]+str(utter["utter_index"]))]
            else:
                convM1Input=[0.0]*self.TOTALSIZEOFM1DEFINEDFeature
                convM1Input=self.__calculateM1sInputFeature(self.tagsets,convM1Input, utter)
                
        convInput=None
        if self.isEnableToUseM1sFeature:
            convInput=convClassInput+convSentenceInput+convM1Input
        else: 
            convInput=convClassInput+convSentenceInput
        
        assert len(convInput)==len(self.dictIn.keys()),"Illigal input length"
        return convInput
    
    #TODO implement example
    def __rejisterM1sInputFeatureLabel(self,tagsets,dataset):
        #tagsets 本タスクのフレームラベルのオントロジー詳しくはHandbook6.4.節を参照のこと
        #dataset Log objectsとLabel Objectsのタプル。詳しくはHandbook6.1,6.2節を参照のこと
        #Log objectsとLabel Objectsのアクセスは
        #for (uttr,label) in call:
        #のようにする。この例では、uttrがLog object, labelはLabel Objectsを指す

        #ここで追加する素性のラベル名を登録する
        listFeatureLabels=[]
        #-笹野君用の素性登録例
        listFeatureLabels.append("DS_AskedInMR")
        #-石川さん用の素性登録例
        for topic in tagsets.keys():
            for slot in tagsets[topic].keys():
                if ("WNSim_"+ slot) not in listFeatureLabels:
                    listFeatureLabels.append("WNSim_"+slot)
        return listFeatureLabels
    
    #TODO implement example
    def __calculateM1sInputFeature(self,tagsets,convInput,utter):
        #ここで追加した素性を計算する
        #以下のようにして、計算した結果を入力に反映させる
        #convInput[self.dictIn["素性のラベル"]]]=素性の計算結果
        
        #-笹野君用の素性計算例
        #--直前の発話で質問が行われたか
        if (not hasattr(self,"userDefHistory")) or utter['segment_info']['target_bio'] == 'B':
            self.userDefHistory=[]
        if re.search("\?", utter["transcript"]) != None:
            self.userDefHistory=["Asked"]
        for his in self.userDefHistory:
            if his == "Asked":
                convInput[self.dictIn["DS_AskedInMR"]]=1.0
                self.userDefHistory.remove("Asked")
        
        #-石川さん用の素性計算例
        #--WordNetを用いて、スロットのそれぞれ値と入力発話の意味的な類似度を計算
        #--意味的な類似度とは、ここでは文中の名詞の平均類似度を指す
        #---入力文章のトークン化
        transt=re.sub("\,","",copy.copy(utter["transcript"]))
        transt=re.sub("\?","",transt)
        transt=re.sub("\.","",transt)
        transt=re.sub("(%.+ )?","",transt)
        tokensUtter=transt.split(" ")
        tokensWPOSUtter=nltk.pos_tag(tokensUtter)
        #print tokensWPOSUtter
        tokensWNounUtter=[]
        for token,pos in tokensWPOSUtter:
            #名詞のみ抽出. POSタグの分類:(http://cs.nyu.edu/grishman/jet/guide/PennPOS.html)
            if (pos == "NN") or (pos == "NNS") or (pos == "NNP") or (pos == "NNPS"):
                tokensWNounUtter.append(token)
        #---各スロットと入力発話との類似度を計算
        for topic in tagsets.keys():
            for slot in tagsets[topic].keys():
                tokensValue=slot.split(" ")
                tokensWPOSValue=nltk.pos_tag(tokensValue)
                #print tokensWPOSValue
                tokensWNounValue=[]
                for token,pos in tokensWPOSValue:
                    #名詞のみ抽出. POSタグの分類:(http://cs.nyu.edu/grishman/jet/guide/PennPOS.html)
                    if (pos == "NN") or (pos == "NNS") or (pos == "NNP") or (pos == "NNPS"):
                        tokensWNounValue.append(token)
                #類似度計算
                totalAvrSim=0.0
                for tokenInU in tokensWNounUtter:
                    avrSimtokenInUvsV=0.0
                    for tokenInV in tokensWNounValue:
                        sim=0
                        try:
                            sTU=wordnet.synset(tokenInU+".n.01")
                            sTV=wordnet.synset(tokenInV+".n.01")
                            sim=wordnet.path_similarity(sTU, sTV, simulate_root=False)
                            if sim==None:
                                print "NONE"
                                sim=0.0
                            avrSimtokenInUvsV+=sim
                        except:
                            avrSimtokenInUvsV+=sim
                        #print "Sim "+ tokenInU +" and " + tokenInV +" = " + str(sim)
                    if len(tokensWNounValue) > 0:
                        avrSimtokenInUvsV/=float(len(tokensWNounValue))
                    totalAvrSim+=avrSimtokenInUvsV
                if len(tokensWNounUtter) > 0:
                    totalAvrSim/=float(len(tokensWNounUtter))
                #print "TAvrSim="+str(totalAvrSim)
                convInput[self.dictIn[("WNSim_"+ slot)]]=totalAvrSim

        return convInput
    
    def addUtter(self, utter, call):
        #Pre-processes 
        output = {'utter_index': utter['utter_index']}
        if not self.isStoreFrameOutputedByLSTMAtEachTurnInSubDialog:
            self.frame = {}#As LSTM take the dialogue history consideration, output frame is completely updated by LSTM output at each turn
        if self.isIgnoreUtterancesNotRelatedToMainTask:
            if utter['segment_info']['target_bio'] == "O":
                output['frame_label'] = {}
                return output
        #-mae shori1
        assert self.LSTM is not None, "LSTM is required for tracking, but not existed."
        assert self.dictIn is not None, "Input dictionary is required for tracking, but not existed."
        assert self.dictOut is not None, "Output is required for tracking, but not existed."
        #-mae shori2
        if utter['segment_info']['target_bio'] == 'B':
            self.frame = {}
            if self.isSeparateDialogIntoSubDialog:
                if self.LSTM is not None:
                    self.LSTM.reset() #IF LSTM was trained by SeqDat which insert new seq. at each call, LSTM.reset() must called in same manner

        #Convert input into vector representation, and store sequential data
        convInput=self._translateUtteranceIntoInputVector(utter, call)

        #Estimate frame label of input
        #calc.
        outputVec=self.LSTM.activate(convInput)
        #print str(len(outputVec))
        #Interpret output vector of LSTM into frame_label
        topic = utter['segment_info']['topic']
        if topic in self.tagsets:
            for slot in self.tagsets[topic]:
                for value in self.tagsets[topic][slot]:
                    if outputVec[self.dictOut[topic+"_"+slot+"_"+value]] > 0.50:
                        if slot not in self.frame:
                            self.frame[slot] = []
                        if value not in self.frame[slot]:
                            self.frame[slot].append(value)
        
        #Integrate with Baseline
        if self.isCombineResultWithBaseline:
            if topic in self.tagsets:
                for slot in self.tagsets[topic]:
                    for value in self.tagsets[topic][slot]:
                        ratio = convInput[self.dictIn["CLASS_"+value]]
                        if ratio > 80:
                            if slot not in self.frame:
                                self.frame[slot] = []
                            if value not in self.frame[slot]:
                                self.frame[slot].append(value)
                if topic == 'ATTRACTION' and 'PLACE' in self.frame and 'NEIGHBOURHOOD' in self.frame and self.frame['PLACE'] == self.frame['NEIGHBOURHOOD']:
                    del self.frame['PLACE']
        
        #return
        #print str(len(self.frame.keys()))
        output['frame_label'] = self.frame
        return output

    def reset(self):
        self.frame = {}
        if self.LSTM is not None:
            self.LSTM.reset()
    
    
    #Variables for 
    D2V_MAXITERATION=3500
    D2V_MAXNUMCPU=15
    D2V_VECTORSIZE=300
    @staticmethod
    def constructDoc2vec(nameDataS2V=["dstc4_train","dstc4_dev"], dataPath="data", NameLearnedFile="LearnedDoc2Vec.d2v"):
        #nameDataS2V list of source dialogue data
        #Name of the file for learned doc2vecs.
        
        #label=s<sessionID>u(utteranceIndex)
        #words given bow
        
        print "Start to construct doc2vec from given dialogs."
        print nameDataS2V
        #Make input to doc to vec
        #-Load data
        dataS2V=[]
        for nameData in nameDataS2V:
            dataS2V.append(dataset_walker.dataset_walker(nameData,dataroot=dataPath,labels=True))
        lSentences=[]
        for dataset in dataS2V:
            print dataset
            for call in dataset:
                sid=call.log["session_id"]
                for (uttr,_) in call:
                    uid=uttr["utter_index"]
                    label="s"+str(sid)+"u"+str(uid)#
                    #print label
                    words=LSTMWithBOWTracker.__getRegurelisedBOW(copy.copy(uttr["transcript"]))
                    #print words
                    lSentences.append(TaggedDocument(words=words,tags=[label]))
        #Learn
        #reference: http://rare-technologies.com/doc2vec-tutorial/
        numMaxCPU=multiprocessing.cpu_count()
        if numMaxCPU > LSTMWithBOWTracker.D2V_MAXNUMCPU:
            print "As number of  CPU is exceeded, it rescaled into " + str(LSTMWithBOWTracker.D2V_MAXNUMCPU)
            numMaxCPU=LSTMWithBOWTracker.D2V_MAXNUMCPU
        print "Lern doc2vec with " + str(numMaxCPU)+" CPUs."
        model = Doc2Vec(size=LSTMWithBOWTracker.D2V_VECTORSIZE,workers=numMaxCPU,min_count=0)  # use fixed learning rate
        model.build_vocab(lSentences)
        for epoch in range(LSTMWithBOWTracker.D2V_MAXITERATION):
            model.train(lSentences)
            model.alpha *= 0.995# decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay    pass
            print str(epoch)+ "/"+ str(LSTMWithBOWTracker.D2V_MAXITERATION) + "(epoch/max epoch)"
        model.save(NameLearnedFile)
        print "Doc2Vec was constructed with:" 
        print dataS2V
        print ", stored to " + NameLearnedFile    
    
    @staticmethod
    def loadDoc2VecAndCheckAppropriateness(NameOfLearnedD2VFile):
        d2v=Doc2Vec.load(NameOfLearnedD2VFile)
        assert d2v != None, "Doc2Vec is not loaded"
        assert d2v.vector_size == LSTMWithBOWTracker.D2V_VECTORSIZE, "Illigal vector size"
        print "Doc2vec was loaded."
        return d2v
            
    
        
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
    
    tracker = LSTMWithBOWTracker(tagsets,nameOfODictPickle="dictOutput.pic",nameOfIDictPickle="dictInput.pic",nameOfLSTMFile="LSTM.rnnw")
    for call in dataset:
        this_session = {"session_id":call.log["session_id"], "utterances":[]}
        tracker.reset()
        for (utter,_) in call:
            #sys.stderr.write('%d:%d\n'%(call.log['session_id'], utter['utter_index']))
            tracker_result = tracker.addUtter(utter,call)
            if tracker_result is not None:
                this_session["utterances"].append(tracker_result)
        track["sessions"].append(this_session)
    end_time = time.time()
    elapsed_time = end_time - start_time
    track['wall_time'] = elapsed_time
    
    json.dump(track, track_file, indent=4)
    
    track_file.close()
    
if __name__ =="__main__":
    main(sys.argv)
