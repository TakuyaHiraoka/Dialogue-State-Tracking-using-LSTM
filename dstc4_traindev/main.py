'''
Created on 2015/07/14

@author: takuya-hv2
'''

#Modified from V15: modification for test set

if __name__ == '__main__':
    import sys
    import os
    import glob
    import shutil
    import re
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/scripts')
    from LSTMWithBOW import LSTMWithBOWTracker
    import DataAnalysis
    import score

    #Warning musi
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)

    #DataAnalysis.main(None)
    
    #construct d2V  
    isLearnDoc2Vec4LSTM=True
    if isLearnDoc2Vec4LSTM:
        #LSTMWithBOWTracker.constructDoc2vec(nameDataS2V=["dstc4_train","dstc4_dev"])
        #4 test
        LSTMWithBOWTracker.constructDoc2vec(nameDataS2V=["dstc4_train","dstc4_dev","dstc4_test"])
    
    
    #Train and Evaluate NaiveEnsembler
    import NaiveEnsembler
    import NaiveEnsemblerNBest
    isLearnAndEvaluateNaiveEnsembler=True
    if isLearnAndEvaluateNaiveEnsembler:
        #for test
        #-init i/o dicts required in ensembler
        print "Create o dictionary required in ensembler"
        lstm=LSTMWithBOWTracker()
        lstm.learn(pathdataset=["dstc4_train"], EPOCHS_PER_CYCLE = 1, CYCLES = 1)#the pathdataset which same with that in latter ensembler should be set.
        del lstm#enforce this instance to call destructor to save i/o dics.
        #-learning emsembler
        #ne=NaiveEnsemblerNBest.NaiveEnsembleBasedTrackerWithNBest()
        print "start learning tracker based on commmittee"
        ne=NaiveEnsembler.NaiveEnsembleBasedTracker()
        ne.learn(pathdataset=["dstc4_train"])
        #NaiveEnsembler.main(['--dataset', 'dstc4_dev', '--dataroot', 'data', '--trackfile', 'baseline_dev.json', '--ontology', 'scripts/config/ontology_dstc4.json'])
        #score.main(['--dataset', 'dstc4_dev', '--dataroot', 'data', '--trackfile', 'baseline_dev.json', '--scorefile', 'baseline_dev.score.csv', '--ontology', 'scripts/config/ontology_dstc4.json'])
        NaiveEnsembler.main(['--dataset', 'dstc4_test', '--dataroot', 'data', '--trackfile', 'baseline_dev.json', '--ontology', 'scripts/config/ontology_dstc4.json'])
        #NaiveEnsemblerNBest.errorAnalysis(['--dataset', 'dstc4_dev', '--dataroot', 'data', '--trackfile', 'baseline_dev.json', '--ontology', 'scripts/config/ontology_dstc4.json'])
        print "Evaluation was finished"
        print "Evaluatino fin. Results are described in baseline_dev.score.csv"












    
    
    
    #LEGACY (not for test data)    
    #Train LSTM tracker(i.e., proposed method)
    isLearnLSTM=False
    if isLearnLSTM:
        lstm=LSTMWithBOWTracker()
        lstm.learn()
        #lstm.learn(pathdataset="dstc4_dev")
        #del lstm
    
    
    #LEGACY (not for test data)
    #Evaluate all learned networks and find best one
    ifFindTheBestOneOverLearnedNetworks=False
    if ifFindTheBestOneOverLearnedNetworks:
        import LSTMWithBOW
        import score
        print "find best onr in learned netweorks(i.e., .rnnw)."
        res=open("performances.txt","w")
        learnedNetworkFiles=glob.glob("*.rnnw")
        if "LSTM.rnnw" in learnedNetworkFiles:
            learnedNetworkFiles.remove("LSTM.rnnw")
        bestf1Net=None
        bestf1=-100.0
        bestAccNet=None
        bestAcc=-100.0
        learnedNetworkFiles=sorted(learnedNetworkFiles)
        for learnedNetworkFile in learnedNetworkFiles:
            #-pre-process
            if os.path.exists("LSTM.rnnw"):
                os.remove("LSTM.rnnw")
            shutil.copy(learnedNetworkFile, "LSTM.rnnw")
            #-evaluate
            LSTMWithBOW.main(['--dataset', 'dstc4_dev', '--dataroot', 'data', '--trackfile', 'baseline_dev.json', '--ontology', 'scripts/config/ontology_dstc4.json'])
            score.main(['--dataset', 'dstc4_dev', '--dataroot', 'data', '--trackfile', 'baseline_dev.json', '--scorefile', 'baseline_dev.score.csv', '--ontology', 'scripts/config/ontology_dstc4.json'])
            #-write result
            res.write(learnedNetworkFile+":"+"\n")
            for line in open("baseline_dev.score.csv","r"):
                #Accuracy
                m1=re.search("all, all, 1, acc,",line)#Only see schedule 1 (Consider all turn in evaluation)
                if m1 != None:
                    m2=re.search("([0-9]|\.)+$",line)
                    res.write("-"+line)
                    if bestAcc < float(m2.group(0)):
                        bestAcc=float(m2.group(0))
                        bestAccNet=learnedNetworkFile
                #L1
                m1=re.search("all, all, 1, f1,",line)#Only see schedule 1 (Consider all turn in evaluation)
                if m1 != None:
                    m2=re.search("([0-9]|\.)+$",line)
                    res.write("-"+line)
                    if bestf1 < float(m2.group(0)):
                        bestf1=float(m2.group(0))
                        bestf1Net=learnedNetworkFile
        res.write("Best Acc="+bestAccNet+"\n")
        res.write("Best f1="+bestf1Net)
        res.close()
        print "Evaluatinon have finished. Summary of results are described in performances.txt"
        
        
        
        
        

#     #LEGACY parts: 
#     #tracking
#     isTrackWithLSTM=False
#     if isTrackWithLSTM:
#         import LSTMWithBOW
#         LSTMWithBOW.main(['--dataset', 'dstc4_dev', '--dataroot', 'data', '--trackfile', 'baseline_dev.json', '--ontology', 'scripts/config/ontology_dstc4.json'])
#         #LSTMWithBOW.main(['--dataset', 'dstc4_train', '--dataroot', 'data', '--trackfile', 'baseline_dev.json', '--ontology', 'scripts/config/ontology_dstc4.json'])
#     
#     #-tracking basline
#     isTrackWithBaseline=False
#     if isTrackWithBaseline:
#         import baseline
#         baseline.main(['--dataset', 'dstc4_dev', '--dataroot', 'data', '--trackfile', 'baseline_dev.json', '--ontology', 'scripts/config/ontology_dstc4.json'])
#         #baseline.main(['--dataset', 'dstc4_train', '--dataroot', 'data', '--trackfile', 'baseline_dev.json', '--ontology', 'scripts/config/ontology_dstc4.json'])
# 
#     #evaluate result
#     isEvaluate=False
#     if isEvaluate:
#         score.main(['--dataset', 'dstc4_dev', '--dataroot', 'data', '--trackfile', 'baseline_dev.json', '--scorefile', 'baseline_dev.score.csv', '--ontology', 'scripts/config/ontology_dstc4.json'])
#         #score.main(['--dataset', 'dstc4_train', '--dataroot', 'data', '--trackfile', 'baseline_dev.json', '--scorefile', 'baseline_dev.score.csv', '--ontology', 'scripts/config/ontology_dstc4.json'])
#         print "Evaluatino fin. Results are described in baseline_dev.score.csv"
