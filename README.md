## Todo
Translate manual (and comments on source codes) to English 

## Name
DSTracker4DSTC4

## Overview
Dialogue state tracker for [DSTC4](http://www.colips.org/workshop/dstc4/)

[DSTC4](http://www.colips.org/workshop/dstc4/)用の対話状態追跡器

## Description
DSTracker4DSTC4 is a dialogue state tracker for DSTC4, written in Python. 
This tracker is based on Long-shot term memory, and estimates dialogue states from an input utterance and its past history.
This program 1) constructs trackers from training data, and 2) evaluates these trackers. 

<!-- 
DSTracker4DSTC4はpythonで書かれたDSTC4用の対話状態追跡器（以後、トラッカー）。このトラッカーの基本的な枠組みはLong-short term memoryに基づいて実装されており、過去の履歴と入力された発話から対話の状態を推定する。
このプログラムでは、1)訓練データからトラッカーの構築と2)構築したトラッカーの性能評価を行う。
-->

## Demo
Execute "main.py" in DSTC4 directory.  
<!-- 
DSTC4フォルダのmain.pyを実行
-->

## Requirement
### Mandatory
* Dialogue data whose form follows the specification of DSTC4. 
* Python (version 2.7.6+)
* Pybrain and its dependencies (0.3.3+)
* Scikit-learn (version 1.5+)
* fuzzywuzzy (0.5.0+)
* NLTK (3.0.2+)
* gensim (0.12.1+)

<!-- 
* DSTC4準拠の仕様で記述された対話データ
* Python (version 2.7.6で動作確認済み)
* Pybrainとその依存ライブラリ(0.3.3で動作確認済み)
* Scikit-learn (version 1.5以上)
* fuzzywuzzy (0.5.0で動作確認済み)
* NLTK (3.0.2で動作確認済み)
* gensim (0.12.1で動作確認済み)
-->

### Optional
* python_Levenshtein 

## Usage
* [Construction of trackers:] Set the variable "isLearnLSTM" in "DSTC4/main.py" as "True". 
* [Construction and evaluation of trackers with Sentence2Vec] 1) Set variables "isLearnDoc2vec4LSTM", "isLearnLSTM" and "ifFindTheBestOneOverLearnedNetworks" in "DSTC4/main.py" as "True". 2) Set the variable "isUseSentenceRepresentationInsteadofBOW" in "DSTC4/dstc4_traindev/scripts/LSTMWithBow.py" as "True". 3) Execute "main.py"
* [Evaluation of trackers:] Set the variable "ifFindTheBestOneOverLearnedNetworks" in "DSTC4/main.py" as "True", then exucte "main.py". 
* [Construction and evaluation of trackers with Committee:] Set the variable "isLearnAndEvaluateNaiveEnsembler" in "DSTC4/main.py" as "True", and execute "main.py". 

<!-- 
* [トラッカーの構築:] DSTC4/main.py中のisLearnLSTMにTrueを設定してmain.pyを実行。
* [Sentence2Vecを利用したトラッカーの評価と構築:] 1)DSTC4/main.py中のisLearnDoc2vec4LSTMにTrueを設定、2)DSTC4/main.py中のisLearnLSTMとifFindTheBestOneOverLearnedNetworksをTrueにTrueを設定。また3)DSTC4/dstc4_traindev/scripts/LSTMWithBow.py中のisUseSentenceRepresentationInsteadofBOWをTrueにする。その後、main.pyを実行
* [トラッカーの評価:] DSTC4/main.py中のifFindTheBestOneOverLearnedNetworksをTrueに設定して、main.pyを実行。
* [コミッティーに基づくトラッカーの評価と構築:] DSTC4/main.py中のisLearnAndEvaluateNaiveEnsemblerにTrueを設定して、main.pyを実行
-->

## Install
1. install all requirment in "Mandatory". 
2. download this project and set python binary path to DSTC4 directory. 
3. put dialogue data into DSTC4\dstc4_traindev\data. 

<!-- 
1. RequirmentのMandatoryをインストールする
2. 本プロジェクトをダウンロードしてDSTC4フォルダにPythonの実行パスを通す
3. 対話データをDSTC4\dstc4_traindev\dataフォルダに入れる
-->

## Introducing new feature
In order to append new feature, we need implement following part in "DSTC4\dstc4_traindev\scripts\LSTMWithBOW.py": 

* [Registration of new feature:] __rejisterM1sInputFeatureLabel
* [Calculation of registered feature:] __calculateM1sInputFeature

<!-- 
トラッカーへの新しい特徴量を導入する際には、１）特徴量の登録と2)登録した特徴量の計算が必要となる。その際は、DSTC4\dstc4_traindev\scripts\LSTMWithBOW.pyの以下の箇所に追記する。

* [特徴量の登録:] __rejisterM1sInputFeatureLabel
* [登録した特徴量の計算:] __calculateM1sInputFeature

詳細や追記例はソースコードの該当部を参照されたい。
-->

## Upload to CodaLab (DSTC4 competition)

<!-- 
[トラッカーの評価]を行って作成された、baseline_dev.jsonをanswer.jsonにリネームする。そして、このファイルをzipに圧縮する。最後にzipファイルを以下のサイトに行って提出する。
https://www.codalab.org/competitions/4971#participate.
-->>


## Tips
TBA

## Contribution
TBA
## Licence
TBA

## Author

[TakuyaHiroka](http://isw3.naist.jp/~takuya-h/)