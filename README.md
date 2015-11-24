## Todo
Translate manual (and comments on source codes) to English 

## Name
DSTracker4DSTC4

## Overview
[DSTC4](http://www.colips.org/workshop/dstc4/)用の対話状態追跡器

## Description
DSTracker4DSTC4はpythonで書かれたDSTC4用の対話状態追跡器（以後、トラッカー）。このトラッカーの基本的な枠組みはLong-short term memoryに基づいて実装されており、過去の履歴と入力された発話から対話の状態を推定する。
このプログラムでは、1)訓練データからトラッカーの構築と2)構築したトラッカーの性能評価を行う。

## Demo
DSTC4フォルダのmain.pyを実行

## Requirement
### Mandatory
* DSTC4準拠の仕様で記述された対話データ
* Python (version 2.7.6で動作確認済み)
* Pybrainとその依存ライブラリ(0.3.3で動作確認済み)
* Scikit-learn (version 1.5以上)
* fuzzywuzzy (0.5.0で動作確認済み)
* NLTK (3.0.2で動作確認済み)
* gensim (0.12.1で動作確認済み)

### Optional
* python_Levenshtein 

## Usage

* [トラッカーの構築:] DSTC4/main.py中のisLearnLSTMにTrueを設定してmain.pyを実行。
* [Sentence2Vecを利用したトラッカーの評価と構築:] 1)DSTC4/main.py中のisLearnDoc2vec4LSTMにTrueを設定、2)DSTC4/main.py中のisLearnLSTMとifFindTheBestOneOverLearnedNetworksをTrueにTrueを設定。また3)DSTC4/dstc4_traindev/scripts/LSTMWithBow.py中のisUseSentenceRepresentationInsteadofBOWをTrueにする。その後、main.pyを実行
* [トラッカーの評価:] DSTC4/main.py中のifFindTheBestOneOverLearnedNetworksをTrueに設定して、main.pyを実行。
* [コミッティーに基づくトラッカーの評価と構築:] DSTC4/main.py中のisLearnAndEvaluateNaiveEnsemblerにTrueを設定して、main.pyを実行


## Install
1. ReuirmentのMandatoryをインストールする
2. 本プロジェクトをダウンロードしてDSTC4フォルダにPythonの実行パスを通す
3. 対話データをDSTC4\dstc4_traindev\dataフォルダに入れる

## Introducing new feature
トラッカーへの新しい特徴量を導入する際には、１）特徴量の登録と2)登録した特徴量の計算が必要となる。その際は、DSTC4\dstc4_traindev\scripts\LSTMWithBOW.pyの以下の箇所に追記する。

* [特徴量の登録:] __rejisterM1sInputFeatureLabel
* [登録した特徴量の計算:] __calculateM1sInputFeature

詳細や追記例はソースコードの該当部を参照されたい。

## Upload to CodaLab (DSTC4 competition)
[トラッカーの評価]を行って作成された、baseline_dev.jsonをanswer.jsonにリネームする。そして、このファイルをzipに圧縮する。最後にzipファイルを以下のサイトに行って提出する。
https://www.codalab.org/competitions/4971#participate.

## Tips
TBA

## Contribution
TBA
## Licence
TBA

## Author

[TakuyaHiroka](http://isw3.naist.jp/~takuya-h/)