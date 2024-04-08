# dataset

| weibo | twitter | PHEME |
| ----- | ------- | ----- |
| 4662  | 2073    | 6423  |



# SVM

| dataset | weibo              | twitter            | PHEME              |
|---------|--------------------|--------------------|--------------------|
| acc     | 0.7487135506003431 | 0.8384615384615385 | 0.8342644320297952 |

weibo：

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|    
| false        | 1.00      | 0.50   | 0.67     | 583     |
| non-rumor    | 0.67      | 1.00   | 0.80     | 583     |
| accuracy     |           |        | 0.75     | 1166    |
| macro avg    | 0.83      | 0.75   | 0.73     | 1166    |
| weighted avg | 0.83      | 0.75   | 0.73     | 1166    |

# Navi-Bayes

| dataset | weibo              | twitter | PHEME              |
|---------|--------------------|---------|--------------------|
| acc     | 0.8336192109777015 | 0.85    | 0.8392302917442582 |


# Random Forest

| dataset | weibo              | twitter            | PHEME              |
|---------|--------------------|--------------------|--------------------|
| acc     | 0.7710120068610634 | 0.7826923076923077 | 0.7895716945996276 |

# textCNN

PHEME:

accuracy 0.842023346303502

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| non-rumours  | 0.87      | 0.89   | 0.88     | 839     |
| rumours      | 0.79      | 0.74   | 0.77     | 446     |
| accuracy     |           |        | 0.84     | 1285    |
| macro avg    | 0.83      | 0.82   | 0.82     | 1285    |
| weighted avg | 0.84      | 0.84   | 0.84     | 1285    |

weibo:

accuracy 0.8831725616291533

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| false        | 0.90      | 0.86   | 0.88     | 477     |
| non-rumor    | 0.86      | 0.90   | 0.88     | 456     |
| accuracy     |           |        | 0.88     | 933     |
| macro avg    | 0.88      | 0.88   | 0.88     | 933     |
| weighted avg | 0.88      | 0.88   | 0.88     | 933     |

# GRU

weibo:

accuracy 0.872454448017149

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| False        | 0.90      | 0.85   | 0.87     | 477     |
| True         | 0.85      | 0.90   | 0.87     | 456     |
| accuracy     |           |        | 0.87     | 933     |
| macro avg    | 0.87      | 0.87   | 0.87     | 933     |
| weighted avg | 0.87      | 0.87   | 0.87     | 933     |

PHEME:

accuracy 0.8599221789883269

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| non-rumours  | 0.91      | 0.86   | 0.89     | 819     |
| rumours      | 0.78      | 0.85   | 0.82     | 466     |
| accuracy     |           |        | 0.86     | 1285    |
| macro avg    | 0.85      | 0.86   | 0.85     | 1285    |
| weighted avg | 0.86      | 0.86   | 0.86     | 1285    |

twitter.
0.8216867469879519

   |      |     precision|    recall | f1-score|   support|
|--------------|-----------|--------|----------|---------|
|false      | 0.81     | 0.85   |   0.83   |    212|
|non-rumor     |  0.83   |   0.79  |    0.81    |   203|
|accuracy      |            |        | 0.82 |      415|
|macro avg   |    0.82|      0.82  |    0.82  |     415|
|weighted avg   |    0.82    |  0.82   |   0.82  |     415|

# LSTM

weibo:
accuracy 0.879957127545552

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| false        | 0.87      | 0.90   | 0.88     | 477     |
| non-rumor    | 0.89      | 0.86   | 0.88     | 456     |
| accuracy     |           |        | 0.88     | 933     | 
| macro avg    | 0.88      | 0.88   | 0.88     | 933     | 
| weighted avg | 0.88      | 0.88   | 0.88     | 933     | 



twitter：

accuracy 0.8144578313253013

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| false        | 0.84      | 0.78   | 0.81     | 212     |
| non-rumor    | 0.79      | 0.85   | 0.82     | 203     |
| accuracy     |           |        | 0.81     | 415     |
| macro avg    | 0.82      | 0.82   | 0.81     | 415     |
| weighted avg | 0.82      | 0.81   | 0.81     | 415     |



pheme:

accuracy 0.8521400778210116

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| False        | 0.87      | 0.91   | 0.89     | 839     |       
| True         | 0.82      | 0.74   | 0.78     | 446     |        
| accuracy     |           |        | 0.85     | 1285    |
| macro avg    | 0.84      | 0.83   | 0.83     | 1285    |     
| weighted avg | 0.85      | 0.85   | 0.85     | 1285    |       

# attention-LSTM

weibo:

accuracy 0.862759113652609

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| False        | 0.83      | 0.91   | 0.87     | 701     |
| True         | 0.90      | 0.81   | 0.86     | 698     |
| accuracy     |           |        | 0.86     | 1399    |
| macro avg    | 0.87      | 0.86   | 0.86     | 1399    |
| weighted avg | 0.87      | 0.86   | 0.86     | 1399    |

PHEME:

0.8401660612350804

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| non-rumours  | 0.89      | 0.86   | 0.87     | 1217    |
| rumours      | 0.77      | 0.81   | 0.79     | 710     |
| accuracy     |           |        | 0.84     | 1927    |
| macro avg    | 0.83      | 0.83   | 0.83     | 1927    |
| weighted avg | 0.84      | 0.84   | 0.84     | 1927    |

twitter.txt 
0.7373493975903614

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| false        | 0.68      | 0.91   | 0.78     | 212     |
| non-rumor    | 0.85      | 0.56   | 0.68     | 203     |
| accuracy     |           |        | 0.74     | 415     |
| macro avg    | 0.77      | 0.73   | 0.73     | 415     |
| weighted avg | 0.77      | 0.74   | 0.73     | 415     |

# word-embedding

## attention-lstm

PHEME2 

0.842023346303502

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| non-rumours  | 0.87      | 0.89   | 0.88     | 839     |
| rumours      | 0.78      | 0.76   | 0.77     | 446     |
| accuracy     |           |        | 0.84     | 1285    |
| macro avg    | 0.83      | 0.82   | 0.82     | 1285    |
| weighted avg | 0.84      | 0.84   | 0.84     | 1285    |

weibo
0.8821007502679529

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| false        | 0.88      | 0.89   | 0.89     | 477     |
| non-rumor    | 0.88      | 0.87   | 0.88     | 456     |
| accuracy     |           |        | 0.88     | 933     |
| macro avg    | 0.88      | 0.88   | 0.88     | 933     |
| weighted avg | 0.88      | 0.88   | 0.88     | 933     |

twitter

twitter.txt accuracy 0.7927710843373494
              precision    recall  f1-score   support

       false       0.77      0.85      0.81       212
   non-rumor       0.82      0.73      0.78       203

    accuracy                           0.79       415
   macro avg       0.80      0.79      0.79       415
weighted avg       0.80      0.79      0.79       415

## lstm

PHEME2
0.8443579766536965

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| non-rumours  | 0.87      | 0.89   | 0.88     | 839     |
| rumours      | 0.79      | 0.75   | 0.77     | 446     |
| accuracy     |           |        | 0.84     | 1285    |
| macro avg    | 0.83      | 0.82   | 0.83     | 1285    |
| weighted avg | 0.84      | 0.84   | 0.84     | 1285    |

weibo

0.887459807073955

|              | precision | recal | l  f1-score | support |
|--------------|-----------|-------|-------------|---------|
| false        | 0.90      | 0.87  | 0.89        | 477     |
| non-rumor    | 0.87      | 0.90  | 0.89        | 456     |
| accuracy     |           |       | 0.89        | 933     |
| macro avg    | 0.89      | 0.89  | 0.89        | 933     |
| weighted avg | 0.89      | 0.89  | 0.89        | 933     |

# BRET

acc：

PHEME2
0.846

|              | precision | recall | f1-sco | re   support |
|--------------|-----------|--------|--------|--------------|
| 0            | 0.79      | 0.84   | 0.81   | 243          |
| 1            | 0.90      | 0.86   | 0.88   | 400          |
| accuracy     |           |        | 0.85   | 643          |
| macro avg    | 0.84      | 0.85   | 0.84   | 643          |
| weighted avg | 0.85      | 0.85   | 0.85   | 643          |

weibo
0.936

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.95      | 0.91   | 0.93     | 215     |
| 1            | 0.93      | 0.96   | 0.94     | 252     |
| accuracy     |           |        | 0.94     | 467     |
| macro avg    | 0.94      | 0.93   | 0.94     | 467     |
| weighted avg | 0.94      | 0.94   | 0.94     | 467     |

twitter

0.802

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|    
| 0            | 0.80      | 0.70   | 0.75     | 93      |
| 1            | 0.78      | 0.86   | 0.82     | 115     |
| accuracy     |           |        | 0.79     | 208     |
| macro avg    | 0.79      | 0.78   | 0.78     | 208     |
| weighted avg | 0.79      | 0.79   | 0.79     | 208     |



# 学长的代码

## textCNN

weibo

 Test Acc: 86.70%

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|      
| false        | 0.8340    | 0.8982 | 0.8649   | 442     |
| non-rumor    | 0.9013    | 0.8388 | 0.8689   | 490     |
| accuracy     |           |        | 0.8670   | 932     |
| macro avg    | 0.8677    | 0.8685 | 0.8669   | 932     |
| weighted avg | 0.8694    | 0.8670 | 0.8670   | 932     |



PHEME

Test Acc: 86.70%

||precision  |  recall | f1-score |  support|
|--------------|-----------|--------|----------|---------|   
|false |    0.8340 |   0.8982 |   0.8649  |     442|
|non-rumor  |   0.9013|    0.8388 |   0.8689     |  490|
|accuracy        |        |       |  0.8670    |   932|
|macro avg  |   0.8677   | 0.8685    |0.8669 |      932|
|weighted avg   |  0.8694   | 0.8670  |  0.8670  |     932|



## textCNN-embedding

weibo

 86.70%

|   |      precision  |  recall | f1-score  | support|
|--------------|-----------|--------|----------|---------|       
|  false    | 0.8340  |  0.8982 |   0.8649     |  442|
|   non-rumor  |   0.9013   | 0.8388  |  0.8689  |     490|
|  accuracy     |          |    |      0.8670  |     932|
|  macro avg|     0.8677   | 0.8685    |0.8669   |    932|
|weighted avg|     0.8694   | 0.8670 |   0.8670      | 932|



## transformer-embedding

weibo

Test Acc: 79.61%

| |precision |   recall | f1-score |  support|
|--------------|-----------|--------|----------|---------|   
|false|     0.7625   | 0.8281 |   0.7939    |   442|
|non-rumor  |   0.8319|    0.7673  |  0.7983 |      490|
|accuracy        |       |        |  0.7961  |     932|
|macro avg   |  0.7972   | 0.7977 |   0.7961  |     932|
|weighted avg   |  0.7990  |  0.7961    |0.7962 |      932|



# BERT-CNN

weibo

 Loss:  0.323 | Acc:  91.515 % | AUC:0.913569108625045
         0: Precision: 0.935251798561151 | recall: 0.8823529411764706 | f1 score: 0.9080325960419091
         1: Precision: 0.8988326848249028 | recall: 0.9447852760736196 | f1 score: 0.9212362911266202
         macro avg: Precision: 0.9170422416930268 | recall: 0.913569108625045 | f1 score: 0.9146344435842646
         weighted avg: Precision: 0.9161229622378154 | recall: 0.9151450053705693 | f1 score: 0.9149677269725468

PHEME

   Loss:  0.590 | Acc:  88.785 % | AUC:0.8755902947402703
         0: Precision: 0.869281045751634 | recall: 0.8260869565217391 | f1 score: 0.8471337579617834
         1: Precision: 0.8981818181818182 | recall: 0.9250936329588015 | f1 score: 0.9114391143911439
         macro avg: Precision: 0.8837314319667261 | recall: 0.8755902947402703 | f1 score: 0.8792864361764636
         weighted avg: Precision: 0.8873102659358846 | recall: 0.8878504672897196 | f1 score: 0.8872494826501928


twitter

   Loss:  0.486 | Acc:  87.198 % | AUC:0.8672895224251518
         0: Precision: 0.9166666666666666 | recall: 0.7979274611398963 | f1 score: 0.853185595567867
         1: Precision: 0.8414634146341463 | recall: 0.9366515837104072 | f1 score: 0.886509635974304
         macro avg: Precision: 0.8790650406504065 | recall: 0.8672895224251518 | f1 score: 0.8698476157710855
         weighted avg: Precision: 0.8765219355092102 | recall: 0.8719806763285024 | f1 score: 0.8709745156882113

# Bert-RCNN

weibo

 Loss:  0.522 | Acc:  92.266 % | AUC:0.9226836558124901
         0: Precision: 0.9147982062780269 | recall: 0.9230769230769231 | f1 score: 0.9189189189189189
         1: Precision: 0.9298969072164949 | recall: 0.9222903885480572 | f1 score: 0.9260780287474332
         macro avg: Precision: 0.9223475567472609 | recall: 0.9226836558124902 | f1 score: 0.9224984738331761
         weighted avg: Precision: 0.9227286732585971 | recall: 0.9226638023630505 | f1 score: 0.9226791817611782

PHEME

  Loss:  0.564 | Acc:  87.461 % | AUC:0.8596358589030793
         0: Precision: 0.8577777777777778 | recall: 0.7991718426501035 | f1 score: 0.827438370846731
         1: Precision: 0.8836930455635491 | recall: 0.920099875156055 | f1 score: 0.9015290519877674
         macro avg: Precision: 0.8707354116706634 | recall: 0.8596358589030793 | f1 score: 0.8644837114172492
         weighted avg: Precision: 0.8739445452983408 | recall: 0.8746105919003115 | f1 score: 0.8736584920258355

twitter

 Loss:  0.567 | Acc:  87.923 % | AUC:0.8790003047851265
         0: Precision: 0.8666666666666667 | recall: 0.8756476683937824 | f1 score: 0.8711340206185567
         1: Precision: 0.8904109589041096 | recall: 0.8823529411764706 | f1 score: 0.8863636363636365
         macro avg: Precision: 0.8785388127853881 | recall: 0.8790003047851265 | f1 score: 0.8787488284910966
         weighted avg: Precision: 0.8793417598658814 | recall: 0.8792270531400966 | f1 score: 0.8792638396515583

