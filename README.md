# Fast-supervision-model
Quickly run supervised models on your data set, such as CNN, GRU, LSTM, BERT...



tensorflow文件夹需要下载:

bert-base-cased模型

bert-base-chinese模型

 sgns.weibo.word词嵌入

glove词嵌入按需下载



# method

1.sklearn_svm中是不同数据集同时训练

2.tensorflow中

`mode`：中英文

`filename`：输入文件

`embedding_dir`：词嵌入文件



如果需要更换数据集，请按照原数据集的处理格式处理
