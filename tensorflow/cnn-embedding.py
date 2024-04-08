import os
import re

import numpy as np
import pandas as pd
import jieba as jb
from keras import Input, Model
from keras.layers import Embedding, Conv1D, Dense, MaxPooling1D, concatenate, Flatten, Dropout
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from nltk import word_tokenize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

mode = "english"
filename = "PHEME2.txt"


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


def remove_punctuation(input_string):
    chinese_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    return "".join(chinese_pattern.findall(input_string))


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


df = pd.read_csv(filename, sep='\t', header=None)
stopwords = stopwordslist(mode + "StopWords.txt")
df = df[[1, 2]]
print("在 1 列中总共有 %d 个空值." % df[1].isnull().sum())
print("在 2 列中总共有 %d 个空值." % df[2].isnull().sum())
df = df[pd.notnull(df[2])]
print(len(df))
df[3] = df[2].factorize()[0]
id_df = df[[2, 3]].drop_duplicates().sort_values(2).reset_index(drop=True)
noise_pattern = re.compile("|".join(["http\S+", "\@\w+", "\#\w+", r"'"]))
df[1] = df[1].apply(lambda x: re.sub(noise_pattern, '', x))
df[1] = df[1].apply(lambda x: remove_emoji(x))
df[1] = df[1].str.replace('#', '', regex=False)

if mode == "chinese":
    df[4] = df[1].apply(remove_punctuation)
    df[4] = df[4].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
else:
    df[4] = df[1].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stopwords]))

MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 300
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df[4].values)
word_index = tokenizer.word_index
print('共有 %s 个不相同的词语.' % len(word_index))
X = tokenizer.texts_to_sequences(df[4].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = pd.get_dummies(df[3]).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

embeddings_index = {}
embedding_dir = ""
if mode == "chinese":
    embedding_dir = "sgns.weibo.word"
else:
    embedding_dir = 'glove.840B.300d.txt'
f = open(embedding_dir, encoding="UTF-8")
print("loading dict...it may take a long time...")
for line in f:
    values = line.split()
    word = values[0]
    try:
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    except:
        continue
f.close()
print('Found %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
count = 0
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        count += 1

print("find %d existed" % count)
MAX_NB_WORDS = len(word_index) + 1

main_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')  # int32
# 词嵌入（使用预训练的词向量）
embedder = Embedding(MAX_NB_WORDS,
                     EMBEDDING_DIM,
                     weights=[embedding_matrix],
                     input_length=X.shape[1])
embed = embedder(main_input)
# 词窗大小分别为3,4,5
cnn1 = Conv1D(512, 7, padding='same', strides=1, activation='relu')(embed)
cnn1 = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - 2)(cnn1)
cnn2 = Conv1D(512, 8, padding='same', strides=1, activation='relu')(embed)
cnn2 = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - 3)(cnn2)
cnn3 = Conv1D(512, 9, padding='same', strides=1, activation='relu')(embed)
cnn3 = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH - 4)(cnn3)

# 合并三个模型的输出向量
cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
flat = Flatten()(cnn)
drop = Dropout(0.4)(flat)
main_output = Dense(2, activation='softmax')(drop)
model = Model(inputs=main_input, outputs=main_output)

print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=800, epochs=8, validation_split=0.1)

y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)
y_test = Y_test.argmax(axis=1)
target_name = list(map(str, id_df[2].values))
print(filename + ' accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, target_names=target_name, digits=4))


def predict(text):
    if mode == "chinese":
        text = remove_punctuation(text)
        text = [" ".join([w for w in list(jb.cut(text)) if w not in stopwords])]
    else:
        text = [' '.join([word for word in word_tokenize(text) if word not in stopwords])]
    seq = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    id = pred.argmax()
    return id_df.loc[id_df[3] == id][2].values[0]


print(predict("砖家证实接吻也可怀孕，哥表示鸭梨很大！"))
