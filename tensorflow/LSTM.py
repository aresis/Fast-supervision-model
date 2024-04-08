import os
import re
import pandas as pd
import jieba as jb
from keras import Sequential, Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Layer
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize


os.environ["PATH"] += os.pathsep + 'D:/Graphviz/bin/'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mode = "english"
filename = "PHEME2.txt"


# 文本处理函数

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


# 数据预处理

df = pd.read_csv(filename, sep='\t', header=None)
stopwords = stopwordslist(mode + "StopWords.txt")
df = df[[1, 2]]
print("在 1 列中总共有 %d 个空值." % df[1].isnull().sum())
print("在 2 列中总共有 %d 个空值." % df[2].isnull().sum())
df = df[pd.notnull(df[2])]
print(len(df))
df[3] = df[2].factorize()[0]
id_df = df[[2, 3]].drop_duplicates().sort_values(2).reset_index(drop=True)
noise_pattern = re.compile("|".join([r'http\S+', r'\@\w+', r'\#\w+', r'\d+', r"'"]))
df[1] = df[1].apply(lambda x: re.sub(noise_pattern, '', x))
df[1] = df[1].apply(lambda x: remove_emoji(x))

if mode == "chinese":
    df[4] = df[1].apply(remove_punctuation)
    df[4] = df[4].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
else:
    df[4] = df[1].apply(lambda x: ' '.join([word for word in word_tokenize(x)]))
    # df[4] = df[1].apply(lambda x: " ".join([word for word in x.split() if word.lower() not in stopwords]))
    # df[4] = df[1].apply(lambda x: x.split())

# MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 300
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df[4].values)
word_index = tokenizer.word_index
print('共有 %s 个不相同的词语.' % len(word_index))
X = tokenizer.texts_to_sequences(df[4].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = pd.get_dummies(df[3]).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

MAX_NB_WORDS = len(word_index) + 1

epochs = 12
batch_size = 128


# 普通LSTM模型

def lstm():
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, dropout=0.4))
    model.add(Dense(2, activation='softmax'))
    """

    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # Embed each integer in a 128-dimensional vector
    x = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1])(inputs)
    # Add 2 bidirectional LSTMs
    x = Bidirectional(LSTM(128, dropout=0.2, return_sequences=True))(x)
    x = Bidirectional(LSTM(64, dropout=0.2))(x)
    # Add a classifier
    outputs = Dense(2, activation="softmax")(x)
    model = Model(inputs, outputs)
"""
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
              callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    return model


# attention-LSTM模型

# 重定义attention()层
def attention_lstm():
    # Add attention layer to the deep learning network
    class attention(Layer):
        def __init__(self, **kwargs):
            super(attention, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                     initializer='random_normal', trainable=True)
            self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                     initializer='zeros', trainable=True)
            super(attention, self).build(input_shape)

        def call(self, x):
            # Alignment scores. Pass them through tanh function
            e = K.tanh(K.dot(x, self.W) + self.b)
            # Remove dimension of size 1
            e = K.squeeze(e, axis=-1)
            # Compute the weights
            alpha = K.softmax(e)
            # Reshape to tensorFlow format
            alpha = K.expand_dims(alpha, axis=-1)
            # Compute the context vector
            context = x * alpha
            context = K.sum(context, axis=1)
            return context

    def create_LSTM_with_attention():
        x = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedding = Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1])(x)
        # drop = SpatialDropout1D(0.1)(embedding)
        LSTM_layer = LSTM(128, dropout=0.4, return_sequences=True)(embedding)
        attention_layer = attention()(LSTM_layer)
        outputs = Dense(2, activation='softmax')(attention_layer)
        model = Model(x, outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    model_attention = create_LSTM_with_attention()
    model_attention.summary()
    model_attention.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='loss', patience=3, min_delta=0.0001)])

    return model_attention


def predict(model, text):
    if mode == "chinese":
        text = remove_punctuation(text)
    text = [" ".join([w for w in list(jb.cut(text)) if w not in stopwords])]
    seq = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    id = pred.argmax()
    return id_df.loc[id_df[3] == id][2].values[0]


model = lstm()
# model = attention_lstm()
y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)
y_test = Y_test.argmax(axis=1)
target_name = list(map(str, id_df[2].values))
print(filename + ' accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, target_names=target_name, zero_division="warn",digits=4))
if mode == "chinese":
    print(predict(model, "FBI 承认罗斯威尔事件 是飞碟坠毁后，罗斯威尔飞碟坠毁事件幸存外星人视频又被曝光"))
else:
    print(predict(model, "a photo of black nurses saving the life of a kkk member"))
