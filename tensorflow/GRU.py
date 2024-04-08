import re
import pandas as pd
import jieba as jb
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, SpatialDropout1D, GRU, Dense
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from nltk import word_tokenize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import os

mode = "chinese"
filename = "weibo.txt"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
    df[4] = df[1].apply(lambda x: ' '.join([word for word in word_tokenize(x)]))

MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df[4].values)
word_index = tokenizer.word_index
print('共有 %s 个不相同的词语.' % len(word_index))
X = tokenizer.texts_to_sequences(df[4].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = pd.get_dummies(df[3]).values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(GRU(128, dropout=0.3,recurrent_dropout=0.4))
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

y_pred = model.predict(X_test)
y_pred = y_pred.argmax(axis=1)
y_test = Y_test.argmax(axis=1)
target_name = list(map(str, id_df[2].values))
print(filename+' accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, target_names=target_name,digits=4))


def predict(text):
    if mode == "chinese":
        text = remove_punctuation(text)
    text = [" ".join([w for w in list(jb.cut(text)) if w not in stopwords])]
    seq = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    id = pred.argmax()
    return id_df.loc[id_df[3] == id][2].values[0]


print(predict("a photo of black nurses saving the life of a kkk member"))
