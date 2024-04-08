import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import svm


class LanguageDetector:
    def __init__(self, classifier=svm.SVC()):
        self.classifier = classifier
        self.vectorizer = CountVectorizer(ngram_range=(1, 2), preprocessor=self._remove_noise)

    def _remove_noise(self, document):
        noise_pattern = re.compile("|".join(["http\S+", "\@\w+", "\#\w+", "URL"]))
        clean_text = re.sub(noise_pattern, "", document)
        return clean_text

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)


def run(filename, sentence):
    filename = filename
    with open(filename + ".txt", "r", encoding="UTF-8") as f:
        dataset = []
        for line in f:
            s = line.strip().split("\t")
            sentence = " ".join(s[1:-1])
            last_word = s[-1]
            if not len(last_word) == 0:
                dataset.append((sentence, last_word))
        x, y = zip(*dataset)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    language_detector = LanguageDetector()
    language_detector.fit(x_train, y_train)
    print(filename + ":" + str(language_detector.score(x_test, y_test)))
    print(language_detector.predict(sentence))
    y_pred = []
    for xtest in x_test:
        result = language_detector.predict(xtest)
        y_pred.append(result)
    print(classification_report(y_test, y_pred, zero_division="warn",digits=4))


run("weibo", "全特么是《中国达人秀》节目组造假~~乌达木[泪] 赔！！！")  # 0.7487135506003431
run("twitter", "2022 is the end of the word")  # 0.8184523809523809
run("PHEME2", "2022 is the end of the word")  # 0.8152173913043478
