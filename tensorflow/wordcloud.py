import jieba
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import numpy as np

font = 'C:\Windows\Fonts\FZSTK.TTF'  # 电脑自带的字体

path = "weibo.txt"


def tcg(texts):
    cut = jieba.cut(texts)  # 分词
    string = ' '.join(cut)
    return string


text = (open(path, 'r', encoding='utf-8')).read()
string = tcg(text)

wc = WordCloud(
    background_color='white',
    width=1000,
    height=800,
    mask=img_array,  # 设置背景图片
    font_path=font,
    stopwords=stopword
)
