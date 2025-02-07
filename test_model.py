from pythainlp.corpus.common import thai_stopwords
from pythainlp import word_tokenize
import pandas as pd


def text_process(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ"))
    final = word_tokenize(final)
    final = " ".join(word for word in final)
    final = " ".join(word for word in final.split() 
                     if word.lower not in thai_stopwords)
    return final


df = pd.read_csv("sentence.csv")
thai_stopwords = list(thai_stopwords())
df['categorize'].value_counts().plot.bar()

