import pandas as pd
import datetime
import numpy as np
import string
from collections import Counter
from reader import split_data

data = pd.read_csv('SmsCollection.csv', delimiter = ';', usecols=[0,1])

def text_transform(text):
    if isinstance(text, str):
        text = text.translate(str.maketrans('','',string.punctuation))
        text = [word for word in text.lower().split(" ") if word.isalpha()]
        return text

data.text = data.text.apply(text_transform)    

cnt = Counter()
for i in data.text:
    try:
        for word in i:
            cnt[word] += 1
    except TypeError:
        pass
print(cnt)