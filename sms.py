import pandas as pd
import datetime
import numpy as np
import string
from collections import Counter
from reader import split_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv('SmsCollection.csv', delimiter = ';', usecols=[0,1])
data['label'] = data.label.map({'ham': 0, 'spam': 1})

def text_transform(text):
    if isinstance(text, str):
        text = text.translate(str.maketrans('','',string.punctuation))
        text = " ".join([word for word in text.lower().split(" ") if word.isalpha()])
        return text

data.text = data.text.apply(text_transform)    

data = data[data['text'].notnull()]

train, test = split_data(data)

count_vect = CountVectorizer()
word_counts = count_vect.fit_transform(train.text)

clf = MultinomialNB().fit(word_counts, train['label'])

new_counts = count_vect.transform(test.text)
predicted = clf.predict(new_counts)

print("Accuracy: ", np.sum(np.array(test['label']) == np.array(predicted))/len(predicted))