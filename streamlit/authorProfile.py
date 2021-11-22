import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import re  # regular expression
import nltk
import pickle
import os

from nltk.corpus import stopwords
stopwords=set(stopwords.words('english'))

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

def loadfile():
    for dirname, _, filenames in os.walk('input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    data = pd.read_csv("models/blogtext.csv")
    print(data.head(10))
    print(data.isna().any())
    print(data.shape)
    data = data.head(10000)
    print(data.info())
    data.drop(['id', 'date'], axis=1, inplace=True)
    print(data.head())
    data['age'] = data['age'].astype('object')
    print(data.info())
    data['clean_data'] = data['text'].apply(lambda x: re.sub(r'[^A-Za-z]+', ' ', x))
    data['clean_data'] = data['clean_data'].apply(lambda x: x.lower())
    data['clean_data'] = data['clean_data'].apply(lambda x: x.strip())
    print("Actual data=======> {}".format(data['text'][1]))
    print("Cleaned data=======> {}".format(data['clean_data'][1]))
    data['clean_data'] = data['clean_data'].apply(
        lambda x: ' '.join([words for words in x.split() if words not in stopwords]))
    print(data['clean_data'][6])
    data['labels'] = data.apply(lambda col: [col['gender'], str(col['age']), col['topic'], col['sign']], axis=1)
    print(data.head())
    data = data[['clean_data', 'labels']]
    print(data.head())
    X = data['clean_data']
    print('===================')
    print(X)
    print('************')
    print(type(X))
    Y = data['labels']
    vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
    X = vectorizer.fit_transform(X)
    print('************')
    print(type(X))
    print(X[1])
    print('^^^^^^^^^^^^^^^^^^^^^^^^')
    print(vectorizer.get_feature_names()[:5])
    label_counts = dict()

    for labels in data.labels.values:
        for label in labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
    print(label_counts)
    binarizer = MultiLabelBinarizer(classes=sorted(label_counts.keys()))
    Y = binarizer.fit_transform(data.labels)
    print(X)
    print('########################')
    print(type(X))
    print('================')
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)
    model = LogisticRegression(solver='lbfgs')
    model = OneVsRestClassifier(model)
    model.fit(Xtrain, Ytrain)
    print(Xtest.shape)
    print(Ytest.shape)
    print(Xtest)
    Ypred = model.predict(Xtest)
    print(Ypred)
    filename = 'models/author_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    Ypred_inversed = binarizer.inverse_transform(Ypred)
    y_test_inversed = binarizer.inverse_transform(Ytest)
    for i in range(5):
        print('Text:\t{}\nTrue labels:\t{}\nPredicted labels:\t{}\n\n'.format(
            Xtest[i],
            ','.join(y_test_inversed[i]),
            ','.join(Ypred_inversed[i])
        ))
    print('Accuracy score: ', accuracy_score(Ytest, Ypred))
    print('F1 score: ', f1_score(Ytest, Ypred, average='micro'))
    print('Average precision score: ', average_precision_score(Ytest, Ypred, average='micro'))
    print('Average recall score: ', recall_score(Ytest, Ypred, average='micro'))

if __name__ == '__main__':
    print('Started  .. ')
    loadfile()
