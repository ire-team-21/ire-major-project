import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

lr_clf = nb_clf = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=(1, 3), max_features = 2500, stop_words = 'english')), ('clf', MultinomialNB())])
svm_clf = Pipeline([('vect', CountVectorizer(analyzer='word', ngram_range=(1, 3), max_features = 2500, stop_words = 'english')), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])

dataset_num_labels = [('news',25), ('enron', 15), ('spooky', 3)]

for dataset, num_labels in dataset_num_labels:
  train_df = pd.read_csv(f'{dataset}/train.csv')
  test_df = pd.read_csv(f'{dataset}/test.csv')
  val_df = pd.read_csv(f'{dataset}/val.csv')

  train_texts = train_df.texts.tolist()
  train_labels = train_df.labels.tolist()
  val_texts = val_df.texts.tolist()
  val_labels = val_df.labels.tolist()
  test_texts = test_df.texts.tolist()
  test_labels = test_df.labels.tolist()

  lr_clf.fit(train_texts, train_labels)
  svm_clf.fit(train_texts, train_labels)

  lr_pred = nb_clf.predict(test_texts)
  svm_pred = svm_clf.predict(test_texts)
  print(f'{dataset} NB - ', precision_recall_fscore_support(test_labels, lr_pred, average = 'weighted'))
  print(f'{dataset} SVM - ', precision_recall_fscore_support(test_labels, svm_pred, average = 'weighted'))

lr_clf = nb_clf = Pipeline([('vect', CountVectorizer(analyzer='char', ngram_range=(2, 4), max_features = 70, stop_words = 'english')), ('clf', MultinomialNB())])
svm_clf = Pipeline([('vect', CountVectorizer(analyzer='char', ngram_range=(2, 4), max_features = 70, stop_words = 'english')), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])

for dataset, num_labels in dataset_num_labels:
  train_df = pd.read_csv(f'{dataset}/train.csv')
  test_df = pd.read_csv(f'{dataset}/test.csv')
  val_df = pd.read_csv(f'{dataset}/val.csv')

  train_texts = train_df.texts.tolist()
  train_labels = train_df.labels.tolist()
  val_texts = val_df.texts.tolist()
  val_labels = val_df.labels.tolist()
  test_texts = test_df.texts.tolist()
  test_labels = test_df.labels.tolist()

  lr_clf.fit(train_texts, train_labels)
  svm_clf.fit(train_texts, train_labels)

  lr_pred = nb_clf.predict(test_texts)
  svm_pred = svm_clf.predict(test_texts)
  print(f'{dataset} NB - ', precision_recall_fscore_support(test_labels, lr_pred, average = 'weighted'))
  print(f'{dataset} SVM - ', precision_recall_fscore_support(test_labels, svm_pred, average = 'weighted'))

lr_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
svm_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])

for dataset, num_labels in dataset_num_labels:
  train_df = pd.read_csv(f'{dataset}/train.csv')
  test_df = pd.read_csv(f'{dataset}/test.csv')
  val_df = pd.read_csv(f'{dataset}/val.csv')

  train_texts = train_df.texts.tolist()
  train_labels = train_df.labels.tolist()
  val_texts = val_df.texts.tolist()
  val_labels = val_df.labels.tolist()
  test_texts = test_df.texts.tolist()
  test_labels = test_df.labels.tolist()

  lr_clf.fit(train_texts, train_labels)
  svm_clf.fit(train_texts, train_labels)

  lr_pred = nb_clf.predict(test_texts)
  svm_pred = svm_clf.predict(test_texts)
  print(f'{dataset} NB - ', precision_recall_fscore_support(test_labels, lr_pred, average = 'weighted'))
  print(f'{dataset} SVM - ', precision_recall_fscore_support(test_labels, svm_pred, average = 'weighted'))