import numpy as np
import pandas as pd
import re
import pickle

############################################## CODE FOR PREPROCESSING ##########################################################

'''model = {}
with open("glove.6B.50d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        model[word] = vector

print('Embeddings Done!')

def decontract(s):
  s = re.sub(r"won't", "will not", s)
  s = re.sub(r"can\'t", "can not", s)

  s = re.sub(r"n\'t", " not", s)
  s = re.sub(r"\'re", " are", s)
  s = re.sub(r"\'s", " is", s)
  s = re.sub(r"\'d", " would", s)
  s = re.sub(r"\'ll", " will", s)
  s = re.sub(r"\'t", " not", s)
  s = re.sub(r"\'ve", " have", s)
  s = re.sub(r"\'m", " am", s)

  return s

def preprocess(l):
  for i in range(len(l)):
    l[i] = re.sub(r'Message-ID(.*\n)*(X-FileName.*\n)', '', l[i])
    l[i] = decontract(l[i])
    l[i] = re.sub('\\n', ' ', l[i])
    l[i] = re.sub('\\t', ' ', l[i])
    l[i] = re.sub('\\r', ' ', l[i])
    l[i] = re.sub('\\"', ' ', l[i])
    l[i] = re.sub('[!#$%&;()*+,/:;.@-_<=>?[\\]^`{|}~]+', ' ', l[i])
  
    vec = np.zeros(50)
    cnt = 0
    for word in l[i].split():
      try:
        vec += model[word]
        cnt += 1
      except:
        pass
    if cnt > 0:
      vec /= cnt
    l[i] = vec
    print('\r {}'.format(i), end="")

  return l

data = pd.read_csv('./datasets/emails.csv')

labels = list(data['file'])

for i in range(len(labels)):
  labels[i] = labels[i][:labels[i].find('/')]

with open('labels_enron.txt','w') as o:
  for i in range(83600):
    o.write(labels[i]+'\n')

messages = preprocess(list(data['message'])[:83600])

print('Preprocessing Done!')

vecs = np.zeros((83600, 50))
for i in range(len(messages)):
  vecs[i] += messages[i]

with open('avgw2v_short.npy','wb') as f:
  np.save(f, vecs)'''

##############################################################################################################################################

vecs = ''
labels = []

###################### COMMENT THIS IF YOU ARE PRERPOCESSING #################################################################################
with open('./Embeddings/avgw2v_short.npy','rb') as f:
    vecs = np.load(f)

with open('./Embeddings/labels/labels_enron.txt','r') as o:
    while True:
        line = o.readline().rstrip('\n')
        if not line:
            break
        labels.append(line)
##############################################################################################################################################

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(vecs, labels, test_size=0.2, random_state=42)
model = ''

########################################### HYPERPARAMETER TESTING ###########################################################################

'''for i in range(5, 65, 5):
  model = DecisionTreeClassifier(max_depth=i)
  model.fit(X_train, y_train)
  pred = model.predict(X_test)
  print(accuracy_score(pred, y_test))'''

##############################################################################################################################################

########################################### CREATING THE MODEL(DECISION TREE) ################################################################

model = DecisionTreeClassifier(max_depth=60)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f1_score(y_test, pred, average='micro'))

##############################################################################################################################################

########################################### LOADING THE PRE-TRAINED DECISION TREE MODEL ######################################################

'''with open('./Models/DT_model_enron.sav','rb') as f:
  model = pickle.load(f)
pred = model.predict(X_test)
print(f1_score(y_test, pred, average='micro'))'''

##############################################################################################################################################

########################################### HYPERPARAMETER TESTING ###########################################################################

'''C = [1, 10, 50, 100, 1000]
for i in C:
  model = SVC(C=i)
  model.fit(X_train, y_train)
  pred = model.predict(X_test)
  print(accuracy_score(pred, y_test))'''

##############################################################################################################################################

########################################### CREATING THE MODEL(SVM) ##########################################################################

model = SVC(C=100)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f1_score(y_test, pred, average='micro'))

##############################################################################################################################################

########################################### LOADING THE PRE-TRAINED SVM MODEL ################################################################

'''with open('./Models/SVM_model_enron.sav','rb') as f:
  model = pickle.load(f)
pred = model.predict(X_test)
print(f1_score(y_test, pred, average='micro'))'''

##############################################################################################################################################