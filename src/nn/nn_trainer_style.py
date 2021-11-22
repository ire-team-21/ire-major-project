# -*- coding: utf-8 -*-
"""nn_trainer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BLjUX9cA3ucmq4G-VaSukmxpW8arq8z4
"""

import torch
import time
import random
import spacy
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings

from torchtext.legacy import data
from torchtext.legacy import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle 
warnings.filterwarnings("ignore")

from copy import deepcopy

import pandas as pd 
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import string

punctuations = string.punctuation

def get_vectorizer(train_texts, v_type, n, max_features):
    count_vec = CountVectorizer(analyzer="word", ngram_range=(n, n), max_features=max_features, stop_words = 'english')
    count_vec.fit(train_texts)
    return count_vec

def ngrams(texts, vectorizer, n):
    features = vectorizer.transform(texts)
    return features.toarray()

def get_texts_from_batch(batch_texts, itos, batch_first = False):
    if(not(batch_first)):
      batch_texts = batch_texts.view(batch_texts.shape[1], batch_texts.shape[0])
    words = [[itos[i] for i in text if itos[i]!='<pad>'] for text in batch_texts]
    return words

def get_stylo_features(batch_texts, itos, batch_first = True):
    texts = get_texts_from_batch(batch_texts, itos, batch_first)
    features = []
    for words in texts:            
        avg_word_length = np.mean([len(word) for word in words])
        total_short_words = len([word for word in words if len(word) < 4])/(len(words)+1)
        complexity = len(set(words))/(len(words)+1)
        punctuation_ratio = len([i for word in words for i in word if i in punctuations])/(len([i for word in words for i in word])+1)
        uppercase_ratio = len([i for word in words for i in word if i.isupper()])/(len([i for word in words for i in word])+1)
        
        features.append([avg_word_length, total_short_words, complexity, punctuation_ratio, uppercase_ratio])
        
    features = np.array(features, dtype=np.float)
    features = features.reshape(len(texts), 5)
    features = torch.from_numpy(features)
    features = features.to(device)
    return features

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def logger(text):
  log_file = open('nn/logs_style.txt', 'a')
  log_file.write(text)
  log_file.close()

EMBEDDING_DIM = 50
N_FILTERS = 100
FILTER_SIZES = [2,3,4]
DROPOUT = 0.5

class CNN(nn.Module):
    def __init__(self, vocab_size=None, embedding_dim=EMBEDDING_DIM, n_filters=N_FILTERS, filter_sizes=FILTER_SIZES, output_dim=None, 
                 dropout = DROPOUT, stylo_features = 0):
        
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear((len(filter_sizes) * n_filters)+stylo_features, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, stylo_features = None):        
        text = text.permute(1, 0)        
        embedded = self.embedding(text)        
        embedded = embedded.unsqueeze(1)        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        output = self.dropout(torch.cat(pooled, dim = 1))
        if(type(stylo_features)!=type(None)):
          output = torch.cat((output, stylo_features), dim = 1)
          output = output.float()
        output = self.fc(output)
        return output

EMBEDDING_DIM = 50
N_LAYERS = 2
HIDDEN_UNITS = 50
BIDIRECTIONAL = True
DROPOUT = 0.5

class RNN(nn.Module):
    def __init__(self, rnn_type = None, vocab_size=None, embedding_dim=EMBEDDING_DIM, n_layers=N_LAYERS, hidden_units=HIDDEN_UNITS, bidirectional=BIDIRECTIONAL, 
                 output_dim=None, dropout=DROPOUT, stylo_features = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if(rnn_type == 'lstm'):
            self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_units, num_layers=n_layers, dropout=dropout, bidirectional = bidirectional, batch_first = True)
        elif(rnn_type == 'gru'):
            self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_units, num_layers=n_layers, dropout=dropout, bidirectional = bidirectional, batch_first = True)
        self.fc = nn.Linear(hidden_units*2+stylo_features, output_dim)
    
    def forward(self, text, stylo_features=None):        
        emb = self.embedding(text)
        output, _ = self.rnn(emb)
        output = torch.mean(output, dim=1)
        if(type(stylo_features)!=type(None)):
          output = torch.cat((output, stylo_features), dim = 1)
          output = output.float()
        output = self.fc(output)
        return output

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def compute_metrics(pred, labels = None):
    if(labels == None):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
    else:
        preds = pred.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train(model, iterator, optimizer, criterion, stylo = None):
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    model.train()
    stylo_features = None 

    for batch in iterator:        
        optimizer.zero_grad()
        if(stylo !=  None):
            stylo_features = get_stylo_features(batch.texts, stylo[0], stylo[1])

        predictions = model(batch.texts, stylo_features)
        loss = criterion(predictions, batch.labels)
        d = compute_metrics(predictions.cpu(), batch.labels.cpu())
        acc = d['accuracy']
        f1 = d['f1']
        loss.backward()
        
        optimizer.step()        

        epoch_loss += loss
        epoch_acc += acc
        epoch_f1 += f1

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)

def evaluate(model, iterator, criterion, stylo = False):
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    epoch_recall = 0
    epoch_precision = 0 

    stylo_features = None 

    model.eval()  
    with torch.no_grad():
        for batch in iterator:
            if(stylo != None):
                stylo_features = get_stylo_features(batch.texts, stylo[0], stylo[1])

            predictions = model(batch.texts, stylo_features)
            loss = criterion(predictions, batch.labels)
            d = compute_metrics(predictions.cpu(), batch.labels.cpu())            
            acc = d['accuracy']
            f1 = d['f1']
            recall = d['recall']
            precision = d['precision']
            epoch_loss += loss
            epoch_acc += acc
            epoch_f1 += f1
            epoch_recall += recall
            epoch_precision += precision
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator), epoch_recall / len(iterator), epoch_precision / len(iterator)

def grid_search(model_type, pretrained_embeddings, input_dim, output_dim, train_data, val_data, style_utils = None):
  best_model = None 
  best_loss = np.inf 
  best_f1 = 0
  best_prec = 0
  best_rec = 0
  best_acc = 0
  best_bs = 0
  best_lr = 0
  if(type(style_utils) != type(None)):
    stylo_features = 5 
  else:
    stylo_features = 5 

  for batch_size in [16, 32]:
    train_iter, val_iter = data.BucketIterator.splits(
                            (train_data, val_data), batch_sizes=(batch_size, 16),
                            sort_key=lambda x: len(x.texts), device=device, )
    for lr in [1e-4, 5e-4, 1e-3, 5e-3]:
      previous_loss = np.inf
      print('*'*80)
      logger('*'*80+'\n')
      print(f"BATCH SIZE - {batch_size}, LEARNING RATE - {lr}")
      logger(f"BATCH SIZE - {batch_size}, LEARNING RATE - {lr}'\n")
      print('*'*80)
      logger('*'*80+'\n')

      if(model_type == 'cnn'):
        model = CNN(vocab_size = input_dim, output_dim=output_dim, stylo_features = stylo_features)
      else:
        model = RNN(rnn_type = model_type, vocab_size = input_dim, output_dim = output_dim, stylo_features=stylo_features)

      model.embedding.weight.data.copy_(pretrained_embeddings)
      optimizer = optim.AdamW(model.parameters(), lr=lr)
      criterion = nn.CrossEntropyLoss()
      model = model.to(device)
      criterion = criterion.to(device)
      for epoch in range(1):
          start_time = time.time()
          
          train_loss, train_acc, train_f1 = train(model, train_iter, optimizer, criterion, style_utils)
          valid_loss, valid_acc, valid_f1, valid_prec, valid_rec = evaluate(model, val_iter, criterion, style_utils)

          if(valid_f1 > best_f1):
            best_model = deepcopy(model.state_dict())
            best_loss = valid_loss
            best_bs = batch_size
            best_f1 = valid_f1
            best_lr = lr
            best_acc = valid_f1
            best_prec = valid_prec
            best_rec = valid_rec 

          if(valid_loss > previous_loss):
            print("EARLY STOPPING")
            break 

          end_time = time.time()
          epoch_mins, epoch_secs = epoch_time(start_time, end_time)
          previous_loss = valid_loss
          print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
          logger(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n')
          print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1}')
          logger(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1}\n')
          print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | Val. F1: {valid_f1} | Val. Prec: {valid_prec} | Val. Rec.: {valid_rec}')
          logger(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | Val. F1: {valid_f1} | Val. Prec: {valid_prec} | Val. Rec.: {valid_rec}\n')

  
  return best_model, best_f1, best_prec, best_rec, best_acc, best_lr, best_bs

dataset_num_labels = [('news',25), ('enron', 15), ('spooky', 3)]

all = []

for dataset, num_labels in dataset_num_labels:
  for model_type in ['cnn', 'lstm', 'gru']:
    if(model_type in ['lstm',  'gru']):
      batch_first = True
    else:
      batch_first = False
              
    TEXT = data.Field(tokenize = 'spacy', tokenizer_language = 'en_core_web_sm', batch_first = batch_first)
    LABEL = data.LabelField(sequential=False, use_vocab = False)

    train_data, val_data, test_data = data.TabularDataset.splits(
      path=f'./{dataset}', train='train.csv', validation = 'val.csv', test = 'test.csv', format='csv',
      fields=[('texts', TEXT), ('labels', LABEL)], skip_header=True)
    TEXT.build_vocab(train_data, vectors = "glove.6B.50d", unk_init = torch.Tensor.normal_)

    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    pretrained_embeddings = TEXT.vocab.vectors
    pretrained_embeddings[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    pretrained_embeddings[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    style_utils = (TEXT.vocab.itos, batch_first)
    best = grid_search(model_type, pretrained_embeddings, len(TEXT.vocab), num_labels, train_data, val_data, style_utils)
    print(dataset, model_type, best[1:])
    logger(" ".join([dataset, model_type]+[str(i) for i in best[1:]])+'\n')
    
    train_iter, test_iter = data.BucketIterator.splits(
                            (train_data, test_data), batch_sizes=(16, 64),
                            sort_key=lambda x: len(x.texts), device=device, )
    
    if(model_type == 'cnn'):
        model = CNN(vocab_size = len(TEXT.vocab), output_dim=num_labels, stylo_features=5)
    else:
      model = RNN(rnn_type = model_type, vocab_size = len(TEXT.vocab), output_dim = num_labels, stylo_features=5)

    model.load_state_dict(best[0])
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    criterion.to(device)

    test_all = evaluate(model, test_iter, criterion, style_utils)
    print(test_all)
    logger(" ".join([str(i) for i in test_all])+'\n')
    all.append(test_all)
    torch.save({'model_state_dict': best[0]}, f'{dataset}_{model_type}_stylo.pth')

import pickle 
final_results = open('results_style.b', 'wb')
pickle.dump(all, final_results)