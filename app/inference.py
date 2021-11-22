from cnn import CNN 
from rnn import RNN 
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from collections import defaultdict
import numpy as np 
import torch 
import spacy 
import pickle 
import en_core_web_sm
from torch.nn.functional import softmax 

nlp = en_core_web_sm.load()
stoi = pickle.load(open("app/stoi.pk", "rb"))
id2label = pickle.load(open('app/id2label.b', 'rb'))

root = "Models/"
def get_model(model_type):
    if(model_type == 'cnn'):
        model = CNN()
        weights = torch.load(f'{root}/news_cnn.pth', map_location=torch.device('cpu'))['model_state_dict']
        model.load_state_dict(weights)
    elif(model_type in ['lstm', 'gru']):
        model = RNN(rnn=model_type)
        weights = torch.load(f'{root}/news_{model_type}.pth', map_location=torch.device('cpu'))['model_state_dict']
        model.load_state_dict(weights)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(f'{root}/news_{model_type}')

    return model 

def get_inp(text, model_type):
    text_tokens = [str(i) for i in nlp(text) if str(i) in stoi]
    token_idx = [stoi[token] for token in text_tokens]
    out = torch.tensor(token_idx)
    if(model_type == 'cnn'):
        out = out.reshape(len(token_idx), 1)
    else:
        out = out.reshape(1, len(token_idx))
    return out 

def nn_inference(model_type, text):
    model = get_model(model_type)
    inp = get_inp(text, model_type)
    out = model(inp)
    scores = softmax(out).reshape(25).detach().numpy()
    top_3 = np.argsort(scores)[::-1][:3]
    pairs = [(id2label[i], np.round(scores[i], decimals=2)) for i in top_3]
    pairs_dict = {'Author':[], 'Confidence':[]}
    for i in pairs:
        pairs_dict['Author'].append(i[0])
        pairs_dict['Confidence'].append(i[1])
    return pairs_dict

def transformer_inference(model_type, text):
    text = text[:500]
    model = get_model(model_type)
    tokenizer = AutoTokenizer.from_pretrained(f'{root}/{model_type}_tokenizer')
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
    out = pipeline(text)[0]
    top_3_scores = []
    for o in out:
        top_3_scores.append((int(o['label'][6:]), o['score']))
    top_3_scores.sort(key = lambda x: x[1], reverse=True)
    pairs = [(id2label[i[0]], np.round(i[1], decimals=2)) for i in top_3_scores[:3]]
    pairs_dict = {'Author':[], 'Confidence':[]}
    for i in pairs:
        pairs_dict['Author'].append(i[0])
        pairs_dict['Confidence'].append(i[1])
    return pairs_dict