import pandas as pd 
import numpy as np 
import pickle 

def get_emails():
    data = pd.read_csv("../../data/original/emails.csv")
    data.rename(columns={"from":"author"}, inplace = True)

    data["text"].replace('', np.nan, inplace = True)
    data.dropna(inplace = True)

    data["length"] = data["text"].apply(lambda x: len(x.split()))
    data = data.drop(data[data.length > 500].index)
    data = data.drop(data[data.length < 10].index)

    items = data.author.value_counts().to_dict().items()
    data = data[data.author.isin([key for key, val in items if val > 2000])]

    texts = data.text.tolist()
    labels = data.author.tolist()
    label2id = {i: idx for (idx, i) in enumerate(sorted(set(labels)))}
    id2label = {label2id[i]: i for i in label2id}
    labels = [label2id[i] for i in labels]

    d = {'texts':texts, 'labels':labels}
    new_data = pd.DataFrame.from_dict(d)
    new_data.to_csv('../../data/pruned/emails.csv')
    pickle.dump(id2label, open('../../data/labels/email_labels.b', 'wb'))

def get_wapo():
    data = pd.read_csv("../../data/original/wapo.csv")
    data["article"].replace('', np.nan, inplace = True)
    data.dropna(inplace = True)

    items = data.author.value_counts().to_dict().items()
    data = data[data.author.isin([key for key, val in items if val > 99])]

    texts = data.article.tolist()
    labels = data.author.tolist()

    label2id = {i: idx for (idx, i) in enumerate(sorted(set(labels)))}
    id2label = {label2id[i]: i for i in label2id}

    labels = [label2id[i] for i in labels]

    d = {'texts':texts, 'labels':labels}
    new_data = pd.DataFrame.from_dict(d)
    new_data.to_csv('../data/pruned/wapo.csv')
    pickle.dump(id2label, open('../../data/labels/wapo_labels.b', 'wb'))
    
def get_spooky():
    data = pd.read_csv("../../data/original/spooky.csv")

    data["text"].replace('', np.nan, inplace = True)
    data.dropna(inplace = True)

    texts = data.text.tolist()
    labels = data.author.tolist()
    label2id = {i: idx for (idx, i) in enumerate(sorted(set(labels)))}
    id2label = {label2id[i]: i for i in label2id}
    labels = [label2id[i] for i in labels]

    d = {'texts':texts, 'labels':labels}
    new_data = pd.DataFrame.from_dict(d)
    new_data.to_csv('../../data/pruned/spooky.csv')
    pickle.dump(id2label, open('../../data/labels/spooky_labels.b', 'wb'))

if __name__=="__main__":
    get_wapo()
    get_emails()
    get_spooky()
