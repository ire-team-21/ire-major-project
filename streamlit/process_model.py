import torch
import pickle
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, TextClassificationPipeline


def load_enron_models():
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=15)
    model.load_state_dict(torch.load('./models/transformer_model_enron.bin', map_location=torch.device('cpu')))
    return model.eval()

def predict_enron_models(model, input):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    return pipeline(input)


def load_spooky_model():
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels = 3)
    model.load_state_dict(torch.load('./models/transformer_model_spooky.bin', map_location=torch.device('cpu')))
    return model.eval()

def predict_spooky_model(model, input):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    return pipeline(input)


def load_wapo_model():
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=25)
    model.load_state_dict(torch.load('./models/transformer_model_wapo.bin', map_location=torch.device('cpu')))
    return model.eval()

def predict_wapo_model(model, input):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    return pipeline(input)


def unpickle_spooky_dictionary():
    file_to_read = open("models/id2label_spooky.b", "rb")
    return pickle.load(file_to_read)

def unpickle_enron_dictionary():
    file_to_read = open("models/id2label_enron.b", "rb")
    return pickle.load(file_to_read)

def unpickle_wapo_dictionary():
    file_to_read = open("models/id2label_wapo.b", "rb")
    return pickle.load(file_to_read)



if __name__ == "__main__":
    input = 'This process, however, afforded me no means of ascertaining the dimensions of my dungeon; as I might make its circuit'

    ##  enron_model = load_enron_models()
    ## prediction = predict_enron_models(enron_model, input)
    ## enron_label_dict = unpickle_enron_dictionary()

    # enron_model = load_wapo_model()
    # prediction = predict_wapo_model(enron_model, input)
    # wapo_label_dict = unpickle_wapo_dictionary()

    enron_model = load_spooky_model()
    prediction = predict_spooky_model(enron_model, input)
    spooky_label_dict = unpickle_spooky_dictionary()

    print('prediction', prediction[0]['label'])
    index = prediction[0]['label']
    num = int(index.replace('LABEL_', ''))
    print(num)
    print('author', spooky_label_dict[num])
