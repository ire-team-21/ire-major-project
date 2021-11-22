from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torchtext.legacy import data

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

def train(model, iterator, optimizer, criterion, stylo = False):
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    model.train()
    stylo_features = None 

    for batch in iterator:        
        optimizer.zero_grad()
        if(stylo):
            stylo_features = get_stylo_features(batch.texts)

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
            if(stylo):
                stylo_features = get_stylo_features(batch.texts)

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
