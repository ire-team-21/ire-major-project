from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split 

def compute_metrics(pred, labels = None):
    if(not(labels)):
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

def get_train_val_test(texts, labels):
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=.15, random_state = 42)
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.15, random_state = 42)
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
