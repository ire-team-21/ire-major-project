from copy import deepcopy 
import warnings
import argparse
from transformers import Trainer, TrainingArguments
from train_utils import compute_metrics
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, EarlyStoppingCallback
from datasets import Dataset
import pandas as pd 
import torch

warnings.filterwarnings("ignore")

class customDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def tokenize_function(examples):
        return tokenizer(examples["text"])

for model_name in ['bert-base-cased','roberta-base','distilbert-base-cased']:
    for dataset, num_labels in [('email', 15),('spooky',3),('news', 25)]:
        name = model_name+dataset
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_df = pd.read_csv(f'{dataset}/train.csv')
        val_df = pd.read_csv(f'{dataset}/val.csv')
        test_df = pd.read_csv(f'{dataset}/test.csv')

        train_texts = train_df['texts'].tolist()
        val_texts = val_df['texts'].tolist()
        test_texts = test_df['texts'].tolist()

        train_labels = train_df['labels'].tolist()
        val_labels = val_df['labels'].tolist()
        test_labels = test_df['labels'].tolist()

        train_encodings = tokenizer(train_texts, truncation=True, padding = True)
        test_encodings = tokenizer(test_texts, truncation=True, padding = True)
        val_encodings = tokenizer(val_texts, truncation=True, padding = True)

        train_dataset = customDataset(train_encodings, train_labels)
        test_dataset = customDataset(test_encodings, test_labels)
        val_dataset = customDataset(val_encodings, val_labels)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = num_labels)

        training_args = TrainingArguments(
                        output_dir=f'./{name}_{model_name}_results',          
                        num_train_epochs=15,              
                        per_device_train_batch_size=16,  
                        per_device_eval_batch_size=32,   
                        warmup_steps=100,                
                        weight_decay=0.01,               
                        logging_dir=f'./{name}_{model_name}_logs',            
                        logging_steps=1000,
                        save_strategy = 'epoch',
                        evaluation_strategy="epoch",
                        load_best_model_at_end=True,
                        save_total_limit=1
                        )

        trainer = Trainer(
                model=model,                         
                args=training_args,                  
                train_dataset=train_dataset,         
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback]
                )

        trainer.train()
        trainer.evaluate()

        _, _, results = trainer.predict(test_dataset)
        print(results)
