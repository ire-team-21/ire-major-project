import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 50
DROPOUT = 0.5
N_FILTERS = 100
FILTER_SIZES = [2,3,4]
DROPOUT = 0.5

class CNN(nn.Module):
    def __init__(self, vocab_size = None, embedding_dim = EMBEDDING_DIM, pretrained_weights = None, n_filters=N_FILTERS, filter_sizes=FILTER_SIZES, output_dim=None, dropout=DROPOUT, stylo_features = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim).from_pretrained(pretrained_weights, freeze=False)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes)*n_filters+stylo_features, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, stylo_features = None):        
        text = text.permute(1, 0)        
        embedded = self.embedding(text)        
        embedded = embedded.unsqueeze(1)        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        if(stylo_features):
            cat = torch.cat((cat, stylo_features))
        return self.fc(cat)