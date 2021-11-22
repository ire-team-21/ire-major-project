import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 50
N_LAYERS = 2
HIDDEN_UNITS = 50
BIDIRECTIONAL = True
DROPOUT = 0
VOCAB = 58361
LABELS = 25

class RNN(nn.Module):
    def __init__(self, rnn = None, vocab_size=VOCAB, embedding_dim=EMBEDDING_DIM, n_layers=N_LAYERS, hidden_units=HIDDEN_UNITS, bidirectional=BIDIRECTIONAL, output_dim=LABELS, dropout=DROPOUT, stylo_features = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if(rnn == 'lstm'):
            self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_units, num_layers=n_layers, dropout=dropout, bidirectional = bidirectional, batch_first = True)
        elif(rnn == 'gru'):
            self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_units, num_layers=n_layers, dropout=dropout, bidirectional = bidirectional, batch_first = True)
        self.fc = nn.Linear(hidden_units*2+stylo_features, output_dim)
    
    def forward(self, text, stylo_features=None):        
        emb = self.embedding(text)
        output, _ = self.rnn(emb)
        output = torch.mean(output, dim=1)
        if(stylo_features):
            output = torch.cat((output, stylo_features))
        output = self.fc(output)
        return output
