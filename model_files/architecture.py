import torch
from torch import nn

class CharRNN(nn.Module):
    
    def __init__(self, tokens, n_hidden=512, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        ## TODO: define the layers of the model
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                           dropout=drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(p=.5)
        self.fc1 = nn.Linear(n_hidden, 256)
        self.fc2 = nn.Linear(256, len(self.chars))
    
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
                
        ## TODO: Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        # stack up the LSTM outputs
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc1(out)
        out = self.fc2(out)
        # return the final output and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
        return hidden


def load_model():
    with open('/home/martineliteai/flask/flask_story_maker/model_files/rnn_20_epoch.net', 'rb') as f:
        checkpoint = torch.load(f)
    
    model = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model