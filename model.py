import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 3
        vocab_size = len(dataset.unique_words)

        # The vocab size indicates the number of emmbeddings
        # The embedding dimension indicates the dimension of each embedding that is stored
        # But how does it know what embedding to store? Is this learned?
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(input_size = self.lstm_size,
                            hidden_size = self.lstm_size,
                            num_layers = 3,
                            dropout=0.2)
        self.fc = nn.Linear(self.lstm_size, vocab_size)
    def forward(self, x, prev_state):
        ''' The forward functions defines the structure of the model. That is, what transformations are being run
        Parameters:
        -  x : A one dimensional tensor containing the indeces for the word embeddings
        - prev_state: state_h, state_c tuple
        '''
        embedding = self.embedding(x)
        output, state = self.lstm(embedding, prev_state)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))
