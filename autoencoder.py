# Imports
import torch.nn.functional as F
import torch.nn as nn
import torch

class Autoencoder(nn.Module):

    def __init__(self, input_size  = 32,
                       n_nodes_fc1 = 128,
                       n_nodes_fc2 = 64,
                       embedding_size = 10):
        """Autoencoder for categorical embedding

        Args:
            input_size (int): size of input to be embedded
            n_nodes_fc1 (int, optional): Number of nodes in the first layer. Defaults to 128.
            n_nodes_fc2 (int, optional): Number of nodes in the second layer. Defaults to 64.
            embedding_size (int, optional): size of embedding vector. Defaults to 10.
        """

        super(Autoencoder, self).__init__()

        # Hyperparameters
        self.input_size = input_size
        self.n_nodes_fc1 = n_nodes_fc1
        self.n_nodes_fc2 = n_nodes_fc2
        self.embedding_size = embedding_size
        
        # Encoder
        self.encoder_fc1 = nn.Linear(self.input_size,  self.n_nodes_fc1)
        self.encoder_fc2 = nn.Linear(self.n_nodes_fc1, self.n_nodes_fc2)
        self.encoder_fc3 = nn.Linear(self.n_nodes_fc2, self.embedding_size)

        # Decoder
        self.decoder_fc1 = nn.Linear(self.embedding_size, self.n_nodes_fc2)
        self.decoder_fc2 = nn.Linear(self.n_nodes_fc2, self.n_nodes_fc1)
        self.decoder_fc3 = nn.Linear(self.n_nodes_fc1, self.input_size)

    def encode(self, x):
        x = F.relu(self.encoder_fc1(x))
        x = F.relu(self.encoder_fc2(x))
        x = F.relu(self.encoder_fc3(x))

        return x

    def decode(self, x):
        x = F.relu(self.decoder_fc1(x))
        x = F.relu(self.decoder_fc2(x))
        x = F.relu(self.decoder_fc3(x))

        return x

    def forward(self, x):

        x = self.encode(x)
        x = self.decode(x)
        return x

