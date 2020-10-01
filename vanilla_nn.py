# Imports
import torch
import torch.nn as nn
from math import ceil
import torch.nn.functional as F

# Model
class Vanilla_NN(nn.Module):

    def __init__(self,  input_size, 
                        n_nodes_layer1 = None,
                        n_nodes_layer2 = None,
                        n_nodes_layer3 = None):
        """Simple 3 hidden layer deep neural network.

        Args:
            input_size (int): Length of input vector
            n_nodes_layer1 (int, optional): Number of nodes in first hidden layer. Defaults to None.
            n_nodes_layer2 (int, optional): Number of nodes in second hidden layer. Defaults to None.
            n_nodes_layer3 (int, optional): Number  of nodes in third hidden layer. Defaults to None.
        """

        super(Vanilla_NN,self).__init__()
        
        # Hyperparameters
        self.input_size = input_size
        self.n_nodes_layer1 = n_nodes_layer1 if n_nodes_layer1 is not None else ceil(2 * self.input_size)
        self.n_nodes_layer2 = n_nodes_layer2 if n_nodes_layer2 is not None else ceil(3 * self.input_size)
        self.n_nodes_layer3 = n_nodes_layer3 if n_nodes_layer3 is not None else ceil(2 * self.input_size)

        # Layer Declarations
        self.fc1 = nn.Linear(self.input_size,     self.n_nodes_layer1)
        self.fc2 = nn.Linear(self.n_nodes_layer1, self.n_nodes_layer2)
        self.fc3 = nn.Linear(self.n_nodes_layer2, self.n_nodes_layer3)
        self.fc4 = nn.Linear(self.n_nodes_layer3, 1)

    def forward(self, input):
        """Feedforward pass through the network

        Args:
            input (Tensor[input_size, 1]): Input vector to the network

        Returns:
            float: Regression output
        """
        # Feed Forward
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = self.fc4(x)

        return output

    
    