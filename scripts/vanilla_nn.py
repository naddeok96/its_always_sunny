# Imports
from torch import nn

# Model
class Vanilla_NN(nn.Module):

    def __init__(self,  input_size, 
                        n_nodes_layer1 = None,
                        n_nodes_layer2 = None,
                        n_nodes_layer3 = None):
        """
        Simple 3 hidden layer neural network.

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
        self.fc1 = nn.Linear(input_size, n_nodes_layer1)
        self.fc2 = nn.Linear(n_nodes_layer1, n_nodes_layer2)
        self.fc3 = nn.Linear(n_nodes_layer2, n_nodes_layer3)
        self.fc4 = nn.Linear(n_nodes_layer3, 1)

    def forward(self, input):
        """
        Feedforward pass through the network

        Args:
            input (Tensor[input_size, 1]): Input vector to the network

        Returns:
            float: Regression output
        """
        # Feed Forward
        x = self.fc1(input)
        x = self.fc2(x)
        x = self.fc3(x)
        output = fc4(x)

        return output

    
    