class Vanilla_NN(nn.Module):

    def __init__(self,  input_size, 
                        n_nodes_layer1 = None,
                        n_nodes_layer2 = None,
                        n_nodes_layer3 = None):
        '''
        Vanila Neural Network Object

        Parameters
        ----------
        input_size : Int
            length of input vector to network
        n_nodes_layer1 : Int, optional
            DESCRIPTION. The default is None.
        n_nodes_layer2 : TYPE, optional
            DESCRIPTION. The default is None.
        n_nodes_layer3 : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''

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
        '''
        Parameters
        ----------
        input : Tensor(N, 1)
            N number of features that will be the input to the NN

        Returns
        -------
        output : Tensor(1)
            NN's regression output

        '''
        # Feed Forward
        x = self.fc1(input)
        x = self.fc2(x)
        x = self.fc3(x)
        output = fc4(x)

        return output

    
    