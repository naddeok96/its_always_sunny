# Imports
import torch
import math
import copy
import wandb
from sleeper import Sleeper
# sleeper = Sleeper()


class Academy:
    def __init__(self,
                 net, 
                 data,
                 gpu = False,
                 autoencoder_trainer = False,
                 wandb_on = False,
                 wandb_sweep = False):
        """
        Trains and test models on datasets

        Args:
            net (Pytorch model class):  Network to train and test
            data (Tensor Object):       Data to train and test with
            gpu (bool, optional):       If True then GPU's will be used. Defaults to False.
            autoencoder_trainer (bool, optional): If True labels will equal inputs. Defaults to False.
        """
        super(Academy,self).__init__()

        # Declare GPU usage
        self.gpu = gpu

        # Push net to CPU or GPU
        self.net = net if self.gpu == False else net.cuda()
        
        # Declare data
        self.data = data

        # Declare if training is for an autoencoder
        self.autoencoder_trainer = autoencoder_trainer

        # Declare if Weights and Biases are used
        self.wandb_on = wandb_on
        self.wandb_sweep = wandb_sweep

        # Hyperparameter Sweep
        if self.wandb_on and self.wandb_sweep: 
            sweep_config = {
                        'method': 'random', #grid, random
                        'metric': {
                        'name': 'loss',
                        'goal': 'minimize'   
                        },
                        'parameters': {
                            'epochs': {
                                'values': [100, 500, 1000]
                            },
                            'batch_size': {
                                'values': [256, 128, 64, 32]
                            },
                            'n_nodes_fc1' : {
                                'values': [64, 128, 256]
                            },
                            'n_nodes_fc2' : {
                                'values'  : [32, 64, 128]
                            },
                            'embedding_size': {
                                'values': [8, 16, 32]
                            },
                            'momentum': {
                                'values': [0, 0.85, 0.9, 0.95]
                            },
                            'weight_decay': {
                                'values': [0, 0.0005, 0.005, 0.05]
                            },
                            'learning_rate': {
                                'values': [1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5]
                            },
                            'optimizer': {
                                'values': ['adam', 'sgd', 'rmsprop']
                            }
                        }
                    }
            sweep_id = wandb.sweep(sweep_config, entity="sweep", project="AutoEnc")
            wandb.agent(sweep_id, train)

    def train(self, batch_size = 124, 
                    n_epochs = 1, 
                    learning_rate = 1e-12, 
                    momentum = 0.1, 
                    weight_decay = 0.0001):
        """Fits model to data

        Args:
            batch_size (int, optional):         Number of samples in each batch. Defaults to 124.
            n_epochs (int, optional):           Number of cycles through the data. Defaults to 1.
            learning_rate (float, optional):    Learning rate coeffiecent in the optimizer. Defaults to 0.001.
            momentum (float, optional):         Momentum coeffiecent in the optimizer. Defaults to 0.9.
            weight_decay (float, optional):     Weight decay in the optimizer. Defaults to 0.0001.
        """

        #Get training data
        train_loader = self.data.get_train_loader(batch_size)

        #Create optimizer and loss functions
        optimizer = torch.optim.SGD(self.net.parameters(), 
                                    lr = learning_rate, 
                                    momentum = momentum,
                                    weight_decay = weight_decay)
        criterion = torch.nn.MSELoss()

        # Weights and Biases Setup
        if self.wandb_on:
            config_defaults = {
                    'epochs': n_epochs,
                    'batch_size': batch_size,
                    'n_nodes_fc1': 128,
                    'n_nodes_fc2': 64,
                    'embedding_size' : 8,
                    'momentum'   : momentum,
                    'weight_decay' : weight_decay,
                    'learning_rate': learning_rate,
                    'optimizer': 'sgd'
                }
            wandb.init(config = config_defaults)
            config = wandb.config

            self.net.n_nodes_fc1 = config.n_nodes_fc1
            self.net.n_nodes_fc2 = config.n_nodes_fc2
            self.net.embedding_size = config.embedding_size

            if config.optimizer=='sgd':
                optimizer = torch.optim.SGD(self.net.parameters(),lr=config.learning_rate, momentum=config.momentum)
            elif config.optimizer=='adam':
                optimizer = torch.optim.Adam(self.net.parameters(),lr=config.learning_rate)
            elif config.optimizer=='rmsprop':
                optimizer = torch.optim.RMSprop(self.net.parameters(),lr=config.learning_rate, weight_decay = config.weight_decay, momentum = config.momentum)

            n_epochs = config.n_epochs

        # Loop for n_epochs
        total_instances = 0
        for epoch in range(n_epochs):    
            for i, data in enumerate(train_loader, 0): 
                # Check to sleep
                # sleeper.check()    

                # Get labels and inputs from train_loader
                if self.autoencoder_trainer:
                    labels, inputs = data, data
                else:
                    labels, inputs = data[:, 0].float(), data[:, 1:].float()
                    
                # Push to gpu
                if self.gpu == True:
                    inputs, labels = inputs.cuda(), labels.cuda()

                #Set the parameter gradients to zero
                optimizer.zero_grad()   

                #Forward pass
                outputs = self.net(inputs).squeeze(1)      # Forward pass
                loss = criterion (outputs, labels) # Calculate loss 
                
                # Backward pass and optimize
                loss.backward()                   # Find the gradient for each parameter
                optimizer.step()                  # Parameter update
                
                if self.wandb_on:
                    total_instances += len(outputs)
                    wandb.log({"epoch": epoch, "batch_loss": loss.item()}, step=total_instances)
            # Display 
            if epoch % (n_epochs/100) == 0:
                
                print("Epoch: ", epoch + 1, "\t Loss: ", loss.item())

        wandb.log({"loss": loss.item()})
        

    def test(self):
        """Test the model on the unseen data in the test set

        Returns:
            float: Mean loss on test dataset
        """
        #Create loss functions
        criterion = torch.nn.MSELoss()

        # Initialize
        total_tested = 0
        total_loss = 0
        
        # Test data in test loader
        for i, data in enumerate(self.data.test_loader, 0):
            # # Check sleep
            # sleeper.check()

            # Get labels and inputs from train_loader
            if self.autoencoder_trainer:
                labels, inputs = data, data
            else:
                labels, inputs = data[:, 0].float(), data[:, 1:].float()

            # Push to gpu
            if self.gpu == True:
                inputs, labels = inputs.cuda(), labels.cuda()

            #Forward pass
            outputs = self.net(inputs).squeeze(1)
            loss = criterion(outputs, labels) # Calculate loss 

            # Update runnin sum
            total_tested += labels.size(0)
            total_loss   += loss.item()
        
        # Test Loss
        test_loss = (total_loss/total_tested)
        return test_loss

