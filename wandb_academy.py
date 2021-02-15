# Imports
import torch
import math
import copy
import wandb
from autoencoder import Autoencoder
from sleeper import Sleeper
from dice_loss import DiceLoss
# sleeper = Sleeper()


class WandB_Academy:
    def __init__(self,
                 sweep_config,
                 data,
                 gpu = False,
                 autoencoder_trainer = False):
        """
        Trains and test models on datasets

        Args:
            net (Pytorch model class):  Network to train and test
            data (Tensor Object):       Data to train and test with
            gpu (bool, optional):       If True then GPU's will be used. Defaults to False.
            autoencoder_trainer (bool, optional): If True labels will equal inputs. Defaults to False.
        """
        super(WandB_Academy,self).__init__()

        # Declare GPU usage
        self.gpu = gpu
        
        # Declare data
        self.data = data

        # Declare if training is for an autoencoder
        self.autoencoder_trainer = autoencoder_trainer

        # Declare Losses
        self.mse_criterion = torch.nn.MSELoss()
        self.cos_criterion = torch.nn.CosineEmbeddingLoss(reduction='mean',
                                                            margin=0.5)
        self.dice_criterion = DiceLoss(smooth=1)

        # Declare if Weights and Biases are used
        
        self.sweep_config = sweep_config["parameters"]
        sweep_id = wandb.sweep(sweep_config, project="Year_AutoEnc_Trunc10000_Loss")
        wandb.agent(sweep_id, self.train)

    def train(self):
        """Fits model to data

        Args:
            batch_size (int, optional):         Number of samples in each batch. Defaults to 124.
            epochs (int, optional):           Number of cycles through the data. Defaults to 1.
            learning_rate (float, optional):    Learning rate coeffiecent in the optimizer. Defaults to 0.001.
            momentum (float, optional):         Momentum coeffiecent in the optimizer. Defaults to 0.9.
            weight_decay (float, optional):     Weight decay in the optimizer. Defaults to 0.0001.
        """

        # Weights and Biases Setup
        config_defaults = {
                    'epochs': self.sweep_config["epochs"]['values'][0],
                    'batch_size': self.sweep_config["batch_size"]['values'][0],
                    'n_nodes_fc1': self.sweep_config["n_nodes_fc1"]['values'][0],
                    'n_nodes_fc2': self.sweep_config["n_nodes_fc2"]['values'][0],
                    'embedding_size' : self.sweep_config["embedding_size"]['values'][0],
                    'momentum'   : self.sweep_config["momentum"]['values'][0],
                    'weight_decay' :  self.sweep_config["weight_decay"]['values'][0],
                    'learning_rate': self.sweep_config["learning_rate"]['values'][0],
                    'optimizer': self.sweep_config["optimizer"]['values'][0],
                    'criterion': self.sweep_config["criterion"]['values'][0],
                    'activation_func': self.sweep_config["activation_func"]['values'][0],
                    'output_activation_func': self.sweep_config['output_activation_func']['values'][0]
                }
        wandb.init(config = config_defaults, entity="naddeok", project="Year_AutoEnc_Trunc10000_Loss")
        config = wandb.config

        #Get training data
        train_loader = self.data.get_train_loader(config.batch_size)

        # Push net to CPU or GPU
        net = Autoencoder(input_size  = self.data.num_unique_embeddings,
                            n_nodes_fc1 = config.n_nodes_fc1,
                            n_nodes_fc2 = config.n_nodes_fc2,
                            embedding_size = config.embedding_size,
                            activation_func = config.activation_func,
                            output_activation_func = config.output_activation_func)

        self.net = net if self.gpu == False else net.cuda()

        # Setup Optimzier
        if config.optimizer=='sgd':
            optimizer = torch.optim.SGD(self.net.parameters(),lr=config.learning_rate, momentum=config.momentum)
        elif config.optimizer=='adam':
            optimizer = torch.optim.Adam(self.net.parameters(),lr=config.learning_rate)
        elif config.optimizer=='rmsprop':
            optimizer = torch.optim.RMSprop(self.net.parameters(),lr=config.learning_rate, weight_decay = config.weight_decay, momentum = config.momentum)

        # Setup Criterion
        if config.criterion=="mse":
            criterion = self.mse_criterion
        elif config.criterion=="cossim":
            criterion = self.cos_criterion
        elif config.criterion=="dice":
            criterion = self.dice_criterion
            

        # Loop for epochs
        total_instances = 0
        for epoch in range(config.epochs):    
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
                if config.criterion=="cossim":
                    loss = criterion(outputs.view(1, -1), labels.view(1, -1), torch.Tensor(labels.size(0)).cuda().fill_(1.0)) # Calculate loss 
                else:
                    loss = criterion (outputs, labels) # Calculate loss 
                
                # Backward pass and optimize
                loss.backward()                   # Find the gradient for each parameter
                optimizer.step()                  # Parameter update
                
                total_instances += len(outputs)
                
            # Display 
            if epoch % (config.epochs/100) == 0:
                
                print("Epoch: ", epoch + 1, "\t Loss: ", loss.item())

                wandb.log({"epoch"  : epoch, 
                            "MSE"   : self.mse_criterion(outputs, labels).item(),
                            "CosSim": self.cos_criterion(outputs.view(1, -1), labels.view(1, -1), torch.Tensor(labels.size(0)).cuda().fill_(1.0)).item(),
                            "Dice"  : self.dice_criterion(outputs, labels).item()}, step=total_instances)


        wandb.log({ "MSE"   : self.mse_criterion(outputs, labels).item(),
                    "CosSim": self.cos_criterion(outputs.view(1, -1), labels.view(1, -1), torch.Tensor(labels.size(0)).cuda().fill_(1.0)).item(),
                    "Dice"  : self.dice_criterion(outputs, labels).item()})

        # Save Model
        # Define File Names
        filename  = "net_w_mse_loss_" + str(int(round(self.mse_criterion(outputs, labels).item(), 3))) + ".pt"
        torch.save(self.net.state_dict(), filename)
        

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

