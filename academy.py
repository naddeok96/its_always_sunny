# Imports
import torch
import math
import copy

class Academy:
    def __init__(self,
                 net, 
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
        super(Academy,self).__init__()

        # Declare GPU usage
        self.gpu = gpu

        # Push net to CPU or GPU
        self.net = net if self.gpu == False else net.cuda()
        
        # Declare data
        self.data = data

        # Declare if training is for an autoencoder
        self.autoencoder_trainer = autoencoder_trainer

    def train(self, batch_size = 124, 
                    n_epochs = 1, 
                    learning_rate = 1e-12, 
                    momentum = 0.1, 
                    weight_decay = 0.0001,
                    autoencoder_trainer = False):
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

        # Loop for n_epochs
        for epoch in range(n_epochs):            
            for i, data in enumerate(train_loader, 0):
                
                # Get labels and inputs from train_loader
                if self.autoencoder_trainer == True:
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
                # print(i, (10 ** (math.floor(math.log(len(train_loader), 10)) - 2)))
                if i % (10 ** (math.floor(math.log(len(train_loader), 10)) - 2)) == 0:
                    print("Epoch: ", epoch + 1, "\tBatch: ", i, " of ", len(train_loader), "\t Loss: ", loss.item())

                # Backward pass and optimize
                loss.backward()                   # Find the gradient for each parameter
                optimizer.step()                  # Parameter update

                # Check if gradients have exploded
                # if torch.isnan(self.net.fc2.weight.grad).any().item():
                #     print("Gradients have exploded!")
                #     exit()
           
                
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
            # Get labels and inputs from train_loader
            if self.autoencoder_trainer == True:
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

