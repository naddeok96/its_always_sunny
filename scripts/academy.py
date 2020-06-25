# Imports
import torch
from torch import nn
import torchvision
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import math
import operator

class Academy:
    def __init__(self,
                 net, 
                 data,
                 gpu = False):
        """
        Trains and test models on datasets

        Args:
            net (Pytorch model class): Network to train and test
            data (Tensor Object): Data to train and test with
            gpu (bool, optional): If True then GPU's will be used. Defaults to False.
        """
        super(Academy,self).__init__()

        # Declare GPU usage
        self.gpu = gpu

        # Push net to CPU or GPU
        self.net = net if self.gpu == False else net.cuda()
        
        # Declare data
        self.data = data

    def train(self, batch_size = 124, 
                    n_epochs = 1, 
                    learning_rate = 0.001, 
                    momentum = 0.9, 
                    weight_decay = 0.0001):
        """
        Trains network on data 

        Args:
            batch_size (int, optional): Number of samples in each batch. Defaults to 124.
            n_epochs (int, optional): Number of cycles through the data. Defaults to 1.
            learning_rate (float, optional): Learning rate coeffiecent in the optimizer. Defaults to 0.001.
            momentum (float, optional): Momentum coeffiecent in the optimizer. Defaults to 0.9.
            weight_decay (float, optional): Weight decay in the optimizer. Defaults to 0.0001.
        """
        #Get training data
        train_loader = self.data.get_train_loader(batch_size)
        n_batches = len(train_loader)

        #Create optimizer and loss functions
        optimizer = torch.optim.SGD(self.net.parameters(), 
                                    lr = learning_rate, 
                                    momentum = momentum,
                                    weight_decay = weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        #Loop for n_epochs
        for epoch in range(n_epochs):            
            for i, data in enumerate(train_loader, 0):
                
                # Get inputs and labels from train_loader
                inputs, labels = data

                # Push to gpu
                if self.gpu == True:
                    inputs, labels = inputs.cuda(), labels.cuda()

                #Set the parameter gradients to zero
                optimizer.zero_grad()   

                #Forward pass
                outputs = self.net(inputs)        # Forward pass
                loss = criterion (outputs, labels) # Calculate loss

                # Freeze layers
                if frozen_layers != None:
                    self.freeze(frozen_layers) 

                # Backward pass and optimize
                loss.backward()                   # Find the gradient for each parameter
                optimizer.step()                  # Parameter update
                
    def test(self):
        '''
        Test the model on the unseen data in the test set
        '''
        # Initialize
        total_tested = 0
        correct = 0

        # Test images in test loader
        for inputs, labels in self.data.test_loader:
            # Push to gpu
            if self.gpu == True:
                inputs, labels = inputs.cuda(), labels.cuda()

            #Forward pass
            outputs = self.net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Update runnin sum
            total_tested += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate accuracy
        accuracy = (correct/total_tested)
        return accuracy

