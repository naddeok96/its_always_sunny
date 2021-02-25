# Imports
from torch.utils.data import DataLoader, random_split
from data_loader import WeatherDataset
import numpy as np
import torch
import math

class Data:
    """Opens, splits and puts data in a dataloader"""
    def __init__(self,  csv_file,
                        gpu = False,
                        test_batch_size  = 100,
                        train_percentage = 0.8,
                        autoencoder_data = False,
                        one_hot_embedding_size = None):
        """
        Args:
            csv_file (string): Relative path to the csv file
            gpu (bool, optional): If True use GPU's. Defaults to False.
            test_batch_size (int, optional): Size of batches in the test set. Defaults to 100.
            train_percentage (float, optional): Percentage of dataset to be allocated for training. Defaults to 0.8.
            category(int, optional): Column to have an autoencoder dataset. 
                                        # 1 is year   (100 Types)
                                        # 2 is month
                                        # 3 is day
                                        # 4 is geoid
                                        # Defaults to None.
        """
        super(Data,self).__init__()

        # Hyperparameters
        self.gpu = gpu
        self.test_batch_size = test_batch_size

        # Pull in data
        self.weather_dataset = WeatherDataset(csv_file = csv_file, 
                                              autoencoder_data = autoencoder_data)

        # Convert dataset for autoencoder
        if autoencoder_data:
            categorical_data = torch.tensor(range(len(self.weather_dataset[:])))
            
            if one_hot_embedding_size is None:
                one_hot_embedding_size = categorical_data.size(0)

            # One hot encode
            self.weather_dataset = torch.nn.functional.one_hot(categorical_data.type(torch.LongTensor), one_hot_embedding_size).type(torch.FloatTensor)
            self.num_unique_embeddings = self.weather_dataset.size(0)
            
        # Find size of feature space
        self.n_features = len(self.weather_dataset[0]) - 1

        # Split Data
        train_proportion = math.ceil(train_percentage * len(self.weather_dataset))
        test_proportion  = len(self.weather_dataset) - train_proportion

        self.train_set, self.test_set = random_split(self.weather_dataset.data, [train_proportion, test_proportion])

        #Test and validation loaders have constant batch sizes, so we can define them 
        if self.gpu:
            self.test_loader = DataLoader(self.test_set, 
                                            batch_size = test_batch_size,
                                            shuffle = False,
                                            num_workers = 8,
                                            pin_memory = True)
        else:
            self.test_loader = DataLoader(self.test_set, 
                                            batch_size = test_batch_size,
                                            shuffle = False)

    # Fucntion to break training set into batches
    def get_train_loader(self, batch_size):
        """Loads training data into a dataloader

        Args:
            batch_size (int): Size of batches in the train set

        Returns:
            Pytorch Dataloader object: Training data in a dataloader
        """
        if self.gpu:
            self.train_loader = DataLoader(self.train_set,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8,
                                            pin_memory=True)
        else:
            self.train_loader = DataLoader(self.train_set, 
                                            batch_size = batch_size,
                                            shuffle = True)
            

        return self.train_loader


    