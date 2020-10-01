# Imports
from data_loader import WeatherDataset
from torch.utils.data import DataLoader, random_split
import math

class Data:
    """Opens, splits and puts data in a dataloader"""
    def __init__(self,  csv_file,
                        gpu = False,
                        test_batch_size  = 100,
                        train_percentage = 0.8):
        """
        Args:
            csv_file (string): Relative path to the csv file
            gpu (bool, optional): If True use GPU's. Defaults to False.
            test_batch_size (int, optional): Size of batches in the test set. Defaults to 100.
            train_percentage (float, optional): Percentage of dataset to be allocated for training. Defaults to 0.8.
        """

        super(Data,self).__init__()

        # Hyperparameters
        self.gpu = gpu
        self.test_batch_size = test_batch_size

        # Pull in data
        self.weather_dataset = WeatherDataset(csv_file = csv_file)
        self.n_features = len(self.weather_dataset[0]) - 1

        # Split Data
        train_portion = math.ceil(train_percentage * len(self.weather_dataset))
        test_portion  = len(self.weather_dataset) - train_portion
        self.train_set, self.test_set = random_split(self.weather_dataset.data, [train_portion, test_portion])


        #Test and validation loaders have constant batch sizes, so we can define them 
        if self.gpu == False:
            self.test_loader = DataLoader(self.test_set, 
                                            batch_size = test_batch_size,
                                            shuffle = False)

        else:
            self.test_loader = DataLoader(self.test_set, 
                                            batch_size = test_batch_size,
                                            shuffle = False,
                                            num_workers = 8,
                                            pin_memory = True)

    # Fucntion to break training set into batches
    def get_train_loader(self, batch_size):
        """Loads training data into a dataloader

        Args:
            batch_size (int): Size of batches in the train set

        Returns:
            Pytorch Dataloader object: Training data in a dataloader
        """
        if self.gpu == False:
            self.train_loader = DataLoader(self.train_set, 
                                            batch_size = batch_size,
                                            shuffle = True)

        else:
            self.train_loader = DataLoader(self.train_set,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            num_workers = 8,
                                            pin_memory=True)

        return self.train_loader


    