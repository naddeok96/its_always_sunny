# Imports
import pandas as pd
import torch

# Class
class WeatherDataset(torch.utils.data.Dataset):
    """Pytorch Dataloader for weather data."""

    def __init__(self,  csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Load data
        self.data = pd.read_csv(csv_file)

        # Temporary testbed cleans
        # self.data.drop("Unnamed: 0", axis = 1, inplace = True) # This is just the index col saved by accident
        self.data[self.data != self.data] = 0 # This is to convert NaN's to 0
        self.data.drop(["year", "month", "day", "geoid"], axis = 1, inplace = True) # Remove categorical features

        # Transforms
        self.data = torch.tensor(self.data.values) # Convert to a Tensor
            # Norm Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :]