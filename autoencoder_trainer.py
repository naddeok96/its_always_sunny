
"""
This script will take a model and dataset and put them in an academy to train
"""
# Imports
from autoencoder import Autoencoder
from data_setup import Data
from academy import Academy
from wandb_academy import WandB_Academy
import torch
import os

# Sleep function
# from sleeper import Sleeper
# sleeper = Sleeper()
# sleeper.check()

# Hyperparameters
csv_file = "data/weatherdata_merged2_truncated_10000.csv" # "../../../data/WeatherData/weatherdata_merged2.csv" # 
gpu = True
save_model = False

# Push to GPU if necessary
if gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Load data
print("Loading Data...")
data_set = Data(csv_file = csv_file, 
                gpu = gpu,
                category = 1) 
print("...done")

# Enter student network and curriculum data into an academy
print("\nLoading Academy...")
academy  = WandB_Academy(data_set, gpu,
                    autoencoder_trainer = True)
print("...done")
