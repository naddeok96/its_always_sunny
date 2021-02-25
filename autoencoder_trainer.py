
"""
This script will take a model and dataset and put them in an academy to train
"""
# Imports
from autoencoder import Autoencoder
from data_setup import Data
from academy import Academy
from wandb_academy import WandB_Academy
from sweep_config import sweep_config
import torch
import os

# Sleep function
# from sleeper import Sleeper
# sleeper = Sleeper()
# sleeper.check()

# Hyperparameters
csv_file = "data/unique_day.csv"
gpu = True

# Push to GPU if necessary
if gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Load data
print("Loading Data...")
data = Data(csv_file = csv_file, 
                gpu = gpu,
                autoencoder_data = True) 
print("...done")

net = Autoencoder(input_size  = data.num_unique_embeddings)

# Enter student network and curriculum data into an academy
print("\nLoading Academy...")
academy  = WandB_Academy("Day AutoEncoder",
                        sweep_config, data, gpu,
                    autoencoder_trainer = True)
# academy  = Academy(net, 
#                  data,
#                  gpu = True,
#                  autoencoder_trainer = True)
# academy.train()
print("...done")
