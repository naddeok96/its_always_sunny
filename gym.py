"""
This script will take a model and dataset and put them in an academy to train
"""
# Imports
import os
from data_setup import Data
from vanilla_nn import Vanilla_NN
from academy import Academy
import torch

# Hyperparameters
csv_file = "../data/Weather/weatherdata_merged2_training.csv"
n_epochs = 2
gpu = False
save_model = True


# Push to GPU if necessary
if gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load data
data = Data(csv_file = csv_file, 
            gpu = gpu)

# Initalize network
net = Vanilla_NN(input_size = data.n_features)

# Enter student network and curriculum data into an academy
academy  = Academy(net, data, gpu)

# Fit Model
academy.train(n_epochs = n_epochs,
              batch_size = 5)

# Calculate accuracy on test set
test_loss = academy.test()
print(test_loss)

# Save Model
if save_model:
    # Define File Names
    filename  = "net_w_loss_" + str(int(round(test_loss * 100, 3))) + ".pt"
    
    # Save Models
    torch.save(academy.net.state_dict(), filename)