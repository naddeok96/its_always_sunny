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
csv_file = "../data/Weather/weatherdata_merged2.csv"
n_epochs = 2
gpu = False
categorical = False
save_model = True

#   add run time!!!!

# Push to GPU if necessary
if gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load data
data = Data(csv_file = csv_file, 
            gpu = gpu)
print("Data Loaded...")

# Initalize network
net = Vanilla_NN(input_size = data.n_features)
print("Network Loaded...")

# Enter student network and curriculum data into an academy
academy  = Academy(net, data, gpu)
print("Entered into Academy...")


# Fit Model
print("Start Training...")
academy.train(n_epochs = n_epochs)

# Calculate accuracy on test set
test_loss = academy.test()
print("Final Loss: ", test_loss)

# Save Model
if save_model:
    # Define File Names
    filename  = "net_w_loss_" + str(int(round(test_loss, 3))) + ".pt"
    
    # Save Models
    torch.save(academy.net.state_dict(), filename)