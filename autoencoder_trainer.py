
"""
This script will take a model and dataset and put them in an academy to train
"""
# Imports
from autoencoder import Autoencoder
from data_setup import Data
from academy import Academy
import torch
import os

# Hyperparameters
csv_file = "weatherdata_merged2_training_truncated_50.csv"
n_epochs = 1
gpu = False
save_model = True

# Push to GPU if necessary
if gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load data
data_set = Data(csv_file = csv_file, 
                gpu = gpu,
                category = 1) 

# Initalize network
net = Autoencoder()

# Enter student network and curriculum data into an academy
academy  = Academy(net, data_set, gpu,
                    autoencoder_trainer = True)

# Fit Model
academy.train(n_epochs = n_epochs)

# Calculate accuracy on test set
test_loss = academy.test()
print("Final Loss: " + "{:.2e}".format(test_loss))

for data in data_set.test_loader:
    print(data[0])
    print(net(data[0]))
    continue

# Save Model
if save_model:
    # Define File Names
    filename  = "net_w_loss_" + "{:.2e}".format(test_loss) + ".pt"
    
    # Save Models
    torch.save(academy.net.state_dict(), filename)