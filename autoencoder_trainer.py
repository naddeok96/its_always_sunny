
"""
This script will take a model and dataset and put them in an academy to train
"""
# Imports
from autoencoder import Autoencoder
from data_setup import Data
from academy import Academy
import torch
import os

# Sleep function
from sleeper import Sleeper
sleeper = Sleeper()
sleeper.check()

# Hyperparameters
csv_file = "../data/Weather/weatherdata_merged2.csv"
n_epochs = 1000
gpu = False
save_model = True

# Push to GPU if necessary
if gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load data
number_of_unique_categories = 100 # For years

data_set = Data(csv_file = csv_file, 
                gpu = gpu,
                category = 1,
                one_hot_embedding_size = number_of_unique_categories) 

# Initalize network
net = Autoencoder(input_size  = number_of_unique_categories)

# Enter student network and curriculum data into an academy
academy  = Academy(net, data_set, gpu,
                    autoencoder_trainer = True)

# Fit Model
academy.train(n_epochs = n_epochs)

# Calculate accuracy on test set
test_loss = academy.test()
print("Final Loss: " + "{:.2e}".format(test_loss))

# Check one sample
for data in data_set.test_loader:
    print(data[0])
    print(net(data[0]))
    break

# Save Model
if save_model:
    # Define File Names
    filename  = "net_w_loss_" + "{:.2e}".format(test_loss) + ".pt"
    
    # Save Models
    torch.save(academy.net.state_dict(), filename)