
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
# from sleeper import Sleeper
# sleeper = Sleeper()
# sleeper.check()

# Hyperparameters
csv_file = "data/weatherdata_merged2_truncated_500.csv" # "../../../data/WeatherData/weatherdata_merged2.csv"
n_epochs = 1000
gpu = True
save_model = False
wandb_on = True
wandb_sweep = True

# Push to GPU if necessary
if gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load data
number_of_unique_categories = 100 # For years

print("Loading Data...")
data_set = Data(csv_file = csv_file, 
                gpu = gpu,
                category = 1,
                one_hot_embedding_size = number_of_unique_categories) 
print("...done")

# Initalize network
print("\nLoading Network...")
net = Autoencoder(input_size  = number_of_unique_categories)
print("...done")

# Enter student network and curriculum data into an academy
print("\nLoading Academy...")
academy  = Academy(net, data_set, gpu,
                    autoencoder_trainer = True,
                    wandb_on = wandb_on,
                    wandb_sweep = wandb_sweep)
print("...done")

# Fit Model
if wandb_sweep == False:
    print("Training...")
    academy.train(  batch_size = 124, 
                    n_epochs = n_epochs, 
                    learning_rate = 1e-12, 
                    momentum = 0.1, 
                    weight_decay = 0.0001,
                    wandb_on = wandb_on)

    # Calculate accuracy on test set
    test_loss = academy.test()
    print("Final Loss: " + "{:.2e}".format(test_loss))

    # Check one sample
    for data in data_set.test_loader:
        test_data = data[0]
        print(test_data)

        if gpu:
            net = net.cuda()
            test_data = test_data.cuda()
            print(net(test_data))
        else:
            print(net(test_data))

    # Save Model
    if save_model:
        # Define File Names
        filename  = "net_w_loss_" + "{:.2e}".format(test_loss) + ".pt"
        
        # Save Models
        torch.save(academy.net.state_dict(), filename)