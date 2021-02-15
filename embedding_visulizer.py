
# Imports
from autoencoder import Autoencoder
from data_setup import Data
from academy import Academy
import matplotlib.pyplot as plt
from wandb_academy import WandB_Academy
from sweep_config import sweep_config
import torch
import os

# Sleep function
# from sleeper import Sleeper
# sleeper = Sleeper()
# sleeper.check()

# Hyperparameters
csv_file = "data/weatherdata_merged2_truncated_10000.csv" # "../../../data/WeatherData/weatherdata_merged2.csv" # 
gpu = False

# Push to GPU if necessary
if gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# Load data
print("Loading Data...")
data = Data(csv_file = csv_file, 
                gpu = gpu,
                category = 1) 
print("...done")

net = Autoencoder(input_size  = data.num_unique_embeddings,
                    n_nodes_fc1 = 512,
                    n_nodes_fc2 = 256,
                    embedding_size = 2,
                    activation_func = "relu",
                    output_activation_func = "softmax")

net.load_state_dict(torch.load("happy1.pt", map_location=torch.device('cpu')))
net.eval()

if gpu:
    net = net.cuda()

# train_loader = data.get_train_loader(config.batch_size)
for year, one_hot in zip(data.categorical_data, torch.unique(data.weather_dataset, dim = 0)):
    print(year)
    x, y = net.encode(one_hot)
    
    plt.plot(x.item(), y.item())
    plt.show()
    exit()