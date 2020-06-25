"""
This script will take a put a model and dataset in an academy to train
"""
# Imports
from scripts.vanilla_nn import Vanilla_NN
from scripts.academy import Academy
import pandas as pd

# Load data
data = pd.read_csv("data/weatherdata_merged_truncated_50.csv")