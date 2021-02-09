''' 
Opens up data
'''
# Imports
import pandas as pd

# Hyperparameters
truncation_size = 10000
save_truncd = True
file_name = "weatherdata_merged2"

# Load data
data = pd.read_csv("../../../data/WeatherData/" + file_name + ".csv")

# Preview
# print(data.head())

# print("Target Max: ", data["mean_swrad"].max())
# print("Target Min: ", data["mean_swrad"].min())
# print("Target Mean: ", data["mean_swrad"].mean())
# print("Target Std: ", data["mean_swrad"].std())

# print("Datatypes")
# print(data.dtypes)

if save_truncd:
    # Create a small truncated version for hacking
    truncd_data = data[:truncation_size]
    
    # Save truncation
    truncd_data.to_csv("data/" + file_name + "_truncated_" + str(truncation_size) + ".csv")

