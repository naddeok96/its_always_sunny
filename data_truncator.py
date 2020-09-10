''' 
Opens up data
'''
# Imports
import pandas as pd

# Hyperparameters
truncation_size = 50
save_truncd = True
file_name = "weatherdata_merged2_training"

# Load data
data = pd.read_csv("../data/Weather/" + file_name + ".csv")

# Preview
print(data.head())

# print("Datatypes")
# print(data.dtypes)

if save_truncd:
    # Create a small truncated version for hacking
    truncd_data = data[:truncation_size]
    
    # Save truncation
    truncd_data.to_csv(file_name + "_truncated_" + str(truncation_size) + ".csv")

