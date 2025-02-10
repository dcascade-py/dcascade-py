# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:57:28 2024

@author: Sahansila
"""


import pickle

path = "E:\\Sahansila\\cascade_results\\"
name_output = "EH_new_inputs.p"

data_output = pickle.load(open(path + name_output, "rb"))

output_name = 'Mobilised volume per provenance [m^3/d]'

datasets = data_output[output_name]

import pandas as pd
# Sum rows for each DataFrame in the list and retain column indices
summed_datasets = [pd.DataFrame(df.sum(axis=0)).T for df in datasets]


    
#plots

import matplotlib.pyplot as plt

# Assuming 'datasets' is your list of DataFrames
N = 19 # Choose the index of the DataFrame to plot
df_to_plot = summed_datasets[N ]  # Access the chosen DataFrame


df_to_plot.columns = [int(col) + 1 if str(col).isdigit() else col for col in df_to_plot.columns]


# Initialize the x and y lists for plotting
x_values = []
y_values = []

# Loop through 0 to 64
for i in range(1, 66):
    if i in df_to_plot.columns:  # Check if the column exists in the DataFrame
        x_values.append(i)  # Add the column index to x
        y_values.append(df_to_plot[i])  

# Plotting
plt.plot(x_values, y_values, marker='o')
plt.xticks(range(65), rotation=90)  # Rotate x-axis labels
plt.grid(axis='x', linestyle='--', alpha=0.5)  # Add faint vertical grid lines
plt.xlabel("Reach", fontsize = 12)
plt.ylabel("Sum of Mobilized volume", fontsize = 12)
plt.title(f"{output_name} for the year 2019 for reach {N+1}", fontsize = 15)

# to plot a vertical line and dynamically place the text in the line
#detemine the y position
y_min, y_max = plt.ylim()  # Get the current y-axis limits
y_position = (y_min + y_max) / 2  # Place the text at the middle of the y-axis
#plot a vertical line in an outlet reach
plt.axvline(x= N +1, color='red', linestyle='--', linewidth=1.5, label='Vertical Line')
# Add text near the vertical line
plt.text(N+1, y_position, 'chosen reach', color='red', rotation=90, verticalalignment='bottom', fontsize = 14)

#plot a vertical line to differentiate tributaries
plt.axvline(x=45, color='red', linestyle='--', linewidth=1.5, label='Vertical Line')

# Add text near the vertical line
plt.text(45, y_position, 'Tributaries', color='red', rotation=90, verticalalignment='bottom', fontsize =14)


plt.grid(True)
plt.show()
    