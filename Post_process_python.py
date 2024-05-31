# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:53:56 2023

Plot d-cascade results 

@author: Diane Doolaeghe
"""

# libraries 
import pickle
import numpy as np 
from matplotlib import pyplot as plt  
import pandas as pd
import geopandas as gpd
import matplotlib
from matplotlib.pyplot import text
import copy
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
#matplotlib.use('Qt5Agg') 
 
     
    
#----------------------directories 
path = "E:\\cascade\\combined_random_sampling\\"
name_output = "output_change_width_-15_slope_-20.pkl"
path_Q = "E:\\cascade\\input\\"
name_q = 'Q_latest_reordered.csv' # the order is incresing FromN - ToNode in ReachData 

path_river_network = "E:\\UNIPD\\shp_file_slopes_hydro_and_LR\\"
name_river_network = "Po_rivernet_grainsze_new_d.shp"

figure_folder = "E:\\cascade\\combined_random_sampling\\fig-15-20\\"


#---------------------loading data

# load network for reference 
ReachData = gpd.GeoDataFrame.from_file(path_river_network + name_river_network) #read shapefine from shp format
ReachData = ReachData.sort_values(by = 'FromN', ignore_index = True )
ReachData_Po = ReachData[ReachData['River'] == 'Po'] # select Po
Po_idx = ReachData_Po['FromN'].values
Po_idx = Po_idx.astype(int)

# load outputs  
data_output = pickle.load(open( path + name_output , "rb"))
# ext_output = pickle.load(open(path + name_ext_output , "rb"))=

#exclude variables not included in the plotting yet (sediment divided into classes)
data_output_t = copy.deepcopy(data_output)
# variable_names = [data for data in data_output_t.keys() if data.endswith('per class [m^3/s]')]
# for item in variable_names: 
#     del data_output_t[item]
    
    
#-----------------calculate eroded/deposited and add it to data_output_t
data_output_t['Delta volume [m^3]'] = data_output_t['Transported [m^3]'] - data_output_t['Mobilized volume [m^3]']
data_output_t['Delta z [m]'] = data_output_t['Delta volume [m^3]']/(ReachData['Wac'].values*ReachData['Length'].values)



#------------------usefull function

def rename_names(output_name):
    '''For exemple, renames 'Mobilized volume [m^3]' into 'Mobilized_volume' 
    to use for saving csv and plots
    '''
    new_name = ''
    for c in output_name:
        if c == ' ':
            new_name = new_name+'_'
        elif c == '[':
            break
        else:
            new_name = new_name+c
    new_name = new_name[:-1]
    return new_name

  
#--------------------plots 

""" Write .csv files from data outputs """
output_name_list = list(data_output_t.keys())

for output_name in output_name_list:
    df = pd.DataFrame(data_output_t[output_name])
    new_output_name = rename_names(output_name)
    df.to_csv(figure_folder+str(new_output_name)+".csv", sep = ';')
    print(output_name+ ' saved as csv')




d = data_output_t['D50 mobilised layer [m]']

indices_values = np.argwhere(d > 0.01)

for index in indices_values:
    print("Index:", index, "Value:", d[index[0], index[1]])


## Sum over 5 classes##
data_lists = data_output['Active layer sed in the reach - per class [m^3/s]']

V_act_sum = np.sum([data for data in data_lists], axis=0)

data_output_t['Summed Active layer sed'] = V_act_sum

specific_rows = np.array([array[26] for array in data])


# Get the number of columns dynamically
num_columns = specific_rows.shape[1]

# Plotting
for col_index in range(num_columns):
    plt.plot(specific_rows[:, col_index], label=f'Class {col_index}')
plt.xlabel('timestep')
plt.ylabel('Tr_cap')
plt.title('Tr_cap throughout the timescale for different sediment classes')
plt.legend()


specific_rows = np.array([array[:,9] for array in Fi_r_act[:365]])

# Get the number of columns dynamically
num_columns = specific_rows.shape[1]

# Plotting
for col_index in range(num_columns):
    plt.plot(specific_rows[:, col_index], label=f'Class {col_index}')
plt.xlabel('Timestep')
plt.ylabel('Fi_r_act')
plt.title('Fi_r_act throughout the timescale for different sediment classes')
plt.legend()


""" Basic plots: along reach id, one line per time step"""
output_name = 'D50 active layer [m]' #'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'

#select data according to output_name and Po reaches
my_data = data_output_t[output_name][:,Po_idx-1]  # -1 because python start at 0

# #select time list:
# label = 'all_times'
# time_list = [i for i in range(0,my_data.shape[0],1)]

label = '2019'
time_list = [i for i in range(0,365,1)]

# label = '2020'
# time_list = [i for i in range(365, 729,1)]

# label = '2021'
# time_list = [i for i in range(731, 1095,1)]

#label = 'yearly_dates'
# time_list = [0,365,728]

#create figure and graph axes
fig = plt.figure()
ax = plt.subplot(111)
color = iter(plt.cm.turbo(np.linspace(0, 1, len(time_list))))

for idx, t in enumerate(time_list):
    c=next(color)    
    ax.plot(Po_idx, my_data[t,:], label = 't='+str(t), linewidth = 1, color=c)

#add tributaries ToNode position on same plot as vertical lines:
trib_data = ReachData[ReachData['River'] != 'Po']
for _, row in trib_data.iterrows():
    ax.axvline(x=row['ToN'], color = 'grey', linestyle = '--')
    plt.text(row['ToN'], 0.01, row['River'], rotation=90, verticalalignment='center', color = 'grey')

#add more graduation along x axis
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter('{x:.0f}')
ax.xaxis.set_minor_locator(MultipleLocator(1))    
    
ax.set_title('Po ('+str(label+')'), fontsize = 20)
ax.set_xlabel('FromN', fontsize = 14)
ax.set_ylabel(output_name,fontsize = 14)

ax.legend(fontsize = 5)

fig.set_tight_layout(True)
fig.set_size_inches(900./fig.dpi,600./fig.dpi)
new_name = rename_names(output_name)
fig.savefig(figure_folder+str(new_name)+'_'+str(label))
fig.clf()
plt.close("all")


""" Basic plots: along time, one line per reach"""

output_name = 'D50 active layer [m]' #'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
# output_name = V_act_sum
#select data according to output_name
my_data = data_output_t[output_name]  
# my_data = [output_name]  

#select your reaches:
label = 'all_reaches'   
reach_list = Po_idx #FromN index

# reach_list = [7] #FromN index

#create figure and graph axes
fig = plt.figure()
ax = plt.subplot(111)
color = iter(plt.cm.turbo(np.linspace(0, 1, len(reach_list))))

for idx, r in enumerate(reach_list):
    c=next(color)    
    ax.plot(np.arange(1,len(my_data[:,r-1])+1,1), my_data[:,r-1], label = 'FromN='+str(r), linewidth = 1, color=c)
    
ax.set_title('Po ('+str(label)+')', fontsize = 20)
ax.set_xlabel('Time', fontsize = 14)
ax.set_ylabel(output_name,fontsize = 14)

ax.legend(fontsize = 5)

fig.set_tight_layout(True)
fig.set_size_inches(900./fig.dpi,600./fig.dpi)
new_name = rename_names(output_name)
fig.savefig(figure_folder+str(new_name)+'_'+str(label))
fig.clf()
plt.close("all")


""" Average of years on same plot """

output_name = 'D50 mobilised layer [m]' #'D5O mobilised layer [m]', 'Mobilized volume [m^3]', 'Transported [m^3]'
#select data according to output_name and Po reaches
my_data = data_output_t[output_name][:,Po_idx-1]  # -1 because python start at 0

#create figure and graph axes
fig = plt.figure()
ax = plt.subplot(111)

label = '2019'
time_list = [i for i in range(0,365,1)]
avg = np.nanmean(my_data[time_list, :], axis = 0)      
ax.plot(Po_idx, avg, color = 'crimson', label = label, linewidth = 2)

label = '2020' 
time_list = [i for i in range(365,731,1)]
avg = np.nanmean(my_data[time_list, :], axis = 0)      
ax.plot(Po_idx, avg, color = 'darkblue', label = label, linewidth = 2)

label = '2021' 
time_list = [i for i in range(731, 1095,1)]
avg = np.nanmean(my_data[time_list, :], axis = 0)      
ax.plot(Po_idx, avg, color = 'green', label = label, linewidth = 2)


#add tributaries ToNode position on same plot as vertical lines:
trib_data = ReachData[ReachData['River'] != 'Po']
for _, row in trib_data.iterrows():
    ax.axvline(x=row['ToN'], color = 'grey', linestyle = '--')
    plt.text(row['ToN'], 0.01, row['River'], rotation=90, verticalalignment='center', color = 'grey')

#add more graduation along x axis
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter('{x:.0f}')
ax.xaxis.set_minor_locator(MultipleLocator(1))    
    
ax.set_title('Po', fontsize = 20)
ax.set_xlabel('FromN', fontsize = 14)
ax.set_ylabel(output_name,fontsize = 14)
ax.legend(fontsize = 10, loc = 'upper right')


fig.set_tight_layout(True)
fig.set_size_inches(900./fig.dpi,600./fig.dpi)
new_name = rename_names(output_name)
fig.savefig(figure_folder+str(new_name)+'_average_per_years')
fig.clf()
plt.close("all")


""" Bar plots showing sum per year"""

output_name =  'Mobilized volume [m^3]' #, 'Transported [m^3]', 'Delta z [m]'
#select data according to output_name and Po reaches
my_data = data_output_t[output_name][:,Po_idx-1]  # -1 because python start at 0

#create figure and graph axes
fig = plt.figure()
ax = plt.subplot(111)

# dx = 0.25

label = '2019'
time_list = [i for i in range(0,9,1)]
my_sum = np.nansum(my_data[time_list, :], axis = 0)      
# ax.bar(Po_idx-dx, my_sum, width = dx, align = 'edge', color = 'crimson', label = label)
ax.plot(Po_idx, my_sum, color = 'crimson', label = label)

# label = '2020' 
# time_list = [i for i in range(365,731,1)]
# my_sum = np.nansum(my_data[time_list, :], axis = 0)      
# # ax.bar(Po_idx, my_sum, width = dx, align = 'edge', color = 'darkblue', label = label)
# ax.plot(Po_idx, my_sum, color = 'darkblue', label = label)

# label = '2021' 
# time_list = [i for i in range(731, 1095,1)]
# my_sum = np.nansum(my_data[time_list, :], axis = 0)      
# # ax.bar(Po_idx+dx, my_sum, width = dx, align = 'edge', color = 'green', label = label)
# ax.plot(Po_idx, my_sum, color = 'green', label = label)


#add tributaries ToNode position on same plot as vertical lines:
trib_data = ReachData[ReachData['River'] != 'Po']
for _, row in trib_data.iterrows():
    ax.axvline(x=row['ToN'], color = 'grey', linestyle = '--')
    plt.text(row['ToN'], 5000000, row['River'], rotation=90, verticalalignment='center', color = 'grey')

#add more graduation along x axis
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter('{x:.0f}')
ax.xaxis.set_minor_locator(MultipleLocator(1))    
    
ax.set_title('Po', fontsize = 20)
ax.set_xlabel('FromN', fontsize = 14)
ax.set_ylabel(output_name,fontsize = 14)
ax.legend(fontsize = 10, loc = 'upper right')


fig.set_tight_layout(True)
fig.set_size_inches(1200./fig.dpi,600./fig.dpi)
new_name = rename_names(output_name)
fig.savefig(figure_folder+str(new_name)+'_sum_per_year')
fig.clf()
plt.close("all")




