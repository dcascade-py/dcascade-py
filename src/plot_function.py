"""
Created on Tue Jan 10 10:00:14 2023

Dynamic plot of the variables within data_output
Note 03/23 The variables divided by sediment class are not implemented yet (commented lines are the 'work in progress stage'..)

This script was adapted from the Matlab version by Marco Tangi

@author: Elisa Bozzolan
"""

from tkinter import Label, OptionMenu, StringVar, Tk  # for widget

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider


def dynamic_plot(data_output, ReachData, psi, **kwargs):
    '''
    Plot input data and show reach features and sediment transport processes by clicking on it
    '''

    #

    # ##input data
    # dmi = 2**(-psi)


    # List of output available for display
    list_outputs = ['D50 active layer [m]', 'D50 volume out [m]', 'Sediment budget [m^3]', 'Transport capacity [m^3]', 'Volume in [m^3]', 'Volume out [m^3]']


    # # output selection
    # def output_selection(event):
    #     global indx
    #     if clicked.get() == "D50 active layer [m]":
    #         indx = "D50 active layer [m]"
        # elif clicked.get() == "Daily trasport capacity [m^3/day]":
        #     indx = "Daily trasport capacity [m^3/day]"
        # elif clicked.get() == "D50 deposited layer [m]":
        #     indx = "D50 deposited layer [m]"
        # elif clicked.get() == "D50 mobilised layer [m]":
        #     indx = "D50 mobilised layer [m]"
        # elif clicked.get() == "Deposited volume[m^3]":
        #     indx = "Deposited volume[m^3]"
        # elif clicked.get() == "Mobilized volume [m^3]":
        #     indx = "Mobilized volume [m^3]"
        # #elif clicked.get() == "Transported + deposited sed - per class [m^3/s]":
        #  #   indx = "Transported + deposited sed - per class [m^3/s]"
        # #elif clicked.get() == "Transported + deposited sed in the reach [m^3]":
        #  #   indx = "Transported + deposited sedin the reach [m^3]"
        # elif clicked.get() == "Channel Width [m]":
        #     indx = "Channel Width [m]"
        # elif clicked.get() == "Reach Slope":
        #     indx = "Reach Slope"
        # root.destroy()


    def output_selection(event):
        global indx
        selection = clicked.get()
        if selection in list_outputs:
            indx = selection
        else:
            indx = None  # or some default fallback
        root.destroy()



    root = Tk() # create the little window
    myLabel = Label(root, text = 'Outputs')
    myLabel.pack()
    root.geometry("300x300")
    #indx_tr_cap = 3 # default value
    # options = ["D50 active layer [m]"]#, "Daily trasport capacity [m^3/day]","D50 deposited layer [m]", "D50 mobilised layer [m]",
    #            #  "Deposited volume[m^3]", "Mobilized volume [m^3]",
    #            # "Channel Width [m]", "Reach Slope"]

    clicked = StringVar()
    clicked.set("Choose an option")

    drop = OptionMenu(root, clicked, *list_outputs, command = output_selection)
    drop.pack(pady=20)
    root.mainloop()

    #start_time = 1
    plot_class = indx # default plot variables


    # lengths_values = {key: len(value) for key, value in data_output.items() if key != 'Transported + deposited sed in the reach [m^3/s]' }
    # lengths_values = {key: len(value) for key, value in data_output.items()}

    lengths_values = {key: len(data_output[key]) for key in list_outputs}
    sim_length = min(lengths_values.values())

    ## define plot variables

    #define sediment classes
    n_class = 10
    i_class = 100/n_class - 0.00001 #interval between classestext

    #cClass = {key: np.unique(np.percentile(value[np.nonzero(value)], np.arange(0,100,i_class))) for key, value in data_output.items() if key != 'Transported + deposited sed in the reach [m^3/s]'}
    # cClass = {key: np.unique(np.percentile(value[np.nonzero(value)], np.arange(0,100,i_class))) for key, value in data_output.items()}
    cClass = {key: np.unique(np.percentile(data_output[key][np.nonzero(data_output[key])], np.arange(0,100,i_class))) for key in list_outputs}

    # add tot_sed_class for the class defined in def_sed_class
    def_sed_class = 0
    #data_output['Transported + deposited sed in the reach [m^3/s]'] = data_output['Transported + deposited sed in the reach [m^3/s]'][def_sed_class]
    #cClass['Transported + deposited sed in the reach [m^3/s]'] = np.unique(np.percentile(data_output['Transported + deposited sed in the reach [m^3/s]'][np.nonzero(data_output['Transported + deposited sed in the reach [m^3/s]'])], np.arange(0,100,i_class)))


    # plot starting river network
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(left=0.1, bottom= 0.3, right = 0.6) # to make space for the sliding bar

    plotvariable = data_output[plot_class]
    plot_network_stat(ReachData,plotvariable, CClass = cClass[plot_class])

    # interactively change the time step
    # create a time slider

    axSlider = plt.axes([0.1,0.1,0.5,0.05])
    axpos1 = plt.axes([0.07, 0.15, 0.03, 0.03])
    axpos2 = plt.axes([0.6, 0.15, 0.03, 0.03])


    slider = Slider(axSlider, 'Time step', valmin = 1, valmax = len(plotvariable)-1, valfmt='%0.0f', color = 'grey')
    button1 = Button(axpos1, '<', color='w', hovercolor='b')
    button2 = Button(axpos2, '>', color='w', hovercolor='b')

    def value_update(t):
        time_step = round(t)
        plot_network_dyn(ReachData,plotvariable,time = time_step, CClass = np.unique(np.around(cClass[plot_class], 5)),  ax=ax, fig=fig)
        #fig.canvas.draw_idle()
    def forward(vl):
        pos = slider.val
        slider.set_val(pos +1)
    def backward(vl):
        pos = slider.val
        slider.set_val(pos -1)


    # implement the interactivity
    slider.on_changed(value_update)
    button1.on_clicked(backward)
    button2.on_clicked(forward)

    return slider, button1, button2




# Supporting functions

def plot_network_dyn(ReachData,  plotvariable, time,  CClass, ax, fig):
    # default settings

    def_linewidth = 4
    def_cMap = 'turbo'

    #CClass =  kwargs['CClass']
    #define color map
    cMapLength = len(np.unique(CClass))
    colormap = cm.get_cmap(def_cMap, cMapLength)

    # loop through all classes
    for c in range(len(CClass)):
        # find all observations that have an attribute value that falls
        # within the current c class
        if c ==0 :
            cClassMem = np.where(plotvariable[time,:]<=CClass[c])[0]
        elif 0 < c <= len(CClass):
            cClassMem = np.where(np.logical_and(plotvariable[time,:]>CClass[c-1], plotvariable[time,:]<=CClass[c]))[0]
        else:
            cClassMem = np.where(plotvariable[time,:]>CClass[c-1])[0]

        for ll in np.asarray(cClassMem):
            if ax == None:
                ax = plt.gca()

            ax.plot(*ReachData['geometry'][ll].xy, color=colormap.colors[c,:], linewidth=def_linewidth)
    # drawing updated values
    fig.canvas.draw()




def plot_network_stat(ReachData,  plotvariable,  **kwargs):

    time = 0
    def_linewidth = 4
    cClass =  kwargs['CClass']


    # customise the legend
    def_cMap = 'turbo'
    cClass = np.unique(np.around(cClass, 5))
    cMapLength = len(cClass)
    colormap = cm.get_cmap(def_cMap, cMapLength)
    custom_lines = []
    custom_names = []

    # loop through all classes
    for c in range(len(cClass)):
        # find all observations that have an attribute value that falls
        # within the current c class
        if c ==0 :
            cClassMem = np.where(plotvariable[time,:]<=cClass[c])[0]
        elif 0 < c <= len(cClass):
            cClassMem = np.where(np.logical_and(plotvariable[time,:]>cClass[c-1], plotvariable[time,:]<=cClass[c]))[0]
        else:
            cClassMem = np.where(plotvariable[time,:]>cClass[c-1])[0]

        for ll in np.asarray(cClassMem):
            plt.plot(*ReachData['geometry'][ll].xy, color=colormap.colors[c,:], linewidth=def_linewidth)

    #plt.axis('off')
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)


    for i in range(cMapLength):
        custom_lines.append(Line2D([0], [0], color=colormap.colors[i,:], lw=4))
        if max(cClass)> 10**4:
            custom_names.append(f'{cClass[i]:.2e}')
        else:
            custom_names.append(f'{cClass[i]:.4f}')

    plt.legend(custom_lines, custom_names, loc='right', bbox_to_anchor = (1, 0, 0.3, 1), fontsize = 12)


