"""
Created on Tue Jan 10 14:35:23 2023

PLOT_NETWORK_DYN plots the river network and visualises continuos data.

This script was adapted from the Matlab version by Marco Tangi
@author: Elisa Bozzolan
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider

get_ipython().run_line_magic('matplotlib', 'qt')



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



















