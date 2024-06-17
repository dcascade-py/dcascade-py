# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 15:01:48 2023

Widget for selecting sediment transport capacity and partitioning formulas 

@author:Elisa Bozzolan
"""

from tkinter import Label, Tk, StringVar, OptionMenu  # for widget

def read_user_input(): 
    
    def flow_depth_selection(event):
        global flo_depth
        if clicked.get() == "Manning (----)":
            flo_depth = 1
        if clicked.get() == "Ferguson (2007)":
            flo_depth = 2
        root.destroy()
        
    root = Tk() # create the little window
    myLabel = Label(root, text = 'Flow depth caluculation formulas')  
    myLabel.pack()
    root.geometry("300x300")
    #flo_depth = 2 # default value
    options = ["Manning (----)", "Ferguson (2007)"]
    clicked = StringVar()
    clicked.set("Choose an option")
    
    drop = OptionMenu(root, clicked, *options, command = flow_depth_selection)
    drop.pack(pady=20)    
    root.mainloop() 
    
    def formula_selection(event): 
        global indx_tr_cap
        if clicked.get() == "Engelund and Hansen 1967": 
            indx_tr_cap = 3
        elif clicked.get() == "Wilkock and Crowe 2003": 
            indx_tr_cap = 2
        elif clicked.get() == "Parker and Klingeman 1982":  
            indx_tr_cap = 1
        elif clicked.get() == "Yang formula 1989": 
            indx_tr_cap = 4
        elif clicked.get() == "Wong and Parker 2006": 
            indx_tr_cap = 5
        elif clicked.get() == "Ackers and White formula 1973": 
            indx_tr_cap = 6
        elif clicked.get() == "Rickenmann 2001":
            indx_tr_cap = 7
        root.destroy()
    
    root = Tk() # create the little window
    myLabel = Label(root, text = 'Sediment transport formulas')  
    myLabel.pack()
    root.geometry("300x300")
    #indx_tr_cap = 2 # default value
    options = ["Engelund and Hansen 1967", "Wilkock and Crowe 2003", "Parker and Klingeman 1982",
               "Yang formula 1989", "Wong and Parker 2006", "Ackers and White formula 1973", "Rickenmann 2001"]
    clicked = StringVar()
    clicked.set("Choose an option")
    
    drop = OptionMenu(root, clicked, *options, command = formula_selection)
    drop.pack(pady=20)    
    root.mainloop() 
    
    # fractioning method selection 
    def partitioning_selection(event): 
        global indx_partition
        if clicked.get() == "Direct": 
            indx_partition = 1
        elif clicked.get() == "Bed material fraction (BMF)": 
            indx_partition = 2
        elif clicked.get() == "Transport capacity function (TCF)":  
            indx_partition = 3
        elif clicked.get() == "Shear stress correction approach":  
            indx_partition = 4
        root.destroy()        
        
        
    root = Tk()
    myLabel = Label(root, text = "Partitioning methods")
    myLabel.pack()
    root.geometry("300x300") 
    if indx_tr_cap != 2 and indx_tr_cap != 1: # put here the formulas options that have forced choice in the partitioning 
        options = ["Direct", "Bed material fraction (BMF)", "Transport capacity function (TCF)", "Shear stress correction approach"]
    else: 
        options = ["Shear stress correction approach"]
    clicked = StringVar()
    clicked.set("Choose an option")
    
    drop = OptionMenu(root, clicked, *options, command = partitioning_selection)
    drop.pack(pady=20)
    root.mainloop()
    
    return flo_depth, indx_tr_cap, indx_partition