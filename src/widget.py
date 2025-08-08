"""
Created on Sat Apr 22 15:01:48 2023

Widget for selecting sediment transport capacity and partitioning formulas

@author:Elisa Bozzolan
"""

from tkinter import Label, OptionMenu, StringVar, Tk  # for widget


def read_user_input():


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

    # Commented because the choice for flo depth and slope reduction needs to remain optional:
    # def flow_depth_selection(event):
    #     global flo_depth
    #     if clicked.get() == "Manning":
    #         flo_depth = 1
    #     if clicked.get() == "Ferguson (2007)":
    #         flo_depth = 2
    #     root.destroy()

    # root = Tk() # create the little window
    # myLabel = Label(root, text = 'Flow depth calculation formulas')
    # myLabel.pack()
    # root.geometry("300x300")
    # #flo_depth = 2 # default value
    # options = ["Manning", "Ferguson (2007)"]
    # clicked = StringVar()
    # clicked.set("Choose an option")

    # drop = OptionMenu(root, clicked, *options, command = flow_depth_selection)
    # drop.pack(pady=20)
    # root.mainloop()


    # def slope_reduction_selection(event):
    #     global slopeRed
    #     if clicked.get() == "None":
    #         slopeRed = 1
    #     if clicked.get() == "Rickenmann (2005)":
    #         slopeRed = 2
    #     if clicked.get() == "Rickenmann & Chiari (2011)":
    #         slopeRed = 3
    #     if clicked.get() == "Nitsche et al. (2011)":
    #         slopeRed = 4
    #     root.destroy()

    # root = Tk() # create the little window
    # myLabel = Label(root, text = 'Slope reduction formulas')
    # myLabel.pack()
    # root.geometry("300x300")
    # #slopeRed = 1 # default value
    # options = ["None", "Rickenmann (2005)", "Rickenmann & Chiari (2011)", "Nitsche et al. (2011)"]
    # clicked = StringVar()
    # clicked.set("Choose an option")

    # drop = OptionMenu(root, clicked, *options, command = slope_reduction_selection)
    # drop.pack(pady=20)
    # root.mainloop()

    return indx_tr_cap, indx_partition