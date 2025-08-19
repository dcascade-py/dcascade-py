#!/usr/bin/env python3
"""
Created on Tue Mar 10 17:26:23 2020

@author: Marco Tangi, Elisa Bozzolan and Diane Doolaeghe
"""

import os

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd




def read_network(file_path):
    # Get the file extension
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.csv':
        # Read CSV file using pandas
        return pd.read_csv(file_path)

    elif file_extension == '.shp':
        # Read Shapefile using geopandas
        return gpd.read_file(file_path)

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


def extract_Q(filename_q):
    ''' This function aims at checking automatically the delimiter used in
    the csv indicated by filename_q (; or ,)
    and if it has headers or not.
    It returns the Q matrix used later in the code
    '''
    # First check, on the delimiter
    Q_check = pd.read_csv(filename_q, header = None, sep = ',')
    if Q_check.iloc[0,:].size == 1:
        my_delimiter = ';'
    else:
        my_delimiter = ','
    # Second check, if headers are provided or not
    Q_check2 = pd.read_csv(filename_q, header=None, sep=my_delimiter)
    if Q_check2.iloc[0,0]=='yyyy/mm/dd':
        Q_matrix = pd.read_csv(filename_q, header = 0, sep=my_delimiter, index_col = 'yyyy/mm/dd')
    else:
        print("\n Warning: you did not provide headers with reach names "
              "in the Q file. We will proceed as if the Q file column order "
              "corresponds to the order of the reach as they are in the "
              "input network shape file. \n"
              )
        Q_matrix = pd.read_csv(filename_q, header = None, sep=my_delimiter)

    return Q_matrix


def check_sediment_sizes(reach_data, dmi):
    ''' Warns the user in case the modeled sediment size range excludes input sizes.
    Classes must be compatible with D16, D50, D84 defined for the reaches.
    '''    
    if min(reach_data.D16) * 1000 < np.percentile(dmi, 10, method='midpoint'):
        print("\n Warning: the minimum input D16, " + str(min(reach_data.D16) * 1000) + "mm, is lower "
               "than the 10th percentile of the defined sediment "
               "range, " + str(np.percentile(dmi, 10, method='midpoint')) + "mm \n")
                            
    if max(reach_data.D84) * 1000 > np.percentile(dmi, 90, method='midpoint'):
        print("\n Warning: the maximum input D84, " + str(max(reach_data.D84) * 1000) + "mm, is larger "
               "than the 90th percentile of the defined sediment "
               "range, " + str(np.percentile(dmi, 90, method='midpoint')) + "mm \n")        
        


def write_adj_matrix(FromN, ToN , Lngt):
    '''
    write_adj_matrix function takes a two vectors of from-nodes and to-nodes transfers them into a
     adjacency matrix . Each pair of from-nodes and to-nodes defines a river reach.
     Vectors need to be of equal size.

     The function works both 1) the downstream, and 2) the upstream
     direction. In case to, the toN is the vector of from-nodes and fromN
     must be the vector of to-nodes.

    %%% Input:
    % FromN: Vector of from-nodes
    % ToN: Vector of to-nodes


    %%% Output
    % D: Sparse adjacency matrix
    '''

    D = np.zeros([max(FromN),max(FromN)])
    outlet = FromN[np.argwhere(FromN==ToN)]

    for pos_fromN in np.unique(np.argwhere(FromN>0)):
        pos_toN = np.argwhere(FromN == ToN[pos_fromN])
                # find the position of the to node of the current fromN.
        D[pos_fromN , pos_toN] = Lngt[pos_fromN]

    return D



def extract_graph(from_n, to_n):
    """
    Receives the informations on the from_n and to_n and returns graph,
    a NetworkX direct graph corresponding to the lines in adjacency list
    format.
    """

    nodes = np.zeros([from_n.size, 2])
    nodes[:, 0] = from_n
    nodes[:, 1] = to_n
    np.savetxt("nodes", nodes, fmt = '%d') # save from_n and to_n as text, necessary to run read_adjlst.
    graph = nx.read_adjlist("nodes", create_using = nx.DiGraph, nodetype=int)
    os.remove("nodes")

    return graph



def graph_preprocessing(reach_data):
    """
    Receives the informations about reach and node ID and
    return the struct Network, that describes network connectivity.

    INPUT:
    reach_data = ReachData Class defining the features of the network reaches

    OUTPUT:

     Network  = dict containing for each node info on upstream and downstream nodes

         Attributes:
         - NH : position in the reach hierarchy of reach R. The
             higher the ranking, the higher the number of upstream node of a reach
         - upstream_node : ID of the fromN of the reaches direcly upstream each reach
         - number_upstream_nodes : max number of nodes between a reach and a source node (used to extract the ranking)
         - outlet : IDs of the outlet reach
         - sources: IDs of the source nodes


    """
    from_n = reach_data.from_n - 1
    to_n = reach_data.to_n - 1
    length = reach_data.length

    # graph represent the network as a direct graph, from which to extract
    # informations on the connection between nodes
    graph = extract_graph(from_n, to_n)

    # by running the function inverting the input from_n with to_n, we obtain a
    # graph directed from the outlet to the sources
    graph_down = extract_graph(to_n, from_n)

    # shortest downstream path to each node
    paths = dict(nx.shortest_path(graph))

    # shortest upstream path to each node
    paths_up = dict(nx.shortest_path(graph_down))

    #find the number of upstream nodes
    number_upstream_nodes = np.zeros([reach_data.n_reaches, 1])
    distance_upstream = [None] * reach_data.n_reaches

    for i in paths_up.keys():
        length_dict_old = -1
        dist = np.full((reach_data.n_reaches,), np.nan)
        for j in paths_up[i].keys():
            length_dict = len(paths_up[i][j])
            length_dict = max([length_dict_old, length_dict])
            length_dict_old = length_dict

        collection = list(paths_up[i].keys())

        for w, key in enumerate(collection):
            el = paths_up[i][key]
            add = 0
            if len(el) == 1 and el[0] == key: # source
                dist[int(el[0])] = 0
            else:
                for z in range(len(el) - 1):
                   # find relative distance
                   idx = np.where((to_n == int(el[z])) & (from_n == int(el[z+1])))
                   if len(idx[0]) != 0: # if found the combination of nodes - find the distance between them
                     dist[int(collection[w])] = length [idx] + add
                     add += length[idx]

        distance_upstream[i] = dist
        number_upstream_nodes[i] = length_dict

    """
    # incoming node count
    incoming_node_count ={}

    for node, p in paths_up.items():
        for target_node in p:
            print(target_node)
            #exclude the node itself
            if target_node != node:

                if target_node in incoming_node_count:
                    incoming_node_count[target_node] +=1
                else:
                    incoming_node_count[target_node] +=1 """


    # create upstream distance list
    upstream_distance_list = []
    for i in range(reach_data.n_reaches):
        to_fill = np.empty((1, reach_data.n_reaches)) * np.nan
        index = np.argsort(distance_upstream[i])
        mask = ~(np.isnan(np.sort(distance_upstream[i])))
        keep = index[mask]
        to_fill[0, 0:len(keep)] = keep
        upstream_distance_list.append(to_fill)

    # print('upstream paths and nodes done..')
    # find the number of downstream nodes and their relative distance (EB)
    number_downstream_nodes = np.zeros([np.size(from_n), 1])
    distance_downstream = [None] * reach_data.n_reaches

    for i in paths.keys():
        length_dict_old = -1
        dist = np.full((reach_data.n_reaches,), np.nan)
        for j in paths[i].keys():
            length_dict = len(paths[i][j])
            length_dict = max([length_dict_old, length_dict])
            length_dict = max([length_dict_old, length_dict])
            length_dict_old = length_dict

        collection = list(paths[i].keys())

        for w, key in enumerate(collection):
            el = paths[i][key]
            add = 0
            if len(el) == 1 and el[0] == key:
                dist[el[0]] = 0
            else:
                for z in range(len(el) - 1):
                    # find relative distance
                    idx = np.where((from_n == el[z]) & (to_n == el[z+1]))
                    if len(idx[0]) != 0: # if found the combination of nodes - find the distance between them
                        dist[collection[w]] = length [idx] + add
                        add += length[idx]

        distance_downstream[i] = dist
        number_downstream_nodes[i] = length_dict

    # create upstream distance list
    downstream_distance_list = []
    for i in range(reach_data.n_reaches):
        to_fill = np.empty((1, reach_data.n_reaches)) * np.nan
        index = np.argsort(distance_downstream[i])
        mask = ~(np.isnan(np.sort(distance_downstream[i])))
        keep = index[mask]
        to_fill[0, 0:len(keep)] = keep
        downstream_distance_list.append(to_fill)

    # print('downstream paths and nodes done..')
    # ID of directly upstream nodes
    upstream_node = [None] * np.size(number_upstream_nodes)
    outlet = from_n[np.argwhere(from_n==to_n)]
    sources = np.array(-1)

    for i in range(0, reach_data.n_reaches):
        node_id = from_n[i]
        upstream_node[i] =  from_n[np.argwhere(to_n ==  from_n[i])]
        if node_id == outlet:
            upstream_node[i] = upstream_node[i][upstream_node[i] != node_id].reshape(-1,1)

        # find sources
        if np.size(upstream_node[i]) == 0:
            if np.sum(sources) == -1:
                sources = from_n[i]
            else:
                sources = np.append(sources, from_n[i])

    # ID of downstream nodes
    downstream_node = [None] * reach_data.n_reaches

    for i in range(0, reach_data.n_reaches):
        node_id = from_n[i]
        downstream_node[i] = to_n[np.argwhere(from_n == node_id)]
        if node_id == outlet:
            downstream_node[i] = np.empty([1,1])

    # node hierarchy for CASCADE loop (refers to the position in ReachData, not the reach ID)
    n_hier = np.argsort(number_upstream_nodes.transpose(), kind = 'mergesort')[0]

    #create network dict to collect all output data
    network = {'number_upstream_nodes': number_upstream_nodes,
               'upstream_node': upstream_node,
               'downstream_node': downstream_node,
               'n_hier' : n_hier,
               'outlet': outlet,
               'sources': sources,
               'upstream_distance_list': upstream_distance_list,
               'downstream_distance': distance_downstream,
               'downstream_path': paths
               }
    # print('preprocessing done!')
    return network
