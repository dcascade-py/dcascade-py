#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:26:23 2020

@author: Marco Tangi and Elisa Bozzolan
"""
import numpy as np
import networkx as nx
import os

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

'''##########################################################################################'''

def extract_graph (FromN,ToN):
    '''
    EXTRACT_GRAPH receives the informations on the FromN and ToN and 
    returns G, a NetworkX direct graph corresponding to the lines in adjacency list 
    format.
    
    '''  
    
    nodes = np.zeros([FromN.size,2])
    nodes[:,0] = FromN
    nodes[:,1] = ToN    
    np.savetxt("nodes",nodes,fmt = '%d') #save fromN and ToN as text, necessary to run read_adjlst
    G = nx.read_adjlist("nodes",create_using = nx.DiGraph)
    os.remove("nodes") 
    
    return G
    
'''##########################################################################################'''



def graph_preprocessing( ReachData ):
    '''
    GRAPH_PREPROCESSING receives the informations about reach and node ID and
    return the struct Network, that describes network connectivity
    
     INPUT: 
    
     ReachData    = Dataframe from shapefile defining the features of the network reaches
    
    ---
     OUTPUT: 
    
     Network  = dict containing for each node info on upstream and downstream nodes
         
         Attributes:
         - NH : position in the reach hierarchy of reach R. The 
             higher the ranking, the higher the number of upstream node of a reach
         - Upstream_Node : ID of the fromN of the reaches direcly upstream each reach
         - numberUpstreamNodes : max number of nodes between a reach and a source node (used to extract the ranking)
         - outlet : IDs of the outlet reach
         - sources: IDs of the source nodes
    
      
    '''
    FromN = np.array(ReachData.FromN) -1
    ToN = np.array(ReachData.ToN) -1
    length = np.array(ReachData.Length)
    

    # G represent the network as a direct graph, from which to extract 
    # informations on the connection between nodes
    
    G = extract_graph (FromN,ToN)
    # by running the function inverting the input FromN with ToN, we obtain a 
    # graph directed from the outlet to the sources
    G_down = extract_graph (ToN,FromN) 
    
    #shortest downstream path to each node
    paths = nx.shortest_path(G)
    
    
    #shortest upstream path to each node
    paths_up = nx.shortest_path(G_down)  
    
    
    #find the number of upstream nodes
    numberUpstreamNodes = np.zeros([np.size(FromN),1])
    distanceUpstream = [None]*len(paths_up)

    for i in paths_up.keys():
        length_dict_old = -1  
        dist = np.full((len(FromN),),np.nan)
        for j in paths_up[str(i)].keys():
            length_dict = len(paths_up[i][j])
            length_dict = max([length_dict_old,length_dict])
            length_dict_old = length_dict
        
        collection = list(paths_up[str(i)].keys())
        
        for w, key in enumerate(collection):           
            el = paths_up[str(i)][key]
            add = 0 
            if len(el) == 1 and el[0] == key: # source
                dist[int(el[0])] = 0 
            else:
                for z in range(len(el)-1):
                   # find relative distance 
                   idx = np.where((ToN == int(el[z])) & (FromN == int(el[z+1])))
                   if len(idx[0]) != 0: # if found the combination of nodes - find the distance between them 
                     dist[int(collection[w])] = length [idx] + add
                     add += length[idx]
        
        distanceUpstream[int(i)] = dist
        numberUpstreamNodes[int(i)] = length_dict
        
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
    for i in range(len(ReachData)):
        to_fill = np.empty((1,len(ReachData)))*np.nan
        index = np.argsort(distanceUpstream[i])    
        mask = ~(np.isnan(np.sort(distanceUpstream[i])))
        keep = index[mask]
        to_fill[0, 0:len(keep)] = keep
        upstream_distance_list.append(to_fill)
        
    print('upstream paths and nodes done..')
    #find the number of downstream nodes and their relative distance (EB)
    numberDownstreamNodes = np.zeros([np.size(FromN),1])
    distanceDownstream = [None]*len(paths_up)

    for i in paths.keys():
        length_dict_old = -1  
        dist = np.full((len(FromN),),np.nan)
        for j in paths[str(i)].keys():
            length_dict = len(paths[i][j])
            length_dict = max([length_dict_old,length_dict])
            length_dict = max([length_dict_old,length_dict])
            length_dict_old = length_dict
        
        collection = list(paths[str(i)].keys())
        
        for w, key in enumerate(collection):           
            el = paths[str(i)][key]
            add = 0 
            if len(el) == 1 and el[0] == key:
                dist[int(el[0])] = 0 
            else:
                for z in range(len(el)-1):
                   # find relative distance 
                   idx = np.where((FromN == int(el[z])) & (ToN == int(el[z+1])))
                   if len(idx[0]) != 0: # if found the combination of nodes - find the distance between them 
                     dist[int(collection[w])] = length [idx] + add
                     add += length[idx]
        
        distanceDownstream[int(i)] = dist 
        numberDownstreamNodes[int(i)]  = length_dict        
        
    # create upstream distance list
    downstream_distance_list = []
    for i in range(len(ReachData)):
        to_fill = np.empty((1,len(ReachData)))*np.nan
        index = np.argsort(distanceDownstream[i])    
        mask = ~(np.isnan(np.sort(distanceDownstream[i])))
        keep = index[mask]
        to_fill[0, 0:len(keep)] = keep
        downstream_distance_list.append(to_fill)
    
    print('downstream paths and nodes done..')
    # ID of directly upstream nodes
    Upstream_Node = [None] * np.size(numberUpstreamNodes)
    outlet = FromN[np.argwhere(FromN==ToN)]
    sources = np.array(-1)
    
    for i in range(0,np.size(numberUpstreamNodes)):
        node_id = FromN[i]            
        Upstream_Node[i] =  FromN[np.argwhere(ToN ==  FromN[i])]
        if node_id == outlet:
            Upstream_Node[i] = Upstream_Node[i][Upstream_Node[i]!=node_id].reshape(-1,1)
            
        #find sources
        if np.size(Upstream_Node[i]) == 0:
            if np.sum(sources) == -1:
                sources =  FromN[i]
            else:
                sources = np.append(sources, FromN[i])
    
    
    # ID of downstream nodes
    Downstream_Node = [None] * np.size(numberDownstreamNodes)
    
    for i in range(0,np.size(numberDownstreamNodes)):
        node_id = FromN[i]            
        Downstream_Node[i] =  ToN[np.argwhere(FromN ==  node_id)]
        if node_id == outlet:
            Downstream_Node[i] = np.empty([1,1])
            
    
    # node hierarchy for CASCADE loop (refers to the position in ReachData, not the reach ID)
    Nh = np.argsort(numberUpstreamNodes.transpose() , kind = 'mergesort')[0]

    
    #create Network dict to collect all output data
    Network ={'numberUpstreamNodes':numberUpstreamNodes,'Upstream_Node':Upstream_Node, 'Downstream_Node': Downstream_Node , 
              'NH' : Nh , 'outlet':outlet , 'sources':sources,
              'upstream_distance_list':upstream_distance_list, 'downstream_distance': distanceDownstream,
              'downstream_path': paths}
    print('preprocessing done!')
    return Network
