#import packages
import gzip
import json
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import torch

from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn

import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv, GATConv

def read_data(xbar_number):
    '''
    Read in design data given a design number
    '''
    with gzip.open('NCSU-DigIC-GraphData-2023-07-25/xbar/' + str(xbar_number) + '/xbar.json.gz','rb') as f:
        design = json.loads(f.read().decode('utf-8'))
    instances = pd.DataFrame(design['instances'])
    return instances

def read_cell_data():
    '''
    Read in cell data
    '''
    with gzip.open('NCSU-DigIC-GraphData-2023-07-25/cells.json.gz','rb') as f:
        cells = json.loads(f.read().decode('utf-8'))
    return cells

#functions for accessing GRCs
def buildBST(array,start=0,finish=-1):
    '''
    Create a binary search tree
    '''
    if finish<0:
        finish = len(array)
    mid = (start + finish) // 2
    if mid-start==1:
        ltl=start
    else:
        ltl=buildBST(array,start,mid)
    
    if finish-mid==1:
        gtl=mid
    else:
        gtl=buildBST(array,mid,finish)
        
    return((array[mid],ltl,gtl))

def getGRCIndex(x,y,xbst,ybst):
    '''
    Use a binary search tree to efficiently find a GRC given an x-location and y-location
    '''
    while (type(xbst)==tuple):
        if x < xbst[0]:
            xbst=xbst[1]
        else:
            xbst=xbst[2]
            
    while (type(ybst)==tuple):
        if y < ybst[0]:
            ybst=ybst[1]
        else:
            ybst=ybst[2]
            
    return ybst, xbst

def extract_congestion(instances, xbar_number):
    '''
    Loop through instances, locate their corresponding GRC, and aggregate demand and capacity across 
    all layers for that GRC to assign the instance those demand and capacity values
    '''
    congestion_data = np.load('NCSU-DigIC-GraphData-2023-07-25/xbar/' + str(xbar_number) + '/xbar_congestion.npz')
    xbst=buildBST(congestion_data['xBoundaryList'])
    ybst=buildBST(congestion_data['yBoundaryList'])
    demand = np.zeros(shape = [instances.shape[0],])
    capacity = np.zeros(shape = [instances.shape[0],])
    indices = []
    
    for k in range(instances.shape[0]):
        #print(k)
        xloc = instances.iloc[k]['xloc']; yloc = instances.iloc[k]['yloc']
        i,j=getGRCIndex(xloc,yloc,xbst,ybst)
        d = 0 
        c = 0
        for l in list(congestion_data['layerList']): 
            lyr=list(congestion_data['layerList']).index(l)
            d += congestion_data['demand'][lyr][i][j]
            c += congestion_data['capacity'][lyr][i][j]
        demand[k] = d
        capacity[k] = c
        indices.append((i, j))
        
    instances['routing_demand'] = demand
    instances['routing_capacity'] = capacity
    instances['grc_index'] = indices
    return instances

def extract_features(instances, cells):
    '''
    Access cells dictionary to extract features
    '''
    #extract pin counts, in and out connections, and cell sizes from cells dictionary
    individual_pins = []
    cell_widths = []
    cell_heights = []
    in_connections = []
    out_connections = []
    for k in range(instances.shape[0]):
        cell_type = instances.iloc[k]['cell']
        cell_pins = len(cells[cell_type]['terms'])
        cell_in_connections = 0
        cell_out_connections = 0
        for term in cells[cell_type]['terms']:
            if term['dir'] == 0:
                cell_in_connections += 1
            if term['dir'] == 1:
                cell_out_connections += 1
        cell_width = cells[cell_type]['width']
        cell_height = cells[cell_type]['height']
        individual_pins.append(cell_pins)
        cell_widths.append(cell_width)
        cell_heights.append(cell_height)
        in_connections.append(cell_in_connections)
        out_connections.append(cell_out_connections)

    instances['individual_pins'] = individual_pins
    instances['width'] = cell_widths
    instances['height'] = cell_heights
    instances['in_connections'] = in_connections
    instances['out_connections'] = out_connections
    return instances

def create_features(instances, xbar_number):
    '''
    Engineer features based on GRCs and connectivity
    '''
    instances['grc_pin_count'] = instances['individual_pins'].groupby(instances['grc_index']).transform('sum')
    instances['grc_cell_count'] = instances['individual_pins'].groupby(instances['grc_index']).transform('count')
    #open connectivity data
    conn=np.load('NCSU-DigIC-GraphData-2023-07-25/xbar/' + str(xbar_number) + '/xbar_connectivity.npz')
    A = coo_matrix((conn['data'], (conn['row'], conn['col'])), shape=conn['shape'])
    connectivity = A.toarray()
    instance_connections = []
    for i in range(len(connectivity)):
        connections = np.count_nonzero(connectivity[i])
        instance_connections.append(connections)
    instances['connections'] = instance_connections
    instances 
    return instances

def create_feature_matrix(instances):
    '''
    Create feature matrix by dropping columns and normalizing features
    '''
    feature_matrix = instances.drop(columns=['name', 'id', 'cell', 'orient', 'routing_demand', 'routing_capacity', 'grc_index', 'in_connections', 'out_connections', 'height'])
    scaler = MinMaxScaler()
    normalized_feature_matrix = scaler.fit_transform(feature_matrix[['xloc', 'yloc', 'individual_pins', 'width', 'grc_pin_count', 'grc_cell_count', 'connections']])
    return normalized_feature_matrix

def extract_labels(instances):
    '''
    Take demand column as the labels
    '''
    labels = instances['routing_demand']
    return labels

def to_tensors(feature_matrix, labels):
    '''
    Create tensors to be passed into model
    '''
    X = torch.tensor(feature_matrix, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.float)
    return X, y

def create_edge_index(xbar_number):
    '''
    Create edge index to be passed into model based on connectivity data
    '''
    conn=np.load('NCSU-DigIC-GraphData-2023-07-25/xbar/' + str(xbar_number) + '/xbar_connectivity.npz')

    A = coo_matrix((conn['data'], (conn['row'], conn['col'])), shape=conn['shape'])
    A = A.__mul__(A.T)

    A_coo = A.tocoo()
    edge_index = np.array([A_coo.row, A_coo.col])
    edge_index = torch.tensor(edge_index, dtype=torch.int64)
    return edge_index

def prepare_data(xbar_number):
    '''
    Outputs a dataframe with all columns necessary for processes
    '''
    instances = read_data(xbar_number)
    #nets = pd.DataFrame(design['nets'])
    cells = read_cell_data()
    instances = extract_congestion(instances, xbar_number)
    instances = extract_features(instances, cells)
    instances = create_features(instances, xbar_number)
    return instances