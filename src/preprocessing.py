import gzip
import json
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors


def buildBST(array,start=0,finish=-1):
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


pathname = 'NCSU-DigIC-GraphData-2023-07-25'


def load_data(pathname):
    # Load instance and net data
    with gzip.open(pathname + '/xbar/1/xbar.json.gz', 'rb') as f:
        design = json.loads(f.read().decode('utf-8'))

    instances = pd.DataFrame(design['instances'])
    nets = pd.DataFrame(design['nets'])
    
    # Load cells data
    with gzip.open(pathname + '/cells.json.gz', 'rb') as f:
        cells = json.loads(f.read().decode('utf-8'))
        
    return instances, nets, cells


def load_congestion_data(pathname):
    # Load congestion data
    congestion_data = np.load(pathname + '/xbar/1/xbar_congestion.npz')
    return congestion_data


def assign_demand_congestion_capacity(instances, congestion_data):
    # Get X and Y BST
    xbst = buildBST(congestion_data['xBoundaryList'])
    ybst = buildBST(congestion_data['yBoundaryList'])
    
    # Create empty arrays for demand, congestion, and capacity
    demand = np.zeros(shape=[instances.shape[0],])
    congestion = np.zeros(shape=[instances.shape[0],])
    capacity = np.zeros(shape=[instances.shape[0],])
    
    # Loop through congestion data and assign corresponding demand, capacity, congestion values to instances
    for k in range(instances.shape[0]):
        xloc = instances.iloc[k]['xloc']
        yloc = instances.iloc[k]['yloc']
        i, j = getGRCIndex(xloc, yloc, xbst, ybst)
        
        d = 0
        c = 0
        for l in list(congestion_data['layerList']): 
            lyr=list(congestion_data['layerList']).index(l)
            d += congestion_data['demand'][lyr][i][j]
            c += congestion_data['capacity'][lyr][i][j]
            
        demand[k] = d
        congestion[k] = d - c
        capacity[k] = c
    
    # Add arrays to DataFrames
    instances['routing_demand'] = demand
    instances['congestion'] = congestion
    instances['capacity'] = capacity
    
    return instances, xbst, ybst


def compute_features(instances, cells, xbst, ybst):
    # Empty list for GRC index and pin density
    grc_pos = []
    pin_count = []
    
    for i in range(len(instances)):
        point = instances.loc[i]
        j = getGRCIndex(point.xloc, point.yloc, xbst, ybst)
        grc_pos.append(j)
        pin_count.append(len(cells[instances.loc[i].cell]['terms']))
        
    instances['GRC_Index'] = grc_pos
    instances['pin_count'] = pin_count
    instances['pin_total'] = instances.groupby('GRC_Index')['pin_count'].transform('sum')
    
    return instances


def preprocess_data(pathname, congestion_data):
    # Load all congestion data and assign them to their corresponding instances
    instances, nets, cells = load_data(pathname)
    xbst = buildBST(congestion_data['xBoundaryList'])
    ybst = buildBST(congestion_data['yBoundaryList'])
    instances = assign_demand_congestion_capacity(instances, congestion_data)
    instances = compute_features(instances, cells, xbst, ybst)

    return instances


def save_preprocessed_data(data, output_file):
    # Save features to new preprocessed CSV
    data.to_csv(output_file, index=False)


# def main(pathname):
#     congestion_data = np.load(pathname + '/xbar/1/xbar_congestion.npz')
#     instances, nets, cells = load_data(pathname)
#     instances = assign_demand_congestion_capacity(instances, congestion_data)
#     instances = compute_features(instances, cells)
#     output_file = 'preprocessed_data.csv'
#     save_preprocessed_data(instances, output_file)