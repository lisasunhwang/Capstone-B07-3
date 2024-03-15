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

class GAT(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATConv(dim_in, dim_h, heads=heads)
        self.gat2 = GATConv(dim_h*heads, dim_h, heads=heads)
        self.linear = nn.Linear(dim_h*heads, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0175, weight_decay=5e-4)

    def forward(self, x, edge_index):
        #h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        #h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        h = F.elu(h)
        h = self.linear(h).squeeze(1)

        return h

def train(model, x, y, edge_index):
    criterion = torch.nn.MSELoss()
    optimizer = model.optimizer
    epochs = 500

    model.train()
    for epoch in range(epochs+1):
        # training
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()


        # print metrics every 10 epochs
        if(epoch % 100 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} '
                  )
        if(epoch == epochs):
            print('---Training Complete---')
            #print parameters
            #for name, param in model.named_parameters():
            #    print(name, param.grad.norm())
          
    return model, out

def test(model, x, y, edge_index):
    model.eval()
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        out = model(x, edge_index)
        mse = criterion(out, y)
    return out, mse