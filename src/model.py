#import packages
import gzip
import json
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import torch

import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv

class GAT(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        #arbitrarily set output of second layer to have the same number of dimensions as dim_in
        self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
        #self.out = Linear(dim_in, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=5e-4)

    def forward(self, x, edge_index):
        #h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        #h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        #h = F.elu(h)
        #h = self.out(h)
        return h

#def accuracy(pred_y, y):
#    return ((pred_y == y).sum() / len(y)).item()

def train(model, x, y, edge_index):
    criterion = torch.nn.MSELoss()
    optimizer = model.optimizer
    epochs = 1000

    model.train()
    for epoch in range(epochs+1):
        # Training
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out, y)
        #acc = accuracy(out[train_mask].argmax(dim=1), y[train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        #val_loss = criterion(out[val_mask], y[val_mask])
        #val_acc = accuracy(out[val_mask].argmax(dim=1), y[val_mask])

        # Print metrics every 10 epochs
        if(epoch % 100 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                  )
          
    return model, out

def test(model, data):
    """Evaluate the model on test set and print the accuracy score."""
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc