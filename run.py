#!/usr/bin/env python

import sys
import json
import os

sys.path.insert(0, 'src')

from preprocess import *
from model import *

HIDDEN_LAYERS = 32
TRAIN_DATA = 1
TEST_DATA = 3

def main(targets):
    
    instances = prepare_data(1)
    feature_matrix = create_feature_matrix(instances)
    labels = extract_labels(instances)
    X, y = to_tensors(feature_matrix, labels)
    edge_index = create_edge_index(1)
    gat = GAT(X.shape[1], 32, 1)
    print(gat)
    model, out = train(gat, X, y, edge_index)


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)