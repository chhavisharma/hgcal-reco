import os
import glob
import os.path as osp
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import random_split
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.data import (Data, Dataset)

import sys
import pdb 

def load_graph(filename):
    """ Reade a single graph from NPZ """
    with np.load(filename) as f:
        return sparse_to_graph(**dict(f.items()))

if __name__ == "__main__":

    ''' Read Files '''
    events = []
    root = './../../data/raw/'
    processed_dir = './../../data/node/processed/'

    for file in os.listdir(root):
        if file.endswith(".npz"):
            path = os.path.join(root, file)
            events.append(path)

    for idx,raw_path in enumerate(tqdm(events)):
        
        g = load_graph(raw_path)

        # pdb.set_trace()
        Ro = g.Ro[0].T.astype(np.int64)
        Ri = g.Ri[0].T.astype(np.int64)
        
        i_out = Ro[Ro[:,1].argsort(kind='stable')][:,0]
        i_in  = Ri[Ri[:,1].argsort(kind='stable')][:,0]
                    
        x = g.X.astype(np.float32)
        edge_index = np.stack((i_out,i_in))
        y = g.y.astype(np.int64)

        y_nodes = np.zeros(x.shape[0])
        categories = np.unique(y)
        # print('Found categories: %s', categories)

        for i_category in categories:
            # Get all the edges belonging to this category
            indices_edges_this_category = (y == i_category)
            # Get all the nodes belonging to this category
            # (Use both ingoing and outgoing)
            node_indices_this_category = np.unique(np.concatenate((
                edge_index[0][indices_edges_this_category],
                edge_index[1][indices_edges_this_category]
                )))
            # Set the y value to the category
            y_nodes[node_indices_this_category] = i_category

        outdata = Data(x=torch.from_numpy(x),edge_index=torch.from_numpy(edge_index),y=torch.from_numpy(y))
        outdata.y_nodes = torch.from_numpy(y_nodes.astype(np.int64))
        
        _directed=False
        if not _directed and not outdata.is_undirected():
            rows,cols = outdata.edge_index
            temp = torch.stack((cols,rows))
            outdata.edge_index = torch.cat([outdata.edge_index,temp],dim=-1)
            outdata.y = torch.cat([outdata.y,outdata.y])
    
        torch.save(outdata, osp.join(processed_dir, 'data_{}.pt'.format(idx)))

        # pdb.set_trace()
    
    print('Process completed.')