"""
    PyTorch specification for the hit graph dataset.
"""

# System imports
import os
import glob
import os.path as osp

# External imports
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import random_split
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.data import (Data, Dataset)

# Local imports
from datasets.graph import load_graph

import logging, sys
logger = logging.getLogger('glue')

class HitGraphDataset(Dataset):
    """PyTorch geometric dataset from processed hit information"""
    
    def __init__(self, root,
                 directed = True,
                 categorical = False,
                 transform = None,
                 pre_transform = None):
        self._directed = directed
        self._categorical = categorical
        super(HitGraphDataset, self).__init__(root, transform, pre_transform)
    
    def download(self):
        pass #download from xrootd or something later
    
    @property
    def raw_file_names(self):
        if not hasattr(self,'input_files'):
            self.input_files = glob.glob(self.raw_dir+'/*.npz')
        return [f.split('/')[-1] for f in self.input_files]
    
    @property
    def processed_file_names(self):
        if not hasattr(self,'processed_files'):
            proc_names = ['data_{}.pt'.format(idx) for idx in range(len(self.raw_file_names))]
            self.processed_files = [osp.join(self.processed_dir,name) for name in proc_names]
        return self.processed_files
    
    def __len__(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(self.processed_files[idx])
        return data
    
    def process(self):
        #convert the npz into pytorch tensors and save them
        path = self.processed_dir
        for idx,raw_path in enumerate(tqdm(self.raw_paths)):

            g = load_graph(raw_path)

            Ro = g.Ro[0].T.astype(np.int64)
            Ri = g.Ri[0].T.astype(np.int64)
            
            i_out = Ro[Ro[:,1].argsort(kind='stable')][:,0]
            i_in  = Ri[Ri[:,1].argsort(kind='stable')][:,0]
                        
            x = g.X.astype(np.float32)
            edge_index = np.stack((i_out,i_in))
            y = g.y.astype(np.int64)
            if not self._categorical:
                y = g.y.astype(np.float32)
            
            y_nodes = np.zeros(x.shape[0])
            categories = np.unique(y)
            # logger.debug('Found categories: %s', categories)
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

            # # Testing
            # some_cat3_edge = np.nonzero(y == 3)[0][0] # Get the 0th edge that is category 3
            # logger.debug('some_cat3_edge = %s', some_cat3_edge)
            # # Get the nodes connected to that edge
            # node_in = edge_index[0][some_cat3_edge]
            # node_out = edge_index[1][some_cat3_edge]
            # logger.debug('node_in = %s, y_nodes[node_in] = %s', node_in, y_nodes[node_in])
            # logger.debug('node_out = %s, y_nodes[node_out] = %s', node_out, y_nodes[node_out])
            # assert y_nodes[node_in] == 3
            # assert y_nodes[node_out] == 3

            outdata = Data(x=torch.from_numpy(x),
                           edge_index=torch.from_numpy(edge_index),
                           y=torch.from_numpy(y))
            outdata.y_nodes = torch.from_numpy(y_nodes.astype(np.int64))

            if not self._directed and not outdata.is_undirected():
                rows,cols = outdata.edge_index
                temp = torch.stack((cols,rows))
                outdata.edge_index = torch.cat([outdata.edge_index,temp],dim=-1)
                outdata.y = torch.cat([outdata.y,outdata.y])
        
            torch.save(outdata, osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))