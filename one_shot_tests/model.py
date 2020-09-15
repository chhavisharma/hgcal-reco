'''Python imports'''
import numpy as np
import awkward as ak
from math import sqrt
import networkx as nx

'''Torch imports'''
import torch
from torch import nn
import torch.nn.functional as F
from torch_cluster import knn_graph, radius_graph
from torch_scatter import scatter_mean, scatter_add

import torch_geometric
from torch_geometric.nn import max_pool_x
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.nn import EdgeConv
from torch_geometric.typing import OptTensor, PairTensor

#https://github.com/eldridgejm/unionfind
from unionfind import UnionFind
import pdb 


class SimpleEmbeddingNetwork(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, ncats_out=2, nprops_out=1, output_dim=8,
                 conv_depth=3, edgecat_depth=6, property_depth=3, k=8, aggr='add',
                 norm=torch.tensor([1./500., 1./500., 1./54., 1/25., 1./1000.]), interm_out=6):
        super(SimpleEmbeddingNetwork, self).__init__()
        
        # self.datanorm = nn.Parameter(norm, requires_grad=False)
        self.k = k
        
        start_width = 2 * (hidden_dim )
        middle_width = (3 * hidden_dim ) // 2
        
        '''Main Input Net'''
        # embedding loss
        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            #nn.LayerNorm(hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            #nn.LayerNorm(hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
        )        
        
        '''Main Edge Convolution'''
        self.edgeconvs = nn.ModuleList()
        for i in range(conv_depth):
            convnn = nn.Sequential(
                nn.Linear(start_width, middle_width),
                nn.ELU(),
                nn.Linear(middle_width, middle_width),                                             
                nn.ELU(),
                nn.Linear(middle_width, hidden_dim),                                             
                nn.ELU(),
                #nn.LayerNorm(hidden_dim),
                nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
            )
            self.edgeconvs.append(EdgeConv(nn=convnn, aggr=aggr))
        
        
        '''Embedding Output Net'''
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            #nn.LayerNorm(hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
            # nn.Linear(hidden_dim, interm_out)
        )
        # self.plotlayer = nn.Sequential(
        #     nn.Linear(interm_out, interm_out),
        #     nn.ELU(),
        #     nn.Linear(interm_out, output_dim))

        
        # edge categorization
        '''InputNetCat'''
        self.inputnet_cat =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),            
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False),
            nn.Tanh(),            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False),
            nn.Tanh()            
        )
        
        '''EdgeConcat Convolution'''
        self.edgecatconvs = nn.ModuleList()
        for i in range(edgecat_depth):
            convnn = nn.Sequential(
                nn.Linear(start_width + 2*hidden_dim + 2*input_dim, middle_width),
                nn.ELU(),
                nn.Linear(middle_width, hidden_dim),                                             
                nn.ELU(),
                nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
            )
            self.edgecatconvs.append(EdgeConv(nn=convnn, aggr=aggr))
        
        '''Edge Classifier'''
        self.edge_classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ELU(),
            nn.BatchNorm1d(num_features=hidden_dim//2, track_running_stats=False),
            nn.Linear(hidden_dim//2, ncats_out)
        )
        
        '''InputNet for Cluster Properties'''
        self.inputnet_prop =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),            
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False),
            nn.Tanh(),            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False),
            nn.Tanh()            
        )
        
        '''Convolution for Cluster Properties'''
        self.propertyconvs = nn.ModuleList()
        for i in range(property_depth):
            convnn = nn.Sequential(
                nn.Linear(start_width + 2*hidden_dim + 2*input_dim, middle_width),
                nn.ELU(),
                nn.Linear(middle_width, hidden_dim),                                             
                nn.ELU(),
                nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False)
            )
            self.propertyconvs.append(EdgeConv(nn=convnn, aggr='max'))

        '''Classifier for Cluster Properties'''
        self.property_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ELU(),
            nn.Linear(hidden_dim//2, nprops_out)
        )        

    def forward(self, x, batch: OptTensor=None):
        
        if batch is None:
            batch = torch.zeros(x.size()[0], dtype=torch.int64, device=x.device)
        
        '''Embedding1: Intermediate Latent space features (hiddenDim)'''
        x_emb = self.inputnet(x)   

        '''KNN(k neighbors) over intermediate Latent space features'''     
        for ec in self.edgeconvs:
            edge_index = knn_graph(x_emb, self.k, batch, loop=False, flow=ec.flow)
            x_emb = x_emb + ec(x_emb, edge_index)
    
        '''
        [1]
        Embedding2: Final Latent Space embedding coords from x,y,z to ncats_out
        '''
        out = self.output(x_emb)
        #plot = self.plotlayer(out)


        '''KNN(k neighbors) over Embedding2 features''' 
        # edge_index = knn_graph(out, self.k, batch, loop=False, flow=ec.flow)
        edge_index = radius_graph(out, r=0.5, batch=batch, loop=False)
        
        ''' 
        use Embedding1 to build an edge classifier
        inputnet_cat is residual to inputnet
        '''
        x_cat = self.inputnet_cat(x) + x_emb

        '''
        [2]
        Compute Edge Categories Convolution over Embedding1
        '''
        for ec in self.edgecatconvs:            
            x_cat = x_cat + ec(torch.cat([x_cat, x_emb, x], dim=1), edge_index)
        
        edge_scores = self.edge_classifier(torch.cat([x_cat[edge_index[0]], 
                                                      x_cat[edge_index[1]]], 
                                                      dim=1)).squeeze()
        

        '''
        use the predicted graph to generate disjoint subgraphs
        these are our physics objects
        '''
        objects = UnionFind(x.size()[0])
        good_edges = edge_index[:,torch.argmax(edge_scores, dim=1) > 0]
        good_edges_cpu = good_edges.cpu().numpy() 

        for edge in good_edges_cpu.T:
            objects.union(edge[0],edge[1])
        cluster_map = torch.from_numpy(np.array([objects.find(i) for i in range(x.shape[0])], 
                                                dtype=np.int64)).to(x.device)
        cluster_roots, inverse = torch.unique(cluster_map, return_inverse=True)
        # remap roots to [0, ..., nclusters-1]
        cluster_map = torch.arange(cluster_roots.size()[0], 
                                   dtype=torch.int64, 
                                   device=x.device)[inverse]
        

        ''' 
        [3]
        use Embedding1 to learn segmented cluster properties 
        inputnet_cat is residual to inputnet
        '''
        x_prop = self.inputnet_prop(x) + x_emb
        # now we accumulate over all selected disjoint subgraphs
        # to define per-object properties
        for ec in self.propertyconvs:
            x_prop = x_prop + ec(torch.cat([x_prop, x_emb, x], dim=1), good_edges)        
        props_pooled, cluster_batch = max_pool_x(cluster_map, x_prop, batch)
        cluster_props = self.property_predictor(props_pooled)    

        return out, edge_scores, edge_index, cluster_map, cluster_props, cluster_batch


if __name__ == "__main__":

    '''Test Load Model'''
    import config
    norm = torch.tensor([1./70., 1./5., 1./400.])
    print('test model')
    model = SimpleEmbeddingNetwork(input_dim=config.input_dim, 
                                hidden_dim=config.hidden_dim, 
                                output_dim=config.output_dim,
                                ncats_out=config.ncats_out,
                                nprops_out=config.nprops_out,
                                conv_depth=config.conv_depth, 
                                edgecat_depth=config.edgecat_depth, 
                                k=config.k, 
                                aggr='add',
                                norm=norm,
                                interm_out=config.interm_out
                                ).to('cuda')
    print('Model:\n',model)

    print( 'test config params')
    print( 'home : ',config.home)
    print( 'load_checkpoint_path : ',config.load_checkpoint_path)
    print( 'data_root : ',config.data_root)
    print( 'logfile_name : ',config.logfile_name)
    print( 'total_epochs : ',config.total_epochs)
    print( 'train_samples : ',config.train_samples)
    print( 'test_samples : ',config.test_samples)
    print( 'input_classes : ',config.input_classes)
    print( 'plot_dir_root : ',config.plot_dir_root)
    print( 'plot_dir_name : ',config.plot_dir_name)
    print( 'plot_path : ',config.plot_path)
    print( 'checkpoint_dir : ',config.checkpoint_dir)
    print( ',checkpoint_path : ',config.checkpoint_path)
    print( 'input_dim : ',config.input_dim)
    print( 'hidden_dim : ',config.hidden_dim)
    print( 'interm_out : ',config.interm_out)
    print( 'output_dim : ',config.output_dim)
    print( 'ncats_out : ',config.ncats_out)
    print( 'nprops_out : ',config.nprops_out)
    print( 'k : ',config.k)
    print( 'conv_depth : ',config.conv_depth)
    print( 'edgecat_depth : ',config.edgecat_depth)
    print( 'make_plots : ',config.make_plots)
    print( 'make_test_plots : ',config.make_test_plots)