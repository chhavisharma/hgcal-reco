'''Python imports'''
import numpy as np
import awkward as ak
from math import sqrt
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
# from IPython import display
import time
#https://github.com/eldridgejm/unionfind
from unionfind import UnionFind
import pdb 

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
from typing import List

'''File Imports'''
from particle_margin import TrackMLParticleTrackingDataset

'''Globals'''
torch.manual_seed(1)
norm = torch.tensor([1./70., 1./5., 1./400.])
output_dim = 2


def simple_embedding_truth(coords, truth_label_by_hits, device='cpu'):
    truth_ordering = torch.argsort(truth_label_by_hits)    
    uniques, counts = torch.unique(truth_label_by_hits, return_counts=True)

    out_truths: List[PairTensor] = []
    # for cat in uniques[:-1]:
    for cat in uniques:
        thecat = cat.item()
        in_cat = coords[truth_label_by_hits == thecat]
        not_cat = coords[truth_label_by_hits != thecat]
        
        in_cat_dists = torch.cdist(in_cat, in_cat)
        in_idxs = torch.triu_indices(in_cat_dists.size()[0], in_cat_dists.size()[0], 
                                     offset=1, device=in_cat.device)
        in_idxs = in_idxs[0] + in_cat_dists.size()[0]*in_idxs[1]
        in_cat_dists = in_cat_dists.view(-1)[in_idxs] / (uniques.size()[0] - 1)
        
        # all pairwise distances between in-category and out of category
        # there's a factor of 2 here I need to deal with
        not_cat_dists = torch.cdist(in_cat, not_cat).flatten() / (uniques.size()[0] - 1)
                
        #build the final labelled distance vectors
        dists = torch.cat([in_cat_dists, not_cat_dists], dim=0)
        truth = torch.cat([torch.ones_like(in_cat_dists, dtype=torch.int64),
                           torch.full_like(not_cat_dists, -1, dtype=torch.int64)], dim=0)
        out_truths.append((dists, truth))
        
    return out_truths

def match_cluster_targets(clusters, truth_clusters, data):
    np_truth_clusters = truth_clusters.cpu().numpy()
    true_cluster_labels = np.unique(np_truth_clusters)   
    np_clusters = clusters.cpu().numpy()
    pred_cluster_labels = np.unique(np_clusters)
    pred_cluster_mask = np.ones_like(np_truth_clusters, dtype=np.bool)
        
    #print(data)    
    
    #print('match_cluster_targets')
    #print(np_clusters)
    #print(np_truth_clusters)
    #print(true_cluster_labels)
    #print(pred_cluster_labels)
    indices = np.arange(np_clusters.size, dtype=np.int64)
    #print(indices)
    pred_clusters = []
    for label in pred_cluster_labels:
        pred_clusters.append(indices[np_clusters == label])
    #print(pred_clusters)
    
    # make pt weighting vector
    max_pt = torch.max(data.truth_pt).item()
    #print(max_pt)
    
    matched_pred_clusters = []
    true_cluster_properties = []
    for label in true_cluster_labels:
        true_indices = indices[np_truth_clusters == label]        
        best_pred_cluster = -1
        best_iou = 0
        for i, pc in enumerate(pred_clusters):
            isec = np.intersect1d(true_indices, pc)
            iun = np.union1d(true_indices, pc)
            iou = isec.size/iun.size
            if best_pred_cluster == -1 or iou > best_iou:
                best_pred_cluster = i
                best_iou = iou
        matched_pred_clusters.append(best_pred_cluster)
        
        # now make the properties vector
        thebc = torch.unique(data.y_particle_barcodes[data.y == label]).item()
        select_truth = (data.truth_barcodes == thebc)
        true_cluster_properties.append(1./data.truth_pt[select_truth])
        #[data.truth_eta[select_truth], data.truth_phi[select_truth]]
    matched_pred_clusters = np.array(matched_pred_clusters, dtype=np.int64)
    pred_indices = torch.from_numpy(matched_pred_clusters).to(clusters.device)
    #print(pred_indices)
    
    true_cluster_properties = np.array(true_cluster_properties, dtype=np.float)
    y_properties = torch.from_numpy(true_cluster_properties).to(clusters.device).float()
    print(y_properties)    
    
    #print('match_cluster_targets')
    return pred_indices, y_properties

class SimpleEmbeddingNetwork(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, ncats_out=2, nprops_out=1, output_dim=8,
                 conv_depth=3, edgecat_depth=6, property_depth=3, k=8, aggr='add',
                 norm=torch.tensor([1./500., 1./500., 1./54., 1/25., 1./1000.])):
        super(SimpleEmbeddingNetwork, self).__init__()
        
        # self.datanorm = nn.Parameter(norm, requires_grad=False)
        self.k = k
        
        start_width = 2 * (hidden_dim )
        middle_width = (3 * hidden_dim ) // 2
          
        # embedding loss
        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            #nn.LayerNorm(hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            #nn.LayerNorm(hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim)
        )        
        
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
                nn.BatchNorm1d(num_features=hidden_dim)
            )
            self.edgeconvs.append(EdgeConv(nn=convnn, aggr=aggr))
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            #nn.LayerNorm(hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # edge categorization
        self.inputnet_cat =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),            
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.Tanh(),            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.Tanh()            
        )
        
        self.edgecatconvs = nn.ModuleList()
        for i in range(edgecat_depth):
            convnn = nn.Sequential(
                nn.Linear(start_width + 2*hidden_dim + 2*input_dim, middle_width),
                nn.ELU(),
                nn.Linear(middle_width, hidden_dim),                                             
                nn.ELU(),
                nn.BatchNorm1d(num_features=hidden_dim)
            )
            self.edgecatconvs.append(EdgeConv(nn=convnn, aggr=aggr))
        
        self.edge_classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ELU(),
            nn.BatchNorm1d(num_features=hidden_dim//2),
            nn.Linear(hidden_dim//2, ncats_out)
        )
        
        # property prediction
        self.inputnet_prop =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),            
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.Tanh(),            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.Tanh()            
        )
        
        self.propertyconvs = nn.ModuleList()
        for i in range(property_depth):
            convnn = nn.Sequential(
                nn.Linear(start_width + 2*hidden_dim + 2*input_dim, middle_width),
                nn.ELU(),
                nn.Linear(middle_width, hidden_dim),                                             
                nn.ELU(),
                nn.BatchNorm1d(num_features=hidden_dim)
            )
            self.propertyconvs.append(EdgeConv(nn=convnn, aggr='max'))

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
        
        x_emb = self.inputnet(x)        
        
        for ec in self.edgeconvs:
            edge_index = knn_graph(x_emb, self.k, batch, loop=False, flow=ec.flow)
            x_emb = x_emb + ec(x_emb, edge_index)
        
        out = self.output(x_emb)
        edge_index = knn_graph(out, self.k, batch, loop=False, flow=ec.flow)
        
        # use the embedded space to build an edge classifier
        x_cat = self.inputnet_cat(x) + x_emb
        for ec in self.edgecatconvs:            
            x_cat = x_cat + ec(torch.cat([x_cat, x_emb, x], dim=1), edge_index)
        
        edge_scores = self.edge_classifier(torch.cat([x_cat[edge_index[0]], 
                                                      x_cat[edge_index[1]]], 
                                                      dim=1)).squeeze()
        
        # use the predicted graph to generate disjoint subgraphs
        # these are our physics objects
        objects =UnionFind(x.size()[0])
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
        
        x_prop = self.inputnet_prop(x) + x_emb
        # now we accumulate over all selected disjoint subgraphs
        # to define per-object properties
        for ec in self.propertyconvs:
            x_prop = x_prop + ec(torch.cat([x_prop, x_emb, x], dim=1), good_edges)        
        props_pooled, cluster_batch = max_pool_x(cluster_map, x_prop, batch)
        cluster_props = self.property_predictor(props_pooled)    
        
        return out, edge_scores, edge_index, cluster_map, cluster_props, cluster_batch

def load_data(root_path):
    data = TrackMLParticleTrackingDataset(root=root_path,
                                        layer_pairs_plus=True,
                                        pt_min=0,
                                        n_events=100, n_workers=1)
    print('number of events read = ',len(data))
    print(data.__len__())
    return data


if __name__ == "__main__" :

    root = '/home/csharma/prototyping/data/train_1/'
    data = load_data(root)

    model = SimpleEmbeddingNetwork(input_dim=3, 
                                hidden_dim=32, 
                                output_dim=output_dim,
                                ncats_out=2,
                                nprops_out=1,
                                conv_depth=3, 
                                edgecat_depth=6, 
                                k=8, 
                                aggr='add',
                                norm=norm).to('cuda')

    opt = torch.optim.AdamW([
                            {'params': list(model.inputnet.parameters()) + list(model.edgeconvs.parameters()) + list(model.output.parameters())},
                            {'params': list(model.inputnet_cat.parameters()) + list(model.edgecatconvs.parameters()) + list(model.edge_classifier.parameters()), 'lr': 0.0},
                            {'params': list(model.inputnet_prop.parameters()) + list(model.propertyconvs.parameters()) + list(model.property_predictor.parameters()), 'lr': 0.0}
                            ], lr=2.5e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.65, patience=30)
    # sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20, eta_min=0)

    truth_cache = []

    n_samples = 1

    n = 10*n_samples
    color_cycle = plt.cm.coolwarm(np.linspace(0.1,0.9,n))
    marker_hits = ['^','v','s','h']
    marker_centers = ['+','1','x','3']
    all_loss = []

    converged_embedding = False
    converged_categorizer = False

    make_plots = True

    print('begin training')
    pdb.set_trace()
    for e in range(1000):
        
        if((e+1)%200==0):
            print('epoch',e)
            
        avg_loss = 0
        
        if make_plots:
            plt.clf()

        if opt.param_groups[0]['lr'] < 1e-4 and not converged_embedding:
            converged_embedding = True
            opt.param_groups[1]['lr'] = 1e-4
            opt.param_groups[1]['lr'] = 1e-3
            
        if opt.param_groups[1]['lr'] < 1e-4 and not converged_categorizer and converged_embedding:
            converged_categorizer = True
            opt.param_groups[2]['lr'] = 1e-3
        
        
        print('data len', len(data))
        print(data, data)
        
        for idata, d in enumerate(data[0:n_samples]):
            d_gpu = d.to('cuda')
            
            y_orig = d_gpu.y
            
            d_gpu.x = d_gpu.x[d_gpu.y < 3] # just take the first three tracks
            d_gpu.x = (d_gpu.x - torch.min(d_gpu.x, axis=0).values)/(torch.max(d_gpu.x, axis=0).values - torch.min(d_gpu.x, axis=0).values) # Normalise
            d_gpu.y_particle_barcodes = d_gpu.y_particle_barcodes[d_gpu.y < 3]
            d_gpu.y = d_gpu.y[d_gpu.y < 3]
            
            
            coords, edge_scores, edges, cluster_map, cluster_props, cluster_batch = model(d_gpu.x)
            '''
            #------------  DANIEL TRAINING VERSION ------------------
            reference = coords.index_select(0, e_spatial[1])
            neighbors = coords.index_select(0, e_spatial[0])
            d = torch.sum((reference - neighbors)**2, dim=-1)
            
            hinge_truth = (d_gpu.y[e_spatial[0]] == d_gpu.y[e_spatial[1]]).float()
            hinge_truth[hinge_truth == 0] = -1
            print(hinge_truth)
            
            loss = torch.nn.functional.hinge_embedding_loss(d, hinge_truth, margin=1.0, reduction="mean")
            #==============================================
            '''
            
            #-------------- LINDSEY TRAINING VERSION ------------------
            print('compute hinge loss ...')
            multi_simple_hinge = simple_embedding_truth(coords, d_gpu.y, device='cuda')
            
            print('scatter mean ...')
            centers = scatter_mean(coords, d_gpu.y, dim=0, dim_size=(torch.max(d_gpu.y).item()+1))
            
            if make_plots:
                if output_dim==3:
                    fig = plt.figure(figsize=(20,20))
                    ax = fig.add_subplot(111, projection='3d')
                    for i in range(centers.size()[0]):  
                        ax.scatter(coords[d_gpu.y == i,0].cpu().detach().numpy(), 
                            coords[d_gpu.y == i,1].cpu().detach().numpy(),
                            coords[d_gpu.y == i,2].cpu().detach().numpy(),
                            color=color_cycle[2*idata + i], marker = marker_hits[i%4], s=100);

                        ax.scatter(centers[i,0].cpu().detach().numpy(), 
                            centers[i,1].cpu().detach().numpy(), 
                            centers[i,2].cpu().detach().numpy(), 
                            marker=marker_centers[i%4], color=color_cycle[2*idata+i], s=100); 
                elif output_dim==2:
                    for i in range(centers.size()[0]):
                            plt.scatter(coords[d_gpu.y == i,0].cpu().detach().numpy(), 
                                        coords[d_gpu.y == i,1].cpu().detach().numpy(),
                                        color=color_cycle[2*idata + i], marker = marker_hits[i%4]);
                            plt.scatter(centers[i,0].cpu().detach().numpy(), 
                                        centers[i,1].cpu().detach().numpy(), 
                                        marker=marker_centers[i%4], color=color_cycle[2*idata+i])  ;  
        
                # display.clear_output(wait=True)
                # display.display(plt.gcf())  
                plt.savefig('train_plt.png')   

            
            hinges = torch.cat([F.hinge_embedding_loss(d**2, y, margin=1.0, reduction='mean')[None] 
                                for d, y in multi_simple_hinge],
                            dim=0)
            
            y_edgecat = (d_gpu.y[edges[0]] == d_gpu.y[edges[1]]).long()
            loss_ce = F.cross_entropy(edge_scores, y_edgecat, reduction='mean')
            
            pred_cluster_match, y_properties = match_cluster_targets(cluster_map, d_gpu.y, d_gpu)
            
            loss_mse = F.mse_loss(cluster_props[pred_cluster_match].squeeze(), y_properties, reduction='mean')
            
            loss = hinges.mean() + loss_ce + loss_mse
            
            avg_loss += loss.item()
            all_loss.append(avg_loss)
            
            loss.backward()
                    
            print(e, idata, 'loss / LR /centers -->>\n', 
                loss.item(), loss_ce.item(), loss_mse.item(), 
                '\n', opt.param_groups[0]['lr'], opt.param_groups[1]['lr'], opt.param_groups[2]['lr'],
                '\n', 'N_true_edges / accuracy -->> ', y_edgecat.sum().item(), '/', (torch.argmax(edge_scores, dim=1) == y_edgecat).sum().item()/(y_edgecat.size()[0]),
                '\n centers --> ', centers,
                '\n cluster_properties -> ', cluster_props
                )
            #time.sleep(3.)
        opt.step()
        sched.step(avg_loss)



    print("Training Complted .\n")
    pdb.set_trace()
    print(1/y_properties)
    print(pred_cluster_match)
    print(1/cluster_props[pred_cluster_match].squeeze())

    n_clusters = data[0].y[data[0].y < 3].max().item() + 1

    fig, ax = plt.subplots()

    for i in range(n_clusters):
        mapped_i = pred_cluster_match[i].item()
        r = data[0].x[data[0].y < 3][cluster_map == mapped_i,0].cpu().detach().numpy()
        phi = data[0].x[data[0].y < 3][cluster_map == mapped_i,1].cpu().detach().numpy()
        z = data[0].x[data[0].y < 3][cluster_map == mapped_i,2].cpu().detach().numpy()
        ax.scatter(r*np.cos(phi), 
                r*np.sin(phi),
                color=color_cycle[2*idata + i], marker = marker_hits[i%4], s=100);
        ax.text((r*np.cos(phi)).mean(), (r*np.sin(phi)).mean(), 'pt_pred = %.3f\npt_true = %.3f' % (1./cluster_props[mapped_i].item(), 1/y_properties[i]))

    plt.savefig('learned_clusters.png')

    
    print("UnionFind Roots:")
    objects = UnionFind(coords.size()[0])
    good_edges = edges.t()[torch.argmax(edge_scores, dim=1) > 0].cpu().numpy()
    for edge in good_edges:
        objects.union(edge[0],edge[1])
        #objects.union(edge[1],edge[0])
    roots = np.array([objects.find(i) for i in range(coords.size()[0])], dtype=np.int64)
    print(roots)

    print('exit')

    # print("Plots:")
    # fig = plt.figure(figsize=(20,10))
    # plt.plot(batch_loss, linewidth=5)
    # plt.plot(batchlayer_loss)
    # plt.plot(batchnosched_loss)
    # plt.plot(batcheuclidean_loss)
    # plt.plot(only_radius_graph_loss, linewidth=5)
    # plt.plot(batchlayerallexamples_loss, linewidth=5)
    # plt.plot(batchlayergraph_loss, linewidth=5)
    # plt.plot(batchlayergraphdepth3_loss, linewidth=5)
    # plt.yscale('log')
    # plt.legend(["Batch norm", "Batch and layer norm", "Batch norm and no scheduler", "Batch norm with Euclidean dist (vs. dist^2)", "No norms, No sched, Euclidean dist", "Batch and layer norm (on all pairs)", "Batch and layer norm with message passing (depth 1)", "Batch and layer norm with message passing (depth 3)"])
