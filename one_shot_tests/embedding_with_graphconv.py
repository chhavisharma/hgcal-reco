'''Python imports'''
import numpy as np
import awkward as ak
from math import sqrt
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from  mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.cm as cm

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

'''File imports'''
from particle_margin import TrackMLParticleTrackingDataset

'''Globals'''
torch.manual_seed(1)
norm = torch.tensor([1./70., 1./5., 1./400.])

norm = mpl.colors.Normalize(vmin=-20, vmax=10)
cmap = cm.hot
m = cm.ScalarMappable(norm=norm, cmap=cmap)
ctr = 0

def simple_embedding_truth(coords, truth_label_by_hits, device='cpu'):
    truth_ordering = torch.argsort(truth_label_by_hits)    
    uniques, counts = torch.unique(truth_label_by_hits, return_counts=True)
    out_truths: List[PairTensor] = []

    '''
    for each latent space 2d -coordinates of category cat, compute all incat and outofcat indices,
    then compute pnorm distace with both kind of categories,
    return distances and truths(in or out)
    '''
    for cat in uniques:

        thecat = cat.item()
        in_cat = coords[truth_label_by_hits == thecat]
        not_cat = coords[truth_label_by_hits != thecat]
        
        in_cat_dists = torch.cdist(in_cat, in_cat)
        in_idxs = torch.triu_indices(in_cat_dists.size()[0], in_cat_dists.size()[0], 
                                     offset=1, device=in_cat.device)
        in_idxs = in_idxs[0] + in_cat_dists.size()[0]*in_idxs[1]
        in_cat_dists = in_cat_dists.view(-1)[in_idxs] / (uniques.size()[0] - 1)
        
        '''
        all pairwise distances between in-category and out of category
        there's a factor of 2 here I need to deal with
        '''
        not_cat_dists = torch.cdist(in_cat, not_cat).flatten() / (uniques.size()[0] - 1)
                
        '''build the final labelled distance vectors'''
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
        
        '''now make the properties vector'''
        thebc = torch.unique(data.y_particle_barcodes[data.y == label]).item()
        select_truth = (data.truth_barcodes == thebc)
        true_cluster_properties.append(1./data.truth_pt[select_truth])
        #[data.truth_eta[select_truth], data.truth_phi[select_truth]]
    matched_pred_clusters = np.array(matched_pred_clusters, dtype=np.int64)
    pred_indices = torch.from_numpy(matched_pred_clusters).to(clusters.device)
    #print(pred_indices)
    
    true_cluster_properties = np.array(true_cluster_properties, dtype=np.float)
    y_properties = torch.from_numpy(true_cluster_properties).to(clusters.device).float()
    # print(y_properties)    
    
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
        
        '''Main Input Net'''
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
                nn.BatchNorm1d(num_features=hidden_dim)
            )
            self.edgeconvs.append(EdgeConv(nn=convnn, aggr=aggr))
        
        
        '''Embedding Output Net'''
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
        '''InputNetCat'''
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
        
        '''EdgeConcat Convolution'''
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
        
        '''Edge Classifier'''
        self.edge_classifier = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ELU(),
            nn.BatchNorm1d(num_features=hidden_dim//2),
            nn.Linear(hidden_dim//2, ncats_out)
        )
        
        '''InputNet for Cluster Properties'''
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
        
        '''Convolution for Cluster Properties'''
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

        '''KNN(k neighbors) over Embedding2 features''' 
        edge_index = knn_graph(out, self.k, batch, loop=False, flow=ec.flow)
        
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

def load_data(root_path):
    data = TrackMLParticleTrackingDataset(root=root_path,
                                        layer_pairs_plus=True,
                                        pt_min=0,
                                        n_events=100, n_workers=1)
    print('Number of events read = ',len(data))
    return data

def plot_event(my_data,y_t):

    x,y,z = my_data[:,0], my_data[:,1], my_data[:,2]

    fig = plt.figure(figsize = (15, 10)) 
    ax1 = fig.add_subplot(111,projection='3d')
    
    #Axis 1 - hits 
    ax1.set_xlabel('Z-axis', fontweight ='bold')  
    ax1.set_ylabel('Y-axis', fontweight ='bold')  
    ax1.set_zlabel('X-axis', fontweight ='bold')  
    ax1.scatter3D(z, y, x, s=10, color= m.to_rgba(y_t), edgecolors='black')      

    global ctr
    plt.savefig(plot_path+'event_'+str(ctr)+'.pdf') 
    ctr = ctr+1
    ctr = ctr%10
    plt.close(fig)
    

if __name__ == "__main__" :

    root = '/home/csharma/prototyping/data/train_1/'
    plot_folder_name = 'event1_epoch50_classes3'
    plot_path       = './plots/'+plot_folder_name+'/'

    total_epochs = 50
    n_samples  = 1
    input_classes = 3

    input_dim  = 3
    hidden_dim = 32
    output_dim = 2
    
    ncats_out  = 3
    nprops_out = 1
    
    conv_depth = 3
    k          = 8 
    edgecat_depth = 6 

    


    '''Load Data'''
    data = load_data(root)

    '''Load Model'''
    model = SimpleEmbeddingNetwork(input_dim=input_dim, 
                                hidden_dim=hidden_dim, 
                                output_dim=output_dim,
                                ncats_out=ncats_out,
                                nprops_out=nprops_out,
                                conv_depth=conv_depth, 
                                edgecat_depth=edgecat_depth, 
                                k=k, 
                                aggr='add',
                                norm=norm).to('cuda')
    
    '''Set Optimizer'''
    opt = torch.optim.AdamW([
                            {'params': list(model.inputnet.parameters()) + list(model.edgeconvs.parameters()) + list(model.output.parameters())},
                            {'params': list(model.inputnet_cat.parameters()) + list(model.edgecatconvs.parameters()) + list(model.edge_classifier.parameters()), 'lr': 0.0},
                            {'params': list(model.inputnet_prop.parameters()) + list(model.propertyconvs.parameters()) + list(model.property_predictor.parameters()), 'lr': 0.0}
                            ], lr=2.5e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.65, patience=30)
    # sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 20, eta_min=0)

    truth_cache = []
    all_loss    = []
    sep_loss_avg = []
    
    color_cycle = plt.cm.coolwarm(np.linspace(0.1,0.9,10*n_samples))
    marker_hits =    ['^','v','s','h','<','>']
    marker_centers = ['+','1','x','3','2','4']

    converged_embedding = False
    converged_categorizer = False
    make_plots = True

    print('-------------------')
    print('Root      : ',root)
    print('Epochs    : ',total_epochs)
    print('Samples   : ',n_samples)
    print('Track Kind: ', input_classes)

    print('InputdDim : ',input_dim)
    print('HiddenDim : ',hidden_dim)
    print('OutputDim : ',output_dim)

    print('Model Parameters (trainable):',  sum(p.numel() for p in model.parameters() if p.requires_grad))
    # pdb.set_trace()

    print('--------------------')
    print('Begin training :')

    for epoch in range(total_epochs):
        
        sep_loss = np.zeros((total_epochs,3), dtype=np.float)
        avg_loss_track = np.zeros(len(data), dtype=np.float)
        avg_loss = 0
        
        if make_plots:
            plt.clf()

        opt.zero_grad()
        if opt.param_groups[0]['lr'] < 1e-4 and not converged_embedding:
            converged_embedding = True
            opt.param_groups[1]['lr'] = 1e-4
            opt.param_groups[1]['lr'] = 1e-3
            
        if opt.param_groups[1]['lr'] < 1e-4 and not converged_categorizer and converged_embedding:
            converged_categorizer = True
            opt.param_groups[2]['lr'] = 1e-3
        

        for idata, d in enumerate(data[0:n_samples]):            
            
            d_gpu = d.to('cuda')
            y_orig = d_gpu.y

            d_gpu.x = d_gpu.x[d_gpu.y < input_classes] # just take the first three tracks
            d_gpu.x = (d_gpu.x - torch.min(d_gpu.x, axis=0).values)/(torch.max(d_gpu.x, axis=0).values - torch.min(d_gpu.x, axis=0).values) # Normalise
            d_gpu.y_particle_barcodes = d_gpu.y_particle_barcodes[d_gpu.y < input_classes]
            d_gpu.y = d_gpu.y[d_gpu.y < input_classes]
            # plot_event(d_gpu.x.detach().cpu().numpy(), d_gpu.y.detach().cpu().numpy())
            

            '''
            project data to some 2d plane where it is seperable usinfg the deep model
            compute edge net scores and seperated cluster properties in that latent space
            '''
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
            '''
            
            #-------------- LINDSEY TRAINING VERSION ------------------
            
            # print('compute latent space distances for hinge loss ...')
            multi_simple_hinge = simple_embedding_truth(coords, d_gpu.y, device='cuda')
            
            # print('predicted centers in latent space - for plotting')
            centers = scatter_mean(coords, d_gpu.y, dim=0, dim_size=(torch.max(d_gpu.y).item()+1))
            
            # if (make_plots==True and (epoch==0 or epoch==total_epochs-1) and (idata==0 or idata==n_samples-1)) :
            if (make_plots==True and (epoch%10==0 or epoch==total_epochs-1)) :            
            # if (make_plots==True) :

                fig = plt.figure(figsize=(8,8))
                if output_dim==3:
                    ax = fig.add_subplot(111, projection='3d')
                    for i in range(centers.size()[0]):  
                        ax.scatter(coords[d_gpu.y == i,0].detach().cpu().numpy(), 
                            coords[d_gpu.y == i,1].detach().cpu().numpy(),
                            coords[d_gpu.y == i,2].detach().cpu().numpy(),
                            color=color_cycle[2*idata + i], marker = marker_hits[i%6], s=100);

                        ax.scatter(centers[i,0].detach().cpu().numpy(), 
                            centers[i,1].detach().cpu().numpy(), 
                            centers[i,2].detach().cpu().numpy(), 
                            marker=marker_centers[i%6], color=color_cycle[2*idata+i], s=100); 
                elif output_dim==2:
                    for i in range(int(centers.size()[0])):
                            plt.scatter(coords[d_gpu.y == i,0].detach().cpu().numpy(), 
                                        coords[d_gpu.y == i,1].detach().cpu().numpy(),
                                        color=color_cycle[2*idata + i], 
                                        marker = marker_hits[i%6] )


                            plt.scatter(centers[i,0].detach().cpu().numpy(), 
                                        centers[i,1].detach().cpu().numpy(), 
                                        color=color_cycle[2*idata+i],  
                                        edgecolors='black',
                                        marker=marker_centers[i%6]) 
        
                # display.clear_output(wait=True)
                # display.display(plt.gcf())  
                plt.savefig(plot_path+'train_plot_epoch_'+str(epoch)+'_ex_'+str(idata)+'.pdf')   
                plt.close(fig)

            # Hinge: embedding distance based loss
            hinges = torch.cat([F.hinge_embedding_loss(dis**2, y, margin=1.0, reduction='mean')[None] 
                                for dis, y in multi_simple_hinge],
                            dim=0)
            
            # Cross Entropy: Edge categories loss
            y_edgecat = (d_gpu.y[edges[0]] == d_gpu.y[edges[1]]).long()
            loss_ce = F.cross_entropy(edge_scores, y_edgecat, reduction='mean')
            
            # MSE: Cluster loss
            pred_cluster_match, y_properties = match_cluster_targets(cluster_map, d_gpu.y, d_gpu)
            loss_mse = F.mse_loss(cluster_props[pred_cluster_match].squeeze(), y_properties, reduction='mean')
            
            # Combined loss
            loss = hinges.mean() + loss_ce + loss_mse
            avg_loss_track[idata] = loss.item()
            
            avg_loss += loss.item()

            sep_loss[idata,0]= hinges.mean().detach().cpu().numpy()
            sep_loss[idata,1]= loss_ce.detach().cpu().numpy()
            sep_loss[idata,2]= loss_mse.detach().cpu().numpy()

            # Loss Backward
            loss.backward()
            
            '''Per example stats'''
            # print(epoch, idata, 'loss / LR /centers -->>\n', 
            #     loss.item(), loss_ce.item(), loss_mse.item(), 
            #     '\n', opt.param_groups[0]['lr'], opt.param_groups[1]['lr'], opt.param_groups[2]['lr'],
            #     '\n', 'N_true_edges / accuracy -->> ', y_edgecat.sum().item(), '/', (torch.argmax(edge_scores, dim=1) == y_edgecat).sum().item()/(y_edgecat.size()[0]),
            #     '\n centers --> ', centers,
            #     '\n cluster_properties -> ', cluster_props,
            #     '\n hinges.mean() :', hinges.mean().detach().cpu().numpy(),
            #     '\n loss_ce       :', loss_ce.detach().cpu().numpy(),
            #     '\n loss_mse      :', loss_mse.detach().cpu().numpy()
            #     )
        
        '''track Epoch Updates'''
        all_loss.append(avg_loss_track.mean())
        sep_loss_avg.append([sep_loss[:,0].mean(), sep_loss[:,1].mean(), sep_loss[:,2].mean()])

        '''Per Epoch Stats'''
        print("---------------------------------------------------------")
        print("Epoch: {}\nLosses:\nCombined : {:.5e}\nHinge_distance : {:.5e}\nCrossEntr_Edges : {:.5e}\nMSE_centers : {:.5e}\n".format(
                epoch,all_loss[epoch],sep_loss_avg[epoch][0],sep_loss_avg[epoch][1],sep_loss_avg[epoch][2]))
        print("LR: opt.param_groups \n[0] : {:.9e}  \n[1] : {:.9e}  \n[2] : {:.9e}".format(opt.param_groups[0]['lr'], opt.param_groups[1]['lr'], opt.param_groups[2]['lr']))
        opt.step()
        sched.step(avg_loss)

    print("Training Complted!")
    print(1/y_properties)
    print(pred_cluster_match)
    print(1/cluster_props[pred_cluster_match].squeeze())
    
    '''Plot Learning Curve'''
    fig = plt.figure(figsize=(15,10))
    plt.plot(np.arange(total_epochs), [x[0] for x in sep_loss_avg], color='brown', linewidth=1, label="Hinge")
    plt.plot(np.arange(total_epochs), [x[1] for x in sep_loss_avg], color='green', linewidth=1, label="CrossEntropy")
    plt.plot(np.arange(total_epochs), [x[2] for x in sep_loss_avg], color='olive', linewidth=1, label="MSE")
    plt.plot(np.arange(total_epochs), all_loss, color='red', linewidth=2, label="Combined")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.legend()
    plt.title(plot_folder_name)
    plt.savefig(plot_path + plot_folder_name+'_Learning_curve.pdf')
    plt.close(fig)

    print("Plot learned clusters")
    pdb.set_trace()
    n_clusters = data[0].y[data[0].y < input_classes].max().item() + 1
    fig, ax = plt.subplots()
    for i in range(n_clusters):
        mapped_i = pred_cluster_match[i].item()
        r = data[0].x[data[0].y < input_classes][cluster_map == mapped_i,0].detach().cpu().numpy()
        phi = data[0].x[data[0].y < input_classes][cluster_map == mapped_i,1].detach().cpu().numpy()
        z = data[0].x[data[0].y < input_classes][cluster_map == mapped_i,2].detach().cpu().numpy()
        ax.scatter(r*np.cos(phi), 
                r*np.sin(phi),
                color=color_cycle[2*idata + i], marker = marker_hits[i%6], s=100);
        ax.text((r*np.cos(phi)).mean(), (r*np.sin(phi)).mean(), 'pt_pred = %.3f\npt_true = %.3f' % (1./cluster_props[mapped_i].item(), 1/y_properties[i]))
    plt.savefig(plot_path+'learned_clusters.pdf')
    plt.close(fig)

    print("UnionFind Roots:")
    objects = UnionFind(coords.size()[0])
    good_edges = edges.t()[torch.argmax(edge_scores, dim=1) > 0].cpu().numpy()
    for edge in good_edges:
        objects.union(edge[0],edge[1])
        #objects.union(edge[1],edge[0])
    roots = np.array([objects.find(i) for i in range(coords.size()[0])], dtype=np.int64)
    print(roots)
    print('exit')