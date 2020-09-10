'''Python imports'''
import os
import shutil
import numpy as np
import awkward as ak
from math import sqrt
import networkx as nx
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from  mpl_toolkits import mplot3d
from typing import List
import pickle
from sklearn.metrics import confusion_matrix
from os.path import expanduser
home = expanduser("~")

# from IPython import display
import time
from datetime import datetime
from timeit import default_timer as timer
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

'''File Imports'''
from particle_margin import TrackMLParticleTrackingDataset

'''Globals'''
torch.manual_seed(1)
norm = torch.tensor([1./70., 1./5., 1./400.])
norm = mpl.colors.Normalize(vmin=-20, vmax=10)
cmap = cm.hot
m = cm.ScalarMappable(norm=norm, cmap=cmap)
ctr = 0


'''
SET CONFIG / Move to Config File 
--------------------------------
'''
load_checkpoint_path = False

data_root    = home+'/prototyping/data/train_1_/'
logfile_name = 'training.log'

total_epochs  = 3000
train_samples = 100
test_samples  = 100
input_classes = 10

plot_dir_root   = './plots/'
plot_dir_name   = 'train_event'+str(train_samples)+'_epoch'+str(total_epochs)+'_classes'+str(input_classes)
plot_path       = plot_dir_root+plot_dir_name+'/'

checkpoint_dir  = './checkpoints/'
checkpoint_path = checkpoint_dir+plot_dir_name

# Embedding Dim
input_dim  = 3
hidden_dim = 32
interm_out = None
output_dim = 2

# Regressor and Classifier Output Dim
ncats_out  = 2
nprops_out = 1

# EdgeCat Settings
k             = 8 
conv_depth    = 3
edgecat_depth = 6  # TRY DEPTH==3,tried - kills edgenet's performance
make_plots   = True
make_test_plots = True


def logtofile(path, filename, logs):
    filepath = path + '/'+ filename
    if os.path.exists(filepath):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w'
    logfile = open(filepath,append_write)
    logfile.write(logs)
    logfile.write('\n')
    logfile.close()

def save_checkpoint(model_state, is_best, checkpoint_dir, checkpoint_name):
    f_path = os.path.join(checkpoint_dir,checkpoint_name + '_checkpoint.pt')
    torch.save(model_state, f_path)
    if is_best:
        best_fpath = os.path.join(checkpoint_dir, 'best_model_checkpoint.pt')
        shutil.copyfile(f_path, best_fpath)

def load_checkpoint(load_checkpoint_path, model, optimizer, scheduler):

    checkpoint = torch.load(load_checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['converged_categorizer'], \
                                    checkpoint['converged_embedding'], checkpoint['best_loss'] 

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
        #[data.truth_eta[select_truth], data.truth_phi[select_truth]] ### TRY_THIS ETA+PHI
    matched_pred_clusters = np.array(matched_pred_clusters, dtype=np.int64)
    pred_indices = torch.from_numpy(matched_pred_clusters).to(clusters.device)
    #print(pred_indices)
    
    true_cluster_properties = np.array(true_cluster_properties, dtype=np.float)
    y_properties = torch.from_numpy(true_cluster_properties).to(clusters.device).float()
    # print(y_properties)    
    
    #print('match_cluster_targets')
    return pred_indices, y_properties ## Also predict Eta-Phi

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
        #plot = self.plotlayer(out)


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

def load_data(root_path, samples):
    print('Loading data ...')
    data = TrackMLParticleTrackingDataset(root=root_path,
                                        layer_pairs_plus=True,
                                        pt_min=0,
                                        n_events=samples, n_workers=1)
    print('{} events read.'.format(data))
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

def training(data, model, opt, sched, lr_param_gp_1, lr_param_gp_2, lr_param_gp_3, \
          lr_threshold_1, lr_threshold_2, converged_embedding, converged_categorizer, start_epoch, best_loss):

    model.train()

    combo_loss_avg = []
    sep_loss_avg = []
    pred_cluster_properties = []
    edge_acc_track = np.zeros(train_samples, dtype=np.float)
    
    color_cycle = plt.cm.coolwarm(np.linspace(0.1,0.9,input_classes*k))
    marker_hits =    ['^','v','s','h','<','>']
    marker_centers = ['+','1','x','3','2','4']

    print('\n[TRAIN]:')

    t1 = timer()
    
    for epoch in range(start_epoch, start_epoch+total_epochs):
        
        '''book-keeping'''
        sep_loss_track = np.zeros((train_samples,3), dtype=np.float)
        avg_loss_track = np.zeros(train_samples, dtype=np.float)
        edge_acc_track = np.zeros(train_samples, dtype=np.float)
        edge_acc_conf  = np.zeros((train_samples,ncats_out,ncats_out), dtype=np.int)
        pred_cluster_properties = []
        avg_loss = 0
        
        if make_plots:
            plt.clf()

        opt.zero_grad()
        if opt.param_groups[0]['lr'] < lr_threshold_1 and not converged_embedding:
            converged_embedding = True
            opt.param_groups[1]['lr'] = lr_threshold_1
            opt.param_groups[2]['lr'] = lr_threshold_2
            
        if opt.param_groups[1]['lr'] < lr_threshold_1 and not converged_categorizer and converged_embedding:
            converged_categorizer = True
            opt.param_groups[2]['lr'] = lr_threshold_2
        

        for idata, d in enumerate(data[0:train_samples]):            
            
            d_gpu = d.to('cuda')
            y_orig = d_gpu.y

            d_gpu.x = d_gpu.x[d_gpu.y < input_classes] 
            d_gpu.x = (d_gpu.x - torch.min(d_gpu.x, axis=0).values)/(torch.max(d_gpu.x, axis=0).values - torch.min(d_gpu.x, axis=0).values) # Normalise
            d_gpu.y_particle_barcodes = d_gpu.y_particle_barcodes[d_gpu.y < input_classes]
            d_gpu.y = d_gpu.y[d_gpu.y < input_classes]
            # plot_event(d_gpu.x.detach().cpu().numpy(), d_gpu.y.detach().cpu().numpy())
            
            '''
            project embedding to some 2d latent space where it is seperable using the deep model
            compute edge net scores and seperated cluster properties with the ebedding
            '''
            coords, edge_scores, edges, cluster_map, cluster_props, cluster_batch = model(d_gpu.x)

            #-------------- LINDSEY TRAINING VERSION ------------------         
            '''Compute latent space distances'''
            multi_simple_hinge = simple_embedding_truth(coords, d_gpu.y, device='cuda')
            # multi_simple_hinge += simple_embedding_truth(coords_interm, d_gpu.y, device='cuda')
            
            '''Compute centers in latent space '''
            centers = scatter_mean(coords, d_gpu.y, dim=0, dim_size=(torch.max(d_gpu.y).item()+1))
            
            '''Compute Losses'''
            # Hinge: embedding distance based loss
            hinges = torch.cat([F.hinge_embedding_loss(dis**2, y, margin=1.0, reduction='mean')[None] 
                                for dis, y in multi_simple_hinge],dim=0)
            
            #Cross Entropy: Edge categories loss
            y_edgecat = (d_gpu.y[edges[0]] == d_gpu.y[edges[1]]).long()
            loss_ce = F.cross_entropy(edge_scores, y_edgecat, reduction='mean')
            
            #MSE: Cluster loss
            pred_cluster_match, y_properties = match_cluster_targets(cluster_map, d_gpu.y, d_gpu)
            loss_mse = F.mse_loss(cluster_props[pred_cluster_match].squeeze(), y_properties, reduction='mean')
            
            #Combined loss
            loss = hinges.mean() + loss_ce + loss_mse
            avg_loss_track[idata] = loss.item()
            
            avg_loss += loss.item()

            '''Track Losses, Acuracies and Properties'''   
            sep_loss_track[idata,0]= hinges.mean().detach().cpu().numpy()
            sep_loss_track[idata,1]= loss_ce.detach().cpu().numpy()
            sep_loss_track[idata,2]= loss_mse.detach().cpu().numpy()

            true_edges = y_edgecat.sum().item()
            edge_accuracy = (torch.argmax(edge_scores, dim=1) == y_edgecat).sum().item() / (y_edgecat.size()[0])
            edge_acc_track[idata] = edge_accuracy

            edge_acc_conf[idata,:,:] = confusion_matrix(y_edgecat.detach().cpu().numpy(), torch.argmax(edge_scores, dim=1).detach().cpu().numpy())

            true_prop = y_properties.detach().cpu().numpy()
            pred_prop = cluster_props[pred_cluster_match].squeeze().detach().cpu().numpy()
            pred_cluster_properties.append([1/true_prop,1/pred_prop])

            '''Plot Training Clusters'''
            # if (make_plots==True):
            if (make_plots==True and (epoch==0 or epoch==start_epoch+total_epochs-1) and idata%(train_samples/10)==0):     
                fig = plt.figure(figsize=(8,8))
                if output_dim==3:
                    ax = fig.add_subplot(111, projection='3d')
                    for i in range(centers.size()[0]):  
                        ax.scatter(coords[d_gpu.y == i,0].detach().cpu().numpy(), 
                            coords[d_gpu.y == i,1].detach().cpu().numpy(),
                            coords[d_gpu.y == i,2].detach().cpu().numpy(),
                            color=color_cycle[(i*k)%(train_samples*k - 1)], marker = marker_hits[i%6], s=100)

                        ax.scatter(centers[i,0].detach().cpu().numpy(), 
                            centers[i,1].detach().cpu().numpy(), 
                            centers[i,2].detach().cpu().numpy(), 
                            marker=marker_centers[i%6], color=color_cycle[(i*k)%(train_samples*k - 1)], s=100); 
                elif output_dim==2:
                    for i in range(int(centers.size()[0])):
                            plt.scatter(coords[d_gpu.y == i,0].detach().cpu().numpy(), 
                                        coords[d_gpu.y == i,1].detach().cpu().numpy(),
                                        color=color_cycle[(i*k)%(train_samples*k - 1)], 
                                        marker = marker_hits[i%6] )

                            plt.scatter(centers[i,0].detach().cpu().numpy(), 
                                        centers[i,1].detach().cpu().numpy(), 
                                        color=color_cycle[(i*k)%(train_samples*k - 1)],  
                                        edgecolors='b',
                                        marker=marker_centers[i%6]) 
        
                plt.title('train_plot_epoch_'+str(epoch)+'_ex_'+str(idata)+'_EdgeAcc_'+str('{:.5e}'.format(edge_accuracy)))
                plt.savefig(plot_path+'train_plot_epoch_'+str(epoch)+'_ex_'+str(idata)+'.pdf')   
                plt.close(fig)

            '''Loss Backward''' 
            loss.backward()
        
        '''track Epoch Updates'''
        combo_loss_avg.append(avg_loss_track.mean())
        sep_loss_avg.append([sep_loss_track[:,0].mean(), sep_loss_track[:,1].mean(), sep_loss_track[:,2].mean()])
        true_0_1 = edge_acc_conf.sum(axis=2)
        pred_0_1 = edge_acc_conf.sum(axis=1) 
        total_true_0_1 =   true_0_1.sum(axis=0)
        total_pred_0_1 =   pred_0_1.sum(axis=0)
        
        # print('true_0_1:',true_0_1)
        # pdb.set_trace()

        if(epoch%10==0 or epoch==start_epoch or epoch==start_epoch+total_epochs-1):
            '''Per Epoch Stats'''
            print('--------------------')
            print("Epoch: {}\nLosses:\nCombined: {:.5e}\nHinge_distance: {:.5e}\nCrossEntr_Edges: {:.5e}\nMSE_centers: {:.5e}".format(
                    epoch,combo_loss_avg[epoch-start_epoch],sep_loss_avg[epoch-start_epoch][0],sep_loss_avg[epoch-start_epoch][1],sep_loss_avg[epoch-start_epoch][2]))
            print("LR: opt.param_groups \n[0]: {:.9e}  \n[1]: {:.9e}  \n[2]: {:.9e}".format(opt.param_groups[0]['lr'], opt.param_groups[1]['lr'], opt.param_groups[2]['lr']))
            print("[TRAIN] Average Edge Accuracies over {} events: {:.5e}".format(train_samples,edge_acc_track.mean()) )
            print("Total true edges [class_0: {:6d}] [class_1: {:6d}]".format(total_true_0_1[0],total_true_0_1[1]))
            print("Total pred edges [class_0: {:6d}] [class_1: {:6d}]".format(total_pred_0_1[0],total_pred_0_1[1]))
        
            if(epoch==start_epoch+total_epochs-1 or epoch==start_epoch):
                logtofile(plot_path, logfile_name, "Epoch: {}\nLosses:\nCombined: {:.5e}\nHinge_distance: {:.5e}\nCrossEntr_Edges: {:.5e}\nMSE_centers: {:.5e}".format(
                    epoch,combo_loss_avg[epoch-start_epoch],sep_loss_avg[epoch-start_epoch][0],sep_loss_avg[epoch-start_epoch][1],sep_loss_avg[epoch-start_epoch][2]))
                logtofile(plot_path, logfile_name,"LR: opt.param_groups \n[0]: {:.9e}  \n[1]: {:.9e}  \n[2]: {:.9e}".format(opt.param_groups[0]['lr'], opt.param_groups[1]['lr'], opt.param_groups[2]['lr']))
                logtofile(plot_path, logfile_name,"Average Edge Accuracies over {} events, {} Tracks: {:.5e}".format(train_samples, input_classes,edge_acc_track.mean()) )                    
                logtofile(plot_path, logfile_name,"Total true edges [class_0: {:6d}] [class_1: {:6d}]".format(total_true_0_1[0],total_true_0_1[1]))
                logtofile(plot_path, logfile_name,"Total pred edges [class_0: {:6d}] [class_1: {:6d}]".format(total_pred_0_1[0],total_pred_0_1[1]))                
                # logtofile(plot_path, logfile_name,'Properties:\n')
                # logtofile(plot_path, logfile_name,str(pred_cluster_properties))
                logtofile(plot_path, logfile_name,'--------------------------')

            if(combo_loss_avg[epoch-start_epoch] < best_loss):
                best_loss = combo_loss_avg[epoch-start_epoch]
                is_best = True
                checkpoint = {
                    'epoch': epoch+1,
                    'state_dict': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scheduler': sched.state_dict(),
                    'converged_embedding':False,
                    'converged_categorizer':False,
                    'best_loss':best_loss
                }
                checkpoint_name = 'event'+str(train_samples)+'_classes' + str(input_classes) + '_epoch'+str(epoch) + '_loss' + '{:.5e}'.format(combo_loss_avg[epoch-start_epoch]) + '_edgeAcc' + '{:.5e}'.format(edge_acc_track.mean())
                save_checkpoint(checkpoint, is_best, checkpoint_path, checkpoint_name)

        '''Update Weights'''
        opt.step()
        sched.step(avg_loss)

    t2 = timer()
    print('--------------------')
    print("Training Complted in {:.5f}mins.".format((t2-t1)/60.0))

    # print('1/properties: ',1/y_properties)
    # print('pred cluster matches: ',pred_cluster_match)
    # print('1/cluster_prop[cluster_match]: ',1/cluster_props[pred_cluster_match].squeeze())    

    return combo_loss_avg, sep_loss_avg, edge_acc_track, pred_cluster_properties, edge_acc_conf

def testing(data, model):

    model.eval()

    combo_loss_avg     = []
    sep_loss_avg = []
    pred_cluster_properties = []
    edge_acc_track = np.zeros(test_samples, dtype=np.float)
    
    
    color_cycle = plt.cm.coolwarm(np.linspace(0.1,0.9,input_classes*k))
    marker_hits =    ['^','v','s','h','<','>']
    marker_centers = ['+','1','x','3','2','4']

    print('\n[TEST]:')

    t1 = timer()
    epoch=0

    with torch.no_grad():
        
        '''book-keeping'''
        sep_loss_track = np.zeros((test_samples,3), dtype=np.float)
        avg_loss_track = np.zeros(test_samples, dtype=np.float)
        edge_acc_track = np.zeros(test_samples, dtype=np.float)
        edge_acc_conf  = np.zeros((test_samples,ncats_out,ncats_out), dtype=np.int)
        pred_cluster_properties = []
        avg_loss = 0
        
        if make_plots:
            plt.clf()

        for idata, d in enumerate(data[train_samples:train_samples+test_samples]):            
            
            d_gpu = d.to('cuda')
            y_orig = d_gpu.y

            d_gpu.x = d_gpu.x[d_gpu.y < input_classes] # just take the first three tracks
            d_gpu.x = (d_gpu.x - torch.min(d_gpu.x, axis=0).values)/(torch.max(d_gpu.x, axis=0).values - torch.min(d_gpu.x, axis=0).values) # Normalise
            d_gpu.y_particle_barcodes = d_gpu.y_particle_barcodes[d_gpu.y < input_classes]
            d_gpu.y = d_gpu.y[d_gpu.y < input_classes]

            '''
            project data to some 2d plane where it is seperable usinfg the deep model
            compute edge net scores and seperated cluster properties in that latent space
            '''
            coords, edge_scores, edges, cluster_map, cluster_props, cluster_batch = model(d_gpu.x)
            
            '''Compute latent space distances'''
            multi_simple_hinge = simple_embedding_truth(coords, d_gpu.y, device='cuda')
            # multi_simple_hinge += simple_embedding_truth(coords_interm, d_gpu.y, device='cuda')
            
            '''Predict centers in latent space '''
            centers = scatter_mean(coords, d_gpu.y, dim=0, dim_size=(torch.max(d_gpu.y).item()+1))
            
            
            '''LOSSES'''
            
            '''Hinge: embedding distance based loss'''
            hinges = torch.cat([F.hinge_embedding_loss(dis**2, y, margin=1.0, reduction='mean')[None] 
                                for dis, y in multi_simple_hinge],dim=0)
            
            '''Cross Entropy: Edge categories loss'''
            y_edgecat = (d_gpu.y[edges[0]] == d_gpu.y[edges[1]]).long()
            loss_ce = F.cross_entropy(edge_scores, y_edgecat, reduction='mean')
            
            '''MSE: Cluster loss'''
            pred_cluster_match, y_properties = match_cluster_targets(cluster_map, d_gpu.y, d_gpu)
            loss_mse = F.mse_loss(cluster_props[pred_cluster_match].squeeze(), y_properties, reduction='mean')
            
            '''Combined loss'''
            loss = hinges.mean() + loss_ce + loss_mse
            avg_loss_track[idata] = loss.item()
            
            avg_loss += loss.item()

            '''Track Losses, Acuracies and Properties'''   
            sep_loss_track[idata,0]= hinges.mean().detach().cpu().numpy()
            sep_loss_track[idata,1]= loss_ce.detach().cpu().numpy()
            sep_loss_track[idata,2]= loss_mse.detach().cpu().numpy()

            true_edges = y_edgecat.sum().item()
            edge_accuracy = (torch.argmax(edge_scores, dim=1) == y_edgecat).sum().item() / (y_edgecat.size()[0])
            edge_acc_track[idata] = edge_accuracy
            edge_acc_conf[idata,:,:] = confusion_matrix(y_edgecat.detach().cpu().numpy(), torch.argmax(edge_scores, dim=1).detach().cpu().numpy())


            true_prop = y_properties.detach().cpu().numpy()
            pred_prop = cluster_props[pred_cluster_match].squeeze().detach().cpu().numpy()
            pred_cluster_properties.append([1/true_prop,1/pred_prop])

            '''Plot test clusters'''
            if (make_test_plots==True):
                
                fig = plt.figure(figsize=(8,8))
                if output_dim==3:
                    ax = fig.add_subplot(111, projection='3d')
                    for i in range(centers.size()[0]):  
                        ax.scatter(coords[d_gpu.y == i,0].detach().cpu().numpy(), 
                            coords[d_gpu.y == i,1].detach().cpu().numpy(),
                            coords[d_gpu.y == i,2].detach().cpu().numpy(),
                            color=color_cycle[(i*k)%(test_samples*k - 1)], marker = marker_hits[i%6], s=100);

                        ax.scatter(centers[i,0].detach().cpu().numpy(), 
                            centers[i,1].detach().cpu().numpy(), 
                            centers[i,2].detach().cpu().numpy(), 
                            marker=marker_centers[i%6], color=color_cycle[(i*k)%(test_samples*k - 1)], s=100); 
                elif output_dim==2:
                    for i in range(int(centers.size()[0])):
                            plt.scatter(coords[d_gpu.y == i,0].detach().cpu().numpy(), 
                                        coords[d_gpu.y == i,1].detach().cpu().numpy(),
                                        color=color_cycle[(i*k)%(test_samples*k - 1)], 
                                        marker = marker_hits[i%6] )

                            plt.scatter(centers[i,0].detach().cpu().numpy(), 
                                        centers[i,1].detach().cpu().numpy(), 
                                        color=color_cycle[(i*k)%(test_samples*k - 1)],  
                                        edgecolors='b',
                                        marker=marker_centers[i%6]) 
        
                plt.title('test_plot_'+'_ex_'+str(idata)+'_EdgeAcc_'+str('{:.5e}'.format(edge_accuracy)))
                plt.savefig(plot_path+'test_plot_'+'_ex_'+str(idata)+'.pdf')   
                plt.close(fig)
        
        '''track test Updates'''
        combo_loss_avg.append(avg_loss_track.mean())
        sep_loss_avg.append([sep_loss_track[:,0].mean(), sep_loss_track[:,1].mean(), sep_loss_track[:,2].mean()])

        true_0_1 = edge_acc_conf.sum(axis=2)
        pred_0_1 = edge_acc_conf.sum(axis=1) 
        total_true_0_1 =   true_0_1.sum(axis=0)
        total_pred_0_1 =   pred_0_1.sum(axis=0)

        '''Test Stats'''
        print('--------------------')
        print("Losses:\nCombined: {:.5e}\nHinge_distance: {:.5e}\nCrossEntr_Edges: {:.5e}\nMSE_centers: {:.5e}".format(
                combo_loss_avg[epoch],sep_loss_avg[epoch][0],sep_loss_avg[epoch][1],sep_loss_avg[epoch][2]))
        print("[TEST] Average Edge Accuracies over {} events: {:.5e}".format(test_samples,edge_acc_track.mean()) )
        print("Total true edges [class_0: {:6d}] [class_1: {:6d}]".format(total_true_0_1[0],total_true_0_1[1]))
        print("Total pred edges [class_0: {:6d}] [class_1: {:6d}]".format(total_pred_0_1[0],total_pred_0_1[1]))
        
        logtofile(plot_path, logfile_name,'\nTEST:')
        logtofile(plot_path, logfile_name, "Losses:\nCombined: {:.5e}\nHinge_distance: {:.5e}\nCrossEntr_Edges: {:.5e}\nMSE_centers: {:.5e}".format(
                                                                combo_loss_avg[epoch],sep_loss_avg[epoch][0],sep_loss_avg[epoch][1],sep_loss_avg[epoch][2]))
        logtofile(plot_path, logfile_name,"Average Edge Accuracies over {} events, {} Tracks: {:.5e}".format(test_samples,input_classes,edge_acc_track.mean()) )                    
        logtofile(plot_path, logfile_name,"Total true edges [class_0: {:6d}] [class_1: {:6d}]".format(total_true_0_1[0],total_true_0_1[1]))
        logtofile(plot_path, logfile_name,"Total pred edges [class_0: {:6d}] [class_1: {:6d}]".format(total_pred_0_1[0],total_pred_0_1[1]))
        logtofile(plot_path, logfile_name,'\nProperties:')
        logtofile(plot_path, logfile_name,str(pred_cluster_properties))
        logtofile(plot_path, logfile_name,'--------------------------')

    t2 = timer()

    print("Testing Complted in {:.5f}mins.\n".format((t2-t1)/60.0))
    return combo_loss_avg, sep_loss_avg, edge_acc_track, pred_cluster_properties, edge_acc_conf

if __name__ == "__main__":

    '''Plots'''
    if not os.path.exists(plot_dir_root):
        os.makedirs(plot_dir_root)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    '''Checkpoint'''
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)    


    '''Load Data'''
    data = load_data(data_root, train_samples+test_samples)

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
                                norm=norm,
                                interm_out=interm_out
                                ).to('cuda')
    
    lr_threshold_1    = 1e-4 #5e-3
    lr_threshold_2    = 7.5e-4 #1e-3

    lr_param_gp_1     = 5e-3
    lr_param_gp_2     = 0   
    lr_param_gp_3     = 0  

    '''Set Optimizer'''
    opt = torch.optim.AdamW([
                            {'params': list(model.inputnet.parameters()) + list(model.edgeconvs.parameters()) + list(model.output.parameters())},
                            {'params': list(model.inputnet_cat.parameters()) + list(model.edgecatconvs.parameters()) + list(model.edge_classifier.parameters()), 'lr': lr_param_gp_2},
                            {'params': list(model.inputnet_prop.parameters()) + list(model.propertyconvs.parameters()) + list(model.property_predictor.parameters()), 'lr': lr_param_gp_3}
                            ], lr=lr_param_gp_1, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.70, patience=30)

    print('[CONFIG]')
    print('Epochs   : ', total_epochs)
    print('Samples  : ', train_samples)
    print('TrackKind: ', input_classes)
    print('InputdDim: ', input_dim)
    print('HiddenDim: ', hidden_dim)
    print('OutputDim: ', output_dim)
    print('IntermOut: ', interm_out)
    print('NCatsOut : ', ncats_out)
    print('NPropOut : ', nprops_out)

    print('Model Parameters (trainable):',  sum(p.numel() for p in model.parameters() if p.requires_grad))


    logtofile(plot_path, logfile_name, '\nStart time: '+datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    logtofile(plot_path, logfile_name, "\nCONFIG: {}\nEpochs:{}\nEvents:{}\nTracks: {}".format(plot_dir_name, total_epochs, train_samples, input_classes))
    logtofile(plot_path, logfile_name, "MODEL:\nInputDim={}\nHiddenDim={}\nOutputDim={}\ninterm_out={}\nNcatsOut={}\nNPropsOut={}\nConvDepth={}\nKNN_k={}\nEdgeCatDepth={}".format(
                                                input_dim,hidden_dim,output_dim,interm_out,ncats_out,nprops_out,conv_depth,k,edgecat_depth))
    logtofile(plot_path, logfile_name, "LEARNING RATE:\nParamgp1:{:.3e}\nParamgp2:{:.3e}\nParamgp3:{:.3e}".format(lr_param_gp_1, lr_param_gp_2, lr_param_gp_3))
    logtofile(plot_path, logfile_name, "threshold_1={:.3e}\nthreshold_2={:.3e}\n".format(lr_threshold_1, lr_threshold_2))


    converged_embedding = False
    converged_categorizer = False
    start_epoch = 0
    best_loss = np.inf

    if (load_checkpoint_path != False):

        model, opt, sched, start_epoch, converged_categorizer, converged_embedding, best_loss = \
                                            load_checkpoint(load_checkpoint_path, model, opt, sched)

        print('\nloaded checkpoint:')
        print('\tstart_epoch :',start_epoch)
        print('\tbest_loss   :',best_loss)
        logtofile(plot_path, logfile_name, '\nloaded checkpoint with start epoch {} and loss {} \n'.format(start_epoch,best_loss))

    ''' Train '''
    combo_loss_avg, sep_loss_avg, edge_acc_track, pred_cluster_properties, edge_acc_conf = training(data, model, opt, sched, \
                                                                        lr_param_gp_1, lr_param_gp_2, lr_param_gp_3, \
                                                                        lr_threshold_1, lr_threshold_2, converged_embedding, \
                                                                        converged_categorizer, start_epoch, best_loss)

    ''' Test '''
    test_combo_loss_avg, test_sep_loss_avg, test_edge_acc_track, test_pred_cluster_properties, test_edge_acc_conf = testing(data, model)    


    '''Save Losses'''
    training_dict = {  
        'Combined_loss':combo_loss_avg,
        'Seperate_loss':sep_loss_avg,
        'Edge_Accuracies': edge_acc_track,
        'Pred_cluster_prop':pred_cluster_properties,
        'Edge_acc_conf_matrix':edge_acc_conf
    }
    with open(plot_path+'/training.pickle', 'wb') as handle:
        pickle.dump(training_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    testing_dict = {  
        'Combined_loss':test_combo_loss_avg,
        'Seperate_loss':test_sep_loss_avg,
        'Edge_Accuracies': test_edge_acc_track,
        'Pred_cluster_prop':test_pred_cluster_properties,
        'Edge_acc_conf_matrix':test_edge_acc_conf
    }
    with open(plot_path+'/testing.pickle', 'wb') as handle:
        pickle.dump(training_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    '''Learning Curve / Clusters / Centers'''
    if(make_plots==True):

        '''Plot Learning Curve'''
        fig = plt.figure(figsize=(20,10))
        ax1 = fig.add_subplot(121)
        ax1.plot(np.arange(start_epoch, start_epoch+total_epochs), [x[0] for x in sep_loss_avg], color='brown', linewidth=1, label="Hinge")
        ax1.plot(np.arange(start_epoch, start_epoch+total_epochs), [x[1] for x in sep_loss_avg], color='green', linewidth=1, label="CrossEntropy")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Losses")
        ax1.legend()

        ax2 = fig.add_subplot(122)
        ax2.plot(np.arange(start_epoch, start_epoch+total_epochs), [x[2] for x in sep_loss_avg], color='olive', linewidth=1, label="MSE")
        ax2.plot(np.arange(start_epoch, start_epoch+total_epochs), combo_loss_avg, color='red', linewidth=2, label="Combined")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Losses")
        ax2.legend()

        plt.title(plot_dir_name)
        ax1.set_title(plot_dir_name+': indivudual losses')
        ax2.set_title(plot_dir_name+': combined loss')
        plt.savefig(plot_path + plot_dir_name+'_Learning_curve.pdf')
        plt.close(fig)

        # pdb.set_trace()
        # print("Plot learned clusters")
        # n_clusters = data[0].y[data[0].y < input_classes].max().item() + 1
        # print("Number of clusters: ", n_clusters)

        # fig, ax = plt.subplots()
        # for i in range(n_clusters):
        #     mapped_i = pred_cluster_match[i].item()
        #     r = test_data[0].x[test_data[0].y < input_classes][cluster_map == mapped_i,0].detach().cpu().numpy()
        #     phi = test_data[0].x[test_data[0].y < input_classes][cluster_map == mapped_i,1].detach().cpu().numpy()
        #     z = test_data[0].x[test_data[0].y < input_classes][cluster_map == mapped_i,2].detach().cpu().numpy()
        #     ax.scatter(r*np.cos(phi), 
        #             r*np.sin(phi),
        #             color=color_cycle[2*idata + i], marker = marker_hits[i%6], s=100);
        #     ax.text((r*np.cos(phi)).mean(), (r*np.sin(phi)).mean(), 'pt_pred = %.3f\npt_true = %.3f' % (1./cluster_props[mapped_i].item(), 1/y_properties[i]))
        # plt.savefig(plot_path+'learned_clusters.pdf')
        # plt.close(fig)

    # print("UnionFind Roots:")
    # objects = UnionFind(coords.size()[0])
    # good_edges = edges.t()[torch.argmax(edge_scores, dim=1) > 0].cpu().numpy()
    # for edge in good_edges:
    #     objects.union(edge[0],edge[1])
    #     #objects.union(edge[1],edge[0])
    # roots = np.array([objects.find(i) for i in range(coords.size()[0])], dtype=np.int64)
    # print(roots)
