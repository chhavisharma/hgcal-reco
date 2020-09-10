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
from sklearn.metrics import confusion_matrix

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

load_checkpoint_path = '/home/csharma/workspace/hgcal-reco/one_shot_tests/checkpoints/event500_epoch3000_classes10' + '/best_model_checkpoint.pt'
data_root    = '/home/csharma/prototyping/data/train_1_/'
logfile_name = 'testing.log'

train_samples = 500
test_samples  = 50
input_classes = 10

plot_dir_root   = './plots/'
plot_dir_name   = 'test_event'+str(test_samples)+'_classes'+str(input_classes)
plot_path       = plot_dir_root+plot_dir_name+'/'

input_dim  = 3
hidden_dim = 32
interm_out = None
output_dim = 2

ncats_out  = 2
nprops_out = 1

conv_depth = 3
k          = 8 
edgecat_depth = 6  
make_test_plots = False


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

def load_checkpoint(load_checkpoint_path, model):

    checkpoint = torch.load(load_checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint['epoch'], checkpoint['best_loss'] 

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

def testing(data, model):

    model.eval()

    combo_loss_avg     = []
    sep_loss_avg = []
    pred_cluster_properties = []
    edge_acc_track = np.zeros(test_samples, dtype=np.float)
    edge_acc_conf = np.zeros((test_samples,2,2), dtype=np.int)
    
    
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
        pred_cluster_properties = []
        avg_loss = 0
        
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

            true_prop = y_properties.detach().cpu().numpy()
            pred_prop = cluster_props[pred_cluster_match].squeeze().detach().cpu().numpy()
            pred_cluster_properties.append([1/true_prop,1/pred_prop])

            edge_acc_conf[idata,:,:] = confusion_matrix(y_edgecat.detach().cpu().numpy(), torch.argmax(edge_scores, dim=1).detach().cpu().numpy())

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
        
                plt.title('test_plot_'+'_ex_'+str(idata)+'_EdgeAcc_'+str('{:.3e}'.format(edge_accuracy))+'_HingeLoss_'+'{:.3e}'.format(sep_loss_track[idata,0]))
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
        pdb.set_trace()
        total_0 = edge_acc_conf.sum()
        logtofile(plot_path, logfile_name,'Total Edge Stats:\nTrue_0: {} \tTrue_1: {}\nPred_0: {} \tPred_1: {}'.format())
        logtofile(plot_path, logfile_name,"Total true edges [class_0: {:6d}] [class_1: {:6d}]".format(total_true_0_1[0],total_true_0_1[1]))
        logtofile(plot_path, logfile_name,"Total pred edges [class_0: {:6d}] [class_1: {:6d}]".format(total_pred_0_1[0],total_pred_0_1[1]))        
        logtofile(plot_path, logfile_name,'\nProperties:')
        logtofile(plot_path, logfile_name,str(pred_cluster_properties))
        logtofile(plot_path, logfile_name,'--------------------------')

        # pdb.set_trace()

    t2 = timer()

    print("Testing Complted in {:.5f}mins.\n".format((t2-t1)/60.0))
    return combo_loss_avg, sep_loss_avg, edge_acc_track, pred_cluster_properties, sep_loss_track, edge_acc_conf

if __name__ == "__main__":

    '''Plots'''
    if not os.path.exists(plot_dir_root):
        os.makedirs(plot_dir_root)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

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


    print('[TEST CONFIG]')
    print('Train Samples : ', train_samples)
    print('Test Samples  : ', test_samples)
    print('TrackKind: ', input_classes)
    print('InputdDim: ', input_dim)
    print('HiddenDim: ', hidden_dim)
    print('OutputDim: ', output_dim)
    print('IntermOut: ', interm_out)
    print('Model Parameters :',  sum(p.numel() for p in model.parameters() if p.requires_grad))


    logtofile(plot_path, logfile_name, '\nStart time: '+datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    logtofile(plot_path, logfile_name, "\nTEST CONFIG: {}\nEvents:{}\nTracks: {}".format(plot_dir_name, train_samples, input_classes))
    logtofile(plot_path, logfile_name, "MODEL:\nInputDim={}\nHiddenDim={}\nOutputDim={}\ninterm_out={}\nNcatsOut={}\nNPropsOut={}\nConvDepth={}\nKNN_k={}\nEdgeCatDepth={}".format(
                                                input_dim,hidden_dim,output_dim,interm_out,ncats_out,nprops_out,conv_depth,k,edgecat_depth))


    model,  start_epoch, best_loss = load_checkpoint(load_checkpoint_path, model)
    print('\nloaded checkpoint:')
    print('\tstart_epoch :',start_epoch)
    print('\tbest_loss   :',best_loss)
    logtofile(plot_path, logfile_name, '\nloaded checkpoint with start epoch {} and loss {} \n'.format(start_epoch,best_loss))

    ''' Test '''
    test_combo_loss_avg, test_sep_loss_avg, test_edge_acc_track, test_pred_cluster_properties, test_sep_loss_track, edge_acc_conf = testing(data, model)    

    colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00']
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(131)
    ax1.hist(test_sep_loss_track[:,0], color=colors[1], label="Hinge Loss Per Example")
    ax1.set_xlabel("Loss")
    ax1.set_ylabel("Examples")
    ax1.legend()

    ax2 = fig.add_subplot(132)
    ax2.hist(test_sep_loss_track[:,1], color=colors[3], label="EdgeCE Loss Per Example")
    ax2.set_xlabel("Loss")
    ax2.set_ylabel("Examples")
    ax2.legend()

    ax3 = fig.add_subplot(133)
    ax3.hist(test_edge_acc_track, color=colors[4], label="EdgeClsf Acc Per Example")
    ax3.set_xlabel("Acc")
    ax3.set_ylabel("Examples")
    ax3.legend()

    pdb.set_trace()
    ax1.set_title('Hinge Loss Dist: Avg='+str('{:.3e}'.format(test_sep_loss_avg[0][0])))
    ax2.set_title('EdgeCE Loss Dist: Avg='+str('{:.3e}'.format(test_sep_loss_avg[0][1])))
    ax3.set_title('Edge Acc Dist: Avg='+str('{:.3e}'.format(test_edge_acc_track.mean())))
    # plt.title(plot_dir_name)
    # fig.title(plot_dir_name)
    plt.savefig(plot_path+plot_dir_name+'_Loss_distributions.pdf')
    plt.close(fig)    



    pdb.set_trace()
    print('EXIT.')

