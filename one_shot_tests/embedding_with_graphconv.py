'''Python imports'''
import os
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
import config
from utils import logtofile, save_checkpoint, load_checkpoint
from particle_margin import TrackMLParticleTrackingDataset
from model import SimpleEmbeddingNetwork

'''Globals'''
torch.manual_seed(1)
color_range = mpl.colors.Normalize(vmin=-20, vmax=10)
cmap = cm.hot
m = cm.ScalarMappable(norm=color_range, cmap=cmap)

'''Data Norm'''
data_norm = torch.tensor([1./70., 1./5., 1./400.])

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

def load_data(root_path, samples):
    print('Loading data ...')
    data = TrackMLParticleTrackingDataset(root=root_path,
                                        layer_pairs_plus=True,
                                        pt_min=0,
                                        n_events=samples, n_workers=1)
    print('{} events read.'.format(data))
    return data

def training(data, model, opt, sched, lr_param_gp_1, lr_param_gp_2, lr_param_gp_3, \
          lr_threshold_1, lr_threshold_2, converged_embedding, converged_categorizer, start_epoch, best_loss):

    model.train()

    combo_loss_avg = []
    sep_loss_avg = []
    pred_cluster_properties = []
    edge_acc_track = np.zeros(config.train_samples, dtype=np.float)
    
    color_cycle = plt.cm.coolwarm(np.linspace(0.1,0.9,config.input_classes*config.k))
    marker_hits =    ['^','v','s','h','<','>']
    marker_centers = ['+','1','x','3','2','4']

    print('\n[TRAIN]:')

    t1 = timer()
    
    for epoch in range(start_epoch, start_epoch+config.total_epochs):
        
        '''book-keeping'''
        sep_loss_track = np.zeros((config.train_samples,3), dtype=np.float)
        avg_loss_track = np.zeros(config.train_samples, dtype=np.float)
        edge_acc_track = np.zeros(config.train_samples, dtype=np.float)
        edge_acc_conf  = np.zeros((config.train_samples,config.ncats_out,config.ncats_out), dtype=np.int)
        pred_cluster_properties = []
        avg_loss = 0

        opt.zero_grad()
        # if opt.param_groups[0]['lr'] < lr_threshold_1 and not converged_embedding:
        #     converged_embedding = True
        #     opt.param_groups[1]['lr'] = lr_threshold_1
        #     opt.param_groups[2]['lr'] = lr_threshold_2
            
        # if opt.param_groups[1]['lr'] < lr_threshold_1 and not converged_categorizer and converged_embedding:
        #     converged_categorizer = True
        #     opt.param_groups[2]['lr'] = lr_threshold_2
        

        for idata, d in enumerate(data[0:config.train_samples]):            
            
            d_gpu = d.to('cuda')
            y_orig = d_gpu.y

            d_gpu.x = d_gpu.x[d_gpu.y < config.input_classes] 
            d_gpu.x = (d_gpu.x - torch.min(d_gpu.x, axis=0).values)/(torch.max(d_gpu.x, axis=0).values - torch.min(d_gpu.x, axis=0).values) # Normalise
            d_gpu.y_particle_barcodes = d_gpu.y_particle_barcodes[d_gpu.y < config.input_classes]
            d_gpu.y = d_gpu.y[d_gpu.y < config.input_classes]
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
            # if (config.make_plots==True):
            if (config.make_plots==True and (epoch==0 or epoch==start_epoch+config.total_epochs-1) and idata%(config.train_samples/10)==0):     
                fig = plt.figure(figsize=(8,8))
                if config.output_dim==3:
                    ax = fig.add_subplot(111, projection='3d')
                    for i in range(centers.size()[0]):  
                        ax.scatter(coords[d_gpu.y == i,0].detach().cpu().numpy(), 
                            coords[d_gpu.y == i,1].detach().cpu().numpy(),
                            coords[d_gpu.y == i,2].detach().cpu().numpy(),
                            color=color_cycle[(i*config.k)%(config.train_samples*config.k - 1)], marker = marker_hits[i%6], s=100)

                        ax.scatter(centers[i,0].detach().cpu().numpy(), 
                            centers[i,1].detach().cpu().numpy(), 
                            centers[i,2].detach().cpu().numpy(), 
                            marker=marker_centers[i%6], color=color_cycle[(i*config.k)%(config.train_samples*config.k - 1)], s=100); 
                elif config.output_dim==2:
                    for i in range(int(centers.size()[0])):
                            plt.scatter(coords[d_gpu.y == i,0].detach().cpu().numpy(), 
                                        coords[d_gpu.y == i,1].detach().cpu().numpy(),
                                        color=color_cycle[(i*config.k)%(config.train_samples*config.k - 1)], 
                                        marker = marker_hits[i%6] )

                            plt.scatter(centers[i,0].detach().cpu().numpy(), 
                                        centers[i,1].detach().cpu().numpy(), 
                                        color=color_cycle[(i*config.k)%(config.train_samples*config.k - 1)],  
                                        edgecolors='b',
                                        marker=marker_centers[i%6]) 
        
                plt.title('train_plot_epoch_'+str(epoch)+'_ex_'+str(idata)+'_EdgeAcc_'+str('{:.5e}'.format(edge_accuracy)))
                plt.savefig(config.plot_path+'train_plot_epoch_'+str(epoch)+'_ex_'+str(idata)+'.pdf')   
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

        if(epoch%10==0 or epoch==start_epoch or epoch==start_epoch+config.total_epochs-1):
            '''Per Epoch Stats'''
            print('--------------------')
            print("Epoch: {}\nLosses:\nCombined: {:.5e}\nHinge_distance: {:.5e}\nCrossEntr_Edges: {:.5e}\nMSE_centers: {:.5e}".format(
                    epoch,combo_loss_avg[epoch-start_epoch],sep_loss_avg[epoch-start_epoch][0],sep_loss_avg[epoch-start_epoch][1],sep_loss_avg[epoch-start_epoch][2]))
            print("LR: opt.param_groups \n[0]: {:.9e}  \n[1]: {:.9e}  \n[2]: {:.9e}".format(opt.param_groups[0]['lr'], opt.param_groups[1]['lr'], opt.param_groups[2]['lr']))
            print("[TRAIN] Average Edge Accuracies over {} events: {:.5e}".format(config.train_samples,edge_acc_track.mean()) )
            print("Total true edges [class_0: {:6d}] [class_1: {:6d}]".format(total_true_0_1[0],total_true_0_1[1]))
            print("Total pred edges [class_0: {:6d}] [class_1: {:6d}]".format(total_pred_0_1[0],total_pred_0_1[1]))
        
            if(epoch==start_epoch+config.total_epochs-1 or epoch==start_epoch):
                logtofile(config.plot_path, config.logfile_name, "Epoch: {}\nLosses:\nCombined: {:.5e}\nHinge_distance: {:.5e}\nCrossEntr_Edges: {:.5e}\nMSE_centers: {:.5e}".format(
                    epoch,combo_loss_avg[epoch-start_epoch],sep_loss_avg[epoch-start_epoch][0],sep_loss_avg[epoch-start_epoch][1],sep_loss_avg[epoch-start_epoch][2]))
                logtofile(config.plot_path, config.logfile_name,"LR: opt.param_groups \n[0]: {:.9e}  \n[1]: {:.9e}  \n[2]: {:.9e}".format(opt.param_groups[0]['lr'], opt.param_groups[1]['lr'], opt.param_groups[2]['lr']))
                logtofile(config.plot_path, config.logfile_name,"Average Edge Accuracies over {} events, {} Tracks: {:.5e}".format(config.train_samples, config.input_classes,edge_acc_track.mean()) )                    
                logtofile(config.plot_path, config.logfile_name,"Total true edges [class_0: {:6d}] [class_1: {:6d}]".format(total_true_0_1[0],total_true_0_1[1]))
                logtofile(config.plot_path, config.logfile_name,"Total pred edges [class_0: {:6d}] [class_1: {:6d}]".format(total_pred_0_1[0],total_pred_0_1[1]))                
                # logtofile(config.plot_path, config.logfile_name,'Properties:\n')
                # logtofile(config.plot_path, config.logfile_name,str(pred_cluster_properties))
                logtofile(config.plot_path, config.logfile_name,'--------------------------')

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
                checkpoint_name = 'event'+str(config.train_samples)+'_classes' + str(config.input_classes) + '_epoch'+str(epoch) + '_loss' + '{:.5e}'.format(combo_loss_avg[epoch-start_epoch]) + '_edgeAcc' + '{:.5e}'.format(edge_acc_track.mean())
                save_checkpoint(checkpoint, is_best, config.checkpoint_path, checkpoint_name)

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
    edge_acc_track = np.zeros(config.test_samples, dtype=np.float)
    
    
    color_cycle = plt.cm.coolwarm(np.linspace(0.1,0.9,config.input_classes*config.k))
    marker_hits =    ['^','v','s','h','<','>']
    marker_centers = ['+','1','x','3','2','4']

    print('\n[TEST]:')

    t1 = timer()
    epoch=0

    with torch.no_grad():
        
        '''book-keeping'''
        sep_loss_track = np.zeros((config.test_samples,3), dtype=np.float)
        avg_loss_track = np.zeros(config.test_samples, dtype=np.float)
        edge_acc_track = np.zeros(config.test_samples, dtype=np.float)
        edge_acc_conf  = np.zeros((config.test_samples,config.ncats_out,config.ncats_out), dtype=np.int)
        pred_cluster_properties = []
        avg_loss = 0
        
        if config.make_plots:
            plt.clf()

        for idata, d in enumerate(data[config.train_samples:config.train_samples+config.test_samples]):            
            
            d_gpu = d.to('cuda')
            y_orig = d_gpu.y

            d_gpu.x = d_gpu.x[d_gpu.y < config.input_classes] # just take the first three tracks
            d_gpu.x = (d_gpu.x - torch.min(d_gpu.x, axis=0).values)/(torch.max(d_gpu.x, axis=0).values - torch.min(d_gpu.x, axis=0).values) # Normalise
            d_gpu.y_particle_barcodes = d_gpu.y_particle_barcodes[d_gpu.y < config.input_classes]
            d_gpu.y = d_gpu.y[d_gpu.y < config.input_classes]

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
            if (config.make_test_plots==True):
                
                fig = plt.figure(figsize=(8,8))
                if config.output_dim==3:
                    ax = fig.add_subplot(111, projection='3d')
                    for i in range(centers.size()[0]):  
                        ax.scatter(coords[d_gpu.y == i,0].detach().cpu().numpy(), 
                            coords[d_gpu.y == i,1].detach().cpu().numpy(),
                            coords[d_gpu.y == i,2].detach().cpu().numpy(),
                            color=color_cycle[(i*config.k)%(config.test_samples*config.k - 1)], marker = marker_hits[i%6], s=100);

                        ax.scatter(centers[i,0].detach().cpu().numpy(), 
                            centers[i,1].detach().cpu().numpy(), 
                            centers[i,2].detach().cpu().numpy(), 
                            marker=marker_centers[i%6], color=color_cycle[(i*config.k)%(config.test_samples*config.k - 1)], s=100); 
                elif config.output_dim==2:
                    for i in range(int(centers.size()[0])):
                            plt.scatter(coords[d_gpu.y == i,0].detach().cpu().numpy(), 
                                        coords[d_gpu.y == i,1].detach().cpu().numpy(),
                                        color=color_cycle[(i*config.k)%(config.test_samples*config.k - 1)], 
                                        marker = marker_hits[i%6] )

                            plt.scatter(centers[i,0].detach().cpu().numpy(), 
                                        centers[i,1].detach().cpu().numpy(), 
                                        color=color_cycle[(i*config.k)%(config.test_samples*config.k - 1)],  
                                        edgecolors='b',
                                        marker=marker_centers[i%6]) 
        
                plt.title('test_plot_'+'_ex_'+str(idata)+'_EdgeAcc_'+str('{:.5e}'.format(edge_accuracy)))
                plt.savefig(config.plot_path+'test_plot_'+'_ex_'+str(idata)+'.pdf')   
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
        print("[TEST] Average Edge Accuracies over {} events: {:.5e}".format(config.test_samples,edge_acc_track.mean()) )
        print("Total true edges [class_0: {:6d}] [class_1: {:6d}]".format(total_true_0_1[0],total_true_0_1[1]))
        print("Total pred edges [class_0: {:6d}] [class_1: {:6d}]".format(total_pred_0_1[0],total_pred_0_1[1]))
        
        logtofile(config.plot_path, config.logfile_name,'\nTEST:')
        logtofile(config.plot_path, config.logfile_name, "Losses:\nCombined: {:.5e}\nHinge_distance: {:.5e}\nCrossEntr_Edges: {:.5e}\nMSE_centers: {:.5e}".format(
                                                                combo_loss_avg[epoch],sep_loss_avg[epoch][0],sep_loss_avg[epoch][1],sep_loss_avg[epoch][2]))
        logtofile(config.plot_path, config.logfile_name,"Average Edge Accuracies over {} events, {} Tracks: {:.5e}".format(config.test_samples,config.input_classes,edge_acc_track.mean()) )                    
        logtofile(config.plot_path, config.logfile_name,"Total true edges [class_0: {:6d}] [class_1: {:6d}]".format(total_true_0_1[0],total_true_0_1[1]))
        logtofile(config.plot_path, config.logfile_name,"Total pred edges [class_0: {:6d}] [class_1: {:6d}]".format(total_pred_0_1[0],total_pred_0_1[1]))
        logtofile(config.plot_path, config.logfile_name,'\nProperties:')
        logtofile(config.plot_path, config.logfile_name,str(pred_cluster_properties))
        logtofile(config.plot_path, config.logfile_name,'--------------------------')

    t2 = timer()

    print("Testing Complted in {:.5f}mins.\n".format((t2-t1)/60.0))
    return combo_loss_avg, sep_loss_avg, edge_acc_track, pred_cluster_properties, edge_acc_conf

if __name__ == "__main__":

    '''Plots'''
    if not os.path.exists(config.plot_dir_root):
        os.makedirs(config.plot_dir_root)
    if not os.path.exists(config.plot_path):
        os.makedirs(config.plot_path)

    '''Checkpoint'''
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)
    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)    


    '''Load Data'''
    data = load_data(config.data_root, config.train_samples+config.test_samples)

    '''Load Model'''
    model = SimpleEmbeddingNetwork(input_dim=config.input_dim, 
                                hidden_dim=config.hidden_dim, 
                                output_dim=config.output_dim,
                                ncats_out=config.ncats_out,
                                nprops_out=config.nprops_out,
                                conv_depth=config.conv_depth, 
                                edgecat_depth=config.edgecat_depth, 
                                k=config.k, 
                                aggr='add',
                                norm=data_norm,
                                interm_out=config.interm_out
                                ).to('cuda')
    
    '''Not used a tthe moment'''
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
    print('Epochs   : ', config.total_epochs)
    print('Samples  : ', config.train_samples)
    print('TrackKind: ', config.input_classes)
    print('InputdDim: ', config.input_dim)
    print('HiddenDim: ', config.hidden_dim)
    print('OutputDim: ', config.output_dim)
    print('IntermOut: ', config.interm_out)
    print('NCatsOut : ', config.ncats_out)
    print('NPropOut : ', config.nprops_out)

    print('Model Parameters (trainable):',  sum(p.numel() for p in model.parameters() if p.requires_grad))


    logtofile(config.plot_path, config.logfile_name, '\nStart time: '+datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    logtofile(config.plot_path, config.logfile_name, "\nCONFIG: {}\nEpochs:{}\nEvents:{}\nTracks: {}".format(config.plot_dir_name, config.total_epochs, config.train_samples, config.input_classes))
    logtofile(config.plot_path, config.logfile_name, "MODEL:\nInputDim={}\nHiddenDim={}\nOutputDim={}\nconfig.interm_out={}\nNcatsOut={}\nNPropsOut={}\nConvDepth={}\nKNN_k={}\nEdgeCatDepth={}".format(
                                                config.input_dim,config.hidden_dim,config.output_dim,config.interm_out,config.ncats_out,config.nprops_out,config.conv_depth,config.k,config.edgecat_depth))
    logtofile(config.plot_path, config.logfile_name, "LEARNING RATE:\nParamgp1:{:.3e}\nParamgp2:{:.3e}\nParamgp3:{:.3e}".format(lr_param_gp_1, lr_param_gp_2, lr_param_gp_3))
    logtofile(config.plot_path, config.logfile_name, "threshold_1={:.3e}\nthreshold_2={:.3e}\n".format(lr_threshold_1, lr_threshold_2))


    converged_embedding = False
    converged_categorizer = False
    start_epoch = 0
    best_loss = np.inf

    if (config.load_checkpoint_path != False):

        model, opt, sched, start_epoch, converged_categorizer, converged_embedding, best_loss = \
                                            load_checkpoint(config.load_checkpoint_path, model, opt, sched)

        print('\nloaded checkpoint:')
        print('\tstart_epoch :',start_epoch)
        print('\tbest_loss   :',best_loss)
        logtofile(config.plot_path, config.logfile_name, '\nloaded checkpoint with start epoch {} and loss {} \n'.format(start_epoch,best_loss))

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
    with open(config.plot_path+'/training.pickle', 'wb') as handle:
        pickle.dump(training_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    testing_dict = {  
        'Combined_loss':test_combo_loss_avg,
        'Seperate_loss':test_sep_loss_avg,
        'Edge_Accuracies': test_edge_acc_track,
        'Pred_cluster_prop':test_pred_cluster_properties,
        'Edge_acc_conf_matrix':test_edge_acc_conf
    }
    with open(config.plot_path+'/testing.pickle', 'wb') as handle:
        pickle.dump(training_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    '''Learning Curve / Clusters / Centers'''
    if(config.make_plots==True):

        '''Plot Learning Curve'''
        fig = plt.figure(figsize=(20,10))
        ax1 = fig.add_subplot(121)
        ax1.plot(np.arange(start_epoch, start_epoch+config.total_epochs), [x[0] for x in sep_loss_avg], color='brown', linewidth=1, label="Hinge")
        ax1.plot(np.arange(start_epoch, start_epoch+config.total_epochs), [x[1] for x in sep_loss_avg], color='green', linewidth=1, label="CrossEntropy")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Losses")
        ax1.legend()

        ax2 = fig.add_subplot(122)
        ax2.plot(np.arange(start_epoch, start_epoch+config.total_epochs), [x[2] for x in sep_loss_avg], color='olive', linewidth=1, label="MSE")
        ax2.plot(np.arange(start_epoch, start_epoch+config.total_epochs), combo_loss_avg, color='red', linewidth=2, label="Combined")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Losses")
        ax2.legend()

        plt.title(config.plot_dir_name)
        ax1.set_title(config.plot_dir_name+': indivudual losses')
        ax2.set_title(config.plot_dir_name+': combined loss')
        plt.savefig(config.plot_path + config.plot_dir_name+'_Learning_curve.pdf')
        plt.close(fig)
