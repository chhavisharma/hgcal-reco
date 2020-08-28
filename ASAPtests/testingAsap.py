'''PYTHON'''
import numpy as np
import time
from tqdm import tqdm
import argparse
import pdb
import glob
import math
import os
from sklearn.model_selection import StratifiedKFold
from itertools import product
import os.path as osp

'''TORCH'''
import torch
from torch import tensor
import torch.nn.functional as F
from torch.nn import Linear
from torch.optim import Adam

'''TORCH GEOMETRIC'''
import torch_geometric.transforms as T
from torch_geometric.nn import (ASAPooling, GraphConv, global_mean_pool,
                                JumpingKnowledge, TopKPooling)
from torch.utils.data import random_split
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.data import (Data, Dataset)
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader

'''Local'''
from graph import load_graph

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print('{:02d}/{:03d}: Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
        fold, epoch, val_loss, test_acc))

class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

def get_dataset(name, sparse=True, cleaned=False):
    
    if name=='node':
        path = osp.join(os.environ['GNN_TRAINING_DATA_ROOT'], name)
        print(path)
        dataset = HitGraphDataset2(path, directed=False, categorical=True)
    else:       
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
        dataset = TUDataset(path, name, cleaned=cleaned)
        dataset.data.edge_attr = None

        if dataset.data.x is None:
            max_degree = 0
            degs = []
            for data in dataset:
                degs += [degree(data.edge_index[0], dtype=torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            if max_degree < 1000:
                dataset.transform = T.OneHotDegree(max_degree)
            else:
                deg = torch.cat(degs, dim=0).to(torch.float)
                mean, std = deg.mean().item(), deg.std().item()
                dataset.transform = NormalizedDegree(mean, std)

        if not sparse:
            num_nodes = max_num_nodes = 0
            for data in dataset:
                num_nodes += data.num_nodes
                max_num_nodes = max(data.num_nodes, max_num_nodes)

            # Filter out a few really large graphs in order to apply DiffPool.
            if name == 'REDDIT-BINARY':
                num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
            else:
                num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

            indices = []
            for i, data in enumerate(dataset):
                if data.num_nodes <= num_nodes:
                    indices.append(i)
            dataset = dataset[torch.tensor(indices)]

            if dataset.transform is None:
                dataset.transform = T.ToDense(num_nodes)
            else:
                dataset.transform = T.Compose(
                    [dataset.transform, T.ToDense(num_nodes)])
    
    return dataset

class HitGraphDataset2(Dataset):
    """PyTorch geometric dataset from processed hit information"""
    
    def __init__(self, root,
                 directed = True,
                 categorical = False,
                 transform = None,
                 pre_transform = None):
        self._directed = directed
        self._categorical = categorical
        super(HitGraphDataset2, self).__init__(root, transform, pre_transform)
    
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

            y_nodes = np.zeros(x.shape[0])
            categories = np.unique(y)

            if not self._categorical:
                y = g.y.astype(np.float32)
            #print('y type',y.dtype)

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

def fixed_train_val_set(dataset, model, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, logger=None):

    val_losses, accs, durations = [], [], []
    
    fulllen = len(dataset)
    tv_frac = 0.20
    tv_num = math.ceil(fulllen*tv_frac)
    splits = np.cumsum([fulllen-tv_num,0,tv_num])
    splits = splits.astype(np.int32)
    print('fulllen:', fulllen,' splits:', splits)
   

    train_dataset = torch.utils.data.Subset(dataset,np.arange(start=0,stop=splits[0]).tolist() )
    valid_dataset = torch.utils.data.Subset(dataset,np.arange(start=splits[1],stop=splits[2]).tolist() )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
    val_loader  = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    train_samples = len(train_dataset)
    valid_samples = len(valid_dataset)

    d = dataset
    num_features = d.num_features
    num_classes = d[0].y.dim() if d[0].y.dim() == 1 else d[0].y.size(1)

    # pdb.set_trace()
    # train_dataset = dataset[train_idx]
    # test_dataset = dataset[test_idx]
    # val_dataset = dataset[val_idx]
    # train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

   
    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        train_loss = train(model, optimizer, train_loader)
        val_losses.append(eval_loss(model, val_loader))
        accs.append(eval_acc(model, val_loader))
        eval_info = {
            'fold': None,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_losses[-1],
            'test_acc': accs[-1],
        }

        print('Eval INfo:', eval_info)
        if logger is not None:
            logger(eval_info)

        if (epoch+1) % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()
    durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    
    
    # pdb.set_trace()
    # loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    # loss, argmin = loss.min(dim=1)
    # acc = acc[torch.arange(folds, dtype=torch.long), argmin]

    loss   = loss.min()
    argmin = loss.argmin() 
    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}s'.format(loss_mean, acc_mean, acc_std, duration_mean))

    return loss_mean, acc_mean, acc_std

'''
def cross_validation_with_val_set(dataset, model, folds, epochs, batch_size,
                                  lr, lr_decay_factor, lr_decay_step_size,
                                  weight_decay, logger=None):

    val_losses, accs, durations = [], [], []
    
    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]
        
        # pdb.set_trace()

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            train_loss = train(model, optimizer, train_loader)
            val_losses.append(eval_loss(model, val_loader))
            accs.append(eval_acc(model, test_loader))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_losses[-1],
                'test_acc': accs[-1],
            }

            if logger is not None:
                logger(eval_info)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(folds, dtype=torch.long), argmin]

    loss_mean = loss.mean().item()
    acc_mean = acc.mean().item()
    acc_std = acc.std().item()
    duration_mean = duration.mean().item()
    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(loss_mean, acc_mean, acc_std, duration_mean))

    return loss_mean, acc_mean, acc_std

def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0

        train_indices.append(torch.nonzero(train_mask).view(-1))

    return train_indices, test_indices, val_indices

'''

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

class ASAP(torch.nn.Module):
    def __init__(self, num_classes, num_features, num_layers, hidden, ratio=0.8, dropout=0):
        super(ASAP, self).__init__()
        self.conv1 = GraphConv(num_features, hidden, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden, hidden, aggr='mean')
            for i in range(num_layers - 1)
        ])
        self.pools.extend([
            ASAPooling(hidden, ratio, dropout=dropout)
            for i in range((num_layers) // 2)
        ])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = None
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, edge_weight, batch, _ = pool(
                    x=x, edge_index=edge_index, edge_weight=edge_weight,
                    batch=batch)

        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


def train(model, optimizer, loader):
    model.train()
    print("In Trainer")
    print("data loader size = ", len(loader))

    total_loss = 0
    for data in tqdm(loader):      
        optimizer.zero_grad()
        data = data.to(device)
        
        # print(data)
        pdb.set_trace()

        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        # pdb.set_trace()
    return total_loss / len(loader.dataset)

def eval_acc(model, loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
        # pdb.set_trace()
    return correct / len(loader.dataset)

def eval_loss(model, loader):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)


def main():

    print('Testing ASAP')
    layers = [3]
    hiddens = [256]    
    # datasets = ['PROTEINS']
    datasets = ['node']
    nets = [ASAP]
    results = []

    for dataset_name, Net in product(datasets, nets): 
    # over a list of net definitions / only one here
        
        best_result = (float('inf'), 0, 0)  # (loss, acc, std)
        print('-----\n{} - {}'.format(dataset_name, Net.__name__))
        
        for num_layers, hidden in product(layers, hiddens): 
        # over a number of diffent sizes of this net 

            # dataset = get_dataset(dataset_name, sparse=Net != DiffPool)
            dataset = get_dataset(dataset_name, sparse=Net)

            print(' Dataset  :', dataset_name)
            print(' NumLayers:', num_layers)
            print(' HiddenDim:', hidden)
            
            if dataset_name == 'node':
                num_classes = 4
                num_features = 5
                
            model = Net(num_classes, num_features, num_layers, hidden)
            print(' Model     :\n', model)
            for name, param in model.named_parameters():    
                print(name, param.shape)
            '''
            loss, acc, std = cross_validation_with_val_set(
                dataset,
                model,
                folds=10,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                lr_decay_factor=args.lr_decay_factor,
                lr_decay_step_size=args.lr_decay_step_size,
                weight_decay=0,
                logger=None,
            )
            '''

            loss, acc, std = fixed_train_val_set(
                dataset,
                model,
                folds=None,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                lr_decay_factor=args.lr_decay_factor,
                lr_decay_step_size=args.lr_decay_step_size,
                weight_decay=0,
                logger=None,
            )

            if loss < best_result[0]:
                best_result = (loss, acc, std)

        desc = '{:.3f} ± {:.3f}'.format(best_result[1], best_result[2])
        print('Best result - {}'.format(desc))
        results += ['{} - {}: {}'.format(dataset_name, model, desc)]
    print('-----\n{}'.format('\n'.join(results)))


if __name__ == "__main__" :
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=50)
    parser.add_argument('--plot_results',type=int, default=0) # 'will plot results if 1'

    args = parser.parse_args()

    main()