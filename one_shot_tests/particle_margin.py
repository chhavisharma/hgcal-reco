import os.path as osp
import glob

import multiprocessing as mp
from tqdm import tqdm
import random
import torch
import pandas
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.utils import is_undirected
from torch_geometric.data import Data, Dataset
import gzip
import pdb

class TrackMLParticleTrackingDataset(Dataset):
    r"""The `TrackML Particle Tracking Challenge
    <https://www.kaggle.com/c/trackml-particle-identification>`_ dataset to
    reconstruct particle tracks from 3D points left in the silicon detectors.

    Args:
        root (string): Root directory where the dataset should be saved.

        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)

        n_events (int): Number of events in the raw folder to process



    GRAPH CONSTRUCTION PARAMETERS
    ###########################################################################

        volume_layer_ids (List): List of the volume and layer ids to be included
            in the graph. Layers get indexed by increasing volume and layer id.
            Refer to the following map for the layer indices, and compare them
            to the chart at https://www.kaggle.com/c/trackml-particle-identification/data

                                            41
                        34 --- 39            |        42 --- 47
                                            40

                                            27
                        18 --- 23            |        28 --- 33
                                            24

                                            10
                         0 ---  6            |        11 --- 17
                                             7

        layer_pairs (List): List of which pairs of layers can have edges between them.
            Uses the layer indices described above to reference layers.
            Example for Barrel Only:
            [[7,8],[8,9],[9,10],[10,24],[24,25],[25,26],[26,27],[27,40],[40,41]]

        pt_min (float32): A truth cut applied to reduce the number of nodes in the graph.
            Only nodes associated with particles above this momentum are included.

        eta_range ([min, max]): A cut applied to nodes to select a specific eta

        phi_slope_max (float32): A cut applied to edges to limit the change in phi between
            the two nodes.

        z0_max (float32): A cut applied to edges that limits how far from the center of
            the detector the particle edge can originate from.

        n_phi_sections (int): Break the graph into multiple segments in the phi direction.

        n_eta_sections (int): Break the graph into multiple segments in the eta direction.

        augments (bool): Toggle for turning data augmentation on and off

        intersect (bool): Toggle for interseting lines cut. When connecting Barrel
            edges to the inner most endcap layer, sometimes the edge passes through
            the layer above, this cut removes those edges.

        tracking (bool): Toggle for building truth tracks. Track data is a tensor with
            dimensions (Nx5) with the following columns:
            [r coord, phi coord, z coord, layer index, track number]

        directed (bool): Edges are directed, for an undirected graph, edges are
            duplicated and in reverse direction.

        layer_pairs_plus (bool): Allows for edge connections within the same layer    


    MULTIPROCESSING PARAMETERS
    ###########################################################################

        n_workers (int): Number of worker nodes for multiprocessing

        n_tasks (int): Break the processing into a number of tasks

    """

    url = 'https://www.kaggle.com/c/trackml-particle-identification'

    def __init__(self, root, transform=None, n_events=0,
                 directed=False, layer_pairs_plus=False,
                 volume_layer_ids=[[8, 2], [8, 4], [8, 6], [8, 8]], #Layers Selected
                 layer_pairs=[[7, 8], [8, 9], [9, 10]],             #Connected Layers
                 pt_min=2.0, eta_range=[-5, 5],                     #Node Cuts
                 phi_slope_max=0.0006, z0_max=150,                  #Edge Cuts
                 n_phi_sections=1, n_eta_sections=1,                #N Sections
                 augments=False, intersect=False, tracking=False,   #Toggle Switches
                 n_workers=mp.cpu_count(), n_tasks=1                #multiprocessing
                 ):
        events = glob.glob(osp.join(osp.join(root, 'raw'), 'event*-hits.csv'))
        # print('events found = ', len(events))
        #pdb.set_trace()
        
        events = [e.split(osp.sep)[-1].split('-')[0][5:] for e in events]
        self.events = sorted(events)
        if (n_events > 0):
            self.events = self.events[:n_events]

        self.directed         = directed
        self.layer_pairs_plus = layer_pairs_plus
        self.volume_layer_ids = torch.tensor(volume_layer_ids)
        self.layer_pairs      = torch.tensor(layer_pairs)
        self.pt_min           = pt_min
        self.eta_range        = eta_range
        self.phi_slope_max    = phi_slope_max
        self.z0_max           = z0_max
        self.n_phi_sections   = n_phi_sections
        self.n_eta_sections   = n_eta_sections
        self.augments         = augments
        self.intersect        = intersect
        self.tracking         = tracking
        self.n_workers        = n_workers
        self.n_tasks          = n_tasks

        super(TrackMLParticleTrackingDataset, self).__init__(root, transform)


    @property
    def raw_file_names(self):
        if not hasattr(self,'input_files'):
            self.input_files = sorted(glob.glob(self.raw_dir+'/*.csv'))
        return [f.split('/')[-1] for f in self.input_files]


    @property
    def processed_file_names(self):
        N_sections = self.n_phi_sections*self.n_eta_sections
        if not hasattr(self,'processed_files'):
            proc_names = ['event{}_section{}.pt.gz'.format(idx, i) for idx in self.events for i in range(N_sections)]
            if(self.augments):
                proc_names_aug = ['event{}_section{}_aug.pt.gz'.format(idx, i) for idx in self.events for i in range(N_sections)]
                proc_names = [x for y in zip(proc_names, proc_names_aug) for x in y]
            self.processed_files = [osp.join(self.processed_dir,name) for name in proc_names]
        return self.processed_files


    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download it from {} and move all '
            '*.csv files to {}'.format(self.url, self.raw_dir))


    def len(self):
        N_events = len(self.events)
        N_augments = 2 if self.augments else 1
        return N_events*self.n_phi_sections*self.n_eta_sections*N_augments


    def __len__(self):
        N_events = len(self.events)
        N_augments = 2 if self.augments else 1
        return N_events*self.n_phi_sections*self.n_eta_sections*N_augments


    def read_hits(self, idx):
        hits_filename = osp.join(self.raw_dir, f'event{idx}-hits.csv')
        hits = pandas.read_csv(
            hits_filename, usecols=['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'module_id'],
            dtype={
                'hit_id': np.int64,
                'x': np.float32,
                'y': np.float32,
                'z': np.float32,
                'volume_id': np.int64,
                'layer_id': np.int64,
                'module_id': np.int64
            })
        return hits


    def read_cells(self, idx):
        cells_filename = osp.join(self.raw_dir, f'event{idx}-cells.csv')
        cells = pandas.read_csv(
            cells_filename, usecols=['hit_id', 'ch0', 'ch1', 'value'],
            dtype={
                'hit_id': np.int64,
                'ch0': np.int64,
                'ch1': np.int64,
                'value': np.float32
            })
        return cells


    def read_particles(self, idx):
        particles_filename = osp.join(self.raw_dir, f'event{idx}-particles.csv')
        particles = pandas.read_csv(
            particles_filename, usecols=['particle_id', 'vx', 'vy', 'vz', 'px', 'py', 'pz', 'q', 'nhits'],
            dtype={
                'particle_id': np.int64,
                'vx': np.float32,
                'vy': np.float32,
                'vz': np.float32,
                'px': np.float32,
                'py': np.float32,
                'pz': np.float32,
                'q': np.int64,
                'nhits': np.int64
            })
        return particles


    def read_truth(self, idx):
        truth_filename = osp.join(self.raw_dir, f'event{idx}-truth.csv')
        truth = pandas.read_csv(
            truth_filename, usecols=['hit_id', 'particle_id', 'tx', 'ty', 'tz', 'tpx', 'tpy', 'tpz', 'weight'],
            dtype={
                'hit_id': np.int64,
                'particle_id': np.int64,
                'tx': np.float32,
                'ty': np.float32,
                'tz': np.float32,
                'tpx': np.float32,
                'tpy': np.float32,
                'tpz': np.float32,
                'weight': np.float32
            })
        return truth


    def select_hits(self, hits, particles, truth):
        # print('Selecting Hits')
        valid_layer = 20 * self.volume_layer_ids[:,0] + self.volume_layer_ids[:,1]
        n_det_layers = len(valid_layer)

        layer = torch.from_numpy(20 * hits['volume_id'].values + hits['layer_id'].values)
        index = layer.unique(return_inverse=True)[1]
        hits = hits[['hit_id', 'x', 'y', 'z']].assign(layer=layer, index=index)
        
        valid_groups = hits.groupby(['layer'])
        hits = pandas.concat([valid_groups.get_group(valid_layer.numpy()[i]) for i in range(n_det_layers)])

        pt = np.sqrt(particles['px'].values**2 + particles['py'].values**2)
        particles_mask = pt > self.pt_min
        particles_fail = particles[~particles_mask]
        particles = particles[particles_mask]

        hits = (hits[['hit_id', 'x', 'y', 'z', 'index']].merge(truth[['hit_id', 'particle_id']], on='hit_id'))
        hits['particle_id'].where(hits['particle_id'].isin(particles['particle_id']) | (hits['particle_id'] == 0), -1, inplace=True)
        pids_unique, pids_inverse, pids_counts = np.unique(hits['particle_id'].values, return_inverse=True, return_counts=True)        
        pids_unique = np.arange(pids_unique.size) # make it [not interested, noise, remapped pid]
        hits['remapped_pid'] = pids_unique[pids_inverse]


        #________for testing________________________________________
        print(30 + hits.size%3)
        hits = hits[(hits['remapped_pid'] > 0) & (hits['remapped_pid'] < (30 + hits.size%3))]
        hits['remapped_pid'] = hits['remapped_pid'] - 1
        
        
        r = np.sqrt(hits['x'].values**2 + hits['y'].values**2)
        phi = np.arctan2(hits['y'].values, hits['x'].values)
        theta = np.arctan2(r,hits['z'].values)
        eta = -1*np.log(np.tan(theta/2))
        hits = hits[['z', 'index', 'particle_id', 'remapped_pid']].assign(r=r, phi=phi, eta=eta)

        # Remove duplicate hits
        if not self.layer_pairs_plus:
            hits = hits.loc[hits.groupby(['particle_id', 'index'], as_index=False).r.idxmin()]

        r = torch.from_numpy(hits['r'].values)
        phi = torch.from_numpy(hits['phi'].values)
        z = torch.from_numpy(hits['z'].values)
        eta = torch.from_numpy(hits['eta'].values)
        layer = torch.from_numpy(hits['index'].values)
        particle = torch.from_numpy(hits['particle_id'].values)
        plabel = torch.from_numpy(hits['remapped_pid'].values)
        pos = torch.stack([r, phi, z], 1)
        
        return pos, layer, particle, eta, plabel, particles


    def compute_edge_index(self, pos, layer):
        # print("Constructing Edge Index")
        edge_indices = torch.empty(2,0, dtype=torch.long)

        layer_pairs = self.layer_pairs
        if self.layer_pairs_plus:
            layers = layer.unique()
            layer_pairs_plus = torch.tensor([[layers[i],layers[i]] for i in range(layers.shape[0])])
            layer_pairs = torch.cat((layer_pairs, layer_pairs_plus), 0)

        for (layer1, layer2) in layer_pairs:
            mask1 = layer == layer1
            mask2 = layer == layer2
            nnz1 = mask1.nonzero().flatten()
            nnz2 = mask2.nonzero().flatten()

            dr   = pos[:, 0][mask2].view(1, -1) - pos[:, 0][mask1].view(-1, 1)
            dphi = pos[:, 1][mask2].view(1, -1) - pos[:, 1][mask1].view(-1, 1)
            dz   = pos[:, 2][mask2].view(1, -1) - pos[:, 2][mask1].view(-1, 1)
            dphi[dphi > np.pi] -= 2 * np.pi
            dphi[dphi < -np.pi] += 2 * np.pi

            # Calculate phi_slope and z0 which will be cut on
            phi_slope = dphi / dr
            z0 = pos[:, 2][mask1].view(-1, 1) - pos[:, 0][mask1].view(-1, 1) * dz / dr

            # Check for intersecting edges between barrel and endcap connections
            intersected_layer = dr.abs() < -1
            if (self.intersect):
                if((layer1 == 7 and (layer2 == 6 or layer2 == 11)) or
                   (layer2 == 7 and (layer1 == 6 or layer1 == 11))):
                    z_int =  71.56298065185547 * dz / dr + z0
                    intersected_layer = z_int.abs() < 490.975
                elif((layer1 == 8 and (layer2 == 6 or layer2 == 11)) or
                     (layer2 == 8 and (layer1 == 6 or layer1 == 11))):
                    z_int = 115.37811279296875 * dz / dr + z0
                    intersected_layer = z_int.abs() < 490.975

            adj = (phi_slope.abs() < self.phi_slope_max) & (z0.abs() < self.z0_max) & (intersected_layer == False)

            row, col = adj.nonzero().t()
            row = nnz1[row]
            col = nnz2[col]
            edge_index = torch.stack([row, col], dim=0)

            edge_indices = torch.cat((edge_indices, edge_index), 1)

        return edge_indices

    def compute_y_index(self, edge_indices, particle):
        # print("Constructing y Index")
        pid1 = [ particle[i].item() for i in edge_indices[0] ]
        pid2 = [ particle[i].item() for i in edge_indices[1] ]
        y = np.zeros(edge_indices.shape[1], dtype=np.int64)
        for i in range(edge_indices.shape[1]):
            if pid1[i] == pid2[i]:
                y[i] = 1

        return torch.from_numpy(y)

    def split_detector_sections(self, pos, layer, particle, eta, particle_label, phi_edges, eta_edges):
        pos_sect, layer_sect, particle_sect, particle_label_sect = [], [], [], []

        for i in range(len(phi_edges) - 1):
            phi_mask1 = pos[:,1] > phi_edges[i]
            phi_mask2 = pos[:,1] < phi_edges[i+1]
            phi_mask  = phi_mask1 & phi_mask2
            phi_pos      = pos[phi_mask]
            phi_layer    = layer[phi_mask]
            phi_particle = particle[phi_mask]
            phi_eta      = eta[phi_mask]
            phi_particle_label = particle_label[phi_mask]

            for j in range(len(eta_edges) - 1):
                eta_mask1 = phi_eta > eta_edges[j]
                eta_mask2 = phi_eta < eta_edges[j+1]
                eta_mask  = eta_mask1 & eta_mask2
                phi_eta_pos = phi_pos[eta_mask]
                phi_eta_layer = phi_layer[eta_mask]
                phi_eta_particle = phi_particle[eta_mask]
                phi_eta_particle_label = phi_particle_label[eta_mask]
                pos_sect.append(phi_eta_pos)
                layer_sect.append(phi_eta_layer)
                particle_sect.append(phi_eta_particle)
                particle_label_sect.append(phi_eta_particle_label)

        return pos_sect, layer_sect, particle_sect, particle_label_sect


    def read_event(self, idx):
        hits      = self.read_hits(idx)
        # cells     = self.read_cells(idx)
        particles = self.read_particles(idx)
        truth     = self.read_truth(idx)

        return hits, particles, truth


    def process(self, reprocess=False):
        print('Constructing Graphs using n_workers = ' + str(self.n_workers))
        task_paths = np.array_split(self.processed_paths, self.n_tasks)
        for i in range(self.n_tasks):
            if reprocess or not self.files_exist(task_paths[i]):
                self.process_task(i)


    def process_task(self, idx):
        print('Running task ' + str(idx))
        task_events = np.array_split(self.events, self.n_tasks)
        #with mp.Pool(processes = self.n_workers) as pool:
        #    pool.map(self.process_event, tqdm(task_events[idx]))
        for te in tqdm(task_events[idx]):
            self.process_event(te)

    def process_event(self, idx):
        hits, particles, truth = self.read_event(idx)
        pos, layer, particle, eta, particle_label, tps = self.select_hits(hits, particles, truth)

        tps['pt'] = np.sqrt(tps['px']**2 + tps['py']**2)
        tps['pmag'] = np.sqrt(tps['pt']**2 + tps['pz']**2)
        tps['eta'] = 0.5*(np.log(tps['pmag'] + tps['pz']) - np.log(tps['pmag'] - tps['pz']))
        tps['phi'] = np.arctan2(tps['py'], tps['px'])

        true_barcode = torch.from_numpy(tps['particle_id'].values)
        true_charge = torch.from_numpy(tps['q'].values)
        true_pt = torch.from_numpy(tps['pt'].values)
        true_eta = torch.from_numpy(tps['eta'].values)
        true_phi = torch.from_numpy(tps['phi'].values)

        tracks = torch.empty(0, dtype=torch.long)
        if(self.tracking):
            tracks = self.build_tracks(hits, particles, truth)

        phi_edges = np.linspace(*(-np.pi, np.pi), num=self.n_phi_sections+1)
        eta_edges = np.linspace(*self.eta_range, num=self.n_eta_sections+1)
        pos_sect, layer_sect, particle_sect, particle_label_sect = self.split_detector_sections(pos, layer, particle, eta, particle_label, phi_edges, eta_edges)

        for i in range(len(pos_sect)):
            edge_index = torch.empty((2,0), dtype=torch.int64)# self.compute_edge_index(pos_sect[i], layer_sect[i])
            y = particle_label_sect[i]
            y_particle_barcodes = particle_sect[i]
            
            data = Data(x=pos_sect[i], edge_index=edge_index,
                        y=y,
                        y_particle_barcodes=y_particle_barcodes,
                        truth_barcodes=true_barcode,
                        truth_charge=true_charge,
                        truth_pt=true_pt,
                        truth_eta=true_eta,
                        truth_phi=true_phi,
                        tracks=tracks)
            
            #if not self.directed and not data.is_undirected():
            #    rows,cols = data.edge_index
            #    temp = torch.stack((cols,rows))
            #    data.edge_index = torch.cat([data.edge_index,temp],dim=-1)
            #    data.y = torch.cat([data.y,data.y])

            with gzip.open(osp.join(self.processed_dir, 'event{}_section{}.pt.gz'.format(idx, i)), 'wb') as fout:
                torch.save(data, fout)

            if (self.augments):
                data.x[:,1]= -data.x[:,1]
                with gzip.open(osp.join(self.processed_dir, 'event{}_section{}_aug.pt.gz'.format(idx, i)), 'wb') as fout:
                    torch.save(data, fout)
                    
        # if self.pre_filter is not None and not self.pre_filter(data):
        #     continue
        #
        # if self.pre_transform is not None:
        #     data = self.pre_transform(data)


    def get(self, idx):
        with gzip.open(self.processed_files[idx], 'rb') as fin:
            data = torch.load(fin)
        return data


    def draw(self, idx):
        # print("Making plots for " + str(self.processed_files[idx]))
        width1 = .1
        width2 = .2
        points = .25

        X = self[idx].x.cpu().numpy()
        index = self[idx].edge_index.cpu().numpy()
        y = self[idx].y.cpu().numpy()
        true_index = index[:,y > 0]

        r_co = X[:,0]
        z_co = X[:,2]
        x_co = X[:,0]*np.cos(X[:,1])
        y_co = X[:,0]*np.sin(X[:,1])

        # scale = 12*z_co.max()/r_co.max()
        fig0, (ax0) = plt.subplots(1, 1, dpi=500, figsize=(6, 6))
        fig1, (ax1) = plt.subplots(1, 1, dpi=500, figsize=(6, 6))

        # Adjust axes
        ax0.set_xlabel('Z [mm]')
        ax0.set_ylabel('R [mm]')
        ax0.set_xlim(-1.1*np.abs(z_co).max(), 1.1*np.abs(z_co).max())
        ax0.set_ylim(-1.1*r_co.max(), 1.1*r_co.max())
        ax1.set_xlabel('X [mm]')
        ax1.set_ylabel('Y [mm]')
        ax1.set_xlim(-1.1*r_co.max(), 1.1*r_co.max())
        ax1.set_ylim(-1.1*r_co.max(), 1.1*r_co.max())

        r_co[X[:,1] < 0] *= -1

        #plot points
        ax0.scatter(z_co, r_co, s=points, c='k')
        ax1.scatter(x_co, y_co, s=points, c='k')

        ax0.plot([z_co[index[0]], z_co[index[1]]],
                 [r_co[index[0]], r_co[index[1]]],
                 '-', c='blue', linewidth=width1)
        ax0.plot([z_co[true_index[0]], z_co[true_index[1]]],
                 [r_co[true_index[0]], r_co[true_index[1]]],
                 '-', c='black', linewidth=width2)
        ax1.plot([x_co[index[0]], x_co[index[1]]],
                 [y_co[index[0]], y_co[index[1]]],
                 '-', c='blue', linewidth=width1)
        ax1.plot([x_co[true_index[0]], x_co[true_index[1]]],
                 [y_co[true_index[0]], y_co[true_index[1]]],
                 '-', c='black', linewidth=width2)

        fig0_name = self.processed_files[idx].split('.')[0] + '_rz.png'
        fig1_name = self.processed_files[idx].split('.')[0] + '_xy.png'
        fig0.savefig(fig0_name)
        fig1.savefig(fig1_name)


    def build_tracks(self, hits, particles, truth):
        # print('Building Tracks')
        valid_layer = 20 * self.volume_layer_ids[:,0] + self.volume_layer_ids[:,1]
        hits = (hits[['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id']]
                .merge(truth[['hit_id', 'particle_id']], on='hit_id'))
        hits = (hits[['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'particle_id']]
                .merge(particles[['particle_id', 'px', 'py', 'pz']], on='particle_id'))

        layer = torch.from_numpy(20 * hits['volume_id'].values + hits['layer_id'].values)
        r = torch.from_numpy(np.sqrt(hits['x'].values**2 + hits['y'].values**2))
        phi = torch.from_numpy(np.arctan2(hits['y'].values, hits['x'].values))
        z = torch.from_numpy(hits['z'].values)
        pt = torch.from_numpy(np.sqrt(hits['px'].values**2 + hits['py'].values**2))
        particle = torch.from_numpy(hits['particle_id'].values)

        layer_mask = torch.from_numpy(np.isin(layer, valid_layer))
        pt_mask = pt > self.pt_min
        # mask = layer_mask & pt_mask
        mask = pt_mask

        layer = layer.unique(return_inverse=True)[1]
        r = r[mask]
        phi = phi[mask]
        z = z[mask]
        pos = torch.stack([r, phi, z], 1)
        particle = particle[mask]
        layer = layer[mask]

        particle, indices = torch.sort(particle)
        particle = particle.unique(return_inverse=True)[1]
        pos = pos[indices]
        layer = layer[indices]

        tracks = torch.empty(0,5, dtype=torch.float)
        for i in range(particle.max()+1):
            track_pos   = pos[particle == i]
            track_layer = layer[particle == i]
            track_particle = particle[particle == i]
            track_layer, indices = torch.sort(track_layer)
            track_pos = track_pos[indices]
            track_layer = track_layer[:, None]
            track_particle = track_particle[:, None]
            track = torch.cat((track_pos, track_layer.type(torch.float)), 1)
            track = torch.cat((track, track_particle.type(torch.float)), 1)
            tracks = torch.cat((tracks, track), 0)

        return tracks


    def files_exist(self, files):
        return len(files) != 0 and all([osp.exists(f) for f in files])


    def shuffle(self):
        random.shuffle(self.processed_files)


    def sort(self):
        self.processed_files.sort()
