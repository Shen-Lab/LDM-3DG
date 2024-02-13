import os, json
import torch
import torch_geometric as tgeom
import os.path as osp
import pandas as pd
import numpy as np

from tqdm import tqdm
from rdkit import Chem
from itertools import repeat
from torch_geometric.data import InMemoryDataset
from datasets.datasets_utils import mol_to_graph_data_obj_simple_3D
from torch_geometric.nn import radius_graph
import pdb


class Molecule3D(InMemoryDataset):
    def __init__(
        self, root, split='train', split_mode='random', radius_cutoff=5.0,
        transform=None, pre_transform=None, pre_filter=None, args=None):
        assert split in ['train', 'val', 'test']
        assert split_mode in ['random', 'scaffold']
        self.split_mode = split_mode
        self.radius_cutoff = radius_cutoff
        self.root = root
        self.target_df = pd.read_csv(osp.join(self.raw_dir, 'properties.csv'))

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        super(Molecule3D, self).__init__(root, transform, pre_transform, pre_filter)
        
        # self.data, self.slices = torch.load(osp.join(self.processed_dir, '{}_{}.pt'.format(split_mode, split)))

        '''
        # process data
        self.process_yy('train')
        self.process_yy('val')
        assert False
        '''

        self.data, self.slices = torch.load('./processed_data_2d_to_3d/data_' + split + '.pt')
        with open('./processed_data_2d_to_3d/smiles_' + split + '.txt', 'r') as f:
            self.smiles_list = f.read().split('\n')[:-1]
        self.smiles2emb_dict = torch.load('/scratch/user/yuning.you/project/graph_latent_diffusion/hgraph2graph/hiervae_smiles2emb_dict.pt', map_location='cpu')

        self.args=args

        return

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        name = ''
        return name

    @property
    def processed_file_names(self):
        return ['random_train.pt', 'random_val.pt', 'random_test.pt',
                'scaffold_train.pt', 'scaffold_val.pt', 'scaffold_test.pt']

    def download(self):
        pass

    def process_yy(self, train_or_val='train'):
        with open('/scratch/user/yuning.you/project/graph_latent_diffusion/data/smiles_mol3d_chembl_' + train_or_val + '.txt', 'r') as f:
            smiles_list = f.read().split('\n')[:-1]

        data_list = []
        data_smiles_list = []
        sdf_paths = [osp.join(self.raw_dir, 'combined_mols_0_to_1000000.sdf'),
                     osp.join(self.raw_dir, 'combined_mols_1000000_to_2000000.sdf'),
                     osp.join(self.raw_dir, 'combined_mols_2000000_to_3000000.sdf'),
                     osp.join(self.raw_dir, 'combined_mols_3000000_to_3899647.sdf')]
        suppl_list = [Chem.SDMolSupplier(p, removeHs=False, sanitize=True) for p in sdf_paths]
        
        abs_idx = -1
        for i, suppl in enumerate(suppl_list):
            for j in tqdm(range(len(suppl)), desc=f'{i+1}/{len(sdf_paths)}'):
                abs_idx += 1
                mol = suppl[j]

                # added by YY: filter
                mol_rmhs = Chem.RemoveHs(mol)
                smiles_rmhs = Chem.MolToSmiles(mol_rmhs, isomericSmiles=False)
                smiles_rmhs = Chem.CanonSmiles(smiles_rmhs)
                if not smiles_rmhs in smiles_list: continue

                data, _ = mol_to_graph_data_obj_simple_3D(mol)

                # added by YY: add radius edge index
                radius_edge_index = radius_graph(data.positions, r=self.radius_cutoff, loop=False)
                data.radius_edge_index = radius_edge_index

                data_list.append(data)
                data_smiles_list.append(smiles_rmhs)

        torch.save(self.collate(data_list), './processed_data_2d_to_3d/data_' + train_or_val + '.pt')
        with open('./processed_data_2d_to_3d/smiles_' + train_or_val + '.txt', 'w') as f:
            for smi in data_smiles_list:
                f.write(smi + '\n')

        return data_list, data_smiles_list

    def pre_process(self):
        data_list = []
        data_smiles_list = []
        sdf_paths = [osp.join(self.raw_dir, 'combined_mols_0_to_1000000.sdf'),
                     osp.join(self.raw_dir, 'combined_mols_1000000_to_2000000.sdf'),
                     osp.join(self.raw_dir, 'combined_mols_2000000_to_3000000.sdf'),
                     osp.join(self.raw_dir, 'combined_mols_3000000_to_3899647.sdf')]
        suppl_list = [Chem.SDMolSupplier(p, removeHs=False, sanitize=True) for p in sdf_paths]
        
        abs_idx = -1
        for i, suppl in enumerate(suppl_list):
            for j in tqdm(range(len(suppl)), desc=f'{i+1}/{len(sdf_paths)}'):
                abs_idx += 1
                mol = suppl[j]
                smiles = Chem.MolToSmiles(mol)
                data, _ = mol_to_graph_data_obj_simple_3D(mol)

                # added by YY: add radius edge index
                radius_edge_index = radius_graph(data.positions, r=self.radius_cutoff, loop=False)
                data.radius_edge_index = radius_edge_index

                data_list.append(data)
                data_smiles_list.append(smiles)

        return data_list, data_smiles_list

    def process(self):
        dir_ = os.path.join(self.root, "processed")
        os.makedirs(dir_, exist_ok=True)
        print("dir: ", dir_)
        saver_path = os.path.join(dir_, "geometric_data_processed.pt")
        if not os.path.exists(saver_path):
            full_list, full_smiles_list = self.pre_process()
            index_list = np.arange(len(full_list))

            data_list = [self.get_data_prop(full_list, idx) for idx in index_list]
            print("Saving to {}.".format(saver_path))
            torch.save(self.collate(data_list), saver_path)

            data_smiles_series = pd.Series(full_smiles_list)
            saver_path = os.path.join(dir_, "smiles.csv")
            print("Saving to {}.".format(saver_path))
            data_smiles_series.to_csv(saver_path, index=False, header=False)
        else:
            # TODO: this is for fast preprocessing. will add loader later.    
            full_list, full_smiles_list = self.pre_process()

        print("len of full list: {}".format(len(full_list)))
        print("len of full smiles list: {}".format(len(full_smiles_list)))
        print("target_df:", self.target_df.shape)

        print('making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
                
        for m, split_mode in enumerate(['random', 'scaffold']):
            ind_path = osp.join(self.raw_dir, '{}_split_inds.json').format(split_mode)
            with open(ind_path, 'r') as f:
                 index_list = json.load(f)
            
            for s, split in enumerate(['train', 'valid', 'test']):
                data_list = [self.get_data_prop(full_list, idx) for idx in index_list[split]]
                data_smiles_list = [full_smiles_list[idx] for idx in index_list[split]]
                if self.pre_filter is not None:
                    data_list = [data for data in data_list if self.pre_filter(data)]
                if self.pre_transform is not None:
                    data_list = [self.pre_transform(data) for data in data_list]

                data_smiles_series = pd.Series(data_smiles_list)
                saver_path = os.path.join(self.processed_dir, "{}_{}_smiles.csv".format(split_mode, split))
                print("Saving to {}.".format(saver_path))
                data_smiles_series.to_csv(saver_path, index=False, header=False)

                torch.save(self.collate(data_list), self.processed_paths[s+3*m])
        return

    def get_data_prop(self, full_list, abs_idx):
        data = full_list[abs_idx]
        data.y = torch.FloatTensor(self.target_df.iloc[abs_idx,1:].values)
        return data

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get(self, idx):
        # added by YY: break when graph size > 1
        while True:
            data = self.data.__class__()

            if hasattr(self.data, '__num_nodes__'):
                data.num_nodes = self.data.__num_nodes__[idx]

            for key in self.data.keys:
                item, slices = self.data[key], self.slices[key]
                if torch.is_tensor(item):
                    s = list(repeat(slice(None), item.dim()))
                    s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
                else:
                    s = slice(slices[idx], slices[idx + 1])
                data[key] = item[s]

            if data.x.shape[0] > 1: break
            else: idx = np.random.choice(len(self))

        # added by YY: change dtype of attributes
        # added by YY: only consider atom identities
        data.x = data.x.float()[:, :118]
        data.edge_attr = data.edge_attr.float()

        data.n_nodes = data.x.shape[0]
        data.n_edges = data.edge_index.shape[1]

        # added by YY: HierVAE embedding
        smiles = self.smiles_list[idx]
        emb_2d = self.smiles2emb_dict[smiles].view(1, -1)

        data.emb_2d = emb_2d

        data.idx = idx
        return data

