import argparse
import math
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from models_diffusion.ddpm import Model as DDPMModel
from models_diffusion.ddpm_utils import DDPMSampler
import pdb
import os, sys
from datasets.datasets_utils import mol_to_graph_data_obj_simple_2D
import torch_geometric as tgeom
from torch_geometric.data import Batch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
# from utils_for_3d_embeddings import Model
from main_2dto3d_encoder_decoder import Model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./logs/debug')

    parser.add_argument('--sample_number', type=int, default=10000)

    # number of nodes for parallel training
    parser.add_argument('--ddp_num_nodes', type=int, default=1)
    # number of devices in each node for parallel training
    parser.add_argument('--ddp_device', type=int, default=1)
    args = parser.parse_args()

    print(args)

    device = 'cuda'

    # 3. decode C~p(C|G,z)
    args2 = torch.load('../AE_geom_uncond_weights_and_data/job16_decoder_2d_to_3d_spatial_graphs/args.pt')
    model = Model.load_from_checkpoint('../AE_geom_uncond_weights_and_data/job16_decoder_2d_to_3d_spatial_graphs/last.ckpt', args=args2).decoder_2dto3d.to(device)
    model.eval()

    samples = torch.load(args.log_dir + '/sample_z.pt')
    smiles = torch.load(args.log_dir + '/sample_smiles.pt')
    mol_batch = []
    data_batch = []
    z_batch = []
    batch_size = 32
    mol3d_list = []
    for idx, smi in enumerate(tqdm(smiles)):
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)

        try:
            AllChem.EmbedMolecule(mol, maxAttempts=5000)
            positions = mol.GetConformers()[0].GetPositions()
        except:
            AllChem.Compute2DCoords(mol)
            positions = mol.GetConformers()[0].GetPositions()

        mol_batch.append(mol)

        
        data = mol_to_graph_data_obj_simple_2D(mol)[0]
        data.x = data.x.float()[:, :118]
        data.edge_attr = data.edge_attr.float()
        data.n_nodes = data.x.shape[0]
        data.n_edges = data.edge_index.shape[1]
        data.pos = torch.tensor(positions).float()

        ### spatial graph
        data.edge_index = tgeom.nn.radius_graph(data.pos, r=5, loop=False)
        edge_attr = torch.exp( - torch.norm(data.pos[data.edge_index[0]] - data.pos[data.edge_index[1]], dim=1) )
        data.edge_attr = torch.einsum('i,j->ij', edge_attr, torch.linspace(1, 5, 16).to(edge_attr.device))
        data.n_edges = torch.tensor(data.edge_index.shape[1]).long()
        ###n spatial graph

        data_batch.append(data)

        z_batch.append(samples[idx].unsqueeze(dim=0))

        if len(data_batch) % batch_size == 0 or idx + 1 == len(smiles):
            batch = Batch.from_data_list(data_batch).to(device)
            emb = torch.cat(z_batch).to(device)

            with torch.no_grad():
                # pos_batch = model(batch, emb[:, 250:], True)[0][-1]
                pos_batch = model(batch, emb[:, 250:])[0][-1]
            pos_batch = tgeom.utils.unbatch(pos_batch, batch.batch)

            # set coordinate for molecules
            for mol, pos in zip(mol_batch, pos_batch):
                # AllChem.Compute2DCoords(mol)
                conf = mol.GetConformer()
                for jdx in range(mol.GetNumAtoms()):
                    conf.SetAtomPosition(jdx, Point3D(pos[jdx,0].item(), pos[jdx,1].item(), pos[jdx,2].item()))
                
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                    mol3d_list.append(mol)
                except:
                    continue

            mol_batch = []
            data_batch = []
            z_batch = []

    torch.save(mol3d_list, args.log_dir + '/sample_conformer.pt')
