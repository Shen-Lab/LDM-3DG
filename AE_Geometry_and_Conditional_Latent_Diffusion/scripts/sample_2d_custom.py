import argparse
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import pdb
import os, sys
from hgraph.hgnn import HierVAE
from hgraph.vocab import PairVocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./')

    parser.add_argument('--sample_number', type=int, default=10000)

    # number of nodes for parallel training
    parser.add_argument('--ddp_num_nodes', type=int, default=1)
    # number of devices in each node for parallel training
    parser.add_argument('--ddp_device', type=int, default=1)
    args = parser.parse_args()

    print(args)

    device = 'cuda'

    samples = torch.load('./samples_latent/sample_z_custom.pt') # [:10000]
    n1, n2, n3 = samples.shape
    samples = samples.reshape((n1*n2, n3))[:, :250]

    # 2. decode G~p(G|z)
    args3 = torch.load('/scratch/user/yuning.you/project/graph_latent_diffusion/hgraph2graph/ckpt/args_new.pt')
    vocab = [x.strip('\r\n ').split() for x in open('/scratch/user/yuning.you/project/graph_latent_diffusion/hgraph2graph/vocab_pocket_aware.txt')]
    args3.vocab = PairVocab(vocab)
    model = HierVAE(args3).to(device)
    ckpt = torch.load('/scratch/user/yuning.you/project/graph_latent_diffusion/hgraph2graph/ckpt/pocket_pretrained/last.ckpt')
    state_dict = {k[6:]:v for k,v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    dataset = torch.utils.data.TensorDataset(samples)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    smiles = []
    for batch in tqdm(dataloader):
        batch = batch[0].to(device)
        with torch.no_grad():
            ss = model.decode(batch)

            # try:
            #     ss = model.decode(batch[:, :250])
            # except:
            #     ss = [None for _ in range(batch.shape[0])]

            smiles += ss

    torch.save(smiles, './samples_latent/sample_smiles_custom.pt')

