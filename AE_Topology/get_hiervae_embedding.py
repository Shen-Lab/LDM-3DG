import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import rdkit
import math, random, sys
import numpy as np
import argparse
import os
from tqdm.auto import tqdm

from hgraph import *
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import networkx as nx

import pdb


lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--save_dir', required=False)
parser.add_argument('--load_model', default=None)
parser.add_argument('--seed', type=int, default=7)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--step_beta', type=float, default=0.001)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)

parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--anneal_iter', type=int, default=25000)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=5000)

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
random.seed(args.seed)

vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
args.vocab = PairVocab(vocab)

model = HierVAE(args).cuda()
print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)



# load model
# ckpt = torch.load('ckpt/mol3d_chembl_pretrained/last.ckpt')
# ckpt = torch.load('ckpt/mol3d_chembl_pretrained_new/last.ckpt')
# ckpt = torch.load('ckpt/mol3d_chembl_pretrained_gssl/last.ckpt')
ckpt = torch.load('ckpt/pocket_pretrained/last.ckpt')

state_dict = {k[6:]:v for k,v in ckpt['state_dict'].items()}
model.load_state_dict(state_dict)

model.eval()


def is_same_molecule(smiles1, smiles2):
    mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
    f1, f2 = Chem.RDKFingerprint(mol1), Chem.RDKFingerprint(mol2)
    sim = DataStructs.FingerprintSimilarity(f1, f2)
    return mol1.HasSubstructMatch(mol2) and mol2.HasSubstructMatch(mol1), sim


emb_dict = {}

dataset = DataFolder(args.train, args.batch_size)
for batch in tqdm(dataset):

    with torch.no_grad():
        outs = model.encode(batch[:3])

    for n, smi in enumerate(batch[-1]):
        emb_dict[smi] = outs[n]


# torch.save(emb_dict, 'hiervae_smiles2emb_dict.pt')
# torch.save(emb_dict, 'hiervae_smiles2emb_dict_new.pt')
# torch.save(emb_dict, 'hiervae_smiles2emb_dict_gssl.pt')
torch.save(emb_dict, 'hiervae_smiles2emb_dict_pocket.pt')


### torch.save(emb_dict, 'hiervae_smiles2emb_dict_random.pt')

