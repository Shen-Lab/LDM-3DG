import os
import torch
import numpy as np
from utils import transforms, reconstruct
import rdkit
from rdkit import Chem
import pdb


r = torch.load('custom_data/result_custom.pt')
mols = []

for idx, (pred_v, pred_pos) in enumerate(zip(r['atom_types'], r['pred_ligand_pos'])):
    if 5 in pred_v: continue
    try:
        pred_aromatic = transforms.is_aromatic_from_index(np.array(pred_v), mode='add_aromatic')
        mol = reconstruct.reconstruct_from_generated(pred_pos, pred_v, pred_aromatic)
        mols.append(mol)
    except:
        continue

os.makedirs('custom_data/mols', exist_ok=True)
# save sdf files
for i, mol in enumerate(mols):
    Chem.MolToMolFile(mol, os.path.join('custom_data/mols', f'mol_{i}.sdf'))

