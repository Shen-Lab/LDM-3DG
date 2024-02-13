import argparse
import os
import shutil
import time

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from utils.evaluation import atom_num
from rdkit.Chem import AllChem
from utils import reconstruct, transforms
import pdb


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior', emb=None, feats=None, lig_batch=None):
    sample_num_atoms = 'ref'
    pos_only = True

    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in range(1):
        n_data = batch_size
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

        t1 = time.time()
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                batch_ligand = lig_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            else:
                raise ValueError

            # init ligand pos
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
            batch_center_pos = center_pos[batch_ligand]
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

            # init ligand v
            if pos_only:
                init_ligand_v = feats
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
                init_ligand_v = log_sample_categorical(uniform_logits)
            
            # pdb.set_trace()

            r = model.reconstruc_3d(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,

                init_ligand_pos=init_ligand_pos,
                emb_lat=emb,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode
            )
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]

            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('-i', '--data_id', type=int)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./outputs')
    args = parser.parse_args()

    logger = misc.get_logger('sampling')

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # Load dataset
    dataset, subsets = get_dataset(
        config=ckpt['config'].data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load the model! {config.model.checkpoint}')



    # start sampling
    from rdkit import Chem
    from datasets.protein_ligand import get_ligand_atom_features

    zs = torch.load('./samples_latent/sample_z.pt')
    smiles = torch.load('./samples_latent/sample_smiles.pt')
    batch_size = zs.shape[1]

    n_batch = 2
    import math
    batch_size2 = math.ceil(float(batch_size) / n_batch)

    for data_id in tqdm(range(100)):
        data = test_set[data_id]
        mols2 = []
        zzs = zs[data_id][:,250:].to('cuda')
        smiss = smiles[data_id*batch_size : (data_id+1)*batch_size]
        for i in range(n_batch):
            zz = zzs[i*batch_size2:(i+1)*batch_size2]
            smis = smiss[i*batch_size2:(i+1)*batch_size2]

            feats = []
            lig_batch = []
            mols = []
            for idx, ss in enumerate(smis):
                mol = Chem.MolFromSmiles(ss)
                AllChem.Compute2DCoords(mol)
                feat = torch.tensor(get_ligand_atom_features(mol)[:,0])
                feats.append(feat)
                lig_batch.append(torch.ones(feat.shape[0], dtype=torch.int64) * idx)
                mols.append(mol)

            feats = torch.cat(feats).to('cuda')
            lig_batch = torch.cat(lig_batch).to('cuda')

            pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
                model, data, config.sample.num_samples,
                batch_size=batch_size2, device=args.device,
                num_steps=config.sample.num_steps,
                pos_only=config.sample.pos_only,
                center_pos_mode=config.sample.center_pos_mode,
                sample_num_atoms=config.sample.sample_num_atoms,
                emb=zz,
                feats=feats,
                lig_batch=lig_batch
            )

            from rdkit.Geometry import Point3D
            idxs = []
            for idx, mol in enumerate(mols):
                try:
                    atom_type = [mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in range(mol.GetNumAtoms())]
                    pred_pos = pred_pos[idx]
                    pred_aromatic = transforms.is_aromatic_from_index(np.array(atom_type), mode='add_aromatic')
                    mol = reconstruct.reconstruct_from_generated(pred_pos, atom_type, pred_aromatic)

                    idxs.append( idx )
                    mols2.append( mol )
                except:
                    continue

        result = {
            'data': data,
            'mols': mols2
        }
        logger.info('Sample done!')

        result_path = args.result_path
        os.makedirs(result_path, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
        torch.save(result, os.path.join(result_path, f'result_{data_id}.pt'))

